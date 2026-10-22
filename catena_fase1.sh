#!/usr/bin/env bash
# Catena della fase 1 del termometro (Qwen2.5-1.5B-Instruct), da lanciare staccata con sveglio.py.
# 1. Parte solo se la premessa è congelata (CONGELATA.txt) e il codice è IDENTICO a quello del commit di congelamento.
# 2. Aspetta che il registro della verità abbia finito (analisi.json presente) e che il lucchetto della GPU sia
#    libero; un lucchetto più vecchio di 8 ore non viene preso: la catena si ferma e lo scrive.
# 3. SFT -> campione del passo 0 -> cancelli G0, G1, G3 subito -> termometro (seme 1 completo, seme 2 di controllo)
#    -> tutti i cancelli. Se un cancello fallisce si ferma (CANCELLO_FALLITO.txt) senza correggere niente da sola.
# 4. Mette la previsione su git (sha256 + commit dei soli file della previsione) PRIMA della prima corsa.
# 5. 12 corse (semi 101-112, 40 passi), poi FASE1_FINITA.txt. L'analisi la lancia Claude, non la catena.
# Si può rilanciare: salta i passi che hanno già un risultato.
set -u
cd "$(dirname "$0")"
PY="C:/Users/emari/Desktop/MIA IA LOCALE/registro-env/Scripts/python.exe"
BASE="C:/Users/emari/Desktop/MIA IA LOCALE/esterni/modelli/Qwen2.5-1.5B-Instruct"
REG="../registro_verita"
LOCK="C:/Users/emari/Desktop/MIA IA LOCALE/strumenti/risultati/.gpu_lock"
GIT=(git -c "safe.directory=C:/Users/emari/Desktop/MIA IA LOCALE")
OUT="risultati/q15"
mkdir -p "$OUT/log"
echo "inizio $(date '+%d/%m %H:%M')"
[ -f CONGELATA.txt ] || { echo "premessa non congelata: mi fermo"; exit 1; }
FREEZE=$("${GIT[@]}" log -n1 --format=%H -- CONGELATA.txt)
[ -n "$FREEZE" ] || { echo "CONGELATA.txt non è in un commit: mi fermo"; exit 1; }
if ! "${GIT[@]}" diff --quiet "$FREEZE" -- ./*.py ./*.sh sft_countdown.jsonl PREMESSA.md PREMESSA_FASE2.md; then
  echo "il codice è cambiato dopo il congelamento ($FREEZE): mi fermo"; exit 1
fi
echo "codice identico al congelamento $FREEZE"

until [ -f "$REG/analisi.json" ] && [ ! -f "$LOCK" ]; do
  if [ -f "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 28800 ]; then
    echo "lucchetto più vecchio di 8 ore: non lo prendo, mi fermo"; cat "$LOCK"; exit 1
  fi
  sleep 60
done
sleep 120
[ -f "$LOCK" ] && { echo "lucchetto preso da altri: mi fermo"; exit 1; }
TOKEN="Claude | termometro fase 1 (catena_fase1.sh) | $(date '+%d/%m %H:%M') | $$"
printf '%s\n' "$TOKEN" > "$LOCK"
libera() { [ -f "$LOCK" ] && grep -qF "$TOKEN" "$LOCK" && rm -f "$LOCK"; }
trap libera EXIT
echo "registro finito, lucchetto preso $(date '+%H:%M')"

[ -f "$OUT/sft/adapter_config.json" ] || "$PY" -X utf8 termometro.py sft "$BASE" "$OUT/sft" > "$OUT/log/sft.log" 2>&1
echo "sft: $(tail -1 "$OUT/log/sft.log" | cut -c1-150)"
[ -f "$OUT/pool.json" ] || "$PY" -X utf8 termometro.py pool "$OUT/sft" "$OUT/pool.json" 7 > "$OUT/log/pool.log" 2>&1
echo "$(grep '^pool:' "$OUT/log/pool.log" | cut -c1-400)"
if ! "$PY" -X utf8 termometro.py gates_pool "$OUT/pool.json" "$OUT/gates_pool.json" > "$OUT/log/gates_pool.log" 2>&1; then
  cp "$OUT/gates_pool.json" CANCELLO_FALLITO.txt 2>/dev/null || echo "gates_pool non calcolati" > CANCELLO_FALLITO.txt
  echo "CANCELLO FALLITO (campione): mi fermo $(date '+%H:%M')"; cat "$OUT/log/gates_pool.log" | head -n 20; exit 3
fi
[ -f "$OUT/thermo_s1.json" ] || "$PY" -X utf8 termometro.py thermo "$OUT/sft" "$OUT/pool.json" "$OUT" 1 50 1 > "$OUT/log/thermo_s1.log" 2>&1
[ -f "$OUT/thermo_s2.json" ] || "$PY" -X utf8 termometro.py thermo "$OUT/sft" "$OUT/pool.json" "$OUT" 2 25 0 > "$OUT/log/thermo_s2.log" 2>&1
grep '^PREVISIONE' "$OUT/log/thermo_s1.log" "$OUT/log/thermo_s2.log" | cut -c1-300
if ! "$PY" -X utf8 termometro.py gates "$OUT/pool.json" "$OUT/thermo_s1.json" "$OUT/thermo_s2.json" "$OUT/gates.json" > "$OUT/log/gates.log" 2>&1; then
  cp "$OUT/gates.json" CANCELLO_FALLITO.txt 2>/dev/null || echo "gates non calcolati" > CANCELLO_FALLITO.txt
  echo "CANCELLO FALLITO: mi fermo $(date '+%H:%M')"; cat "$OUT/log/gates.log" | head -n 30; exit 3
fi
echo "cancelli superati"

# la previsione su git prima della prima corsa (solo questi file; i log sono ignorati da .gitignore, quindi -f)
PREV=("$OUT/thermo_s1.json" "$OUT/thermo_s2.json" "$OUT/gates.json" "$OUT/gates_pool.json" "$OUT/pool.json"
      "$OUT/previsione_sha256.txt" "$OUT/log/sft.log" "$OUT/log/pool.log" "$OUT/log/thermo_s1.log" "$OUT/log/thermo_s2.log")
if "${GIT[@]}" ls-files --error-unmatch "$OUT/previsione_sha256.txt" > /dev/null 2>&1; then
  # ripresa dopo un'interruzione: la previsione è già su git e deve essere identica a quella su disco
  "${GIT[@]}" diff --quiet HEAD -- "${PREV[@]}" || { echo "previsione su disco diversa da quella su git: mi fermo"; exit 1; }
  ( cd "$OUT" && grep -v "^codice congelato" previsione_sha256.txt | sha256sum -c --quiet - ) || { echo "impronte della previsione diverse: mi fermo"; exit 1; }
  echo "previsione già su git (ripresa): $("${GIT[@]}" log -n1 --oneline -- "$OUT/previsione_sha256.txt")"
else
  ( cd "$OUT" && sha256sum thermo_s1.json thermo_s2.json gates.json gates_pool.json pool.json vettori_s1.pt gradienti_s1.pt \
      > previsione_sha256.txt && echo "codice congelato: $FREEZE" >> previsione_sha256.txt )
  "${GIT[@]}" add -f "${PREV[@]}" || { echo "git add della previsione fallito: mi fermo"; exit 1; }
  "${GIT[@]}" commit -q -m "[claude] termometro fase 1: previsione congelata prima delle corse (catena notturna)" \
    -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "${PREV[@]}" || { echo "commit della previsione fallito: mi fermo"; exit 1; }
  echo "previsione su git: $("${GIT[@]}" log --oneline -1)"
fi

for s in 101 102 103 104 105 106 107 108 109 110 111 112; do
  f="$OUT/run_s$s.json"
  if [ ! -f "$f" ] || ! "$PY" -c "import json,sys; sys.exit(0 if len(json.load(open('$f'))) == 41 else 1)"; then
    "$PY" -X utf8 termometro.py run "$OUT/sft" "$OUT/pool.json" "$OUT" "$f" $s 40 > "$OUT/log/run_s$s.log" 2>&1 || \
      "$PY" -X utf8 termometro.py run "$OUT/sft" "$OUT/pool.json" "$OUT" "$f" $s 40 > "$OUT/log/run_s${s}_ripetuta.log" 2>&1
  fi
  echo "corsa $s: $(tail -1 "$OUT/log/run_s$s.log" | cut -c1-120) $(date '+%H:%M')"
done
date '+%d/%m %H:%M' > "$OUT/FASE1_FINITA.txt"
echo "fine $(date '+%d/%m %H:%M')"
