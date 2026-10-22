#!/usr/bin/env bash
# Catena della fase 2 del termometro: la STESSA procedura della fase 1 (stesso codice congelato, stesse soglie, stessi
# semi) su altre tre famiglie, una dopo l'altra: Qwen2.5-0.5B-Instruct, OLMo-2-0425-1B-Instruct, SmolLM2-1.7B-Instruct.
# Parte dopo il file-segnale della fase 1 (risultati/q15/FASE1_FINITA.txt) e con il lucchetto libero.
# Se un cancello di una famiglia fallisce, quella famiglia si ferma (CANCELLO_FALLITO_<nome>.txt) e la catena passa
# alla successiva: l'eventuale unica azione permessa (come in fase 1) la decide Emanuele dopo, non la catena.
set -u
cd "$(dirname "$0")"
PY="C:/Users/emari/Desktop/MIA IA LOCALE/registro-env/Scripts/python.exe"
M="C:/Users/emari/Desktop/MIA IA LOCALE/esterni/modelli"
LOCK="C:/Users/emari/Desktop/MIA IA LOCALE/strumenti/risultati/.gpu_lock"
GIT=(git -c "safe.directory=C:/Users/emari/Desktop/MIA IA LOCALE")
echo "inizio $(date '+%d/%m %H:%M')"
[ -f CONGELATA.txt ] || { echo "premessa non congelata: mi fermo"; exit 1; }
FREEZE=$("${GIT[@]}" log -n1 --format=%H -- CONGELATA.txt)
if ! "${GIT[@]}" diff --quiet "$FREEZE" -- ./*.py ./*.sh sft_countdown.jsonl PREMESSA.md PREMESSA_FASE2.md; then
  echo "il codice è cambiato dopo il congelamento ($FREEZE): mi fermo"; exit 1
fi
until [ -f risultati/q15/FASE1_FINITA.txt ] && [ ! -f "$LOCK" ]; do
  if [ -f "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 28800 ]; then
    echo "lucchetto più vecchio di 8 ore: non lo prendo, mi fermo"; cat "$LOCK"; exit 1
  fi
  sleep 60
done
sleep 120
[ -f "$LOCK" ] && { echo "lucchetto preso da altri: mi fermo"; exit 1; }
TOKEN="Claude | termometro fase 2 (catena_fase2.sh) | $(date '+%d/%m %H:%M') | $$"
printf '%s\n' "$TOKEN" > "$LOCK"
libera() { [ -f "$LOCK" ] && grep -qF "$TOKEN" "$LOCK" && rm -f "$LOCK"; }
trap libera EXIT

for fam in "q05:Qwen2.5-0.5B-Instruct" "olmo1:OLMo-2-0425-1B-Instruct" "smol17:SmolLM2-1.7B-Instruct"; do
  nome="${fam%%:*}"; base="$M/${fam#*:}"; OUT="risultati/$nome"
  mkdir -p "$OUT/log"
  echo "== $nome ($(date '+%H:%M'))"
  [ -f "$OUT/sft/adapter_config.json" ] || "$PY" -X utf8 termometro.py sft "$base" "$OUT/sft" > "$OUT/log/sft.log" 2>&1
  [ -f "$OUT/pool.json" ] || "$PY" -X utf8 termometro.py pool "$OUT/sft" "$OUT/pool.json" 7 > "$OUT/log/pool.log" 2>&1
  echo "$(grep '^pool:' "$OUT/log/pool.log" | cut -c1-300)"
  if ! "$PY" -X utf8 termometro.py gates_pool "$OUT/pool.json" "$OUT/gates_pool.json" > "$OUT/log/gates_pool.log" 2>&1; then
    cp "$OUT/gates_pool.json" "CANCELLO_FALLITO_$nome.txt" 2>/dev/null || echo "gates_pool non calcolati" > "CANCELLO_FALLITO_$nome.txt"
    echo "$nome: CANCELLO FALLITO (campione), passo alla famiglia successiva"; continue
  fi
  [ -f "$OUT/thermo_s1.json" ] || "$PY" -X utf8 termometro.py thermo "$OUT/sft" "$OUT/pool.json" "$OUT" 1 50 1 > "$OUT/log/thermo_s1.log" 2>&1
  [ -f "$OUT/thermo_s2.json" ] || "$PY" -X utf8 termometro.py thermo "$OUT/sft" "$OUT/pool.json" "$OUT" 2 25 0 > "$OUT/log/thermo_s2.log" 2>&1
  if ! "$PY" -X utf8 termometro.py gates "$OUT/pool.json" "$OUT/thermo_s1.json" "$OUT/thermo_s2.json" "$OUT/gates.json" > "$OUT/log/gates.log" 2>&1; then
    cp "$OUT/gates.json" "CANCELLO_FALLITO_$nome.txt" 2>/dev/null || echo "gates non calcolati" > "CANCELLO_FALLITO_$nome.txt"
    echo "$nome: CANCELLO FALLITO, passo alla famiglia successiva"; continue
  fi
  PREV=("$OUT/thermo_s1.json" "$OUT/thermo_s2.json" "$OUT/gates.json" "$OUT/gates_pool.json" "$OUT/pool.json"
        "$OUT/previsione_sha256.txt" "$OUT/log/sft.log" "$OUT/log/pool.log" "$OUT/log/thermo_s1.log" "$OUT/log/thermo_s2.log")
  if "${GIT[@]}" ls-files --error-unmatch "$OUT/previsione_sha256.txt" > /dev/null 2>&1; then
    # ripresa dopo un'interruzione: la previsione è già su git e deve essere identica a quella su disco
    "${GIT[@]}" diff --quiet HEAD -- "${PREV[@]}" || { echo "$nome: previsione su disco diversa da quella su git: mi fermo"; exit 1; }
    ( cd "$OUT" && grep -v "^codice congelato" previsione_sha256.txt | sha256sum -c --quiet - ) || { echo "$nome: impronte della previsione diverse: mi fermo"; exit 1; }
    echo "$nome: previsione già su git (ripresa)"
  else
    ( cd "$OUT" && sha256sum thermo_s1.json thermo_s2.json gates.json gates_pool.json pool.json vettori_s1.pt gradienti_s1.pt \
        > previsione_sha256.txt && echo "codice congelato: $FREEZE" >> previsione_sha256.txt )
    "${GIT[@]}" add -f "${PREV[@]}" || { echo "git add fallito: mi fermo"; exit 1; }
    "${GIT[@]}" commit -q -m "[claude] termometro fase 2 ($nome): previsione congelata prima delle corse" \
      -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "${PREV[@]}" || { echo "commit fallito: mi fermo"; exit 1; }
  fi
  for s in 101 102 103 104 105 106 107 108 109 110 111 112; do
    f="$OUT/run_s$s.json"
    if [ ! -f "$f" ] || ! "$PY" -c "import json,sys; sys.exit(0 if len(json.load(open('$f'))) == 41 else 1)"; then
      "$PY" -X utf8 termometro.py run "$OUT/sft" "$OUT/pool.json" "$OUT" "$f" $s 40 > "$OUT/log/run_s$s.log" 2>&1 || \
        "$PY" -X utf8 termometro.py run "$OUT/sft" "$OUT/pool.json" "$OUT" "$f" $s 40 > "$OUT/log/run_s${s}_ripetuta.log" 2>&1
    fi
  done
  date '+%d/%m %H:%M' > "$OUT/FASE_FINITA.txt"
  echo "$nome: corse finite $(date '+%H:%M')"
done
echo "fine $(date '+%d/%m %H:%M')"
