#!/usr/bin/env bash
# Prove brevi prima del congelamento (nessun dato dell'esperimento). Aspetta la fine del registro e il lucchetto
# libero, poi: SFT della partenza (la stessa che userà la fase 1, seme 0), confronto "a lotti" contro "una alla volta"
# sulla GPU vera, campione pilota (seme 8) per G5 ed estrazione di 100 code non vuote. Poi libera la GPU.
set -u
cd "$(dirname "$0")"
PY="C:/Users/emari/Desktop/MIA IA LOCALE/registro-env/Scripts/python.exe"
BASE="C:/Users/emari/Desktop/MIA IA LOCALE/esterni/modelli/Qwen2.5-1.5B-Instruct"
REG="../registro_verita"
LOCK="C:/Users/emari/Desktop/MIA IA LOCALE/strumenti/risultati/.gpu_lock"
OUT="risultati/q15"
mkdir -p "$OUT/log" risultati/pilota
echo "inizio $(date '+%d/%m %H:%M')"
until [ -f "$REG/analisi.json" ] && [ ! -f "$LOCK" ]; do sleep 60; done
sleep 120
[ -f "$LOCK" ] && { echo "lucchetto preso da altri: mi fermo"; exit 1; }
TOKEN="Claude | termometro prove pre-congelamento | $(date '+%d/%m %H:%M') | $$"
printf '%s\n' "$TOKEN" > "$LOCK"
libera() { [ -f "$LOCK" ] && grep -qF "$TOKEN" "$LOCK" && rm -f "$LOCK"; }
trap libera EXIT
[ -f "$OUT/sft/adapter_config.json" ] || "$PY" -X utf8 termometro.py sft "$BASE" "$OUT/sft" > "$OUT/log/sft.log" 2>&1
echo "sft: $(grep 'peak GPU' "$OUT/log/sft.log" | tail -1) | $(tail -1 "$OUT/log/sft.log" | cut -c1-120)"
"$PY" -X utf8 prova_pre_congelamento.py "$OUT/sft" risultati/pilota/prova_gpu_lotti.json > risultati/pilota/prova_gpu_lotti.log 2>&1
grep '^PROVA' risultati/pilota/prova_gpu_lotti.log || tail -n 5 risultati/pilota/prova_gpu_lotti.log
[ -f risultati/pilota/pool_pilota.json ] || "$PY" -X utf8 termometro.py pool "$OUT/sft" risultati/pilota/pool_pilota.json 8 > risultati/pilota/pool_pilota.log 2>&1
grep '^pool:' risultati/pilota/pool_pilota.log | cut -c1-400
"$PY" -X utf8 termometro.py estrai_code risultati/pilota/pool_pilota.json risultati/pilota/code_cieche.jsonl risultati/pilota/etichette_rilevatore.json 100
date '+%d/%m %H:%M' > risultati/pilota/PRE_CONGELAMENTO_FATTO.txt
echo "fine $(date '+%d/%m %H:%M')"
