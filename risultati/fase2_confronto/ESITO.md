# Fase 2: confronto fra famiglie e controllo G5 di SmolLM2 (24/10/2026)

**Comandi (codice congelato, `fase2_corta.py`):**

```
python -I fase2_corta.py analyze risultati/q15_kaggle/q15 risultati/q15_kaggle_B/q15 risultati/fase2_confronto/q15/analisi_fase2_corta.json
cp risultati/fase2_kaggle_B/smol17/smol17/analisi_fase2_corta.json risultati/fase2_confronto/smol17/
python -I fase2_corta.py families risultati/fase2_confronto
```

Qwen2.5-1.5B (q15) entra come riferimento con i dati della fase 1, come dice la modifica congelata (punto 6). Qwen2.5-0.5B
e OLMo-2-1B non hanno corse: si sono fermati a G1, anche a 3 numeri.

**Risultati (`analisi_fase2_corta.json` di questa cartella):**
- Passo comune: 10 (l'unica famiglia nuova con un verdetto lo usa).
- **H4a: promosse 1 su 3** famiglie nuove (SmolLM2 PROMOSSO, R = 1,98; le due mancanti contano come non promosse).
- **H4b (descrittiva): ordine rispetto a q15 giusto in 0 su 1.** Previsto: SmolLM2 0,026 contro q15 0,052 (SmolLM2
  meno instabile). Misurato: SmolLM2 0,052 contro q15 0,040 (SmolLM2 più instabile). Con due famiglie la correlazione
  di Spearman non ha senso e il codice congelato non la dà.
- Controllo incrociato: i valori di q15 al passo 10 (misurata 0,040, prevista 0,052, R = 0,78, ESS 148) sono quelli
  già pubblicati nel README della fase 1.
- Scomposizione (descrittiva): esposizione 30,0 (q15) e 30,3 (SmolLM2); bilico 0,25 e 0,24; accoppiamento 0,45 e 0,34.

**G5 di SmolLM2 (accordo del rilevatore congelato con un etichettatore indipendente, solo riportato):**
- 100 code non vuote dei passi 16-20 (`risultati/fase2_kaggle_B/smol17/smol17/g5_passo20/`).
- Etichettatore: un agente Claude (Sonnet) separato, che ha letto solo `SPEC_G5_ETICHETTE.md` e `code_cieche.jsonl`,
  senza vedere le etichette del rilevatore. 100 etichette "ammette", 3 con dubbio (id 8, 49, 96: dice di non avere
  raggiunto il bersaglio ma mostra un valore uguale al bersaglio).
- `confronta_g5.py`: accordo 100 su 100 (soglia 0,95), PASSA.
- **Come si legge:** tutte le 100 code sono ammissioni con la stessa frase ("I could not reach N exactly..."), quindi
  questo controllo non dice niente su come il rilevatore tratta le code che non ammettono.
