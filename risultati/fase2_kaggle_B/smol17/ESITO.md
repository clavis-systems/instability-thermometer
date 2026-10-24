# Fase 2, SmolLM2-1.7B: esito delle corse (23/10/2026)

- **Corse:** notebook B2 `termometro-fase-2-smol17-corse` (copia D2 senza onnx), partito il 23/10 alle 16:10, finito
  alle 19:30 UTC (segno `FASE2_smol17_FINITA.txt`), 5 h 21 min. 12 corse su 12 (`run_s101` ... `run_s112`), 20 passi.
- **Analisi congelata** (`fase2_corta.py analyze`, regola H1 di `analisi_fase1.verdict`: PROMOSSO se 0,5 <= R <= 2,
  INDIZIO se 0,33 <= R <= 3, altrimenti BOCCIATO), calcolata dal notebook e rifatta in locale il 23/10 alle 21:45 sugli
  stessi file: risultato identico fino all'ultima cifra.
- **Verdetto al passo 10: PROMOSSO.** R = 1,978 (dispersione misurata di D_seq 0,0523 contro prevista 0,0264);
  intervallo al 95% di R: 1,40-3,37; ESS mediano 181, minimo 112 (sopra la soglia di 100, quindi vale il passo 10).
- Passo 5: R = 2,07 (INDIZIO); passo 20, descrittivo: R = 1,86. Sensibilità al pavimento 1e-5: stesso verdetto. H6: R = 1,80,
  PROMOSSO.

**Come si legge.** Per la regola congelata la previsione regge: sta entro un fattore 2. Ma è al limite: la dispersione
vera è quasi il doppio della prevista, a tutti e tre i passi (R tra 1,86 e 2,07), e l'intervallo arriva oltre 2. Il
termometro sembra sottostimare di circa due volte. Sono 12 corse di una sola famiglia: è un indizio, non un risultato.
Le altre due famiglie (Qwen2.5-0.5B e OLMo-2-1B) si sono fermate al cancello G1; Qwen a 3 numeri è partito il 23/10 alle
21:37 (`termometro-fase-2-q05-3n-previsione`).

Gli adattatori LoRA scaricati (2,7 GB) restano fuori da git; in git vanno il registro, l'analisi, le corse e questo
esito.
