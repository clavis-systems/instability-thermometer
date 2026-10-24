# Fase 2 a 3 numeri, OLMo-2-1B: fermo al cancello G1 (23/10/2026)

- Notebook `termometro-fase-2-olmo1-3n-previsione` v1, lanciato il 23/10 alle 22:17 con il notebook congelato
  `fase2_A_olmo1_3n.ipynb` (impronta e57b6a9d..., uguale a `FASE2_3NUMERI_CONGELATA.txt`). Controlli d'avvio passati
  ("fase2_3numeri.py identico al congelamento", "codice identico al congelamento: 5 file").
- **Cancello del campione:** G0 sì, G3 sì (disclosure del campione 55%), **G1 no: tasso di soluzione 0,0** su 416
  tentativi (52 compiti); il notebook si è fermato come previsto. Registro: `kaggle_kernels_logs.txt`.
- Controllo indipendente delle risposte: non fatto per OLMo (uscite non scaricate); per Qwen a 3 numeri lo stesso zero
  è stato confermato (`risultati/q05_3n_kaggle_A2/ESITO.md`).
- **Conseguenza per la premessa:** la correzione a 3 numeri era permessa una volta; OLMo-2-1B esce dalla fase 2.
- **Come si legge:** con 4 numeri entrambi i modelli risolvevano l'1,9% (8 su 416); con 3 numeri lo 0%. Per modelli
  che tirano a indovinare, 3 numeri danno meno combinazioni che arrivano al bersaglio: l'idea della premessa che 3
  numeri fossero più facili era sbagliata per modelli di questa taglia. Della fase 2 resta valutabile una sola famiglia,
  SmolLM2-1.7B (PROMOSSO al limite, R = 1,98).
