# Fase 2 a 3 numeri, Qwen2.5-0.5B: fermo al cancello G1 (23/10/2026)

- Notebook `termometro-fase-2-q05-3n-previsione` v1, lanciato il 23/10 alle 21:37 dopo la fine delle corse di SmolLM2
  (ordine della premessa `PREMESSA_FASE2_3NUMERI.md`), con il notebook congelato `fase2_A_q05_3n.ipynb` (impronta
  38cc9330... uguale a `FASE2_3NUMERI_CONGELATA.txt`). Controlli d'avvio passati: "fase2_3numeri.py identico al
  congelamento", "codice identico al congelamento: 5 file".
- **Cancello del campione:** G0 sì, G3 sì (disclosure del campione 28%), **G1 no: tasso di soluzione 0,0** su 416
  tentativi (52 compiti); il notebook si è fermato come previsto ("CANCELLO FALLITO (campione): mi fermo").
- **Controllo indipendente (Claude, 23/10 22:15, verificatore scritto da zero, espressioni lette con ast senza
  eseguirle):** 0 risposte giuste su 400 di `pool.json`; i 50 compiti distinti sono tutti a 3 numeri e tutti
  risolvibili (ricerca su tutte le combinazioni). Lo zero non viene da un errore del controllo: il modello scrive
  espressioni sbagliate o con numeri ripetuti (esempio: per [21, 10, 26] -> 15 risponde "(21 - 26) * (10 - 21)").
- **Conseguenza per la premessa:** la correzione a 3 numeri era permessa una volta; Qwen2.5-0.5B esce dalla fase 2.
- **Dopo:** OLMo-2-1B a 3 numeri lanciato il 23/10 alle 22:17 (`termometro-fase-2-olmo1-3n-previsione`, notebook
  congelato con impronta e57b6a9d..., uguale al congelamento), decisione di Emanuele del 22/10 22:02.

I pesi scaricati in `base/` (1 GB) restano fuori da git.
