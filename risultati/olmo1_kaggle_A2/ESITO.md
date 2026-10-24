# Fase 2, OLMo-2-1B (olmo1): fermata al cancello G1 (22/10/2026)

- Notebook `emanuelerizzan/termometro-fase-2-olmo1-previsione`, versione 1, lanciato il 22/10 ~19:20 con il notebook
  congelato (`kaggle/push_A2_olmo1/`).
- Campione del passo 0: 416 risposte su 52 compiti, riuscita 8/416 = 1,92%. **G1 non superato**; G0 e G3 superati.
- **DA VERIFICARE PRIMA DI QUALSIASI ALTRA CORSA:** q05 aveva lo stesso identico conteggio (8/416) e la stessa quota di
  lotti informativi (0,4615 = 24/52), pur con ammissione e chiusura diverse. Può essere un caso o un difetto nel conteggio
  delle risposte giuste: controllare quali risposte sono contate giuste nelle due famiglie (servono i file del campione
  su Kaggle) prima di lanciare SmolLM2 o la correzione a 3 numeri.
- **Verifica del 22/10, 21:40 (`risultati/fase2_verifica_g1/`): il conteggio identico è un caso, non un difetto.**
  In entrambe le famiglie i 52 compiti sono quelli di `rl_tasks(7)` nell'ordine giusto; le 800 risposte del pool,
  rivalutate con `countdown.reward` congelato, sono tutte sbagliate; le risposte mancanti per compito (riuscite + non
  terminate + tagliate dal tetto di 400) cadono su compiti in gran parte diversi (in comune 4, solo q05 7, solo olmo1 5).
- Regola congelata: la famiglia si ferma; unica azione permessa (una volta, decisa da Emanuele): compiti a 3 numeri.
- Decisione di Emanuele (22/10/2026, 22:01): compiti a 3 numeri, una volta, come q05 (`PREMESSA_FASE2_3NUMERI.md`).
