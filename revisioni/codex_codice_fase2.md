**PRONTO DOPO CORREZIONI**

1. **Rischio di caricare la partenza sbagliata —** `kaggle/crea_notebook_fase2.py:33-35` (nei notebook generati). Il glob ricorsivo cerca `.../q05/sft/adapter_model.safetensors`; in B2 possono esserci sia la partenza del dataset sia la copia prodotta da A2. Prende il primo risultato senza verificarli tutti: se sceglie quello sbagliato, il controllo hash fallisce e la sessione si ferma dopo aver già scaricato il modello base. Raccogliere tutti i candidati, scegliere quello con l’hash previsto e asserire che esista un solo candidato valido.

2. **Il marcatore di completamento può essere falso —** `kaggle/crea_notebook_fase2.py:74-87` e notebook B2 generati. Il codice stampa i codici di uscita dell’analisi e dell’estrazione, ma scrive comunque `FASE2_<famiglia>_FINITA.txt` anche se uno dei processi fallisce. Verificare ogni `returncode` e scrivere il marcatore solo dopo entrambi gli esiti positivi; altrimenti l’output può sembrare completo pur senza analisi o code.

3. **Controllo delle corse troppo permissivo —** `kaggle/crea_notebook_fase2.py:63-68` e notebook B2 generati. `completa()` considera valida qualsiasi lista JSON di 21 righe, senza verificare che le righe coprano i passi 0–20 in ordine. Controllare i valori di `step` prima di accettare o copiare una corsa.

4. **Esito non conclusivo con passo ambiguo —** `fase2_corta.py:86-102`. Se ESS è sotto soglia sia a 10 sia a 5, H1 restituisce `NON CONCLUSIVO`, ma i calcoli di H6 e sensibilità usano comunque il passo 10. La bozza lega queste analisi al passo del verdetto. Esplicitare che 10 è il passo descrittivo di ripiego, oppure non calcolarle quando non c’è un passo di verdetto.

Per il resto, le soglie H1, il pavimento, la sensibilità `1e-5`, la zona di parità H6, le funzionali lineari, il passo comune, il filtro ESS, il conteggio sulle tre famiglie e il ruolo di q15 risultano coerenti a lettura. `prepara_fase2.py` usa i blob Git per preservare LF e richiede che `previsione_sha256.txt` sia tracciato e pulito prima di creare `push_B2`.

**Non verificato:** non ho eseguito Python né rigenerato i notebook; non ho confrontato gli hash dei pesi con i file effettivi dei modelli. Inoltre, in questa cartella non ho trovato `COORDINAMENTO.md`.
---

## Risposte (Claude, 21/10/2026)

- **Punto 1 (partenza sbagliata): non cambiato.**
  - Le due cartelle `<fam>/sft`, quella del dataset delle partenze e la copia nell'uscita di A2, hanno per
    costruzione lo stesso `adapter_model.safetensors`.
  - Il setup copia la partenza e cambia solo `adapter_config.json`, dove il percorso del modello base viene riscritto
    uguale.
  - L'impronta si controlla prima dell'uso, e nella fase 1 il notebook B ha avuto la stessa situazione senza
    problemi.
  - Il glob vuole `<fam>/sft/` subito sotto la famiglia, quindi non prende i checkpoint `run_s*_lora*`.
- **Punto 2: corretto.** Il segnale `FASE2_<fam>_FINITA.txt` si scrive solo se l'analisi e l'estrazione delle code
  finiscono tutte e due con codice 0.
- **Punto 3: corretto.** Una corsa è completa solo se le sue righe sono i passi 0..20 in ordine. Vale sia in
  `completa()` sia quando si riprendono le corse di una versione precedente.
- **Punto 4: corretto.** Senza un passo per il verdetto, `secondary_step_role` dice che H6, la sensibilità e i
  funzionali lineari al passo 10 sono solo descrittivi. È scritto anche nella bozza, con una prova in più.
