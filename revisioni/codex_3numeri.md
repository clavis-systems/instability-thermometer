**PRONTO DOPO CORREZIONI**

1. **Bloccante — rischio di mescolare risultati a 4 e 3 numeri.** B2 cerca in `q05/` i risultati A2 e importa anche `run_s*.json` completi, senza verificare che siano stati prodotti con compiti a 3 numeri. Le cartelle d’uscita hanno gli stessi nomi nei notebook vecchi e nuovi. Separare le uscite (per esempio `q05_3n`) e far controllare a B2 una prova di provenienza dell’A2 a 3 numeri; escludere gli output precedenti a 4 numeri.

2. **Grave — confronto tra famiglie da qualificare.** La modifica cambia la difficoltà e la distribuzione dei compiti; la partenza SFT resta a 4 numeri. Il confronto q05–olmo1 è leggibile come confronto su compiti a 3 numeri con partenza addestrata a 4, ma le famiglie rimaste a 4 numeri non sono direttamente confrontabili sullo stesso esito. Dichiararlo nei limiti e riportare separatamente i due regimi; niente graduatoria comune come se il compito fosse identico.

3. **Correzione documentale — D2.** La descrizione in `DEVIAZIONI_FASE2.md` riporta l’hash di `model.safetensors` troncato/corrotto (`f55217be…`); i notebook D2 contengono invece quello completo. Correggere il documento e specificare che la verifica attestata riguarda quell’impronta e il filtro `onnx/*`.

4. **Bozza e wrapper.** La lettura “3 numeri nei compiti, SFT invariato a 4” è coerente con la regola; la bozza dichiara bene i risultati G1 già osservati e che SmolLM2 si era fermato prima del campione. Il wrapper sostituisce `countdown.make_instance`, usa `TERMO_CODICE` e avvia `termometro.py` via `runpy`; i comandi previsti non rigenerano l’SFT. La sostituzione globale farebbe però generare dati `sft_data` a 3 numeri se quel generatore fosse invocato.

**Non verificato:** esecuzione dei notebook su Kaggle e confronto automatizzato, cella per cella, con tutte le sei versioni congelate; revisione statica rileva kernel `python3` e trasformazioni attese, ma non prova il flusso runtime o l’associazione effettiva degli input/output.