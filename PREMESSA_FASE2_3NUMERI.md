# Fase 2: la correzione permessa dopo G1, compiti a 3 numeri (scritta e CONGELATA il 22/10/2026, 22:28)

**Stato: CONGELATA il 22/10/2026 alle 22:28** (`FASE2_3NUMERI_CONGELATA.txt`), prima di qualunque dato a 3 numeri, dopo due revisioni di Codex (`revisioni/codex_3numeri.md`: pronto dopo correzioni, applicate; `revisioni/codex_3numeri_2.md`: pronto). Com'era scritto prima:

**Stato precedente: BOZZA.** Scritta il 22/10/2026 alle 19:35, dopo che q05 si è fermata al cancello G1 e prima di qualunque dato
con compiti a 3 numeri. Si congela con un commit (impronte in `FASE2_3NUMERI_CONGELATA.txt`) dopo una revisione
indipendente del testo e del codice, e prima del lancio.

## Cosa è successo

- Il 22/10 alle 18:38 il notebook A2 di Qwen2.5-0.5B (q05), identico a quello congelato (commit 4a30469), ha preso il
  campione del passo 0: 8 risposte giuste su 416 (1,92%). Il cancello G1 chiede tra il 2% e il 60%.
- La catena si è fermata da sola prima del termometro: niente previsione, niente corse (`risultati/q05_kaggle_A2/`).

## La regola e la decisione

- Regola congelata (`PREMESSA_FASE2.md`; tabella dei cancelli in `PREMESSA.md`): la famiglia si ferma e si riporta così;
  l'unica azione permessa, una volta e decisa da Emanuele, per G1 sotto soglia è "compiti con 3 numeri".
- **Decisione di Emanuele, 22/10/2026 19:34:** "Sì: rifai Qwen2.5-0.5B con i compiti a 3 numeri, una volta, dopo OLMo e
  SmolLM2."
- Il 22/10 anche OLMo-2-1B si è fermato a G1 con 8/416. La verifica (`risultati/fase2_verifica_g1/`) mostra che il
  conteggio uguale a q05 è un caso: il contatore è corretto e i compiti risolti sono diversi.
- **Decisioni di Emanuele, 22/10/2026 22:02, prima di qualunque risultato di SmolLM2:** stessa correzione, una volta, per
  OLMo-2-1B ("Sì, come Qwen"); e per SmolLM2-1.7B se anche lui si ferma a G1 ("Sì, stessa regola").

## Cosa cambia (solo questo, proposta da verificare nella revisione)

- I compiti di Countdown hanno **3 numeri invece di 4**, nello stesso intervallo (da 1 a 50) e con gli stessi bersagli
  possibili (da 10 a 200, scelti tra quelli raggiungibili).
- Vale per tutti i compiti che il termometro genera da `countdown.rl_tasks`: campione del passo 0 e insieme dei
  fallimenti, gradienti del termometro, corse.
- Il codice congelato **non si tocca**: un file nuovo, sul modello di `fase1b.py`, sostituisce soltanto
  `countdown.make_instance` con la versione a 3 numeri (`fase2_3numeri.py`) prima di chiamare le funzioni congelate di `termometro.py`.
- **La partenza resta quella di ogni famiglia** del 20/10 (SFT sui dati a 4 numeri, impronte in `risultati/fase2_partenze.json`):
  rifare l'SFT sarebbe una seconda modifica, che la regola non permette.

## Cosa non cambia

Modello, partenza, semi, cancelli G0-G6 e loro soglie, termometro, test al passo 10 con riserva al passo 5, soglie del
verdetto, analisi (`fase2_corta.py`), rilevatore.

## Come si legge (revisione di Codex, punto 2)

- Con 3 numeri i compiti cambiano difficoltà e distribuzione, e la partenza resta addestrata a 4 numeri.
- Quindi i due regimi si riportano separati: le famiglie a 4 numeri da una parte, quelle a 3 numeri dall'altra.
- Il verdetto di ogni famiglia resta quello congelato (R al passo 10, riserva al 5). Non si fa una graduatoria comune
  tra famiglie con compiti diversi, e nei limiti si scrive che il regime a 3 numeri è un compito diverso.
- Le uscite a 3 numeri stanno in cartelle `<famiglia>_3n`, separate da quelle a 4 numeri, e il notebook B2 controlla il
  segno `TRE_NUMERI.txt` scritto da A2.

## Cosa succede dopo

- Se una famiglia supera i cancelli, entra nel confronto della fase 2 come le altre famiglie, con la nota "compiti a 3 numeri".
- Se si ferma di nuovo a un cancello, quella famiglia è fuori dalla fase 2 e si riporta così: la correzione si usa una volta sola.
- Ordine: dopo il primo tentativo di SmolLM2-1.7B; prima q05, poi olmo1, poi smol17 se serve (quota di Kaggle).

## Da decidere nella revisione, prima del congelamento

- Se la partenza a 4 numeri con compiti a 3 numeri è la lettura giusta di "compiti con 3 numeri" (la nostra proposta),
  oppure se la regola intendeva anche l'SFT.
- Che il file nuovo cambi solo il numero dei numeri e che i semi diano compiti diversi da quelli già visti.
