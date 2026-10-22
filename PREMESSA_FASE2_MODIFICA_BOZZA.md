# Modifica della fase 2: il test al passo 10 (scritta il 21/10/2026, CONGELATA il 22/10/2026)

**Stato: CONGELATA il 22/10/2026** (`FASE2_CONGELATA.txt`), dopo il risultato della 1b e prima di qualunque corsa della fase 2.
- Il nome del file resta quello della bozza perché è citato nel codice rivisto (`fase2_corta.py`, `kaggle/crea_notebook_fase2.py`).
- Le tre condizioni scritte sotto sono soddisfatte: OK di Emanuele il 22/10 dopo la 1b ("Sì, congela"), revisioni di Codex del piano e del codice (correzioni fatte), commit di congelamento.

Com'era scritto prima del congelamento:
- È scritta dopo aver visto i risultati della fase 1 (21/10), e lo dichiariamo.
- È scritta prima di qualunque dato della fase 2. Della fase 2 esistono solo le tre partenze SFT, addestrate il 20/10
  con le impronte in `risultati/fase2_partenze.json`: niente termometro, niente corse.
- Per congelarla servono tre cose: l'OK di Emanuele (dopo il risultato della 1b), una revisione indipendente e un
  commit prima di qualunque corsa della fase 2 su Kaggle.
- Fino ad allora vale `PREMESSA_FASE2.md` com'è.
- **Revisione di Codex (21/10): PRONTA DOPO CORREZIONI.** Testo in `revisioni/codex_modifica_fase2.md`.
  - Fatte le correzioni su regola completa, passo comune, costi, potenza e alternativa B.
  - Sul punto "confermativo o esplorativo" rispondiamo sotto.
  - **Decisione di Emanuele (21/10/2026): confermativo sulle 3 famiglie nuove**, con le ragioni scritte sotto.
    L'obiezione di Codex resta riportata.

## In parole semplici

Nella fase 1 lo strumento di lettura (la stima per importanza su 400 fallimenti fissi) si è rotto dopo circa 20
passi. Al passo 40, dove era fissato il test, l'ESS mediana era 15: verdetto NON CONCLUSIVO. La fase 2 congelata
ripete la stessa prova al passo 40 su tre famiglie nuove, quindi molto probabilmente darebbe tre NON CONCLUSIVO, al
costo di circa 12 ore di Kaggle per famiglia.

Questa modifica sposta il test al passo 10, dove nella fase 1 lo strumento funzionava ancora. Così diventa un test
cieco su tre famiglie nuove di una cosa che nella fase 1 abbiamo visto solo dopo: la previsione del passo 0 è buona
finché lo strumento regge.

## Cosa abbiamo visto nella fase 1 (Qwen2.5-1.5B, 12 corse)

| passo | SD misurata | SD prevista | R | ESS mediana | ESS minima |
|---|---|---|---|---|---|
| 5 | 0,024 | 0,034 | 0,69 | 228 | 107 |
| 10 | 0,040 | 0,052 | 0,78 | 148 | 35 |
| 20 | 0,117 | 0,074 | 1,57 | 44 | 5 |
| 40 | 0,206 | 0,104 | 1,99 | 15 | 2,5 |

Il test pre-registrato era solo al passo 40. I valori ai passi 5, 10 e 20 sono stati visti dopo (ricalcolo
indipendente, `risultati/q15_kaggle_B/controllo_ricalcolo/RAPPORTO.md`).

## Cosa cambia (solo questo)

1. **Il passo del test.** H4a (il verdetto H1 per famiglia) si dà al passo **T = 10** invece che al 40.
   - La previsione è `replay.pred_SD.lin_seq["10"]` del termometro congelato.
   - Il termometro non cambia e il replay non si tronca né si ricalcola: va comunque fino a 40 (`SCHED[-1]`), con lo
     stesso seme e le stesse estrazioni. Quindi il valore al passo 10 è esattamente quello che calcola il codice
     congelato.
2. **La regola per famiglia, tutta scritta ora.**
   - Se al passo 10 l'ESS mediana è almeno 100, il verdetto è quello di H1 al passo 10.
   - Se al passo 10 è sotto 100 e al passo 5 è almeno 100, il verdetto è quello di H1 al passo 5.
   - Altrimenti è NON CONCLUSIVO.

   H1 è la stessa regola della fase 1: R = SD misurata (pavimento sottratto) / SD prevista allo stesso passo,
   PROMOSSO con 0,5 ≤ R ≤ 2, INDIZIO con 0,33 ≤ R ≤ 3, altrimenti BOCCIATO.

   Il rapporto dice sempre quale passo è stato usato, anche quando il risultato è NON CONCLUSIVO. Per tutte le
   famiglie si riportano comunque R ed ESS ai passi 5, 10 e 20, qualunque sia il passo del verdetto.

   Quando non c'è un passo per il verdetto, H6, la sensibilità e i funzionali lineari si riportano al passo 10,
   solo come descrizione.
3. **Le corse durano 20 passi invece di 40.**
   - Si usa l'argomento del codice congelato stesso: `termometro.py run ... <seme> 20`.
   - Ne segue che D_seq e D_false si misurano ai passi 5, 10 e 20, D_tf ai passi 1, 2, 5, 10, 15 e 20, i checkpoint
     ai passi 10 e 20, e le code per l'accordo del rilevatore vengono dai passi 16-20.
   - Nessuna riga di `termometro.py` cambia.
   - I compiti sono gli stessi dei primi 20 passi delle corse da 40: `countdown.rl_tasks` li genera uno dopo l'altro
     dallo stesso seme (20.000 + seme della corsa).
4. **Il passo 20 è solo descrittivo:** si riportano R(20) e l'ESS, per vedere se lo strumento si rompe allo stesso
   punto anche nelle altre famiglie.
5. **La mediana dell'ESS** è la mediana vera (`statistics.median`, come nella 1b), non `sorted[n//2]` come nel codice
   della fase 1. Con 12 corse le due possono differire di poco: lo dichiariamo.
6. **H4b e la scomposizione** (esposizione, bilico, accoppiamento) usano gli stessi calcoli.
   - Il confronto tra famiglie si fa a un **passo comune**: il 10 se tutte le famiglie con un verdetto lo usano,
     altrimenti il 5 per tutte.
   - Anche Qwen2.5-1.5B entra al passo comune, con i dati della fase 1, che ha D_seq ai passi 5 e 10.
   - Nell'ordine entrano solo le famiglie con ESS mediana almeno 100 al passo comune. Le altre si elencano come "non
     ordinate".
   - Il conteggio delle promosse è sulle 3 famiglie nuove: una famiglia mancante conta come non promossa.
   - H6 e il pavimento si valutano al passo del verdetto di ogni famiglia. Il pavimento è quello del termometro, non
     dipende dal passo, e al passo 10 è trascurabile: 1,6e-6 contro una SD di circa 0,04 nella fase 1.
7. **L'analisi** è in un file nuovo, `fase2_corta.py`. Riusa le funzioni congelate di `analisi_fase1.py` (`sd`,
   `sd_ci`, `verdict`) e `analisi_fase2.analyse`, ed è provata su dati finti prima dei dati.

## Cosa non cambia

- Codice del termometro, partenze, semi delle corse (101-112), cancelli G0-G6.
- Soglie: PROMOSSO con 0,5 ≤ R ≤ 2, INDIZIO con 0,33 ≤ R ≤ 3, altrimenti BOCCIATO; NON CONCLUSIVO se l'ESS mediana
  è sotto 100.
- Pavimento del rumore e sensibilità a 1e-5.
- H6 (il secondo comportamento) con la sua zona di parità, al passo del test.
- L'accordo del rilevatore con un etichettatore indipendente su 100 code non vuote, solo riportato.
- Piattaforma: Kaggle, 2 T4, float32.

## Cosa è cieco e cosa no

- **Qwen2.5-1.5B non è cieca:** il suo R al passo 10 (0,78) l'abbiamo già visto. Resta il riferimento per l'ordine
  delle famiglie, ma non conta nel verdetto.
- **H4a si riporta come "promosse tra le 3 famiglie nuove".**
- **Qwen2.5-0.5B e SmolLM2-1.7B** sono le sole del tutto cieche: per loro non esiste nessun numero, né nostro né
  dell'articolo.
- **OLMo-2-1B** è cieca per il nostro test, ma l'articolo ne dà la dispersione a 300 passi (già detto in
  `PREMESSA_FASE2.md`).
- **La scelta del passo 10 non è indipendente dalla fase 1.** L'abbiamo scelto perché lì l'ESS era sopra 100, ma
  sapevamo anche che R lì era buono. Per questo Qwen2.5-1.5B esce dal verdetto.

**Confermativo o esplorativo?** La revisione di Codex (21/10) chiede di chiamarlo esplorativo. Secondo lei soglia e
orizzonte sono stati scelti dopo aver visto l'andamento, e la riserva aggiunge un secondo percorso di analisi.

La nostra risposta, decisa da Emanuele il 21/10:
- Le **soglie** non cambiano: sono quelle della fase 1, congelate il 20/10.
- È cambiato l'**orizzonte**, scelto guardando la fase 1. Quindi l'ipotesi "la previsione regge finché lo strumento
  regge" nasce dai dati di Qwen2.5-1.5B.
- La si mette alla prova su tre famiglie i cui dati non esistono ancora, con regole (riserva compresa) tutte scritte
  prima.
- Questo è un test confermativo dell'ipotesi sulle famiglie nuove, non su Qwen2.5-1.5B. Lo scriviamo così, e non
  sommiamo la fase 1 come prova indipendente.

## Cosa può dire e cosa no

- **Può dire** se il termometro del passo 0 prevede la dispersione iniziale tra corse anche in altre famiglie, e se
  ne indovina l'ordine.
- **Non può dire** niente sull'orizzonte lungo: 10 passi sono l'inizio, l'articolo misura a 300. L'orizzonte lungo è
  il compito della 1b.
- **Potenza:** con 12 corse l'intervallo al 95% della SD misurata va da 0,71 a 1,70 volte, a qualunque passo. Come
  nella fase 1, il test smaschera errori di 3 volte o più e non certifica un fattore 2. Non è stato ricalcolato per
  il passo 10.
- **Tre famiglie, tre verdetti:** si riporta il conteggio delle promosse, senza correzione per confronti multipli,
  perché non si combina in un unico test.
- **Rischio dichiarato:** le famiglie più piccole, o con un passo che sposta di più i pesi, possono far crollare l'ESS
  prima. Per questo c'è la riserva al passo 5. Se non basta, il risultato è NON CONCLUSIVO e lo diciamo.

## Costi (stima dai tempi della fase 1, Qwen2.5-1.5B su T4)

| voce | tempo |
|---|---|
| un passo senza misure | 47 s |
| un passo con la sola D_tf | 178 s |
| un passo con D_seq e D_tf | 333 s |
| una corsa da 20 passi (3 passi con D_seq, 3 con sola D_tf, 14 senza misure) | circa 37 minuti |
| 12 corse: 2 GPU, 6 corse per GPU una dopo l'altra | circa 4 ore di orologio (7,4 ore di GPU) |
| termometro e cancelli, da rifare per ogni famiglia (come il notebook A) | circa 3 ore |
| **per famiglia** | **circa 7 ore** |
| **per 3 famiglie** | **circa 21 ore** |

- 21 ore stanno in una settimana di quota Kaggle (30 ore, si azzera il sabato). Nella fase 1 la quota è scesa delle
  ore di sessione: per il notebook B, 12 corse da 40 passi, circa 8,6 ore.
- I tempi sono di Qwen2.5-1.5B. Per le altre famiglie sono un'estrapolazione: la 0,5B dovrebbe essere più veloce,
  la 1,7B un po' più lenta.
- Con corse da 40 passi servirebbero circa 12 ore per famiglia.
- Con corse da 10 passi, circa 5,5 ore, ma si perde il passo 20 descrittivo.
- La prima famiglia sarà Qwen2.5-0.5B, la più piccola, così i tempi veri si vedono subito.

## L'alternativa B (solo se la 1b funziona)

Se la misura senza pesi della 1b (D_samp) dà un verdetto vero, la fase 2 potrebbe usare quella con corse lunghe.
Costa molto di più. Con la stima della regola dell'orizzonte della 1b (150 × 45,45 s + 8 misure da 260 s, cioè circa
2,5 ore per corsa), 12 corse su 2 GPU sono circa 15 ore, più 3 di termometro per famiglia: circa 18 ore per famiglia.
È un'estrapolazione da Qwen2.5-1.5B.
Per tre famiglie fanno circa 54 ore, quasi due settimane di quota. Una via di mezzo: A adesso per tutte e tre, poi B
per una sola famiglia.

## Cosa resta da fare dopo l'OK

1. ~~Scrivere `fase2_corta.py`~~ fatto il 21/10.
   - Prova su dati finti (`fase2_corta.py test`) superata: verdetto al 10, riserva al 5, NON CONCLUSIVO, passo
     comune, Qwen2.5-1.5B fuori dal conteggio.
   - Provato anche sui file veri della fase 1 di Qwen2.5-1.5B, già visti, solo per controllare che legga i file: R
     0,69 / 0,78 / 1,57 ai passi 5 / 10 / 20, uguali al ricalcolo indipendente.
2. ~~Notebook A2 e B2 per famiglia~~ fatti il 21/10.
   - Li genera `kaggle/crea_notebook_fase2.py`, che riusa le celle della fase 1 e controlla che ogni sostituzione
     avvenga esattamente una volta.
   - Rispetto alla fase 1, A2 cambia in 14 righe: famiglia, modello base, impronte, cartella del modello.
   - B2 aggiunge il controllo del codice della fase 2, i 20 passi e l'analisi `fase2_corta.py`.
   - `kaggle/prepara_fase2.py` prepara il dataset del codice e le cartelle di lancio. Il notebook B2 si può
     preparare solo quando la previsione di A2 è già su git.
3. ~~Le partenze su Kaggle~~ ci sono già. Il dataset `termometro-partenze-fase-2` del 20/10 contiene le tre
   partenze, con le stesse dimensioni dei file sul PC; le impronte le controlla il notebook.
4. Revisione indipendente del codice e dei notebook, poi congelamento: commit, impronte e `MANIFEST`.

## Ordine di lancio (dopo il congelamento)

1. `python prepara_fase2.py codice`, poi il caricamento del dataset `termometro-fase2-codice`. Richiede l'OK di
   Emanuele e l'elenco dei file: `fase2_corta.py`, `analisi_fase2.py`, `MANIFEST_FASE2.txt`,
   `dataset-metadata.json`.
2. Per ogni famiglia, prima Qwen2.5-0.5B:
   1. `prepara_fase2.py A <fam>` e il lancio di A2 (circa 3 ore);
   2. `prepara_fase2.py B <fam>`, che scarica e controlla l'uscita di A2;
   3. il commit della previsione su git;
   4. di nuovo `prepara_fase2.py B <fam>`, poi il lancio di B2 (circa 4 ore).
3. Alla fine, sul PC:
   - `fase2_corta.py analyze risultati/q15_kaggle/q15 risultati/q15_kaggle_B/q15 <cartella>/q15/analisi_fase2_corta.json`
     per il riferimento;
   - `fase2_corta.py families <cartella>` per il confronto tra famiglie;
   - poi il controllo indipendente.
