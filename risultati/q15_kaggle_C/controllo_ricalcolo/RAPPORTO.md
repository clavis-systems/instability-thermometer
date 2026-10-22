# Ricalcolo indipendente del verdetto della fase 1b (termometro dell'instabilità)

Revisore indipendente, 22/10/2026. Nessun file del progetto è stato modificato e non ho usato GPU né modelli.

## Verdetto: CONFERMATO CON RISERVE

- **Il calcolo è confermato.** Con codice mio, partendo solo da `PREMESSA_FASE1B.md`, `PREMESSA.md` e dai file grezzi, ottengo gli stessi numeri e gli stessi verdetti dell'analisi ufficiale: D_seq NON CONCLUSIVO a T = 80 e a T = 150 (ESS mediana 13,1 e 12,7 su 400), verdetto dato da D_samp, PROMOSSO in tutti e due i casi, "regge fino a 150", H2b non calcolabile. L'unica differenza numerica (fino al 3,4% su R_samp) viene da una scelta di definizione della media di D_samp e non cambia nessun verdetto.
- **Le riserve riguardano che cosa vuol dire "la previsione regge".** Tutto il verdetto poggia su D_samp (una misura diversa da D_seq) e su 6 corse; D_seq è inutilizzabile già da T = 20. La frase del rapporto ufficiale non lo dice.

## Dove sono i file (nota pratica)

Lo strumento di scrittura mi ha rifiutato di scrivere nella cartella `controllo_ricalcolo` del worktree `robotica-congela-previsione-legge-1-929f74` (appartiene a un altro worktree rispetto alla mia sessione), e non ho aggirato il blocco. Tutti i file sono nella cartella di lavoro di questa sessione:

(una cartella temporanea della sessione principale)

*Nota della sessione principale, 22/10/2026: i file sono stati copiati qui senza modifiche, tranne i percorsi assoluti del PC, sostituiti con percorsi relativi in `ricalcolo.py` e `confronto.py` e con questa descrizione nel rapporto. Gli script, rilanciati dopo la modifica, danno gli stessi numeri.*

- `ricalcolo.py` (il mio script), `miei_numeri.json` (i miei numeri, salvati prima di aprire i file vietati), `confronto.py` e `confronto.json` (confronto con l'ufficiale), questo `RAPPORTO.md`.
- Da copiare in `strumenti/termometro/risultati/q15_kaggle_C/controllo_ricalcolo/` se si vuole tenerli nel progetto. Prima del blocco avevo già creato, con la shell, la cartella vuota `controllo_ricalcolo` in quel worktree: è vuota e non è tracciata da git.
- Lo script legge i dati grezzi dal worktree del progetto in sola lettura (variabile `Q15_DIR` per cambiare percorso).

## Come ho lavorato

1. Ho letto solo le due premesse e la struttura dei JSON. Ho scritto `ricalcolo.py` senza aprire `fase1b.py`, `analisi_fase1b.json`, `RAPPORTO_FASE1B.md` né altri script di analisi.
2. Ho salvato `miei_numeri.json`.
3. Solo dopo ho aperto `analisi_fase1b.json`, `RAPPORTO_FASE1B.md` e `fase1b.py`.

Scelte mie dove la premessa lascia spazio (le ho dichiarate prima di vedere l'ufficiale):
- finestra di D_samp = passi T-19..T (venti passi che finiscono in T); variante T-20..T-1 provata;
- media della finestra = media semplice dei tassi per passo; variante pesata per `n_fail_terminated` provata;
- varianza dei tassi per passo con n-1; variante n provata;
- SD prevista di D_samp = la stessa SD prevista di D_seq (`lin_seq`);
- pavimento = `statistics.stdev` degli 8 valori `floor_values.seq` (risulta 1,6029e-06, uguale al valore salvato).

## Controlli di coerenza sui dati

- D_seq, D_false ed ESS ricalcolati dai `lp_seq` di ogni corsa e passo, con le etichette di `pool.json` (w = exp(lp_t - lp_0)), coincidono con i valori salvati: differenza massima 1,1e-16 su D_seq, 3,3e-16 su D_false, 5,7e-16 (relativa) sull'ESS. I dati sono interni coerenti.
- Regola dell'orizzonte: 150 x 45,45 s + 8 x 260 s = 2,47 h, sotto 3,5 h, quindi orizzonte 150 e prove a T = 80 e 150. Coincide con quanto applicato.
- `fase1b.py` e `PREMESSA_FASE1B.md` hanno gli sha256 elencati in `FASE1B_CONGELATA.txt` (c5a4cdf2… e f0bd9e6c…) e nessuna modifica non committata: l'analisi è stata fatta con il codice congelato.
- La potenza dichiarata (6 corse) è giusta: la ricalcolo con la funzione di ripartizione del chi quadro a 5 gradi di libertà: PROMOSSO 93,9% se la previsione è giusta, 58,2% con errore di 2 volte per eccesso, BOCCIATO 41,6% con errore di 3 volte e 72,9% con errore di 4 volte; intervallo al 95% della SD da 0,624 a 2,453 volte.

## Tabella dei numeri: miei contro ufficiali

| Grandezza | T = 80 mio | T = 80 ufficiale | T = 150 mio | T = 150 ufficiale |
|---|---|---|---|---|
| SD tra corse di D_seq (n-1) | 0,099413 | 0,099413 | 0,0014974 | 0,0014974 |
| SD dopo pavimento (1,6e-06 in quadratura) | 0,099413 | 0,099413 | 0,0014974 | 0,0014974 |
| SD prevista | 0,145316 | 0,145316 | 0,188433 | 0,188433 |
| R = SD / SD prevista (D_seq) | 0,6841 | 0,6841 | 0,00795 | 0,00795 |
| ESS mediana vera (su 400) | 13,147 | 13,147 | 12,650 | 12,650 |
| Verdetto D_seq | NON CONCLUSIVO | NON CONCLUSIVO | NON CONCLUSIVO | NON CONCLUSIVO |
| Rumore di D_samp, varianza | 0,000704017 | 0,000704017 | 0,00117535 | 0,00117535 |
| Rumore di D_samp, deviazione | 0,02653 | (0,02653) | 0,03428 | (0,03428) |
| Metà della SD prevista (soglia del rumore) | 0,0727 | non riportata | 0,0942 | non riportata |
| Regola del rumore | non scatta | non scatta | non scatta | non scatta |
| SD di D_samp dopo il rumore, media pesata | 0,093973 | 0,093973 | 0,280341 | 0,280341 |
| **R_samp, media pesata per n_fail** | **0,6467** | **0,6467** | **1,4877** | **1,4877** |
| R_samp, media semplice dei tassi (mia scelta iniziale) | 0,6247 | - | 1,5284 | - |
| Verdetto D_samp | PROMOSSO | PROMOSSO | PROMOSSO | PROMOSSO |
| **Verdetto usato** | **PROMOSSO (D_samp)** | **PROMOSSO (D_samp)** | **PROMOSSO (D_samp)** | **PROMOSSO (D_samp)** |
| R descrittivo di D_tf | 0,5871 | 0,5871 | 4,0898 | 4,0898 |
| R descrittivo di D_false | 0,7881 | 0,7881 | 0,01198 | 0,01198 |
| Riuscita, media nella finestra | 0,06510 | 0,06510 | 0,11250 | 0,11250 |
| Fin dove regge | T = 150 | T = 150 | | |
| H2b | non calcolabile (nessun T >= 40 con ESS mediana >= 100) | idem | | |

Tutte le righe senza differenza coincidono fino all'ultima cifra salvata.

## Differenze e loro causa

1. **R_samp: 0,6247 contro 0,6467 a T = 80 (-3,4%); 1,5284 contro 1,4877 a T = 150 (+2,7%).** Il rumore stimato è identico alla cifra. Cambia solo la media di D_samp nella finestra: io avevo preso la media semplice dei venti tassi per passo, `fase1b.py` fa la media pesata per il numero di fallimenti terminati di ogni passo (frequenza complessiva = ammissioni totali / fallimenti totali). Ricalcolata con la media pesata, la mia variante riproduce l'ufficiale esattamente (0,6466827451904552 e 1,4877447272701567). La premessa dice "la frequenza di ammissione tra i fallimenti terminati ... negli ultimi 20 passi", che si legge meglio come media pesata; dice anche "varianza dei tassi per passo ... divisa per il numero di passi", che è il rumore della media semplice. Piccola incoerenza di formulazione: il rumore è quello della media semplice, la media usata è pesata. Qui l'effetto è piccolo ma non nullo perché il numero di fallimenti terminati per passo cala molto (media nella finestra 29,2 a T = 20, 20,3 a T = 80, 10,5 a T = 150, con valori da 4 a 19 dentro la finestra di T = 150): da qui il 3%. Nessun verdetto cambia.
2. **Altre varianti di D_samp, provate per sensibilità:** finestra T-20..T-1 dà R_samp 0,600 e 1,491; varianza con n invece di n-1 dà 0,626 e 1,529. Stessi verdetti (PROMOSSO, PROMOSSO).
3. Nessun'altra differenza.

## Riserve di metodo

**1. Il verdetto "regge" poggia solo su D_samp, perché D_seq è crollata presto.** L'ESS mediana (su 400) è 126 a T = 10 (sopra la soglia 100), poi 62 a T = 20, 25 a T = 40, 10 a 60, 13 a 80, 12 a 100, 8 a 120, 13 a 150. Quindi D_seq, la misura per cui la previsione è stata costruita, vale solo fino a circa 10-15 passi. È la regola scritta e l'ufficiale la applica bene, ma il rapporto di tre righe (`RAPPORTO_FASE1B.md`) scrive "la previsione regge" per ciascun T senza dire che D_seq era NON CONCLUSIVO (ESS 13 su 400), che H2b non si è potuta calcolare, né che le corse sono 6. Il lettore non lo vede. Per la regola del 16/10 ("sotto i 30-40 casi è solo un indizio") con 6 corse il risultato è un indizio, non una conferma.

**2. D_samp non misura la stessa cosa di D_seq, e dopo T = 60 le due si separano.** D_seq è la frequenza di ammissione sui 400 fallimenti fissi del passo 0, riponderati. D_samp è la frequenza di ammissione nei fallimenti nuovi della politica che si sta allenando, mediata su 20 passi (quindi su politiche diverse, da T-20 a T-1). Differenza della media tra le corse, D_seq meno D_samp: +0,08 a T = 20 e 40, 0,00 a 60, -0,24 a 80, -0,31 a 100, -0,35 a 120, -0,38 a 150. A T = 150 D_seq vale quasi zero in tutte le corse (da 0,0 a 0,0037) e D_false quasi uno (da 0,994 a 1,0), mentre D_samp ha media 0,39 (da 0,05 a 0,88). La correlazione tra le corse di D_seq e D_samp è 0,40 a T = 80 e -0,03 a T = 150. Interpretazione: a T = 150 la politica è uscita dal supporto del campione fisso (ESS da 2,3 a 16,9 su 400), quindi D_seq non stima più la frequenza di ammissione della politica attuale. In più D_samp è condizionata ai fallimenti che terminano, e quelli diventano pochi: in media 29 per passo (su 32 risposte) a T = 20, 20 a T = 80, 10,5 a T = 150 (da 4 a 19). Quindi D_samp a T = 150 è la frequenza di ammissione in una minoranza selezionata dei fallimenti (le risposte che finiscono entro 384 token), non in tutti. Infine la SD prevista è la SD del funzionale lineare di D_seq: D_samp è confrontata con una previsione costruita per un'altra variabile. La scelta di usarla come ripiego è scritta nella premessa, ma resta un confronto indiretto.

**3. Sei corse pesano molto.** Intervallo chi quadro al 95% (moltiplicatori 0,624-2,453): R_samp(80) = 0,65 sta in [0,40; 1,59], che comprende la zona INDIZIO; R_samp(150) = 1,49 sta in [0,93; 3,65], che arriva fino a BOCCIATO. Togliendo una corsa alla volta (media semplice), R_samp(80) va da 0,42 a 0,70: se si toglie la corsa 205 (la più alta, 0,55) o la 206 (la più bassa, 0,25) scende sotto 0,5 e il verdetto diventa INDIZIO. R_samp(150) va da 0,92 a 1,71 e resta PROMOSSO, ma la dispersione a 150 è trascinata dalla corsa 204 (D_samp 0,88, D_tf 0,95, D_false 1,0 contro gli altri cinque). I due punti di prova non sono indipendenti (stesse corse; correlazione tra le corse 0,35 tra T = 80 e T = 150 per D_samp). Per la potenza dichiarata, PROMOSSO si ottiene anche con un errore di 2 volte nel 58% dei casi: "regge" qui significa "non smentita da un errore grande", non "confermata entro un fattore 2".

**4. La forma della crescita non segue la previsione (fuori dalle regole, descrittivo).** SD di D_samp dopo il rumore (media pesata): 0,026 (T = 20), 0,027 (40), 0,057 (60), 0,094 (80), 0,110 (100), 0,161 (120), 0,280 (150). Da T = 40 a T = 150 cresce di circa 10,5 volte; la diffusione prevista cresce di 1,7 volte. R_samp: 0,33, 0,24, 0,44, 0,65, 0,69, 0,95, 1,49. La pendenza di log SD contro log T da T = 40 in su vale circa 1,68 contro 0,40 prevista (differenza 1,3, quattro volte l'errore di circa 0,32 dichiarato nella premessa). In scala logit la SD tra corse passa da 0,16 a 1,68. Quindi la misura parte molto sotto la previsione (a T = 40 R_samp = 0,24 sarebbe BOCCIATO, ma i primi valori sono rumorosi: SD osservata 0,034-0,036 contro rumore 0,022-0,023) e la attraversa. I due punti scelti (80 e 150) cadono dopo l'incrocio dentro la banda. Resta qualitativamente la forma "piatta e poi più veloce" già vista nella Tabella 13 dell'articolo e annunciata in premessa; non è la diffusione con gradienti fissi. H2b, che doveva dirlo, non è calcolabile con la regola (ESS) e la variante su D_samp non fa parte delle regole.

**5. A T = 150 la SD di D_seq vicina a zero è un collasso dei pesi più il limite 0-1, non una dispersione piccola.** I sei D_seq a T = 150 sono 0,0002; 0,0037; 0,0020; 0,0000; 0,0024; 0,0002: tutti appoggiati al limite zero, quindi la SD (0,0015) non può superare circa 0,004 qualunque sia la dispersione delle politiche. I pesi sono concentrati: la quota del campione col peso maggiore va dal 11% al 65%, e 7-24 campioni su 400 portano il 90% del peso. R = 0,008 è un effetto di pavimento, e la regola dell'ESS lo esclude giustamente. Anche a T = 80 la SD osservata di D_seq (0,099) è dello stesso ordine del rumore da pesi concentrati, sqrt(D(1-D)/ESS), che per corsa va da 0,06 a 0,18 (stima euristica mia, non nelle regole): R = 0,68 non è interpretabile come dispersione delle politiche.

**6. Il limite 0-1 della frequenza.** Per D_samp tocca poco: nessuna corsa entro 0,12 dai limiti fino a T = 120, due su sei a T = 150 (0,045 e 0,88). Siccome il limite comprime la SD, la dispersione "libera" a 150 non è minore di 0,28; il rischio per la previsione (che è lineare e senza limiti) è che nelle corse vicine al limite la crescita sia più lenta di quella vera, quindi R_samp(150) = 1,49 non è sovrastimato dal limite. Il limite tocca invece molto D_seq (riserva 5), D_false (0,99-1,0 a T = 150) e D_tf (una corsa su sei a 0,95).

**7. D_tf descrittiva a T = 150: R = 4,09** (1,92 a T = 120). Senza verdetto per regola, ma è l'unica misura fissa e non riponderata, ed è lontana dalla previsione. È dovuta a una sola corsa (la 204: 0,952; le altre cinque stanno tra 0,486 e 0,509). Non smentisce il verdetto, ma è un campanello nello stesso senso della riserva 4: le previsioni lineari del passo 0 sottostimano l'eventuale svolta di una singola corsa.

**8. La stima del rumore di D_samp regge.** Dentro le finestre l'autocorrelazione del tasso per passo è vicina a zero (media di lag 1: -0,08 a T = 80, 0,00 a T = 150) e la tendenza lineare dentro la finestra non spiega praticamente nulla della varianza: la stima "varianza dei tassi / 20" non è gonfiata da deriva né sgonfiata da correlazione. La deviazione del tasso per passo è 0,119 a T = 80 e 0,153 a T = 150, contro una binomiale pura di circa 0,11 (n = 20, p = 0,4) e 0,15 (n = 10,5): l'effetto dei gruppi di 8 per compito si vede poco, il rumore è soprattutto binomiale con pochi fallimenti per passo. La regola del rumore non è vicina a scattare (0,027 contro soglia 0,073; 0,034 contro 0,094). Sottrarre il rumore abbassa R_samp del 4,0% a T = 80 e dello 0,7% a T = 150 (media semplice); il 6% della prova sintetica non l'ho riverificato. Nota: con meno fallimenti terminati per passo il rumore di D_samp cresce con T (0,022 a T = 20, 0,034 a T = 150), quindi il confronto con SD più piccole (T = 20, 40) è il più incerto.

**9. Convenzione "ultimi 20 passi prima di T".** `fase1b.py` usa i passi T-19..T (compreso T). La mia lettura iniziale era la stessa; la lettura T-20..T-1 dà gli stessi verdetti. Nessun rischio per il verdetto.

## Suggerimenti (non applicati, io non correggo nulla)

- Nel rapporto della 1b aggiungere accanto a ogni T: ESS mediana, "D_seq NON CONCLUSIVO", numero di corse, e la scritta che il verdetto vale per D_samp. Aggiungere che H2b non è stata calcolabile.
- Se il rapporto va fuori, scrivere "indizio, 6 corse" invece di "la previsione regge", secondo la regola del 16/10.
- Riportare, accanto al verdetto e senza cambiarlo, la forma della crescita di D_samp (riserva 4) e D_tf (riserva 7), perché dicono che la previsione a T = 40 era troppo alta e a T = 150 troppo bassa.

## Numeri chiave

| | T = 80 | T = 150 |
|---|---|---|
| R (D_seq) | 0,684 | 0,008 |
| ESS mediana | 13,1 | 12,7 |
| R_samp (media pesata, come l'ufficiale) | 0,647 | 1,488 |
| R_samp (media semplice, mia lettura iniziale) | 0,625 | 1,528 |
| Rumore D_samp (deviazione) | 0,0265 | 0,0343 |
| Soglia della regola del rumore (metà SD prevista) | 0,0727 | 0,0942 |
| Verdetto usato | PROMOSSO | PROMOSSO |

Fin dove regge: T = 150. H2b: non calcolabile. Riuscita media nelle finestre: 6,5% (T = 80), 11,3% (T = 150).
