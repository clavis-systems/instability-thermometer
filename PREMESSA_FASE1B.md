# Premessa, fase 1b: l'orizzonte lungo (20/10/2026, scritta prima di ogni risultato delle corse)

**Stato: CONGELATA il 20/10/2026** (commit che aggiunge `FASE1B_CONGELATA.txt`), approvata da Emanuele ("si va bene approvo"), rivista da un agente indipendente ("pronta
dopo piccole correzioni", correzioni fatte qui e in `fase1b.py`).
- Si congela **prima** che esista qualunque risultato delle corse della fase 1 (notebook B).
- Si può congelare dopo aver visto la previsione del notebook A, perché è solo previsione.
- È stata decisa dopo il congelamento della fase 1: è una prova in più e non cambia niente della fase 1. Lo
  dichiariamo.

**In parole semplici.** La fase 1 controlla il termometro sui primi 40 passi. L'articolo però vede la grande
dispersione dopo 300 passi, e trova che all'inizio cresce piano e poi più in fretta. Qui guardiamo fin dove la
previsione del passo 0 regge, più lontano, e dove si rompe. Il controllo indipendente del 20/10 l'ha indicata come
la prova che interessa di più agli autori.

## Domanda

La dispersione tra corse prevista al passo 0 (replay esatto di Adam sui gradienti del passo 0) vale anche oltre i 40
passi? Se no, la crescita vera è più lenta o più veloce della diffusione con i gradienti fissi del passo 0?

## Assetto

- **Identico alla fase 1:** stessa partenza (sha256 c0506a71…), stesso campione fisso del passo 0 (seme 7, `pool.json`
  di A), stesso codice congelato, Kaggle con 2 T4 in float32.
- **Corse nuove:** 6 corse indipendenti (semi 201-206, mai usati) dalla stessa partenza; i compiti seguono la regola
  della fase 1 (seme 20.000 + seme della corsa).
- **Orizzonte, con una regola scritta prima delle corse:** la durata stimata di una corsa da 150 passi è 150 per la
  mediana dei secondi per lotto del termometro di A (seme 1), più 8 misure da 260 s (la prova di velocità su T4 del
  20/10). Se supera 3,5 ore, l'orizzonte è 100 passi e le prove si fanno a T = 80 e 100; altrimenti 150 passi, con
  prove a T = 80 e 150. Così 3 corse per GPU stanno in una sessione da 12 ore. La regola è applicata dal codice
  (`fase1b.py horizon`), non da noi.
- **Codice:** l'estensione separata `fase1b.py` importa il codice congelato e cambia **solo** l'orizzonte:
  - passi di misura 10, 20, 40, 60, 80, 100 (più 120 e 150 se l'orizzonte è 150);
  - checkpoint a 40, 80 e alla fine;
  - la lista dei passi del replay.

  Nessun altro cambiamento. L'analisi e le frasi del verdetto sono in `fase1b.py`, scritte ora e provate su dati
  sintetici (`fase1b.py test`).

## La previsione (dalla fase 1, estesa)

- Replay di Adam con gli stessi K = 50 gradienti del passo 0 del termometro seme 1 (`gradienti_s1.pt` di A): stesso
  seme del replay (501), stesse direzioni nello stesso ordine, 400 repliche, fino all'orizzonte.
- È la stessa ipotesi della fase 1, cioè una diffusione con i gradienti fissi del passo 0, portata più lontano.
- La previsione è una funzione deterministica di file già su git (codice congelato e uscita del notebook A).
  - Si calcola nella prima cella del notebook della 1b, prima delle corse, e il registro di Kaggle ne riporta
    l'impronta.
  - Se serve una seconda sessione, si riusa la previsione della prima e non si ricalcola.
  - Dopo si mette anche su git.
- Le estrazioni del replay ora coprono tutto l'orizzonte, quindi a T = 40 il valore differisce da quello della fase 1
  solo per rumore Monte Carlo (circa 3,5% con 400 repliche). Si riportano tutti e due.

## Misure

- **D_seq**, con gli stessi pesi per importanza, ai passi di misura, con l'ESS.
- **D_tf** e **D_false** agli stessi passi: si riportano come R descrittivi, senza verdetto.
- **D_samp**, che non dipende dai pesi: la frequenza di ammissione tra i fallimenti terminati delle risposte di
  allenamento negli **ultimi 20 passi** prima di T.
  - Il suo rumore si stima dai dati: varianza dei tassi per passo nella finestra, divisa per il numero di passi, e
    mediata sulle corse. Così include anche l'effetto dei gruppi di 8 risposte per compito, che la formula binomiale
    non vede.
  - Si sottrae in quadratura.
  - La media sulla finestra e la stima del rumore abbassano R_samp di circa il 6% (prova sintetica): lo diciamo e non
    lo correggiamo.
- **Riuscita:** la media della riuscita nelle stesse finestre, solo descrittiva.

## Ipotesi e criteri

- **H1b**, per ciascun T di prova: R(T) = SD tra le corse di D_seq al passo T (pavimento della fase 1 sottratto) / SD
  prevista. Stesse soglie della fase 1: PROMOSSO con 0,5 ≤ R ≤ 2, INDIZIO con 0,33 ≤ R ≤ 3, altrimenti BOCCIATO.
  - **ESS:** si usa la mediana vera tra le corse (`statistics.median`). Se è sotto 100, D_seq è NON CONCLUSIVO e
    per quel T il verdetto si dà con D_samp.
  - **Regola del rumore per D_samp:** se la deviazione del suo rumore supera la metà della SD prevista, anche D_samp
    è NON CONCLUSIVO.
  - **Quando valgono tutte e due:** se D_seq ha ESS ≥ 100, il verdetto è quello di D_seq; D_samp si riporta accanto.
- **Fin dove regge:** è il T di prova più grande per cui tutti i T fino a lui sono PROMOSSI.
  - Se il primo T è BOCCIATO o INDIZIO: "non regge già al primo orizzonte".
  - Se è NON CONCLUSIVO: "non si può dire".
- **H2b (solo descrittiva):** la pendenza di log SD contro log T, misurata e prevista, sui passi da 40 in su con ESS
  mediana ≥ 100 (almeno 3 passi).
  - "Più veloce" o "più lenta" solo oltre ±0,5 dalla prevista, altrimenti "come prevista".
  - Con 6 corse la differenza di pendenza ha un errore di circa 0,32: per questo non c'è verdetto.
  - Una "più lenta" può dipendere anche dal limite tra 0 e 1 della frequenza.
- **Corse:**
  - una corsa che si ferma due volte con lo stesso seme si esclude, come nella fase 1;
  - al massimo due sessioni: l'analisi si fa quando ci sono tutte le 6 corse, oppure alla fine della seconda
    sessione con 4 o 5 corse complete, dichiarando la potenza ridotta;
  - con meno di 4 è NON CONCLUSIVO.
- **Potenza dichiarata, con 6 corse:**
  - l'intervallo al 95% della SD misurata va da 0,62 a 2,45 volte;
  - se la previsione è giusta, PROMOSSO nel 94% dei casi;
  - con un errore di 2 volte, PROMOSSO comunque nel 58%;
  - BOCCIATO nel 42% con un errore di 3 volte e nel 73% con un errore di 4 volte (simulazioni della revisione).

  La 1b quindi smaschera solo errori grandi.

## Costi (stima)

Su T4 in float32: 6 corse, 3 per GPU, in una sessione da 12 ore (la regola dell'orizzonte serve a questo). Circa 11
ore della quota settimanale, che si azzera il sabato.

## Limiti dichiarati prima

- 6 corse soltanto.
- L'ESS può crollare: per questo c'è D_samp, che però è rumorosa.
- I compiti cambiano a ogni passo e tra le corse, e le risposte sono a gruppi di 8 per compito: è varianza in più per
  D_samp.
- La quota dei lotti informativi cresce durante l'allenamento: il replay con i gradienti del passo 0 non lo vede, ed è
  proprio una delle cose che la 1b mette alla prova.
- LoRA e passo di apprendimento 2e-5 invece di 1e-6.
- L'orizzonte arriva al massimo a 150 passi, non a 300.
- Il "termometro rinnovato" (rifare il termometro su un checkpoint del passo 40) è **rinviato**: richiede un campione
  nuovo e renderebbe il confronto ambiguo.
