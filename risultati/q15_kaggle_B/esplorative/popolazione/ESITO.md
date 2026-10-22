# Analisi secondaria "popolazione": esito (ESPLORATIVA)

- **Premessa e codice:** scritti il 20/10/2026 alle 23:05, prima dei dati delle corse. Commit 4f01a81 sul ramo
  claude/situazione-cartella-391884. I file qui accanto sono copie identiche.
- **Dati:** le 12 corse del notebook B, scaricate da Kaggle il 21/10/2026 alle 15:52.
- **Esecuzione:** 21/10/2026, 15:57, con `python -I analisi_popolazione.py` su una copia delle corse. Non fa parte
  della pre-registrazione della fase 1 e non cambia il suo verdetto.

## Numero

**Q = 0,34**, intervallo bootstrap al 90% **0,19–0,50**, premessa del 20/10/2026, 23:05 (commit 4f01a81).
La regola della premessa dà: **"la popolazione non conta"**.

- SD di D_seq al passo 40: 0,206. SD grezza di D_camp (ultimi 5 passi): 0,081. SD corretta per il rumore binomiale:
  0,069.
- Correlazione D_seq / D_camp tra le corse: 0,19.
- Medie: D_seq al passo 40 vale 0,446 (0,485 al passo 0); D_camp vale 0,491.
- La previsione aveva Q circa 1,5.

## Perché il verdetto della regola non si può leggere alla lettera

La regola presume che D_seq al passo 40 sia una buona misura. Nella fase 1 non lo è: l'ESS mediana è 15,7.

| corsa | D_seq 40 | ESS | D_camp |
|---|---|---|---|
| 111 | 0,791 | 2,5 | 0,535 |
| 104 | 0,718 | 2,6 | 0,340 |
| 110 | 0,662 | 7,3 | 0,562 |
| 107 | 0,630 | 7,8 | 0,534 |
| 109 | 0,125 | 9,3 | 0,417 |
| le 4 corse con ESS > 25 | 0,29–0,44 | 27,6–48,4 | 0,43–0,59 |

- I valori estremi di D_seq sono quelli delle corse con l'ESS più basso.
- Il denominatore di Q contiene quindi il rumore della stima per importanza. Q piccolo e correlazione 0,19 dicono
  soprattutto che D_seq al passo 40 è rumorosa, non che la popolazione non conti.
- Il pavimento pre-registrato (perturbazioni da 1e-5) non vede questo rumore: misura la stima vicino ai pesi del
  passo 0, dove i pesi per importanza valgono circa 1.

## Due limiti di D_camp

- Il rumore è stimato con la formula binomiale, che non vede i gruppi di 8 risposte per compito. Quindi la SD
  corretta di D_camp è, se mai, sovrastimata.
- Ogni corsa vede compiti diversi. D_camp include quindi proprio il cambio di popolazione che l'analisi vuole
  misurare.

## Cosa resta

La misura che non dipende dai pesi (D_camp) ha una dispersione tra corse di circa 0,07. È piccola rispetto alla SD
prevista per D_seq (0,104).

È un indizio a posteriori, non un verdetto. Il controllo pre-registrato di questa idea è D_samp della fase 1b, con
finestra di 20 passi e rumore stimato dai dati (`PREMESSA_FASE1B.md`), congelato prima di queste corse.
