# Ricalcolo indipendente del verdetto della fase 1 (21/10/2026, ~16:00)

Un agente nuovo, con contesto pulito, ha scritto il proprio codice (gli script qui accanto) e ha ricalcolato tutto
dai file grezzi. Codex era in sola lettura senza Python e non aveva potuto farlo.

**Verdetto del controllo: CONFERMATO CON RISERVE.** Tutti i numeri tornano e H1 NON CONCLUSIVO è il verdetto giusto
secondo `PREMESSA.md`. Le riserve riguardano alcune frasi del rapporto e due convenzioni di calcolo, non il verdetto.

## Integrità

- 12 corse da 41 righe (passi 0-40), con le cadenze giuste e nessun NaN.
- Nessuna coppia di corse identica.
- Al passo 0 D_seq, D_tf, D_false, ESS e i 400 `lp_seq` sono uguali in tutte le corse e uguali a `logp0` del pool.
- Gli sha256 della previsione tornano con `previsione_sha256.txt` e con il commit f6104eb.
- `analisi_fase1.py` e `rapporto_fase1.py` non sono cambiati dopo il congelamento.

## Numeri (ricalcolo contro `analisi_fase1.json`)

| voce | ricalcolo | JSON |
|---|---|---|
| SD D_seq passo 40 | 0,206224 | uguale |
| pavimento | 1,6e-6 | uguale |
| SD prevista (replay, lin_seq a 40) | 0,103693 | uguale |
| R | 1,989 (IC95 1,41-3,38) | uguale |
| ESS mediana a 40 | 15,13 (mediana vera) | 15,68 (mediana superiore) |
| verdetto H1 | NON CONCLUSIVO | uguale |
| R con pavimento 1e-5 | 1,989 | uguale |
| R dei 7 funzionali lineari | 0,90 / 1,71 / 1,35 / 1,08 / 0,96 / 1,39 / 1,01 (lin_false 0,80) | uguale |
| H2, pendenza misurata contro prevista | 1,215 contro 0,507 | uguale |
| H6, R_false | 2,115, NON CONCLUSIVO | uguale |
| H5, correlazione 20-40 misurata contro prevista | 0,759 contro 0,895 | uguale |
| D_tf | R 1,36, piatta (solo 9% dei p tra 0,02 e 0,98) | uguale |

La "scorciatoia ingenua" dà R 3,27 con la formula dei pesi di Adam e 2,88 con il rapporto del replay. È una
convenzione, non un errore.

## Ricostruzione della stima per importanza (descrittiva)

- D_seq, D_false ed ESS tornano esattamente dai `lp_seq`.
- Peso portato dai 5 esempi più pesanti, media sulle corse:
  - passo 0: 1,3%
  - passo 5: 8%
  - passo 10: 15%
  - passo 20: 28%
  - passo 30: 37%
  - passo 40: **45%** (tra 23% e 73% a seconda della corsa)
- Al passo 40 un solo esempio porta in media il 27% del peso. Nelle corse 104 e 111 porta più del 50%.
- Le 4 corse con ESS più bassa (2,5-7,8) hanno tutte D_seq ≥ 0,63. La correlazione tra log ESS e D_seq è −0,74.
- R(T) di D_seq:
  - T=5: 0,69
  - T=10: 0,78
  - T=20: 1,57
  - T=30: 1,99
  - T=40: 1,99

  Nello stesso tempo la mediana dell'ESS scende da 148 (T=10) a 44 (T=20).

## Riserve, dalla più grave

1. **"Non è un verdetto sul termometro"** è un'ipotesi scritta prima, non un risultato. Lo stesso collasso dei pesi
   spiega anche la linearizzazione rotta (correlazione 0,46 a T=10, 0,18 a T=20, 0,035 a T=40), e questi dati non
   separano la misura dal termometro.
2. **H2 "non regge"** è più forte di quanto i dati permettano. La pendenza nasce quando la misura è già collassata.
   La regola scritta per H2 però non prevede eccezioni per l'ESS, quindi il verdetto resta.
3. **Sensibilità 1e-5.** Il JSON scrive `verdict: PROMOSSO` (solo l'intervallo di R, senza la regola dell'ESS)
   accanto a H1 NON CONCLUSIVO. La frase del rapporto è giusta nella sostanza, ma ambigua: lo ha notato anche
   Codex.
4. **Mediana.** Il codice congelato usa `sorted[n//2]` (15,68) invece della mediana vera (15,13). Non cambia nulla.
   La fase 1b usa già `statistics.median`.
5. **Lineari.** "0,90-1,71" esclude lin_false per scelta. Con lin_false l'intervallo è 0,80-1,71. lin_tf ha IC95
   1,21-2,90, che esclude 1. 5 R su 7 sono sopra 1, con media geometrica 1,17: c'è una lieve sottostima
   sistematica, e i 7 funzionali non sono indipendenti.

## Non verificato

- La correttezza del codice sulla GPU (replay di Adam, gradienti, `lp_seq` dal modello).
- Le etichette del rilevatore.
- I quantili del chi quadro, inseriti a mano: l'IC torna con quello del JSON.
