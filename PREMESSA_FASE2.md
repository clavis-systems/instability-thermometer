# Premessa, fase 2: le famiglie di modelli (scritta il 19/10/2026, prima di ogni dato della fase 1)

**Stato: CONGELATA il 20/10/2026, insieme** alla premessa della fase 1, con lo stesso commit di `CONGELATA.txt`. Così le
famiglie e i criteri sono fissati prima di vedere qualunque risultato.

## Domanda (quella aperta di arXiv 2609.33220)

Quali famiglie di modelli sono più sensibili all'instabilità dell'ammissione degli errori? Il termometro sa dirlo
prima di addestrare?

## Famiglie

Tutte scaricate il 18/10 con l'OK di Emanuele, Apache-2.0:

| Nome | Modello | Dimensione |
|---|---|---|
| q15 | Qwen2.5-1.5B-Instruct | è la fase 1 |
| q05 | Qwen2.5-0.5B-Instruct | |
| olmo1 | OLMo-2-0425-1B-Instruct | |
| smol17 | SmolLM2-1.7B-Instruct | |

## Cosa era già noto prima del congelamento (lo dichiariamo)

Dall'articolo (letto il 19/10 sera, con un controllo indipendente):
- **OLMo-2-1B è già nell'articolo con 20 corse** (Tabella 12): ammissione da 0,072 a 0,903, SD 0,255, contro lo
  0,189 di Qwen2.5-1.5B. Il loro assetto: 300 passi, passo di apprendimento 1e-6, checkpoint esatto e partenza SFT di
  OLMo non indicati. Il nostro è OLMo-2-0425-1B-Instruct.
- Qwen-7B con 19 corse e Qwen2.5-32B con 8; Gemma-2-2B, Llama-3.2-3B (dispersione 0,194), Qwen distillato
  (0,076) e Phi-4 con 3 corse ciascuno (Tabella 16). Per Gemma e Phi la Sezione 7 dice che non mostrano la stessa
  grande dispersione.

Quindi:
- **Il confronto olmo1 contro q15 non è cieco.** In H4b si riporta a parte e non conta tra le "famiglie dalla parte
  giusta". La previsione del termometro resta meccanica (il codice non conosce quei numeri).
- **Unico ancoraggio quantitativo esterno, solo descrittivo:** il rapporto SD(olmo1)/SD(q15) previsto dal termometro e
  quello misurato nelle nostre corse al passo 40, accanto al loro 0,255/0,189 = 1,35. Nessun verdetto: assetti e
  orizzonti diversi.
- Per Qwen2.5-0.5B e SmolLM2-1.7B l'articolo non dà numeri: sono le sole previsioni davvero cieche.

## Procedura

È **identica alla fase 1** in tutto:
- codice, dati, semi, soglie, cancelli;
- 12 corse da 40 passi;
- analisi `analisi_fase1.py` per famiglia (comprese la sensibilità al rumore e H6);
- piattaforma: la stessa della fase 1 (Kaggle, float32, notebook equivalenti ad A e B); la partenza SFT di ogni
  famiglia si addestra con la stessa ricetta e la sua impronta sha256 si scrive prima del campione.

La catena `catena_fase2.sh` le fa una dopo l'altra. Parte dopo il file-segnale della fase 1
(`risultati/q15/FASE1_FINITA.txt`), con lo stesso codice congelato.

Se una famiglia non supera un cancello si ferma e si riporta così. L'eventuale unica azione permessa, la stessa della
fase 1, la decide Emanuele dopo: la catena non corregge niente da sola.

**Il rilevatore è quello congelato con il pilota della fase 1.** Per ogni famiglia si riporta l'accordo con un
etichettatore indipendente su 100 code non vuote del suo campione, senza correggere niente.

## Ipotesi

- **H4a:** per ogni famiglia, lo stesso verdetto H1 della fase 1 (R tra 0,5 e 2 = PROMOSSO). Si riporta quante
  famiglie su 4 sono promosse.
- **H4b (descrittiva, con 4 famiglie non si può fare un test):**
  - la correlazione di Spearman tra la dispersione prevista e quella misurata;
  - quante famiglie stanno dalla parte giusta rispetto a Qwen2.5-1.5B.
- **La risposta alla domanda dell'articolo:** l'ordine delle famiglie per sensibilità misurata, accanto all'ordine
  previsto dal termometro e alla sua scomposizione. I tre fattori sono definiti ora, in `analisi_fase2.py`:
  - **esposizione:** il numero medio di risposte fallite e terminate per lotto al passo 0;
  - **bilico:** D_seq_0 × (1 − D_seq_0);
  - **accoppiamento:** la quota della dispersione prevista per passo che viene dalla zona del ragionamento fallito.

Analisi in `analisi_fase2.py`, scritta prima dei dati e provata su dati finti.

## Limiti

- 4 famiglie piccole (0,5-1,7 miliardi di parametri), LoRA, Countdown nostro.
- Il confronto con le altre famiglie dell'articolo (Gemma-2-2B, Llama-3.2-3B, Qwen distillato, Phi-4) è solo
  qualitativo: quei modelli non sono nella nostra prova (per Llama e Gemma serve accettare una licenza). L'unica
  famiglia in comune, oltre a Qwen2.5-1.5B, è OLMo-2-1B (vedi sopra).
- Tempo: circa 16 ore in tutto sulla GPU da 8 GB (SmolLM2 da sola circa 7 ore), quindi due notti o più.
