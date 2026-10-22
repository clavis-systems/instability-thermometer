# Analisi secondaria "popolazione": scritta prima dei dati (20/10/2026, 23:05, ora del commit 4f01a81)

**Stato:** esplorativa, fuori dal piano congelato della fase 1 (`strumenti/termometro/PREMESSA.md`, commit 979d472). Non
cambia nessun verdetto della fase 1; si riporta a parte, con questa data. Scritta dalla sessione "Punto situazione
cartella" mentre il notebook B girava su Kaggle, prima che qualunque corsa fosse scaricata.

## Da dove viene

Prova 2 del torneo dei generatori di idee (20/10 sera, generatore "negazione delle ipotesi implicite"). L'articolo di
Anthropic arXiv 2609.33220 misura l'ammissione di errore sui **propri fallimenti** di ogni corsa. Ipotesi implicita: i
fallimenti di corse diverse sono una popolazione confrontabile. Negazione: ogni corsa fallisce su problemi diversi,
quindi una parte dello spread tra corse è un cambio di popolazione, non di comportamento.

## Misure (dai file `run_s*.json` della fase 1)

- `D_seq` al passo 40: ammissione stimata sul pool fisso di 400 fallimenti del passo 0 (popolazione fissa).
- `D_camp`: ammissioni diviso fallimenti terminati, sommati sui passi 36-40 (popolazione propria di ogni corsa).
- Il rumore binomiale di `D_camp` (media su corse di p(1-p)/n) si toglie in quadratura dalla varianza tra corse.

## Regola fissata ora

Q = SD corretta di `D_camp` diviso SD di `D_seq`, intervallo bootstrap al 90% sulle corse (2.000 ricampionamenti, seme
20261020).

| Esito | Condizione |
|---|---|
| la popolazione conta | Q ≥ 1,25 e tutto l'intervallo ≥ 1,25 |
| la popolazione non conta | Q ≤ 1,00 e tutto l'intervallo ≤ 1,00 |
| non conclusivo | altrimenti |

Si riporta anche la correlazione tra `D_seq` e `D_camp` sulle corse.

## Previsione mia

"La popolazione conta", Q intorno a 1,5: le corse divergono anche su quali problemi falliscono, e i fallimenti nuovi
hanno testi diversi da quelli del pool. Con 12 corse l'intervallo sarà largo: l'esito più probabile è "non conclusivo",
e in quel caso il risultato è solo un indizio (regola del 16/10: sotto 30-40 casi è un indizio).

## Limiti dichiarati

- `D_camp` usa circa 150 fallimenti per corsa su 5 passi: la correzione binomiale è grossolana.
- `D_seq` è una stima di importanza: vale vicino ai pesi di partenza; la fase 1 riporta l'ESS.
- Le due misure differiscono anche per il rilevatore (lo stesso) applicato a testi diversi: non si separa qui
  "popolazione" da "testi nuovi".
- Il codice `analisi_popolazione.py` è stato collaudato solo su dati finti (`test`).
