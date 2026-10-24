# Ricalcolo indipendente della fase 2 (SmolLM2-1.7B), 24/10/2026

**Verdetto: CONFERMATO.** 0 discrepanze sostanziali; 3 scarti di arrotondamento di 0,01 nei limiti superiori dell'IC.

Codice nuovo (`ricalcolo.py`): legge solo `thermo_s1/s2.json` e `run_s*.json`, 12 corse per famiglia, SD con n-1, IC chi-quadro 95%, pavimento in quadratura.

| voce | atteso | ricalcolato |
|---|---|---|
| Smol p5 | 0,038; 0,018; R 2,07; IC 1,47-3,52; ESS 286 | 0,0377; 0,0182; 2,070; 1,47-**3,51**; 286 |
| Smol p10 | 0,052; 0,026; 1,98; 1,40-3,37; 181 | 0,0523; 0,0264; 1,978; 1,40-**3,36**; 181 |
| Smol p20 | 0,072; 0,039; 1,86; 1,31-3,16; 119 | 0,0718; 0,0387; 1,855; 1,31-**3,15**; 119 |
| Qwen p10 | 0,040; 0,052; 0,78; ESS 148 | 0,0402; 0,0516; 0,780; 148 |
| Ripetibilità p10 | 0,046; R circa 1,14 | 0,0459; R 1,140 |

ESS minima SmolLM2: 192 (p5), 112 (p10), 40 (p20).

**Scarti minori.** Limiti superiori 3,514 / 3,358 / 3,150 (fattore esatto 1,698), non 3,52 / 3,37 / 3,16: il README ha usato 1,70. Nessun verdetto cambia.

**H1 ricalcolato.** ESS mediana al passo 10 = 181, quindi vale il passo 10: R = 1,978, **PROMOSSO**, a 0,02 dalla soglia 2 (al passo 5, R 2,07 sarebbe INDIZIO; al 20, R 1,86, PROMOSSO).

**Ordine: NON rispettato.**
- Prevista: SmolLM2 meno dispersa di Qwen (0,026 contro 0,052 al passo 10).
- Misurata: SmolLM2 più dispersa (0,052 contro 0,040; passo 5: 0,038 contro 0,024).
- Rapporto misurato Smol/Qwen al passo 10: 1,30, IC 95% 0,70-2,42, quindi non distinguibile dalla parità, ma il punto stimato è opposto alla previsione.

**Ripetibilità.** La seconda previsione (seme 2, K = 25) vale 0,0459 al passo 10, 1,74 volte la prima (0,0264). Il pavimento manca in `thermo_s2.json`: usato quello di s1 (2,4e-6, trascurabile).
