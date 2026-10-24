# Provenance

This work was done in a private git repository. Its commit metadata contains a personal e-mail address, so
the history is not published. Below is every commit of that repository that touched this folder, oldest first
(98 commits, up to `f647bcf`). Commit dates are written by the author's computer: they are a record,
not a proof. The checks that do not depend on trusting me are the content hashes and Kaggle's own timestamps
(see README.md, "Provenance"). Commit messages are in Italian; `[claude]` marks commits made by the AI agent.
The messages of the commits about a private letter to the paper's authors are withheld; the commits are listed.

| commit | date (author's clock) | message |
|---|---|---|
| `687d65a` | 2026-10-18T21:35:42+02:00 | [claude] termometro dell'instabilita': Countdown con risolutore esatto, dati di partenza (1.000), codice del termometro e delle corse GRPO, prova a vuoto passata, bozza della premessa |
| `0b19dce` | 2026-10-18T21:52:17+02:00 | [claude] termometro: segnale di fine turno di ogni famiglia, D_tf a passi fissi nelle corse, catena notturna della fase 1 (parte solo a premessa congelata) |
| `cf0d475` | 2026-10-18T22:05:52+02:00 | [claude] termometro: correzione per il momento di Adam, controllo a costo zero sui numeri dell'articolo, esito del controllo di novita' (in parte nuovo) e tensione nei loro dati nella premessa (bozza) |
| `0e5b192` | 2026-10-18T22:36:03+02:00 | [claude] termometro v2 dopo il controllo indipendente: campionamento puro, misura D_seq sull'intera risposta, replay esatto di Adam, cancelli, previsione su git prima delle corse, analisi scritta prima dei dati, 12 corse, premessa bozza 2 |
| `5ee8e89` | 2026-10-18T22:40:46+02:00 | [claude] termometro: prova sulla GPU vera (nessun dato dell'esperimento): campionamento puro confermato, picco 3,95 GB, 30 s per lotto di 32 risposte, 3,4 s per il passo all'indietro |
| `1ea6069` | 2026-10-19T18:14:42+02:00 | [claude] termometro: misure senza gradiente a lotti (stessa funzione per passo 0, corse, pavimento e controlli), prova a vuoto aggiornata e passata |
| `85a27da` | 2026-10-19T18:16:51+02:00 | [claude] termometro fase 2 (famiglie): premessa bozza, catena e analisi scritte prima di ogni dato della fase 1 |
| `5e24a8d` | 2026-10-19T18:18:43+02:00 | [claude] termometro: in modalita' solo processore la GPU viene nascosta prima di caricare torch (una prova a vuoto apriva un contesto CUDA e faceva cadere il registro) |
| `ef341c6` | 2026-10-19T18:19:19+02:00 | [claude] termometro: GPU nascosta con CUDA_VISIBLE_DEVICES=-1 (su Windows la variabile vuota viene cancellata) |
| `e6a93af` | 2026-10-19T18:46:24+02:00 | [claude] termometro: frasi del verdetto scritte prima dei dati (rapporto_fase1.py, provato su numeri finti) |
| `d1f3bc8` | 2026-10-19T18:58:36+02:00 | [claude] termometro: correzioni del secondo controllo indipendente (B1 commit della previsione, B2 G5 come pilota, B3 ESS e G6, B4 SFT a micro-lotti, B5 verifica del congelamento, B6 fase 2), premessa bozza 3 |
| `e8a6cbe` | 2026-10-19T19:05:05+02:00 | [claude] termometro: prove brevi sulla GPU vera prima del congelamento (lotti contro singolo, SFT, campione pilota per G5) |
| `4165a44` | 2026-10-19T21:10:05+02:00 | [claude] termometro: istruzioni per l'etichettatore indipendente di G5 e calcolo dell'accordo, scritti prima dei dati del pilota |
| `f22598f` | 2026-10-19T21:20:47+02:00 | [claude] termometro: le catene delle fasi 1 e 2 riprendono dopo un'interruzione senza fermarsi sul commit della previsione |
| `4ea6402` | 2026-10-19T21:46:42+02:00 | [claude] termometro: correzioni di testo dal controllo indipendente dell'idea, prima del congelamento |
| `cdb4ae6` | 2026-10-20T09:43:38+02:00 | [claude] termometro: taratura prima del congelamento, controlli in più (rumore, rilevatore dopo l'allenamento) e prova di velocità per Kaggle |
| `d6aae67` | 2026-10-20T10:21:40+02:00 | [claude] termometro: decisione B' di Emanuele, soglia bassa di G1 al 2% (taratura: né compiti più facili né più epoche portano la riuscita al 20-40%), limite dichiarato; prova a vuoto passata |
| `a7656b3` | 2026-10-20T10:47:02+02:00 | [claude] termometro: H6 (secondo comportamento, falsa dichiarazione di riuscita), piattaforma Kaggle fp32, correzioni della revisione indipendente |
| `979d472` | 2026-10-20T10:48:05+02:00 | [claude] termometro: CONGELAMENTO della premessa (fasi 1 e 2), del codice e del rilevatore, prima di ogni dato dell'esperimento |
| `0f9c05d` | 2026-10-20T11:51:19+02:00 | [claude] termometro: deviazione 1 (tecnica), lanciatori fase1_A.py/fase1_B.py con le stesse celle dei notebook congelati (l'editor di Kaggle non importava il notebook) |
| `68a337d` | 2026-10-20T13:35:59+02:00 | [claude] termometro: notebook A lanciato su Kaggle con kaggle kernels push (il .ipynb congelato, identico byte per byte) |
| `d79b2c2` | 2026-10-20T13:49:34+02:00 | [claude] (message withheld: it is about a private letter to the paper's authors) |
| `99c8e3c` | 2026-10-20T14:15:53+02:00 | [claude] termometro: fase2_sft.py (partenze delle 3 famiglie con la ricetta congelata, accanto a Numbrix con priorità bassa) |
| `83dc7e3` | 2026-10-20T14:39:49+02:00 | [claude] termometro: prepara_B.py (scarica la previsione di A, controlla le impronte, compila B cambiando una sola riga) |
| `ced396a` | 2026-10-20T14:40:50+02:00 | [claude] termometro: bozza della premessa della fase 1b (orizzonte lungo, 6 corse da 150 passi), scritta prima di ogni risultato delle corse |
| `99d33bd` | 2026-10-20T15:50:00+02:00 | [claude] termometro fase 1b: fase1b.py (estensione dell'orizzonte sul codice congelato, analisi provata su dati sintetici), notebook C, premessa aggiornata |
| `1ef0287` | 2026-10-20T15:54:43+02:00 | [claude] termometro fase 2: impronte delle tre partenze (q05, olmo1, smol17), scritte prima di ogni campione; dataset privato su Kaggle |
| `71330de` | 2026-10-20T16:09:28+02:00 | [claude] termometro fase 1b: correzioni della revisione indipendente (regola dell'orizzonte, mediana vera dell'ESS, rumore di D_samp dalla finestra di 20 passi con la sua regola, H2b descrittiva, regole per meno corse e due sessioni, frasi del verdetto, prova sintetica con previsione indipendente), prepara_C.py |
| `c62b1c2` | 2026-10-20T16:09:58+02:00 | [claude] termometro: CONGELAMENTO della fase 1b (orizzonte lungo), prima di qualunque risultato delle corse della fase 1 |
| `f6104eb` | 2026-10-20T17:17:07+02:00 | [claude] termometro fase 1: PREVISIONE su git prima delle corse (uscita del notebook A su Kaggle: termometri, cancelli tutti superati, impronte) |
| `e2635ba` | 2026-10-20T17:17:31+02:00 | [claude] termometro fase 1: notebook B lanciato su Kaggle (il congelato con le impronte di git, una sola riga diversa) |
| `f8a3097` | 2026-10-21T15:52:54+02:00 | [claude] termometro fase 1: uscita del notebook B (12 corse complete, analisi e rapporto congelati, code dei passi 36-40), scaricata da Kaggle il 21/10 15:52 |
| `e6657cc` | 2026-10-21T15:54:06+02:00 | [claude] termometro fase 1b: notebook C lanciato su Kaggle (congelato con le impronte di git, una sola riga diversa) |
| `05e5368` | 2026-10-21T15:59:03+02:00 | [claude] termometro fase 1: controlli indipendenti (Codex: CONFERMATO CON RISERVE; accordo del rilevatore al passo 40 100/100) e analisi esplorativa "popolazione" (Q 0,34, premessa del 20/10 23:05, commit 4f01a81) |
| `8f87974` | 2026-10-21T16:00:53+02:00 | [claude] termometro fase 1: ricalcolo indipendente da zero (agente pulito con il suo codice): CONFERMATO CON RISERVE, tutti i numeri tornano; ricostruzione del collasso dei pesi |
| `d308ce6` | 2026-10-21T16:21:16+02:00 | [claude] (message withheld: it is about a private letter to the paper's authors) |
| `1affa1f` | 2026-10-21T16:31:58+02:00 | [claude] termometro fase 2: BOZZA di modifica (test al passo 10, riserva al 5, corse da 20 passi), scritta dopo la fase 1 e prima di ogni dato della fase 2; non congelata |
| `9a6cdda` | 2026-10-21T16:34:42+02:00 | [claude] termometro fase 2: bozza di modifica corretta dopo la revisione di Codex (PRONTA DOPO CORREZIONI); revisione e prompt salvati |
| `b70aa1d` | 2026-10-21T16:40:33+02:00 | [claude] termometro fase 2: decisione di Emanuele (confermativo sulle 3 famiglie nuove) e analisi fase2_corta.py con prove su dati finti (non congelata) |
| `6aa3067` | 2026-10-21T17:46:20+02:00 | [claude] termometro fase 2 corta: notebook A2/B2 per famiglia (generatore che riusa le celle della fase 1) e prepara_fase2.py; non congelati, niente caricato |
| `b713ca3` | 2026-10-21T17:48:59+02:00 | [claude] termometro fase 2 corta: correzioni dopo la revisione del codice di Codex (PRONTO DOPO CORREZIONI): segnale di fine solo se l'analisi riesce, corse complete solo con i passi 0..20, ruolo descrittivo del passo 10 senza verdetto |
| `1768c2e` | 2026-10-21T17:57:57+02:00 | [claude] termometro: prove prima del congelamento messe su git (pilota G5 e H6 del 19-20/10, esiti della taratura, impronte delle partenze della fase 2); file creati prima del 20/10 10:47 salvo fase2_partenze.json (14:42, prima di ogni dato) |
| `602d6a6` | 2026-10-21T18:02:24+02:00 | [claude] termometro: bozza del pacchetto pubblico (README inglese, licenza Apache-2.0, esporta.py che costruisce il pacchetto da git senza pubblicare); riuscita al passo 0 corretta al 4% nella nota |
| `e7eebe8` | 2026-10-21T18:03:43+02:00 | [claude] termometro, pacchetto pubblico: versioni congelate dei due file con deviazione 1 in congelati/979d472/, file grandi della previsione dichiarati |
| `5ca25b6` | 2026-10-21T18:04:31+02:00 | [claude] termometro: prepara_fase2.py senza il percorso del PC |
| `c1bdc4f` | 2026-10-21T18:05:24+02:00 | [claude] termometro: elenco dei file del pacchetto pubblico (niente pubblicato) |
| `eb6b67f` | 2026-10-21T18:08:25+02:00 | [claude] termometro, README pubblico corretto dopo la revisione di Codex (PRONTO DOPO CORREZIONI) e dopo il controllo alla fonte dell'articolo (titolo, autori Stanford e Anthropic, Sezione 7, 300 passi, anchoring, partenza 865/135) |
| `0990829` | 2026-10-21T18:08:36+02:00 | [claude] termometro: elenco dei file del pacchetto aggiornato |
| `5818ab9` | 2026-10-21T18:16:20+02:00 | [claude] termometro: piano della pubblicazione di domani con le decisioni di Emanuele (nome, registri Kaggle, data) |
| `2bb3a6a` | 2026-10-22T16:15:24+02:00 | [claude] termometro fase 1b: uscita del notebook C (6 corse complete da 150 passi, previsione estesa, analisi e rapporto congelati), scaricata da Kaggle il 22/10 16:20 |
| `96095dd` | 2026-10-22T16:27:39+02:00 | [claude] termometro fase 1b: ricalcolo indipendente da zero (agente pulito con il suo codice): CONFERMATO CON RISERVE, numeri e verdetti identici |
| `81c681b` | 2026-10-22T16:29:02+02:00 | [claude] termometro: README pubblico con la fase 1b (verdetto congelato, tabella, riserve del ricalcolo indipendente, comando di rianalisi provato) e frase sul lavoro come nell'atlante; registro dello scarico fuori da git (percorso locale) |
| `bae339b` | 2026-10-22T16:30:07+02:00 | [claude] termometro fase 1b: percorsi del PC tolti dai file del ricalcolo indipendente (percorsi relativi, nota nel rapporto); rilanciati, numeri identici al byte |
| `fa84fb6` | 2026-10-22T16:33:40+02:00 | [claude] termometro: README dopo la revisione di Codex sulla 1b (PRONTO DOPO CORREZIONI, numeri confermati; 3 correzioni di formulazione applicate: SD osservata e non vera, crescita detta post-dati, potenza della 1b 42% per un errore di 3 volte) |
| `878ccc5` | 2026-10-22T18:32:01+02:00 | [claude] termometro fase 2: intestazione di stato della modifica (CONGELATA il 22/10, nome del file invariato), prima del commit di congelamento |
| `7949d45` | 2026-10-22T18:32:02+02:00 | [claude] termometro fase 2: manifest del dataset termometro-fase2-codice (prepara_fase2.py codice), prima del congelamento |
| `4a30469` | 2026-10-22T18:32:13+02:00 | [claude] termometro fase 2: CONGELAMENTO della fase 2 corta (FASE2_CONGELATA.txt, 17 file con sha256) dopo la 1b e prima di ogni corsa, con l'OK di Emanuele del 22/10 |
| `d7b2a14` | 2026-10-22T18:32:39+02:00 | [claude] termometro: README con la fase 2 congelata (22/10, FASE2_CONGELATA.txt) |
| `2c00993` | 2026-10-22T18:33:23+02:00 | [claude] termometro: README senza segnaposto (nomi dei notebook Kaggle, link quando saranno pubblici) |
| `7486337` | 2026-10-22T18:39:06+02:00 | [claude] termometro fase 2: notebook A2 di q05 come lanciato su Kaggle il 22/10 (identico al congelato), dopo il congelamento 4a30469 |
| `cb4df78` | 2026-10-22T18:40:24+02:00 | [claude] (message withheld: it is about a private letter to the paper's authors) |
| `80b1ad5` | 2026-10-22T19:18:35+02:00 | [claude] termometro fase 2: q05 fermata al cancello G1 (riuscita al passo 0 1,92% < 2%, 8/416), come da regola; lanciato A2 di OLMo-2-1B (identico al congelato) |
| `1a43d14` | 2026-10-22T19:20:47+02:00 | [claude] termometro: README con i link pubblici di Kaggle (3 notebook, codice congelato, partenza LoRA) e la prima famiglia della fase 2 fermata a G1 |
| `4abcf5b` | 2026-10-22T19:35:10+02:00 | [claude] termometro fase 2: decisione di Emanuele su q05 (compiti a 3 numeri, una volta, dopo OLMo e SmolLM2) e bozza dell'aggiunta, prima di ogni dato a 3 numeri |
| `02e4fef` | 2026-10-22T19:47:55+02:00 | [claude] termometro: fase2_q05_3numeri.py (solo make_instance a 3 numeri, poi termometro.py congelato invariato; prova su CPU superata) (the rest of the message is withheld: it is about a private letter to the paper's authors) |
| `ba885eb` | 2026-10-22T19:48:47+02:00 | [claude] termometro fase 2: OLMo-2-1B fermata al cancello G1 (8/416 come q05: conteggio identico da verificare prima di altre corse) |
| `ad65deb` | 2026-10-22T21:36:36+02:00 | [claude] termometro fase 2: verifica del conteggio di G1 (8/416 identico in q05 e olmo1 = caso: 800 fallimenti rivalutati tutti 0, compiti risolti diversi); lanciato A2 di SmolLM2-1.7B (identico al congelato) |
| `543dcae` | 2026-10-22T21:49:52+02:00 | [claude] termometro fase 2: deviazione D2 (SmolLM2: scarico senza onnx/, 22,7 GB oltre il disco di Kaggle; una sola sostituzione, controllo sha256 del modello invariato), prima di ogni dato di smol17 |
| `ff5bc8b` | 2026-10-22T22:02:08+02:00 | [claude] termometro fase 2: decisioni di Emanuele (22:01, prima di ogni risultato di SmolLM2): compiti a 3 numeri anche per OLMo, e per SmolLM2 se si ferma a G1; aggiunta e codice resi generali (PREMESSA_FASE2_3NUMERI.md, fase2_3numeri.py) |
| `6952fcd` | 2026-10-22T22:02:53+02:00 | [claude] termometro fase 2: testo dell'aggiunta a 3 numeri reso generale con le decisioni delle 22:02 (il commit precedente aveva solo rinominato) |
| `deb20bb` | 2026-10-22T22:04:59+02:00 | [claude] termometro fase 2: generatore e notebook A2/B2 a 3 numeri per q05, olmo1, smol17 (3 modifiche controllate rispetto ai congelati), fase2_3numeri.py con TERMO_CODICE; non congelati, niente caricato |
| `9b53541` | 2026-10-22T22:09:21+02:00 | [claude] termometro fase 2: correzioni dopo Codex sui 3 numeri (PRONTO DOPO CORREZIONI): uscite separate _3n e segno TRE_NUMERI.txt controllato da B2; due regimi riportati separati; impronta completa in D2; niente SFT con fase2_3numeri.py. Non congelato |
| `bc55bd1` | 2026-10-22T22:13:54+02:00 | [claude] (message withheld: it is about a private letter to the paper's authors) |
| `f1409c6` | 2026-10-22T22:17:24+02:00 | [claude] (message withheld: it is about a private letter to the paper's authors) |
| `49eda9c` | 2026-10-22T22:22:21+02:00 | [claude] (message withheld: it is about a private letter to the paper's authors) |
| `43ed4eb` | 2026-10-22T22:26:18+02:00 | [claude] (message withheld: it is about a private letter to the paper's authors) |
| `75763fa` | 2026-10-22T22:28:29+02:00 | [claude] termometro fase 2 a 3 numeri: stato CONGELATA nella premessa dopo il secondo controllo di Codex (PRONTO), prima del commit delle impronte |
| `4a38345` | 2026-10-22T22:28:30+02:00 | [claude] termometro fase 2 a 3 numeri: CONGELAMENTO (FASE2_3NUMERI_CONGELATA.txt, 14 file con sha256), prima di ogni dato a 3 numeri, con le decisioni di Emanuele e due revisioni di Codex |
| `30d3166` | 2026-10-22T22:28:53+02:00 | [claude] termometro fase 2 a 3 numeri: cartella di lancio di A2 q05 3n (kernel nuovo termometro-fase-2-q05-3n-previsione, notebook identico al congelato), non ancora lanciata |
| `c049c58` | 2026-10-23T16:08:22+02:00 | [claude] termometro fase 2: previsione di SmolLM2-1.7B (notebook A2 v2 con la deviazione D2), tutti i cancelli superati; file piccoli e impronte, prima del notebook B2 |
| `adb5768` | 2026-10-23T16:08:56+02:00 | [claude] termometro fase 2: notebook B2 di SmolLM2 come lanciato (copia D2 senza onnx + impronte della previsione: 2 righe diverse dal congelato) |
| `eb10c87` | 2026-10-23T21:40:25+02:00 | [claude] termometro fase 2: SmolLM2-1.7B, 12 corse su 12, verdetto al passo 10 PROMOSSO al limite (R = 1,98, IC 1,40-3,37), analisi rifatta in locale identica; Qwen a 3 numeri lanciato |
| `dd7ff5a` | 2026-10-23T22:18:32+02:00 | [claude] termometro fase 2 a 3 numeri: Qwen2.5-0.5B fermo a G1 (0 su 416; controllo indipendente 0 su 400, 50 compiti tutti risolvibili), esce dalla fase 2; OLMo a 3 numeri lanciato |
| `c23dc29` | 2026-10-23T22:18:48+02:00 | [claude] termometro fase 2 a 3 numeri: registro di Kaggle di Qwen (kaggle kernels logs) |
| `a1e49bd` | 2026-10-23T22:32:28+02:00 | [claude] termometro fase 2 a 3 numeri: OLMo-2-1B fermo a G1 (0 su 416), esce dalla fase 2; resta valutabile solo SmolLM2 |
| `a4483e4` | 2026-10-24T18:00:04+02:00 | [claude] Termometro, fase 2: previsione e cancelli di SmolLM2, code cieche del passo 20 per G5, sorgenti dei dataset Kaggle della fase 2 e dei 3 numeri (i pesi restano fuori da git) |
| `35c353a` | 2026-10-24T18:02:34+02:00 | [claude] Termometro, fase 2: confronto fra famiglie con il codice congelato (promosse 1 su 3; ordine rispetto a Qwen2.5-1.5B sbagliato, 0 su 1) e G5 di SmolLM2 (100 su 100, ma tutte ammissioni) |
| `bf94f2b` | 2026-10-24T18:04:00+02:00 | [claude] Termometro: README con i risultati della fase 2 (bozza, non pubblicata); esportazione esclude la lettera agli autori e le sue revisioni |
| `a5960fa` | 2026-10-24T18:07:45+02:00 | [claude] Termometro: README della fase 2 corretto dopo il primo controllo di Codex (regime a 3 numeri, H6 con la regola di parità, scomposizione) |
| `f2e7e70` | 2026-10-24T18:11:39+02:00 | [claude] Termometro: secondo controllo di Codex sul README della fase 2 (PRONTO); nella provenienza pubblica i messaggi dei commit sulla lettera agli autori sono nascosti |
| `6c56381` | 2026-10-24T18:12:41+02:00 | [claude] Termometro: elenco esatto dei file dell'aggiornamento pubblico della fase 2 (da approvare) |
| `d30834c` | 2026-10-24T18:12:52+02:00 | [claude] Termometro: frase corretta nella provenienza per il commit misto |
| `aa60845` | 2026-10-24T18:47:48+02:00 | [claude] Termometro: nella revisione di Codex del README il link usa il percorso relativo invece di quello del PC (contenuto invariato) |
| `c0dbbe3` | 2026-10-24T19:00:24+02:00 | [claude] Termometro, fase 2: correzioni dopo il controllo indipendente (ripetibilità del termometro, ordine debole, 22,7 GB, notebook privati, file .pt); ricalcolo indipendente CONFERMATO; prompt accorciato; log vuoto escluso |
| `3309a07` | 2026-10-24T19:10:25+02:00 | [claude] Termometro: README precisato dopo il controllo finale (1,7 volte, previsioni pari con la ripetizione, rimandi ai file, ordine debole anche nella lettura, notebook mai lanciati) |
| `ed85fc9` | 2026-10-24T19:11:09+02:00 | [claude] Termometro: elenco esatto dei file rigenerato |
| `365a110` | 2026-10-24T20:24:32+02:00 | [claude] Termometro: README con i link ai notebook e ai dataset Kaggle della fase 2 resi pubblici il 24/10 e al dataset della fase 1b |
| `f647bcf` | 2026-10-24T20:27:58+02:00 | [claude] Termometro: README, quali dataset legge ciascun notebook della fase 2 (dopo il controllo separato) |
