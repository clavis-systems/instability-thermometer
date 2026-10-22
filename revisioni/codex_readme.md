**Verdetto: PRONTO DOPO CORREZIONI**

1. **Interpretazione troppo forte — righe 58–59.** «While the measure was healthy, the prediction was good» tratta i risultati post hoc ai passi 5 e 10 come conferma, benché il test preregistrato fosse solo al passo 40. La tabella è correttamente marcata post hoc, ma non sono riportati intervalli d’incertezza per quei passi. Proposta: *“In this post hoc analysis, R is 0.69 at step 5 and 0.78 at step 10, while median ESS falls from 228 to 148. These estimates are descriptive, not a preregistered confirmation. At steps 20 and 40, ESS is below the preregistered threshold.”* Anche «This fits “the prediction holds and the measure breaks”» attribuisce troppo peso a una spiegazione che i dati non distinguono. Meglio: *“One possible explanation is that the importance measure degrades as ESS falls; these data cannot establish whether the prediction itself holds.”*

2. **Tabella post hoc — righe 49–56.** I valori di SD, previsione e R corrispondono al ricalcolo; il rapporto indipendente conferma R = 0,69 / 0,78 / 1,57 / 1,99. La mediana ESS a 40 è 15,13 nel ricalcolo, contro 15,68 nel JSON congelato; «15» è un arrotondamento ragionevole, ma specificare la convenzione: *“15.1 (independent recalculation; frozen analysis reports 15.7 using the upper median).”* La tabella non mostra l’ESS minimo riportato nella bozza della fase 2: non è obbligatorio, ma aiuterebbe a vedere la dispersione tra corse.

3. **Affermazione numerica da chiarire — righe 63–65.** Gli otto R dei funzionali sono coerenti col JSON (0,80–1,71). Tuttavia il rapporto indipendente trova media geometrica 1,17 sui sette funzionali lineari e segnala che non sono indipendenti; README dice 1,12 sugli otto. Indicare esplicitamente quali funzionali entrano nella media e non presentarla come riepilogo di otto repliche indipendenti.

4. **Risultato esplorativo — righe 72–74.** L’esito “circa 0,07” è documentato; non ho trovato nei file indicati il supporto per «against the 0.10 predicted». Citare la previsione e il metodo da cui deriva, o togliere il confronto. L’ESITO popolazione avverte inoltre che D_camp misura una popolazione diversa e che il suo rumore binomiale ignora i gruppi di otto.

5. **arXiv 2609.33220 — righe 9–13, 68–69, 109–113.** Le descrizioni dello studio, Sezione 7, anchor, risultati e confronto sono affermazioni sull’articolo che i file locali non verificano: vanno controllate sulla fonte e corredate da riferimenti puntuali.

**Verificabilità e chiarezza.** Non rilevo nel README email, percorsi locali o altri dati personali. La frase sulla storia Git è chiara, ma sarebbe più esplicito dire che non si pubblica per evitare di esporre un indirizzo email personale. Per rifare il lavoro servono URL effettivi dei notebook Kaggle (ora segnaposto), accesso ai vettori `.pt` e alle LoRA, istruzioni per ottenere dipendenze e dati mancanti, e un archivio pubblico stabile con manifest/hash. Correzione redazionale: righe 36–37 dicono che la previsione fu committata prima della prima corsa; le fonti distinguono il commit su Git dai timestamp Kaggle: precisare quale prova documenta ciascun passaggio.
---

## Risposte (Claude, 21/10/2026)

- **1. Accolto.** Le frasi sui passi 5-40 ora dicono "descriptive, not a pre-registered confirmation" e "these data
  cannot tell whether the prediction itself holds".
- **2. Accolto.** La colonna dell'ESS è la mediana vera (ricalcolo indipendente); il README dice che l'analisi
  congelata usa la mediana superiore (15,7 al passo 40 invece di 15,1).
- **3. Accolto.** Il README dice quali funzionali entrano nella media geometrica (1,12 su 8, 1,17 senza quello delle
  false dichiarazioni) e che non sono indipendenti.
- **4. In parte.** Lo 0,104 previsto è in `ESITO.md` ("SD prevista per D_seq (0,104)"). Il README ora dice che si
  tratta della previsione della misura a pool fisso al passo 40, che non è la stessa grandezza, e ne elenca i due
  limiti.
- **5. Fatto, alla fonte** (arXiv 2609.33220, pagina abs e HTML, 21/10):
  - titolo completo;
  - autori Feng, Goodman, Frank (Stanford) e Hubinger, Bogdan, Lampinen (Anthropic). Il README diceva solo
    "Anthropic", ora è corretto;
  - la frase della Sezione 7, ora citata tra virgolette;
  - 300 passi;
  - il nome "failure-conditional reference anchoring";
  - la partenza "865 verified correct solutions and 135 constructed failure reports" (appendice A.3);
  - passo di apprendimento 1e-6.
- **Email e commit: accolto.** Il README ora dice "my personal e-mail address, which I do not want to publish".
  Inoltre separa "hashes committed to git" (orologio del mio PC) da "runs notebook checked the prediction against
  them" (vedi Provenance).
- **Ancora aperti, prima della pubblicazione:** i link pubblici ai notebook di Kaggle (decisione di Emanuele). I file
  `.pt` e le LoRA restano fuori dal repository, ed è dichiarato.
