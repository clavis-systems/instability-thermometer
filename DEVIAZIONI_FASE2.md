# Fase 2: deviazioni dal congelamento (FASE2_CONGELATA.txt, commit 4a30469)

## D2 (22/10/2026, 21:55) — SmolLM2-1.7B: scarico del modello senza la cartella onnx/

Scritta dopo il guasto e prima di qualunque dato di SmolLM2: il notebook si è fermato prima del campione del passo 0.

**Fatti.**
- Il notebook A2 di smol17 (versione 1, lanciato il 22/10 alle 21:42 e identico al congelato) si è fermato durante lo
  scarico del modello base con "No space left on device (os error 28)".
- Causa: `snapshot_download('HuggingFaceTB/SmolLM2-1.7B-Instruct', ...)` scarica l'intero repository, che pesa
  22,7 GB. Quasi tutto è nella cartella `onnx/` (dieci varianti ONNX), mentre il modello usato è `model.safetensors`
  (3,42 GB).
- Per q05 e olmo1 lo stesso scarico è andato a buon fine, perché i loro repository non hanno quella cartella.

**Cambio.**
- `kaggle/deviazione_smol17_onnx.py` scrive copie dei notebook congelati A2 e B2 di smol17
  (`kaggle/fase2_A_smol17_d2.ipynb`, `kaggle/fase2_B_smol17_d2.ipynb`).
- Ogni copia ha una sola sostituzione: `ignore_patterns=['onnx/*']` nello scarico.
- Il controllo sha256 di `model.safetensors` resta (f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8, lo stesso del notebook congelato), quindi il modello è identico a quello di casa; il filtro `onnx/*` esclude solo le varianti ONNX.
- Nient'altro cambia: codice, partenze, semi, cancelli e analisi restano quelli congelati.
- La versione 2 del notebook A2 di smol17 si lancia con la copia corretta.
