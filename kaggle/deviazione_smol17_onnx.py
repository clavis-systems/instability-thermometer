"""Fase 2, deviazione D2 (DEVIAZIONI_FASE2.md, 22/10/2026): SmolLM2-1.7B senza la cartella onnx/.

Il notebook A2 congelato di smol17 si è fermato con "No space left on device": snapshot_download scarica tutto il
repository HuggingFaceTB/SmolLM2-1.7B-Instruct (22,7 GB, quasi tutto in onnx/), oltre lo spazio di Kaggle. Questo file
scrive copie dei notebook congelati A2 e B2 di smol17 con UNA sola sostituzione: ignore_patterns=['onnx/*'] nello
scarico. Il controllo sha256 di model.safetensors resta, quindi il modello è lo stesso. Nient'altro cambia.

uso (da strumenti/termometro/kaggle): python deviazione_smol17_onnx.py
"""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
VECCHIO = "snapshot_download('HuggingFaceTB/SmolLM2-1.7B-Instruct', local_dir=f'{W}/base')"
NUOVO = "snapshot_download('HuggingFaceTB/SmolLM2-1.7B-Instruct', local_dir=f'{W}/base', ignore_patterns=['onnx/*'])"

for nome in ("fase2_A_smol17.ipynb", "fase2_B_smol17.ipynb"):
    testo = open(os.path.join(HERE, nome), encoding="utf-8").read()
    n = testo.count(VECCHIO)
    assert n == 1, f"{nome}: la riga dello scarico compare {n} volte"
    nuovo = testo.replace(VECCHIO, NUOVO)
    uscita = os.path.join(HERE, nome.replace(".ipynb", "_d2.ipynb"))
    open(uscita, "w", encoding="utf-8", newline="\n").write(nuovo)
    print(f"{os.path.basename(uscita)}  sha256 {hashlib.sha256(nuovo.encode('utf-8')).hexdigest()}  (1 sostituzione)")
