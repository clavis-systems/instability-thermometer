"""Phase 1b launcher (20/10/2026): after prepara_B.py has downloaded and checked A's prediction, build push_C/ for the
phase-1b notebook with the same git hashes. The ONLY difference from the frozen fase1b_C.ipynb is the GIT_SHA line.
usage (from strumenti/termometro/kaggle): python prepara_C.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "risultati", "q15_kaggle")
PLACEHOLDER = "GIT_SHA = {}  # <- da riempire con il contenuto di previsione_sha256.txt del commit su git (come nel notebook B)"


def main():
    hfile = next(os.path.join(r, f) for r, _, fs in os.walk(SRC) for f in fs if f == "previsione_sha256.txt")
    hashes = {}
    for line in open(hfile, encoding="utf-8"):
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            hashes[parts[1]] = parts[0]
    raw = open(os.path.join(HERE, "fase1b_C.ipynb"), encoding="utf-8").read()
    assert raw.count(PLACEHOLDER) == 1
    filled = raw.replace(PLACEHOLDER, "GIT_SHA = " + json.dumps(hashes).replace('"', '\\"') +
                         "  # impronte della previsione della fase 1, copiate dal commit su git")
    json.loads(filled)
    a, b = raw.splitlines(), filled.splitlines()
    assert len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1
    out = os.path.join(HERE, "push_C")
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "fase1b_C.ipynb"), "w", encoding="utf-8").write(filled)
    meta = {"id": "emanuelerizzan/termometro-fase-1b", "title": "termometro fase 1b", "code_file": "fase1b_C.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": ["emanuelerizzan/termometro-codice-congelato", "emanuelerizzan/termometro-fase1b-codice",
                                "emanuelerizzan/termometro-partenza-q15"],
            "competition_sources": [], "kernel_sources": ["emanuelerizzan/termometro-fase-1-previsione"]}
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    print("push_C pronto (una sola riga diversa dal congelato); lancio: kaggle kernels push -p push_C")


if __name__ == "__main__":
    main()
