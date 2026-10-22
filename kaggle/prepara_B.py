"""After notebook A (20/10/2026): download the small prediction files from Kaggle, check them against A's
previsione_sha256.txt, and build push_B/ for notebook B with the git hashes filled in. The ONLY difference between
push_B/fase1_B_corse.ipynb and the frozen notebook is the GIT_SHA line (checked here). The git commit of the
prediction is done by hand right after this script, BEFORE pushing B.

usage (from strumenti/termometro/kaggle): python prepara_B.py
"""
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KA = "emanuelerizzan/termometro-fase-1-previsione"
DEST = os.path.join(os.path.dirname(HERE), "risultati", "q15_kaggle")
SMALL = ["thermo_s1.json", "thermo_s2.json", "gates.json", "gates_pool.json", "pool.json"]
PLACEHOLDER = "GIT_SHA = {}  # <- da riempire con il contenuto di previsione_sha256.txt del commit su git"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    os.makedirs(DEST, exist_ok=True)
    pat = r"(previsione_sha256\.txt|thermo_s[12]\.json|gates(_pool)?\.json|pool\.json|\.log)$"
    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", KA, "-p", DEST, "--file-pattern", pat, "-o"],
                       capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-800:])
    found = {}
    for root, _, files in os.walk(DEST):
        for f in files:
            found.setdefault(f, os.path.join(root, f))
    assert "previsione_sha256.txt" in found, "previsione_sha256.txt non scaricato: il notebook A è finito bene?"
    hashes = {}
    for line in open(found["previsione_sha256.txt"], encoding="utf-8"):
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            hashes[parts[1]] = parts[0]
    assert set(SMALL) | {"vettori_s1.pt", "gradienti_s1.pt"} <= set(hashes), hashes
    for n in SMALL:
        assert sha(found[n]) == hashes[n], f"impronta diversa per {n}"
    print("file piccoli identici alle impronte di A:", SMALL)
    g = json.load(open(found["gates.json"]))
    print("cancelli:", {k: v for k, v in g.items() if k.startswith("G") or k == "ALL_PASS"})
    nb_path = os.path.join(HERE, "fase1_B_corse.ipynb")
    raw = open(nb_path, encoding="utf-8").read()
    assert raw.count(PLACEHOLDER) == 1
    filled = raw.replace(PLACEHOLDER, "GIT_SHA = " + json.dumps(hashes).replace('"', '\\"') +
                         "  # impronte della previsione, copiate dal commit su git prima di lanciare B")
    out_dir = os.path.join(HERE, "push_B")
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "fase1_B_corse.ipynb"), "w", encoding="utf-8").write(filled)
    json.loads(filled)  # still a valid notebook
    a = raw.splitlines()
    b = filled.splitlines()
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(a) == len(b) and len(diff) == 1, "il notebook B differisce dal congelato in più di una riga"
    meta = {"id": "emanuelerizzan/termometro-fase-1-corse", "title": "termometro fase 1 corse",
            "code_file": "fase1_B_corse.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": ["emanuelerizzan/termometro-codice-congelato", "emanuelerizzan/termometro-partenza-q15"],
            "competition_sources": [], "kernel_sources": [KA]}
    json.dump(meta, open(os.path.join(out_dir, "kernel-metadata.json"), "w"), indent=2)
    print("push_B pronto; unica riga diversa dal congelato:", diff[0][1].strip()[:120], "...")
    print("ADESSO: commit su git di", DEST, "(file piccoli + previsione_sha256.txt), poi kaggle kernels push -p push_B")


if __name__ == "__main__":
    main()
