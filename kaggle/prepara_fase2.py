"""Phase 2, short version (21/10/2026): the steps around the Kaggle notebooks of crea_notebook_fase2.py.

usage (from strumenti/termometro/kaggle):
  python prepara_fase2.py codice      -> codice_fase2/ (fase2_corta.py + analisi_fase2.py as in git, LF, with
                                         MANIFEST_FASE2.txt): the dataset emanuelerizzan/termometro-fase2-codice
  python prepara_fase2.py A FAM       -> push_A2_FAM/ (notebook A2 + kernel-metadata.json)
  python prepara_fase2.py B FAM       -> downloads the small prediction files of A2, checks them against its
                                         previsione_sha256.txt and, only if they are already committed to git,
                                         builds push_B2_FAM/ with the git hashes (the ONLY line that differs from the
                                         frozen notebook B2)
Nothing is uploaded or pushed here: the kaggle commands are printed, to be run after Emanuele's OK.
"""
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TERMO = os.path.dirname(HERE)
GIT = ["git", "-c", "safe.directory=*"]  # the sandbox runs git as another user
FAMS = ["q05", "olmo1", "smol17"]
SMALL = ["thermo_s1.json", "thermo_s2.json", "gates.json", "gates_pool.json", "pool.json"]
PLACEHOLDER = "GIT_SHA = {}  # <- da riempire con il contenuto di previsione_sha256.txt del commit su git"
CODE_FILES = ["fase2_corta.py", "analisi_fase2.py"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def git(*args, check=True):
    return subprocess.run(GIT + list(args), cwd=TERMO, capture_output=True, check=check)


def meta(kid, title, code_file, datasets, kernels):
    return {"id": f"emanuelerizzan/{kid}", "title": title, "code_file": code_file, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": [f"emanuelerizzan/{d}" for d in datasets],
            "competition_sources": [], "kernel_sources": [f"emanuelerizzan/{k}" for k in kernels]}


def codice():
    # analisi_fase2.py must be the one frozen on 20/10 with PREMESSA_FASE2.md (the commit that adds CONGELATA.txt)
    freeze = git("log", "--diff-filter=A", "--format=%H", "--", "CONGELATA.txt").stdout.decode().split()[-1]
    assert git("diff", "--quiet", freeze, "HEAD", "--", "analisi_fase2.py", check=False).returncode == 0, \
        "analisi_fase2.py è cambiato dopo il congelamento del 20/10"
    for f in CODE_FILES:
        assert git("diff", "--quiet", "HEAD", "--", f, check=False).returncode == 0, f"{f}: modifiche non su git"
    out = os.path.join(HERE, "codice_fase2")
    os.makedirs(out, exist_ok=True)
    head = git("rev-parse", "--short", "HEAD").stdout.decode().strip()
    lines = [f"# fase 2 corta: file uguali a git {head} (a capo LF); analisi_fase2.py uguale al congelamento {freeze[:7]}"]
    for f in CODE_FILES:
        blob = git("show", f"HEAD:./{f}").stdout  # the git blob, LF line endings
        open(os.path.join(out, f), "wb").write(blob)
        lines.append(f"{sha(os.path.join(out, f))} {f}")
    open(os.path.join(out, "MANIFEST_FASE2.txt"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    json.dump({"title": "termometro fase2 codice", "id": "emanuelerizzan/termometro-fase2-codice",
               "licenses": [{"name": "apache-2.0"}]}, open(os.path.join(out, "dataset-metadata.json"), "w"), indent=2)
    print(open(os.path.join(out, "MANIFEST_FASE2.txt"), encoding="utf-8").read())
    print("caricamento (dopo l'OK):", f'cd "{out}" && python -m kaggle datasets create -p .')


def push_a(fam):
    out = os.path.join(HERE, f"push_A2_{fam}")
    os.makedirs(out, exist_ok=True)
    nb = f"fase2_A_{fam}.ipynb"
    open(os.path.join(out, nb), "w", encoding="utf-8").write(open(os.path.join(HERE, nb), encoding="utf-8").read())
    json.dump(meta(f"termometro-fase-2-{fam}-previsione", f"termometro fase 2 {fam} previsione", nb,
                   ["termometro-codice-congelato", "termometro-partenze-fase-2"], []),
              open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    print(f"push_A2_{fam} pronto; lancio (dopo l'OK): python -m kaggle kernels push -p \"{out}\"")


def push_b(fam):
    ka = f"emanuelerizzan/termometro-fase-2-{fam}-previsione"
    dest = os.path.join(TERMO, "risultati", "fase2_kaggle_A")
    os.makedirs(dest, exist_ok=True)
    pat = rf"{fam}/(previsione_sha256\.txt|thermo_s[12]\.json|gates(_pool)?\.json|pool\.json|log/.*\.log)$"
    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", ka, "-p", dest, "--file-pattern", pat, "-o"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")  # Windows: not cp1252
    print(r.stdout[-1500:], r.stderr[-800:])
    d = os.path.join(dest, fam)
    hfile = os.path.join(d, "previsione_sha256.txt")
    assert os.path.exists(hfile), "previsione_sha256.txt non scaricato: il notebook A2 è finito bene?"
    hashes = {p[1]: p[0] for p in (l.split() for l in open(hfile, encoding="utf-8")) if len(p) == 2 and len(p[0]) == 64}
    assert set(SMALL) | {"vettori_s1.pt", "gradienti_s1.pt"} <= set(hashes), hashes
    for n in SMALL:
        assert sha(os.path.join(d, n)) == hashes[n], f"impronta diversa per {n}"
    print("file piccoli identici alle impronte di A2:", SMALL)
    g = json.load(open(os.path.join(d, "gates.json")))
    print("cancelli:", {k: v for k, v in g.items() if k.startswith("G") or k == "ALL_PASS"})
    rel = os.path.relpath(hfile, TERMO).replace(os.sep, "/")
    tracked = git("ls-files", "--error-unmatch", rel, check=False).returncode == 0
    clean = git("diff", "--quiet", "HEAD", "--", rel, check=False).returncode == 0
    if not (tracked and clean):
        print(f"PRIMA il commit della previsione su git ({os.path.relpath(d, TERMO)}: file piccoli, log e "
              "previsione_sha256.txt), poi di nuovo questo comando. push_B2 non è stato scritto.")
        sys.exit(1)
    nb = f"fase2_B_{fam}.ipynb"
    raw = open(os.path.join(HERE, nb), encoding="utf-8").read()
    assert raw.count(PLACEHOLDER) == 1
    filled = raw.replace(PLACEHOLDER, "GIT_SHA = " + json.dumps(hashes).replace('"', '\\"') +
                         "  # impronte della previsione, copiate dal commit su git prima di lanciare B2")
    json.loads(filled)
    a, b = raw.splitlines(), filled.splitlines()
    assert len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1, "B2 differisce dal congelato in più di una riga"
    out = os.path.join(HERE, f"push_B2_{fam}")
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, nb), "w", encoding="utf-8").write(filled)
    json.dump(meta(f"termometro-fase-2-{fam}-corse", f"termometro fase 2 {fam} corse", nb,
                   ["termometro-codice-congelato", "termometro-partenze-fase-2", "termometro-fase2-codice"],
                   [f"termometro-fase-2-{fam}-previsione"]),
              open(os.path.join(out, "kernel-metadata.json"), "w"), indent=2)
    print(f"push_B2_{fam} pronto (una sola riga diversa dal congelato); lancio: python -m kaggle kernels push -p \"{out}\"")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "codice":
        codice()
    else:
        assert sys.argv[2] in FAMS, f"famiglia sconosciuta: {sys.argv[2]}"
        push_a(sys.argv[2]) if cmd == "A" else push_b(sys.argv[2])
