"""Fase 2, compiti a 3 numeri (PREMESSA_FASE2_3NUMERI.md, 22/10/2026): notebook A2 e B2 per q05, olmo1, smol17.

Parte dai notebook congelati (per smol17 dalle copie con la deviazione D2, senza onnx/) e fa solo queste modifiche,
ognuna controllata perché avvenga esattamente una volta:
  1. una cella di testo in cima che dice che è la versione a 3 numeri;
  2. dopo la riga che trova il codice congelato (CODE), trova il dataset termometro-fase2-3numeri (CODE3) e controlla
     l'impronta di fase2_3numeri.py con MANIFEST_3NUMERI.txt;
  3. in lancia(): esegue f'{CODE3}/fase2_3numeri.py' al posto di f'{CODE}/termometro.py' e passa TERMO_CODICE=CODE,
     così fase2_3numeri.py usa il termometro.py congelato senza toccarlo;
  4. (revisione di Codex) uscite in {fam}_3n invece di {fam}; A2 scrive TRE_NUMERI.txt con l'impronta di
     fase2_3numeri.py; B2 legge solo da {fam}_3n e controlla quel segno prima di usare l'uscita di A2.

  python crea_notebook_3numeri.py          -> fase2_{A,B}_{fam}_3n.ipynb e codice_3numeri/ (dataset)
"""
import copy
import hashlib
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
TERMO = os.path.dirname(HERE)
FAMS = ["q05", "olmo1", "smol17"]
RIGA_CODE = "CODE = os.path.dirname(glob.glob('/kaggle/input/**/MANIFEST_CONGELAMENTO.txt', recursive=True)[0])"
NUOVE = [
    "CODE3 = os.path.dirname(glob.glob('/kaggle/input/**/MANIFEST_3NUMERI.txt', recursive=True)[0])\n",
    "m3 = [l.split() for l in open(f'{CODE3}/MANIFEST_3NUMERI.txt', encoding='utf-8') if l.strip() and not l.startswith('#')]\n",
    "assert m3 and all(sha(f'{CODE3}/{n}') == h for h, n in m3), 'fase2_3numeri.py DIVERSO DAL CONGELAMENTO'\n",
    "print('compiti a 3 numeri: fase2_3numeri.py identico al congelamento')\n",
    "open(f'{OUT}/TRE_NUMERI.txt', 'w').write(m3[0][0] + '\\n')  # segno di provenienza: uscita a 3 numeri\n",
]
LANCIO_VECCHIO = "f'{CODE}/termometro.py'"
LANCIO_NUOVO = "f'{CODE3}/fase2_3numeri.py'"
ENV_VECCHIO = "env = dict(os.environ, TERMO_FP32='1',"
ENV_NUOVO = "env = dict(os.environ, TERMO_CODICE=CODE, TERMO_FP32='1',"


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sostituzioni(fase, fam):
    """(vecchia, nuova) da applicare esattamente una volta: cartelle d'uscita e di ingresso separate da quelle a 4
    numeri (revisione di Codex, punto 1) e, in B2, il controllo che l'uscita di A2 sia a 3 numeri."""
    out = [(f"OUT = f'{{W}}/{fam}'", f"OUT = f'{{W}}/{fam}_3n'")]
    if fase == "B":
        a_vecchia = f"A = os.path.dirname(glob.glob('/kaggle/input/**/{fam}/previsione_sha256.txt', recursive=True)[0])"
        out += [(a_vecchia, a_vecchia.replace(f"/{fam}/", f"/{fam}_3n/")
                 + "\nassert open(f'{A}/TRE_NUMERI.txt', encoding='utf-8').read().strip() == m3[0][0], "
                   "'l uscita di A2 non e a 3 numeri'"),
                (f"glob.glob('/kaggle/input/**/{fam}/run_s*.json', recursive=True)",
                 f"glob.glob('/kaggle/input/**/{fam}_3n/run_s*.json', recursive=True)")]
    return out


def trasforma(nb, nome, fase, fam):
    nb = copy.deepcopy(nb)
    n_code = n_lancio = n_env = 0
    conti = {v: 0 for v, _ in sostituzioni(fase, fam)}
    for cella in nb["cells"]:
        if cella["cell_type"] != "code":
            continue
        righe, nuove = cella["source"], []
        for r in righe:
            if LANCIO_VECCHIO in r:
                n_lancio += r.count(LANCIO_VECCHIO)
                r = r.replace(LANCIO_VECCHIO, LANCIO_NUOVO)
            if ENV_VECCHIO in r:
                n_env += r.count(ENV_VECCHIO)
                r = r.replace(ENV_VECCHIO, ENV_NUOVO)
            for v, n in sostituzioni(fase, fam):
                if v in r:
                    conti[v] += r.count(v)
                    r = r.replace(v, n)
            nuove.append(r)
            if r.strip() == RIGA_CODE:
                n_code += 1
                if not r.endswith("\n"):
                    nuove[-1] = r + "\n"
                nuove.extend(NUOVE)
        cella["source"] = nuove
    assert (n_code, n_lancio, n_env) == (1, 1, 1), f"{nome}: sostituzioni {n_code}, {n_lancio}, {n_env} invece di 1, 1, 1"
    assert all(c == 1 for c in conti.values()), f"{nome}: cartelle {conti}"
    nota = {"cell_type": "markdown", "metadata": {}, "source": [
        "**Versione a 3 numeri** (`PREMESSA_FASE2_3NUMERI.md`): l'unica correzione permessa dopo il cancello G1. "
        "Uguale al notebook congelato tranne: compiti di Countdown con 3 numeri (`fase2_3numeri.py`, che usa "
        "`termometro.py` congelato senza modificarlo)."]}
    nb["cells"].insert(0, nota)
    return nb


def main():
    for fam in FAMS:
        for fase in "AB":
            base = f"fase2_{fase}_{fam}_d2.ipynb" if fam == "smol17" else f"fase2_{fase}_{fam}.ipynb"
            nb = json.load(open(os.path.join(HERE, base), encoding="utf-8"))
            nuovo = trasforma(nb, base, fase, fam)
            out = os.path.join(HERE, f"fase2_{fase}_{fam}_3n.ipynb")
            testo = json.dumps(nuovo, indent=1, ensure_ascii=False) + "\n"
            open(out, "w", encoding="utf-8", newline="\n").write(testo)
            print(f"{os.path.basename(out)}  da {base}  sha256 {sha_bytes(testo.encode('utf-8'))[:16]}")
    d = os.path.join(HERE, "codice_3numeri")
    os.makedirs(d, exist_ok=True)
    blob = open(os.path.join(TERMO, "fase2_3numeri.py"), "rb").read().replace(b"\r\n", b"\n")
    open(os.path.join(d, "fase2_3numeri.py"), "wb").write(blob)
    open(os.path.join(d, "MANIFEST_3NUMERI.txt"), "w", encoding="utf-8", newline="\n").write(
        f"# fase 2 a 3 numeri: fase2_3numeri.py (a capo LF)\n{sha_bytes(blob)} fase2_3numeri.py\n")
    json.dump({"title": "termometro fase2 3numeri", "id": "emanuelerizzan/termometro-fase2-3numeri",
               "licenses": [{"name": "apache-2.0"}]}, open(os.path.join(d, "dataset-metadata.json"), "w"), indent=2)
    print("codice_3numeri/: fase2_3numeri.py", sha_bytes(blob)[:16])


if __name__ == "__main__":
    main()
