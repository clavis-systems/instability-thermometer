"""Builds the Kaggle notebooks of phase 2, short version (PREMESSA_FASE2_MODIFICA_BOZZA.md, 21/10/2026): for each
family an A2 notebook (the prediction, the same cells as phase-1 A) and a B2 notebook (12 runs of 20 steps, then
fase2_corta.py). The cells of crea_notebook_fase1.py are reused; only the lines that name the family, its base model,
its warm start and the 20 steps change, and every change is checked to happen exactly once.
usage: python crea_notebook_fase2.py   (writes fase2_A_<fam>.ipynb and fase2_B_<fam>.ipynb next to it)"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from crea_notebook_fase1 import A_GATES, A_POOL, A_THERMO, B_CHECK, B_RUNS, SETUP, SFT_SHA, nb  # noqa: E402

# family: (Hugging Face repo, sha256 of model.safetensors, sha256 of the warm-start adapter), from partenze_fase2/MANIFEST.txt
FAMILIES = {
    "q05": ("Qwen/Qwen2.5-0.5B-Instruct", "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
            "9a28754d96f6cef25b0c3e6c644878a1fafaf49183d908e803d2194bc40f0f5b"),
    "olmo1": ("allenai/OLMo-2-0425-1B-Instruct", "36d044c73655bb904f822915e6294ba3dae8e6e1af5e703e9d452f2d6a3a294f",
              "0cb6beedb64846a1b5835e0fc9cc316d6adc4fbdcb144268d468f8d1087484ef"),
    "smol17": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8",
               "d158d49681ca2a78d79d2e314726f779f5f45bdc23bda9de0c695d57a9748f8a"),
}


def sub(text, pairs):
    for old, new in pairs:
        assert text.count(old) == 1, f"expected exactly one occurrence of: {old}"
        text = text.replace(old, new)
    return text


def setup(fam):
    repo, base_sha, sft_sha = FAMILIES[fam]
    return sub(SETUP, [
        ("OUT = f'{W}/q15'", f"OUT = f'{{W}}/{fam}'"),
        ("snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir=f'{W}/qwen')",
         f"snapshot_download('{repo}', local_dir=f'{{W}}/base')"),
        ("assert sha(f'{W}/qwen/model.safetensors').startswith('dd924a11'), 'modello base diverso da casa'",
         f"assert sha(f'{{W}}/base/model.safetensors') == '{base_sha}', 'modello base diverso da casa'"),
        ("glob.glob('/kaggle/input/**/sft/adapter_model.safetensors', recursive=True)",
         f"glob.glob('/kaggle/input/**/{fam}/sft/adapter_model.safetensors', recursive=True)"),
        (f"== '{SFT_SHA}', 'partenza diversa da quella del 19/10'", f"== '{sft_sha}', 'partenza diversa da quella del 20/10'"),
        ("cfg['base_model_name_or_path'] = f'{W}/qwen'", "cfg['base_model_name_or_path'] = f'{W}/base'"),
    ])


B2_CODE = r'''# codice della fase 2 corta (fase2_corta.py e analisi_fase2.py), con le impronte del congelamento
F2 = os.path.dirname(glob.glob('/kaggle/input/**/MANIFEST_FASE2.txt', recursive=True)[0])
m2 = [l.split() for l in open(f'{F2}/MANIFEST_FASE2.txt', encoding='utf-8') if l.strip() and not l.startswith('#')]
assert m2 and all(sha(f'{F2}/{n}') == h for h, n in m2), 'CODICE DELLA FASE 2 DIVERSO DAL CONGELAMENTO'
print('codice della fase 2 identico al congelamento:', open(f'{F2}/MANIFEST_FASE2.txt', encoding='utf-8').readline().strip())
'''


def b_check(fam):
    return sub(B_CHECK, [
        ("'/kaggle/input/**/q15/previsione_sha256.txt'", f"'/kaggle/input/**/{fam}/previsione_sha256.txt'"),
        ("'/kaggle/input/**/q15/run_s*.json'", f"'/kaggle/input/**/{fam}/run_s*.json'"),
        ("if len(json.load(open(f))) == 41:", "if [r['step'] for r in json.load(open(f))] == list(range(21)):"),
    ])


B_RUNS_20 = sub(B_RUNS, [
    ("    return os.path.exists(f) and len(json.load(open(f))) == 41\n",
     "    try:  # complete = rows for steps 0..20 in order; a file cut by a stopped process is incomplete (as in 1b)\n"
     "        return os.path.exists(f) and [r['step'] for r in json.load(open(f))] == list(range(21))\n"
     "    except (json.JSONDecodeError, OSError, KeyError, TypeError):\n"
     "        return False\n"),
    ("stima = max(durate) if durate else 75 * 60", "stima = max(durate) if durate else 45 * 60"),
    ("str(s), '40'], log, gpu)", "str(s), '20'], log, gpu)"),
])


def b_analysis(fam):
    return r'''# analisi della fase 2 corta per questa famiglia e code dei passi 16-20 (solo con tutte le 12 corse)
if len(fatte) == 12:
    env = dict(os.environ, PYTHONPATH=CODE)  # fase2_corta.py imports the frozen analisi_fase1.py
    esiti = []
    for script, args in [(f'{F2}/fase2_corta.py', ['analyze', OUT, OUT, f'{OUT}/analisi_fase2_corta.json']),
                         (f'{CODE}/estrai_code_passo40.py', [OUT, f'{OUT}/g5_passo20'])]:
        r = subprocess.run([sys.executable, '-X', 'utf8', script] + args, capture_output=True, text=True, env=env)
        print(os.path.basename(script), r.returncode); print(r.stdout[-3000:]); print(r.stderr[-1500:])
        esiti.append(r.returncode)
    if esiti == [0, 0]:  # the marker only when both steps worked (review 21/10)
        with open(f'{OUT}/FASE2_FAM_FINITA.txt', 'w') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()) + '\n')
    else:
        print('ANALISI O CODE FALLITE: niente segnale di fine; le corse sono comunque tra gli output')
else:
    print('mancano corse: nuova versione con questo output come ingresso')
shutil.rmtree(f'{W}/base', ignore_errors=True)
'''.replace("FAM", fam)


def cells(fam):
    repo = FAMILIES[fam][0]
    a = [("markdown", f"# Termometro, fase 2 corta, `{fam}` ({repo}), notebook A2: la previsione\nImpostazioni: **GPU T4 x2**, "
                      "**Internet attivo**. Ingressi: il codice congelato e le partenze della fase 2. Le stesse celle del "
                      "notebook A della fase 1: campione del passo 0 (seme 7), cancelli, termometri seme 1 (K=50) e seme 2 "
                      "(K=25), impronte della previsione. Poi la previsione va su git PRIMA del notebook B2."),
         ("code", setup(fam)), ("code", A_POOL), ("code", A_THERMO),
         ("code", sub(A_GATES, [("shutil.rmtree(f'{W}/qwen')", "shutil.rmtree(f'{W}/base')")]))]
    b = [("markdown", f"# Termometro, fase 2 corta, `{fam}` ({repo}), notebook B2: le 12 corse da 20 passi\nImpostazioni: "
                      "**GPU T4 x2**, **Internet attivo**. Ingressi: codice congelato, codice della fase 2, partenze e "
                      "l'output del notebook A2. Controlla che la previsione sia identica a quella su git, poi semi 101-106 "
                      "sulla GPU 0 e 107-112 sulla GPU 1 (`termometro.py run ... 20`), poi `fase2_corta.py analyze` (verdetto "
                      "al passo 10, riserva al 5) e le code dei passi 16-20. Tutto come in `PREMESSA_FASE2_MODIFICA`."),
         ("code", setup(fam)), ("code", B2_CODE), ("code", b_check(fam)), ("code", B_RUNS_20), ("code", b_analysis(fam))]
    return a, b


if __name__ == "__main__":
    for fam in FAMILIES:
        a, b = cells(fam)
        for name, cs in [(f"fase2_A_{fam}.ipynb", a), (f"fase2_B_{fam}.ipynb", b)]:
            for t, s in cs:
                if t == "code":
                    compile(s, name, "exec")  # syntax check of every cell
            json.dump(nb(cs), open(os.path.join(HERE, name), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print("scritto", name)
