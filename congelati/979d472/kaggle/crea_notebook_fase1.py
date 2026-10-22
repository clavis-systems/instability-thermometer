"""Builds the two Kaggle notebooks of phase 1 (20/10/2026). They replace catena_fase1.sh on Kaggle (2 x T4, float32),
with the same steps, seeds and gates. A = pool + gates + the two thermometers (the prediction, timestamped by
Kaggle and then committed to git); B = the 12 runs (6 per GPU), then the frozen analysis and report.
usage: python crea_notebook_fase1.py   (writes fase1_A_previsione.ipynb and fase1_B_corse.ipynb next to it)"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SFT_SHA = "c0506a7111c5c01f0c0ada9cb8e4a8d0c78e2bd91452350ac608f80427111fe6"

SETUP = r'''# ambiente, codice congelato, modello base e partenza (uguale in A e in B)
import os, glob, json, hashlib, shutil, subprocess, sys, time
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True).stdout)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==5.5.4', 'peft==0.18.0'], check=True)
W = '/kaggle/working'; OUT = f'{W}/q15'; os.makedirs(f'{OUT}/log', exist_ok=True)
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()
CODE = os.path.dirname(glob.glob('/kaggle/input/**/MANIFEST_CONGELAMENTO.txt', recursive=True)[0])
man = [l.split() for l in open(f'{CODE}/MANIFEST_CONGELAMENTO.txt', encoding='utf-8') if l.strip() and not l.startswith('#')]
bad = [n for h, n in man if sha(f'{CODE}/{n}') != h]
assert not bad, f'CODICE DIVERSO DAL CONGELAMENTO: {bad}'
print('codice identico al congelamento:', len(man), 'file;', open(f'{CODE}/MANIFEST_CONGELAMENTO.txt', encoding='utf-8').readline().strip())
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir=f'{W}/qwen')
assert sha(f'{W}/qwen/model.safetensors').startswith('dd924a11'), 'modello base diverso da casa'
SFT_IN = os.path.dirname(glob.glob('/kaggle/input/**/sft/adapter_model.safetensors', recursive=True)[0])
shutil.copytree(SFT_IN, f'{OUT}/sft', dirs_exist_ok=True)
assert sha(f'{OUT}/sft/adapter_model.safetensors') == 'SFT_SHA', 'partenza diversa da quella del 19/10'
cfg = json.load(open(f'{OUT}/sft/adapter_config.json')); cfg['base_model_name_or_path'] = f'{W}/qwen'
json.dump(cfg, open(f'{OUT}/sft/adapter_config.json', 'w'), indent=2)
import torch, transformers, peft
print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, '| float32 (TERMO_FP32=1)')
def lancia(args, log, gpu):
    env = dict(os.environ, TERMO_FP32='1', CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1')
    return subprocess.Popen([sys.executable, '-X', 'utf8', f'{CODE}/termometro.py'] + args, stdout=open(log, 'w'),
                            stderr=subprocess.STDOUT, env=env, cwd=W)
def coda(log, n=3):
    return '\n'.join(open(log, encoding='utf-8', errors='replace').read().splitlines()[-n:])
'''.replace("'SFT_SHA'", f"'{SFT_SHA}'")

A_POOL = r'''# campione del passo 0 (seme 7) e cancelli G0, G1, G3
t0 = time.time()
p = lancia(['pool', f'{OUT}/sft', f'{OUT}/pool.json', '7'], f'{OUT}/log/pool.log', '0'); p.wait()
print(coda(f'{OUT}/log/pool.log', 2), f'| {round((time.time() - t0) / 60)} min')
g = lancia(['gates_pool', f'{OUT}/pool.json', f'{OUT}/gates_pool.json'], f'{OUT}/log/gates_pool.log', '0'); g.wait()
print(open(f'{OUT}/log/gates_pool.log', encoding='utf-8').read())
assert g.returncode == 0, 'CANCELLO FALLITO (campione): mi fermo. Unica azione permessa: quella della PREMESSA, decisa da Emanuele.'
'''

A_THERMO = r'''# i due termometri in parallelo, uno per GPU
t0 = time.time()
a = lancia(['thermo', f'{OUT}/sft', f'{OUT}/pool.json', OUT, '1', '50', '1'], f'{OUT}/log/thermo_s1.log', '0')
b = lancia(['thermo', f'{OUT}/sft', f'{OUT}/pool.json', OUT, '2', '25', '0'], f'{OUT}/log/thermo_s2.log', '1')
while a.poll() is None or b.poll() is None:
    time.sleep(300)
    s1, s2 = coda(f'{OUT}/log/thermo_s1.log', 1)[:160], coda(f'{OUT}/log/thermo_s2.log', 1)[:160]
    print(f'{round((time.time() - t0) / 60)} min | s1: {s1} | s2: {s2}', flush=True)
print('fine termometri', a.returncode, b.returncode)
print(coda(f'{OUT}/log/thermo_s1.log', 2)); print(coda(f'{OUT}/log/thermo_s2.log', 2))
assert a.returncode == 0 and b.returncode == 0, 'un termometro si è fermato con un errore'
'''

A_GATES = r'''# cancelli finali e impronte della previsione (questo registro ha l'orario di Kaggle)
g = lancia(['gates', f'{OUT}/pool.json', f'{OUT}/thermo_s1.json', f'{OUT}/thermo_s2.json', f'{OUT}/gates.json'], f'{OUT}/log/gates.log', '0'); g.wait()
print(open(f'{OUT}/log/gates.log', encoding='utf-8').read())
files = ['thermo_s1.json', 'thermo_s2.json', 'gates.json', 'gates_pool.json', 'pool.json', 'vettori_s1.pt', 'gradienti_s1.pt']
with open(f'{OUT}/previsione_sha256.txt', 'w') as f:
    for n in files:
        f.write(f'{sha(f"{OUT}/{n}")}  {n}\n')
    f.write('codice congelato: ' + open(f'{CODE}/MANIFEST_CONGELAMENTO.txt', encoding='utf-8').readline().strip() + '\n')
print('PREVISIONE', time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())); print(open(f'{OUT}/previsione_sha256.txt').read())
shutil.rmtree(f'{W}/qwen')  # il modello base non serve tra gli output
assert g.returncode == 0, 'CANCELLO FALLITO: mi fermo prima delle corse'
print('cancelli superati: la previsione va su git, poi il notebook B')
'''

B_CHECK = r'''# la previsione: identica a quella su git (impronte copiate qui dal commit, prima di lanciare B)
GIT_SHA = {}  # <- da riempire con il contenuto di previsione_sha256.txt del commit su git
A = os.path.dirname(glob.glob('/kaggle/input/**/q15/previsione_sha256.txt', recursive=True)[0])
for n in ['thermo_s1.json', 'thermo_s2.json', 'gates.json', 'gates_pool.json', 'pool.json', 'vettori_s1.pt', 'gradienti_s1.pt']:
    h = sha(f'{A}/{n}')
    assert GIT_SHA and h == GIT_SHA[n], f'PREVISIONE DIVERSA DA QUELLA SU GIT: {n}'
    shutil.copy(f'{A}/{n}', f'{OUT}/{n}')
for f in glob.glob('/kaggle/input/**/q15/run_s*.json', recursive=True):  # corse complete di una versione precedente
    if len(json.load(open(f))) == 41:
        shutil.copy(f, OUT)
        c = f.replace('.json', '_code.jsonl')
        if os.path.exists(c):
            shutil.copy(c, OUT)
print('previsione identica a git; corse già complete:', sorted(os.path.basename(f) for f in glob.glob(f'{OUT}/run_s*.json')))
'''

B_RUNS = r'''# le corse: 6 per GPU, una alla volta per GPU, con il limite di tempo della sessione
T_START, LIMITE = time.time(), 11.0 * 3600
def completa(s):
    f = f'{OUT}/run_s{s}.json'
    return os.path.exists(f) and len(json.load(open(f))) == 41
code = {'0': [s for s in range(101, 107) if not completa(s)], '1': [s for s in range(107, 113) if not completa(s)]}
corrente, tentativi, durate = {}, {}, []
while any(code.values()) or corrente:
    for gpu in ('0', '1'):
        p = corrente.get(gpu)
        if p and p[1].poll() is not None:
            s, proc, t = p
            del corrente[gpu]
            if completa(s):
                durate.append(time.time() - t)
                print(f'corsa {s} finita in {round((time.time() - t) / 60)} min', flush=True)
            elif tentativi.get(s, 0) < 2:
                code[gpu].insert(0, s)
                print(f'corsa {s} fallita, la ripeto una volta', flush=True)
            else:
                print(f'corsa {s} fallita due volte: esclusa (si scrive nel rapporto)', flush=True)
        if gpu not in corrente and code[gpu]:
            stima = max(durate) if durate else 75 * 60
            if time.time() - T_START + stima > LIMITE:
                print(f'tempo quasi finito: la GPU {gpu} non lancia {code[gpu]}', flush=True)
                code[gpu] = []
                continue
            s = code[gpu].pop(0)
            tentativi[s] = tentativi.get(s, 0) + 1
            log = f'{OUT}/log/run_s{s}.log' if tentativi[s] == 1 else f'{OUT}/log/run_s{s}_ripetuta.log'
            corrente[gpu] = (s, lancia(['run', f'{OUT}/sft', f'{OUT}/pool.json', OUT, f'{OUT}/run_s{s}.json', str(s), '40'], log, gpu), time.time())
    time.sleep(60)
fatte = sorted(s for s in range(101, 113) if completa(s))
print('corse complete:', fatte)
'''

B_ANALYSIS = r'''# analisi e rapporto congelati, code dei passi 36-40 (solo se ci sono tutte le 12 corse)
if len(fatte) == 12:
    for script, args in [('analisi_fase1.py', [OUT]), ('rapporto_fase1.py', [OUT]), ('estrai_code_passo40.py', [OUT, f'{OUT}/g5_passo40'])]:
        r = subprocess.run([sys.executable, '-X', 'utf8', f'{CODE}/{script}'] + args, capture_output=True, text=True)
        print(script, r.returncode); print(r.stdout[-3000:]); print(r.stderr[-1500:])
    with open(f'{OUT}/FASE1_FINITA.txt', 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()) + '\n')
else:
    print('mancano corse: nuova versione con questo output come ingresso')
shutil.rmtree(f'{W}/qwen', ignore_errors=True)
'''

A_CELLS = [
    ("markdown", "# Termometro, fase 1, notebook A: la previsione\nImpostazioni: **GPU T4 x2**, **Internet attivo**. "
                 "Ingressi: il dataset del codice congelato e il dataset della partenza. Passi uguali a `catena_fase1.sh`: "
                 "campione del passo 0 (seme 7), cancelli G0/G1/G3, termometro seme 1 (K=50, completo) sulla GPU 0 e seme 2 "
                 "(K=25) sulla GPU 1, cancelli finali, impronte sha256 della previsione. Il registro di Kaggle fa da marca "
                 "temporale; poi la previsione va su git PRIMA del notebook B."),
    ("code", SETUP), ("code", A_POOL), ("code", A_THERMO), ("code", A_GATES),
]
B_CELLS = [
    ("markdown", "# Termometro, fase 1, notebook B: le 12 corse\nImpostazioni: **GPU T4 x2**, **Internet attivo**. Ingressi: "
                 "codice congelato, partenza e l'output del notebook A (la previsione). Prima delle corse controlla che la "
                 "previsione sia identica a quella messa su git. Semi 101-106 sulla GPU 0 e 107-112 sulla GPU 1; una corsa che "
                 "si ferma con un errore si ripete una volta (come `catena_fase1.sh`). Se il tempo non basta, le corse mancanti "
                 "si fanno in una nuova versione con questo output come ingresso. Alla fine: analisi, rapporto e code dei passi "
                 "36-40, tutto congelato."),
    ("code", SETUP), ("code", B_CHECK), ("code", B_RUNS), ("code", B_ANALYSIS),
]


def nb(cells):
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(True),
                       **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells]}


if __name__ == "__main__":
    for name, cells in [("fase1_A_previsione.ipynb", A_CELLS), ("fase1_B_corse.ipynb", B_CELLS)]:
        json.dump(nb(cells), open(os.path.join(HERE, name), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("scritto", name)
