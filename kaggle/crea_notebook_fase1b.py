"""Builds the Kaggle notebook of phase 1b (PREMESSA_FASE1B.md, 20/10/2026), reusing the frozen setup cell of phase 1.
Inputs: the frozen code dataset, the phase-1b code dataset (fase1b.py + MANIFEST_1B.txt), the warm start, and the
output of notebook A. Steps: check every hash, the extended prediction (first, before any run), 6 runs of 150
steps (3 per GPU), the frozen 1b analysis.
usage: python crea_notebook_fase1b.py   (writes fase1b_C.ipynb next to it)"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from crea_notebook_fase1 import SETUP, nb  # noqa: E402  (the frozen setup cell, unchanged)

C_INPUTS = r'''# codice della 1b e previsione della fase 1 (uscita del notebook A), con le impronte
T_START = time.time()  # the session clock starts here (review 20/10)
B1 = os.path.dirname(glob.glob('/kaggle/input/**/MANIFEST_1B.txt', recursive=True)[0])
m1b = [l.split() for l in open(f'{B1}/MANIFEST_1B.txt', encoding='utf-8') if l.strip() and not l.startswith('#')]
assert m1b and all(sha(f'{B1}/{n}') == h for h, n in m1b), 'CODICE DELLA 1B DIVERSO DAL CONGELAMENTO'
print('codice della 1b identico al congelamento:', open(f'{B1}/MANIFEST_1B.txt', encoding='utf-8').readline().strip())
GIT_SHA = {}  # <- da riempire con il contenuto di previsione_sha256.txt del commit su git (come nel notebook B)
A = os.path.dirname(glob.glob('/kaggle/input/**/q15/previsione_sha256.txt', recursive=True)[0])
for n in ['thermo_s1.json', 'pool.json', 'vettori_s1.pt', 'gradienti_s1.pt']:
    assert GIT_SHA and sha(f'{A}/{n}') == GIT_SHA[n], f'PREVISIONE DELLA FASE 1 DIVERSA DA QUELLA SU GIT: {n}'
    shutil.copy(f'{A}/{n}', f'{OUT}/{n}')
def lancia1b(args, log, gpu):
    env = dict(os.environ, TERMO_FP32='1', CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1',
               PYTHONPATH=CODE)  # fase1b.py imports the frozen termometro.py from the frozen code dataset
    return subprocess.Popen([sys.executable, '-X', 'utf8', f'{B1}/fase1b.py'] + args, stdout=open(log, 'w'),
                            stderr=subprocess.STDOUT, env=env, cwd=W)
'''

C_PREDICT = r'''# la previsione estesa, prima di ogni corsa (questo registro ha l'orario di Kaggle); in una seconda sessione si
# riusa quella della prima, senza ricalcolarla
prima = glob.glob('/kaggle/input/**/q15/previsione_1b.json', recursive=True)
if prima:
    shutil.copy(prima[0], f'{OUT}/previsione_1b.json')
    print('previsione della sessione precedente riusata')
else:
    p = lancia1b(['predict', OUT, f'{OUT}/previsione_1b.json'], f'{OUT}/log/previsione_1b.log', '0'); p.wait()
    print(coda(f'{OUT}/log/previsione_1b.log', 3))
    assert p.returncode == 0, 'la previsione della 1b si è fermata con un errore'
P1B = json.load(open(f'{OUT}/previsione_1b.json'))
T_MAX = P1B['T_max']
print('PREVISIONE 1B', time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), sha(f'{OUT}/previsione_1b.json'),
      '| orizzonte', T_MAX, '| regola', P1B['horizon_rule'])
'''

C_RUNS = r'''# le 6 corse, 3 per GPU, con il limite di tempo della sessione
LIMITE = 11.0 * 3600
def completa(s, cartella=OUT):
    f = f'{cartella}/run_s{s}.json'
    try:
        return os.path.exists(f) and len(json.load(open(f))) == T_MAX + 1
    except (json.JSONDecodeError, OSError):  # a file cut by a stopped process counts as incomplete
        return False
for f in glob.glob('/kaggle/input/**/q15/run_s2[0-9][0-9].json', recursive=True):  # corse complete di una versione precedente
    s = int(os.path.basename(f)[5:8])
    if completa(s, os.path.dirname(f)):
        shutil.copy(f, OUT)
code = {'0': [s for s in (201, 202, 203) if not completa(s)], '1': [s for s in (204, 205, 206) if not completa(s)]}
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
            stima = max(durate) if durate else P1B['horizon_rule']['hours_150'] * 3600 * T_MAX / 150
            if time.time() - T_START + stima > LIMITE:
                print(f'tempo quasi finito: la GPU {gpu} non lancia {code[gpu]}', flush=True)
                code[gpu] = []
                continue
            s = code[gpu].pop(0)
            tentativi[s] = tentativi.get(s, 0) + 1
            log = f'{OUT}/log/run_s{s}.log' if tentativi[s] == 1 else f'{OUT}/log/run_s{s}_ripetuta.log'
            corrente[gpu] = (s, lancia1b(['run', f'{OUT}/sft', f'{OUT}/pool.json', OUT, f'{OUT}/run_s{s}.json', str(s)], log, gpu), time.time())
    time.sleep(60)
fatte = sorted(s for s in range(201, 207) if completa(s))
print('corse complete:', fatte)
'''

C_ANALYSIS = r'''# analisi congelata della 1b: con tutte le 6 corse, oppure alla fine della seconda (e ultima) sessione con almeno 4
seconda = bool(glob.glob('/kaggle/input/**/q15/run_s2[0-9][0-9].json', recursive=True))
if len(fatte) == 6 or (seconda and len(fatte) >= 4):
    r = subprocess.run([sys.executable, '-X', 'utf8', f'{B1}/fase1b.py', 'analyze', OUT, f'{OUT}/previsione_1b.json', OUT],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=CODE))
    print(r.returncode); print(r.stdout[-4000:]); print(r.stderr[-1500:])
else:
    print('mancano corse: una seconda (e ultima) sessione con questo output come ingresso')
for f in ['gradienti_s1.pt']:  # a copy of A's file, not needed among the outputs
    if os.path.exists(f'{OUT}/{f}'):
        os.remove(f'{OUT}/{f}')
shutil.rmtree(f'{W}/qwen', ignore_errors=True)
'''

C_CELLS = [
    ("markdown", "# Termometro, fase 1b: l'orizzonte lungo\nImpostazioni: **GPU T4 x2**, **Internet attivo**. Ingressi: "
                 "codice congelato, codice della 1b, partenza e uscita del notebook A. Prima la previsione estesa a 150 "
                 "passi (dal codice congelato e dai gradienti del passo 0 di A), poi 6 corse da 150 passi (semi 201-206), "
                 "poi l'analisi congelata in `fase1b.py`. Tutto come in `PREMESSA_FASE1B.md`."),
    ("code", SETUP), ("code", C_INPUTS), ("code", C_PREDICT), ("code", C_RUNS), ("code", C_ANALYSIS),
]

if __name__ == "__main__":
    json.dump(nb(C_CELLS), open(os.path.join(HERE, "fase1b_C.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("scritto fase1b_C.ipynb")
