# ambiente, codice congelato, modello base e partenza (uguale in A e in B)
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
assert sha(f'{OUT}/sft/adapter_model.safetensors') == 'c0506a7111c5c01f0c0ada9cb8e4a8d0c78e2bd91452350ac608f80427111fe6', 'partenza diversa da quella del 19/10'
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


# la previsione: identica a quella su git (impronte copiate qui dal commit, prima di lanciare B)
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


# le corse: 6 per GPU, una alla volta per GPU, con il limite di tempo della sessione
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


# analisi e rapporto congelati, code dei passi 36-40 (solo se ci sono tutte le 12 corse)
if len(fatte) == 12:
    for script, args in [('analisi_fase1.py', [OUT]), ('rapporto_fase1.py', [OUT]), ('estrai_code_passo40.py', [OUT, f'{OUT}/g5_passo40'])]:
        r = subprocess.run([sys.executable, '-X', 'utf8', f'{CODE}/{script}'] + args, capture_output=True, text=True)
        print(script, r.returncode); print(r.stdout[-3000:]); print(r.stderr[-1500:])
    with open(f'{OUT}/FASE1_FINITA.txt', 'w') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()) + '\n')
else:
    print('mancano corse: nuova versione con questo output come ingresso')
shutil.rmtree(f'{W}/qwen', ignore_errors=True)
