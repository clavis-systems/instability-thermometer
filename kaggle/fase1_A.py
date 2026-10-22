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


# campione del passo 0 (seme 7) e cancelli G0, G1, G3
t0 = time.time()
p = lancia(['pool', f'{OUT}/sft', f'{OUT}/pool.json', '7'], f'{OUT}/log/pool.log', '0'); p.wait()
print(coda(f'{OUT}/log/pool.log', 2), f'| {round((time.time() - t0) / 60)} min')
g = lancia(['gates_pool', f'{OUT}/pool.json', f'{OUT}/gates_pool.json'], f'{OUT}/log/gates_pool.log', '0'); g.wait()
print(open(f'{OUT}/log/gates_pool.log', encoding='utf-8').read())
assert g.returncode == 0, 'CANCELLO FALLITO (campione): mi fermo. Unica azione permessa: quella della PREMESSA, decisa da Emanuele.'


# i due termometri in parallelo, uno per GPU
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


# cancelli finali e impronte della previsione (questo registro ha l'orario di Kaggle)
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
