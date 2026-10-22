"""Phase 2 warm starts (20/10/2026): the frozen SFT recipe (termometro.sft, 2 epochs, seed 0) for the three families
of PREMESSA_FASE2, at home in bf16 like the phase-1 warm start. The sha256 of each adapter is written before any
pool (PREMESSA_FASE2, "Procedura"). It runs NEXT TO the other session's Numbrix 8x8 (CPU only, holding the lock),
with Emanuele's OK ("la A se non disturba numbrix"): low process priority, at most 2 CPU threads, one family at a
time, smallest first. Usage (from strumenti/termometro): python -X utf8 fase2_sft.py [family ...]
"""
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402

torch.set_num_threads(2)
import termometro as tm  # noqa: E402

M = "C:/Users/emari/Desktop/MIA IA LOCALE/esterni/modelli"
FAMILIES = {"q05": "Qwen2.5-0.5B-Instruct", "olmo1": "OLMo-2-0425-1B-Instruct", "smol17": "SmolLM2-1.7B-Instruct"}
OUT = os.path.join(HERE, "risultati", "fase2_partenze.json")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main(names):
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in names:
        out_dir = os.path.join(HERE, "risultati", name, "sft")
        if not os.path.exists(os.path.join(out_dir, "adapter_model.safetensors")):
            t0 = time.time()
            print(f"== {name}: SFT di {FAMILIES[name]} ({time.strftime('%H:%M')})", flush=True)
            tm.sft(f"{M}/{FAMILIES[name]}", out_dir)
            done[name] = {"base": FAMILIES[name], "minutes": round((time.time() - t0) / 60, 1),
                          "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
            torch.cuda.reset_peak_memory_stats()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        done.setdefault(name, {"base": FAMILIES[name]})
        done[name]["adapter_sha256"] = sha(os.path.join(out_dir, "adapter_model.safetensors"))
        done[name]["base_sha256"] = sha(f"{M}/{FAMILIES[name]}/model.safetensors")
        json.dump(done, open(OUT, "w"), indent=1)
        print(name, json.dumps(done[name]), flush=True)
    print("fine", time.strftime("%d/%m %H:%M"))


if __name__ == "__main__":
    main(sys.argv[1:] or list(FAMILIES))
