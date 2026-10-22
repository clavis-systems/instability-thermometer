"""Calibration of the SFT warm start (20/10, before any freeze; no experiment data). The task variants did not raise
the step-0 solve rate above 9% (taratura.py), so the limit is the warm start. The paper picked its SFT reference as
"the earliest checkpoint that passed the frozen format, solve, and reporting checks". Here: one SFT of 6 epochs with
the same recipe (the first 2 epochs equal the current warm start), checkpoints at epochs 3, 4 and 6, and the step-0
solve rate of each on the ORIGINAL task (rl_tasks, seed 951, 48 tasks x 8). Writes risultati/taratura/esito_sft.json.
Usage (from strumenti/termometro): python -X utf8 taratura/taratura_sft.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TERMO = os.path.dirname(HERE)
sys.path.insert(0, TERMO)
import countdown as cd  # noqa: E402
import termometro as tm  # noqa: E402

LOCK = "C:/Users/emari/Desktop/MIA IA LOCALE/strumenti/risultati/.gpu_lock"
BASE = "C:/Users/emari/Desktop/MIA IA LOCALE/esterni/modelli/Qwen2.5-1.5B-Instruct"
OUT_SFT = os.path.join(TERMO, "risultati", "taratura", "sft6")
OUT = os.path.join(TERMO, "risultati", "taratura", "esito_sft.json")


def measure(ckpt, seed=951, n=48):
    tok, model = tm.load(BASE, lora_from=ckpt)
    tm.seed_all(seed)
    ts = cd.rl_tasks(seed, n)
    res, inform = [], []
    for i in range(0, n, tm.PROMPTS_PER_STEP):
        r = tm.rollouts(tok, model, ts[i:i + tm.PROMPTS_PER_STEP])
        inform.append(tm.informative(r))
        res += r
    fails = [r for r in res if r["reward"] == 0 and r["terminated"]]
    del model
    import torch
    torch.cuda.empty_cache()
    return {"solve_rate": sum(r["reward"] for r in res) / len(res),
            "informative_batches": sum(inform) / len(inform),
            "terminated_rate": sum(r["terminated"] for r in res) / len(res),
            "format_ok": sum(r["format_ok"] for r in res) / len(res),
            "disclosure_among_failures": sum(bool(r["disclosed"]) for r in fails) / max(1, len(fails)),
            "false_claim_among_failures": sum(bool(r.get("false_claim")) for r in fails) / max(1, len(fails)),
            "mean_len": sum(len(r["ids"]) for r in res) / len(res)}


def main():
    if os.path.exists(LOCK):
        print("GPU occupata:", open(LOCK, encoding="utf-8-sig").read().strip())
        sys.exit(1)
    token = f"Claude | termometro taratura dell'SFT | {time.strftime('%d/%m %H:%M')} | {os.getpid()}"
    open(LOCK, "w", encoding="utf-8").write(token + "\n")
    try:
        done = json.load(open(OUT)) if os.path.exists(OUT) else {}
        if not os.path.exists(os.path.join(OUT_SFT, "adapter_config.json")):
            t0 = time.time()
            tm.sft(BASE, OUT_SFT, epochs=6, save_epochs=(3, 4))
            done["sft_seconds"] = round(time.time() - t0)
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()
        for ep, ck in [(3, OUT_SFT + "_ep3"), (4, OUT_SFT + "_ep4"), (6, OUT_SFT)]:
            if f"ep{ep}" in done:
                continue
            done[f"ep{ep}"] = measure(ck)
            json.dump(done, open(OUT, "w"), indent=1)
            print(f"ep{ep}", json.dumps(done[f"ep{ep}"]), flush=True)
    finally:
        if os.path.exists(LOCK) and token in open(LOCK, encoding="utf-8-sig").read():
            os.remove(LOCK)
    print("fine", time.strftime("%d/%m %H:%M"))


if __name__ == "__main__":
    main()
