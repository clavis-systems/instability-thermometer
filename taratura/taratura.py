"""Calibration pilot (19/10 night, before any freeze; no experiment data): step-0 solve rate of the existing SFT warm
start on easier Countdown variants. Always four numbers, as in the paper (Section 2); only the number range and the
target range shrink, so every variant is a subset of the SFT training distribution (numbers 1-50, targets 10-200) and
no new SFT is needed. Seeds 931-934 / 941-944 are used nowhere else.

Writes risultati/taratura/esito.json after every variant, so a shutdown keeps what is done.
Usage (from strumenti/termometro): python -X utf8 taratura/taratura.py
"""
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TERMO = os.path.dirname(HERE)
sys.path.insert(0, TERMO)
import countdown as cd  # noqa: E402
import termometro as tm  # noqa: E402

LOCK = "C:/Users/emari/Desktop/MIA IA LOCALE/strumenti/risultati/.gpu_lock"
SFT = os.path.join(TERMO, "risultati", "q15", "sft")
OUT = os.path.join(TERMO, "risultati", "taratura", "esito.json")
N_TASKS = 48
VARIANTS = [  # name, max number, min target, max target, sampling seed, task seed
    ("n25_t100", 25, 10, 100, 931, 941),
    ("n25_t50", 25, 10, 50, 932, 942),
    ("n12_t50", 12, 10, 50, 933, 943),
    ("n50_t100", 50, 10, 100, 934, 944),
]


def tasks(seed, max_n, t_lo, t_hi, n):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        nums = [rng.randint(1, max_n) for _ in range(4)]
        reach = cd.reachable(nums)
        pool = [t for t in range(t_lo, t_hi + 1) if t in reach]
        if pool:
            t = rng.choice(pool)
            out.append({"nums": nums, "target": t, "prompt": cd.prompt(nums, t)})
    return out


def main():
    if os.path.exists(LOCK):
        print("GPU occupata:", open(LOCK, encoding="utf-8-sig").read().strip())
        sys.exit(1)
    token = f"Claude | termometro taratura del compito | {time.strftime('%d/%m %H:%M')} | {os.getpid()}"
    open(LOCK, "w", encoding="utf-8").write(token + "\n")
    try:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        done = json.load(open(OUT)) if os.path.exists(OUT) else {}
        tok, model = tm.load(tm.base_of(SFT), lora_from=SFT)
        for name, max_n, t_lo, t_hi, s_seed, t_seed in VARIANTS:
            if name in done:
                continue
            t0 = time.time()
            tm.seed_all(s_seed)
            ts = tasks(t_seed, max_n, t_lo, t_hi, N_TASKS)
            res, inform = [], []
            for i in range(0, len(ts), tm.PROMPTS_PER_STEP):
                r = tm.rollouts(tok, model, ts[i:i + tm.PROMPTS_PER_STEP])
                inform.append(tm.informative(r))
                res += r
            fails = [r for r in res if r["reward"] == 0 and r["terminated"]]
            per_task = [sum(r["reward"] for r in res[k * tm.G:(k + 1) * tm.G]) for k in range(len(ts))]
            done[name] = {
                "max_number": max_n, "targets": [t_lo, t_hi], "tasks": len(ts), "rollouts": len(res),
                "solve_rate": sum(r["reward"] for r in res) / len(res),
                "tasks_with_any_solve": sum(p > 0 for p in per_task) / len(ts),
                "informative_batches": sum(inform) / len(inform),
                "terminated_rate": sum(r["terminated"] for r in res) / len(res),
                "disclosure_among_failures": sum(bool(r["disclosed"]) for r in fails) / max(1, len(fails)),
                "false_claim_among_failures": sum(bool(r.get("false_claim")) for r in fails) / max(1, len(fails)),
                "mean_len": sum(len(r["ids"]) for r in res) / len(res),
                "seconds": round(time.time() - t0),
            }
            json.dump(done, open(OUT, "w"), indent=1)
            print(name, json.dumps(done[name]), flush=True)
    finally:
        if os.path.exists(LOCK) and token in open(LOCK, encoding="utf-8-sig").read():
            os.remove(LOCK)
    print("fine", time.strftime("%d/%m %H:%M"))


if __name__ == "__main__":
    main()
