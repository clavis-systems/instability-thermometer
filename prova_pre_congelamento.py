"""Checks on the real GPU before the freeze (review 19/10, B4). No experiment data: 16 rollouts with a seed used
nowhere else (99). Compares the batched no-grad log-probabilities with the one-sequence path, for whole sequences
and for the two D_tf routes, and records the peak GPU memory.
Usage: python -I prova_pre_congelamento.py SFT_DIR OUT.json"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import countdown as cd  # noqa: E402
import termometro as tm  # noqa: E402

sft_dir, out = sys.argv[1], sys.argv[2]
tm.seed_all(99)
tok, model = tm.load(tm.base_of(sft_dir), lora_from=sft_dir)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
res = tm.rollouts(tok, model, cd.rl_tasks(99, 2))
gen_s = time.time() - t0
model.eval()
pairs = [(r["prompt_ids"], r["ids"]) for r in res[:16] if r["ids"]]
with torch.no_grad():
    one = [tm.seq_logprob(model, p, s).item() for p, s in pairs]
t0 = time.time()
bat = tm.batch_logprobs(model, pairs)
bat_s = time.time() - t0
diffs = [abs(a - b) for a, b in zip(one, bat)]
fails = [r for r in res if r["reward"] == 0 and r["terminated"] and r["format_ok"]]
route = []
for r in fails[:8]:
    answers = list(cd.ANSWER.finditer(r["text"]))
    item = {"prompt_ids": r["prompt_ids"], "prefix_text": r["text"][:answers[-1].end()].rstrip("\n")}
    shared, s_rep, s_clo = tm.seam_pair(tok, item)
    with torch.no_grad():
        a1 = tm.route_logprob(model, shared, s_rep).item()
        a2 = tm.route_logprob(model, shared, s_clo).item()
    b1, b2 = tm.batch_logprobs(model, [(shared, s_rep), (shared, s_clo)])
    route.append(max(abs(a1 - b1), abs(a2 - b2)))
rep = {"n_sequences": len(pairs), "seconds_generation_16x2": round(gen_s, 1), "seconds_batched_logprobs": round(bat_s, 2),
       "max_abs_diff_seq_logprob": max(diffs), "mean_abs_diff": sum(diffs) / len(diffs),
       "all_finite": all(math.isfinite(x) for x in one + bat), "route_max_abs_diff": max(route) if route else None,
       "solve_rate_16x2": sum(r["reward"] for r in res) / len(res), "mean_len": sum(len(r["ids"]) for r in res) / len(res),
       "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
json.dump(rep, open(out, "w"), indent=1)
print("PROVA PRE-CONGELAMENTO:", json.dumps(rep))
