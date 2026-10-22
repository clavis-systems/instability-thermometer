"""Speed and precision test of the thermometer code, the same at home and on Kaggle (20/10/2026). No experiment data:
tasks from seed 77777, used nowhere else. Measures, on the SFT warm start:
- generation of one GRPO batch (4 prompts x 8 = 32 sequences), 3 times;
- one GRPO loss + backward;
- the batched no-grad log-probabilities of 400 sequences (the pool size);
- the gradient of one sequence log-probability (thermometer cost), 10 times;
- the batched vs one-at-a-time difference in this precision (the 0.47 nats found at home in bf16).
Set TERMO_FP32=1 for float32 (needed on T4/P100).

usage: python velocita.py CODE_DIR BASE_MODEL_DIR SFT_DIR|none OUT.json   (none = a fresh, untrained LoRA)
"""
import json
import os
import sys
import time

code_dir, base_dir, sft_dir, out = sys.argv[1:5]
sys.path.insert(0, code_dir)
import torch  # noqa: E402

import countdown as cd  # noqa: E402
import termometro as tm  # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


rep = {"dtype": str(tm.DTYPE), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
       "torch": torch.__version__}
import transformers, peft  # noqa: E401,E402
rep.update(transformers=transformers.__version__, peft=peft.__version__)
tm.seed_all(77777)
tok, model = tm.load(base_dir, lora_from=None if sft_dir == "none" else sft_dir, trainable=True)
rep["lora"] = "fresh" if sft_dir == "none" else "sft"
torch.cuda.reset_peak_memory_stats()
tasks = cd.rl_tasks(77777, 12)

gen_s, res_all = [], []
for i in range(3):
    sync(); t0 = time.time()
    res = tm.rollouts(tok, model, tasks[4 * i:4 * i + 4])
    sync(); gen_s.append(time.time() - t0)
    res_all += res
rep["seconds_generation_per_batch"] = gen_s
rep["mean_len"] = sum(len(r["ids"]) for r in res_all) / len(res_all)
rep["solve_rate_96"] = sum(r["reward"] for r in res_all) / len(res_all)

model.train()
sync(); t0 = time.time()
tot = tm.grpo_loss(tok, model, res_all[:32])
sync(); rep["seconds_grpo_backward"] = time.time() - t0
rep["grpo_batch_informative"] = tot is not None
for p in model.parameters():
    p.grad = None

model.eval()
pairs = [(r["prompt_ids"], r["ids"]) for r in res_all if r["ids"]]
pairs400 = (pairs * (400 // len(pairs) + 1))[:400]
sync(); t0 = time.time()
tm.batch_logprobs(model, pairs400)
sync(); rep["seconds_batch_logprobs_400"] = time.time() - t0

g_s = []
for p, s in pairs[:10]:
    sync(); t0 = time.time()
    lp = tm.seq_logprob(model, p, s)
    lp.backward()
    sync(); g_s.append(time.time() - t0)
    for q in model.parameters():
        q.grad = None
rep["seconds_grad_per_sequence"] = sum(g_s) / len(g_s)

with torch.no_grad():
    one = [tm.seq_logprob(model, p, s).item() for p, s in pairs[:16]]
bat = tm.batch_logprobs(model, pairs[:16])
diffs = [abs(a - b) for a, b in zip(one, bat)]
rep["batched_vs_single_mean_abs_diff"] = sum(diffs) / len(diffs)
rep["batched_vs_single_max_abs_diff"] = max(diffs)
rep["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
json.dump(rep, open(out, "w"), indent=1)
print("VELOCITA:", json.dumps(rep))
