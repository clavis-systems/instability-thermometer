"""Real-GPU smoke test of termometro.py v2 (no experiment data: untrained LoRA, one batch).
Times one rollout batch, one GRPO backward, D_seq/D_tf on a few items; checks pure sampling on the real model; peak memory."""
import os
import sys
import time

sys.path.insert(0, r"C:\Users\emari\Desktop\MIA IA LOCALE\.claude\worktrees\robotica-congela-previsione-legge-1-929f74\strumenti\termometro")
import torch  # noqa: E402

import countdown as cd  # noqa: E402
import termometro as tm  # noqa: E402

base = r"C:/Users/emari/Desktop/MIA IA LOCALE/esterni/modelli/Qwen2.5-1.5B-Instruct"
t0 = time.time()
tok, model = tm.load(base, trainable=True)
print("device", tm.DEV, "load s", round(time.time() - t0, 1), flush=True)
print("processors on the real model:", tm.sampling_processors(model), "| model default repetition_penalty now",
      model.get_base_model().generation_config.repetition_penalty, flush=True)
params = [p for p in model.parameters() if p.requires_grad]
print("LoRA params:", sum(p.numel() for p in params), flush=True)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
res = tm.rollouts(tok, model, cd.rl_tasks(10_001, tm.PROMPTS_PER_STEP))
t_gen = time.time() - t0
lens = [len(r["ids"]) for r in res]
print(f"rollouts: {len(res)} in {t_gen:.1f}s; mean length {sum(lens) / len(lens):.0f} tokens, max {max(lens)}; "
      f"solve {sum(r['reward'] for r in res)}/{len(res)}; terminated {sum(r['terminated'] for r in res)}", flush=True)
print("example:", repr(res[0]["text"][:200]), flush=True)
# force a mixed group so the backward runs even if nothing is solved by the untrained model
for r in res[:2]:
    r["reward"] = 1
model.train()
t0 = time.time()
tot = tm.grpo_loss(tok, model, res)
torch.cuda.synchronize()
print(f"grpo backward: {time.time() - t0:.1f}s (loss {tot})", flush=True)
tm.zero(params)
fails = [r for r in res if r["reward"] == 0 and r["terminated"]][:3]
pool = [{"prompt_ids": r["prompt_ids"], "ids": r["ids"], "d": i % 2} for i, r in enumerate(fails)]
if pool:
    with torch.no_grad():
        for it in pool:
            it["logp0"] = tm.seq_logprob(model, it["prompt_ids"], it["ids"]).item()
    t0 = time.time()
    D, ess, _ = tm.d_seq(model, pool, grad=True)
    torch.cuda.synchronize()
    print(f"d_seq with gradient on {len(pool)} items: {time.time() - t0:.1f}s; D {D:.3f} ess {ess:.2f}", flush=True)
    tm.zero(params)
probe = [{"prompt_ids": r["prompt_ids"], "prefix_text": "1 + 2 = 3\nAnswer: 1 + 2 + 3 + 4"} for r in res[:3]]
t0 = time.time()
Dt, ps, mass = tm.d_tf(tok, model, probe)
print(f"d_tf on 3 items: {time.time() - t0:.1f}s; p {['%.3g' % p for p in ps]}; route mass {['%.3g' % m for m in mass]}",
      flush=True)
print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
print("real GPU smoke test passed", flush=True)
