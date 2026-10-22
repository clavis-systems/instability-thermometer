"""Smoke test of termometro.py v2 on the CPU with a tiny randomly initialised Qwen2 (no download, no GPU).
- The tiny model gets a copy of the REAL Qwen generation_config.json (repetition_penalty 1.1): the test checks that
  our sampling uses no logits processor at all.
- Real generate() is exercised once; then rollouts are replaced by synthetic mixed groups built from the SFT records
  (a random tiny model never solves anything), to exercise pool, D_seq, D_tf, thermometer (floor, one-step checks,
  regions, Adam replay), gates and a short run.
- Finite-difference checks of grad D_seq and grad D_tf (fp32 tiny model); region gradients must add up to the total.
Usage: python -I prova_fumo.py QWEN_DIR WORK_DIR"""
import json
import os
import random
import shutil
import sys

os.environ["TERMO_CPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # no CUDA context at all (see termometro.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

assert not torch.cuda.is_available(), "the smoke test must not see the GPU"
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM  # noqa: E402

import countdown as cd  # noqa: E402
import termometro as tm  # noqa: E402

qwen_dir, work = sys.argv[1], sys.argv[2]
os.makedirs(work, exist_ok=True)
tok = AutoTokenizer.from_pretrained(qwen_dir)
torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048, tie_word_embeddings=True)
tiny = os.path.join(work, "tiny_base")
Qwen2ForCausalLM(cfg).save_pretrained(tiny)
tok.save_pretrained(tiny)
shutil.copy(os.path.join(qwen_dir, "generation_config.json"), os.path.join(tiny, "generation_config.json"))
print("real Qwen generation config copied:", json.load(open(os.path.join(tiny, "generation_config.json"))))
tm.DTYPE = torch.float32
tm.MAX_NEW = 24
tm.REPLAY_B = 30
tm.N_POOL, tm.N_PROBE = 24, 8
tm.REGION_BATCHES, tm.ONE_STEP_BATCHES = 3, 2

recs = [json.loads(l) for l in open(os.path.join(tm.HERE, "sft_countdown.jsonl"), encoding="utf-8")]
solved = [r for r in recs if r["kind"] == "solved"]
failed = [r for r in recs if r["kind"] == "failure_report"]
sub = os.path.join(work, "sftdata")
os.makedirs(sub, exist_ok=True)
with open(os.path.join(sub, "sft_countdown.jsonl"), "w", encoding="utf-8") as f:
    for r in solved[:16] + failed[:8]:
        f.write(json.dumps(r) + "\n")
real_here, tm.HERE = tm.HERE, sub
sft_dir = os.path.join(work, "sft")
tm.sft(tiny, sft_dir, epochs=1, bs=8)
tm.HERE = real_here

# 1. pure sampling and one real generate() call
tok2, model = tm.load(tm.base_of(sft_dir), lora_from=sft_dir, trainable=True)
procs = tm.sampling_processors(model)
print("logits processors used by our sampling:", procs, "| repetition_penalty in use:",
      model.termo_gc.repetition_penalty, "| model default now:", model.get_base_model().generation_config.repetition_penalty)
assert procs == [], procs
real = tm.rollouts(tok2, model, cd.rl_tasks(5, 1))
print("real generate(): rollouts", len(real), "first text", repr(real[0]["text"][:60]))
assert len(real) == tm.G


def fake_rollouts(tok_, model_, tasks):
    """Groups of G: 3 solved (reward 1), 3 failures with a report, 2 silent failures."""
    rng = random.Random(len(tasks[0]["prompt"]) + tasks[0]["target"])
    out = []
    for t in tasks:
        pids = tok_(tm.chat(tok_, t["prompt"]), add_special_tokens=False)["input_ids"]
        for k in range(tm.G):
            if k < 3:
                text, rew = rng.choice(solved)["response"], 1
            elif k < 6:
                text, rew = rng.choice(failed)["response"], 0
            else:
                fr = rng.choice(failed)["response"]
                text, rew = fr[:fr.index("\n\nI could not")], 0
            ids = tok_(text, add_special_tokens=False)["input_ids"] + [tok_.eos_token_id]
            c = cd.classify(text, t["nums"], t["target"], True)
            c["reward"] = rew
            if rew == 0 and c["disclosed"] is None:
                c["disclosed"] = "could not" in text
            out.append({"task": t, "prompt_ids": pids, "ids": ids, "text": text, **c, "terminated": True})
    return out


tm.rollouts = fake_rollouts

# 2. regions add up to the total gradient
params = [p for p in model.parameters() if p.requires_grad]
model.train()
batch = fake_rollouts(tok2, model, cd.rl_tasks(1, tm.PROMPTS_PER_STEP))
tm.grpo_loss(tok2, model, batch)
g_full = tm.flat_grad(params)
tm.zero(params)
g_sum = torch.zeros_like(g_full)
for reg in (0, 1, 2, 3):
    if tm.grpo_loss(tok2, model, batch, region=reg) is not None:
        g_sum += tm.flat_grad(params)
    tm.zero(params)
rel = ((g_full - g_sum).norm() / g_full.norm()).item()
print(f"regions add up to the total gradient: relative error {rel:.2e}")
assert rel < 1e-4
r = batch[4]
m = tm.region_masks(tok2, r)
print("window16 tokens:", repr(tok2.decode([i for i, c in zip(r["ids"], m) if c == 2])))

# 3. pool (with the fake rollouts) and finite-difference checks of the two gradients
pool_path = os.path.join(work, "pool.json")
tm.make_pool(sft_dir, pool_path, seed=7)
data = json.load(open(pool_path))
# H6 (20/10): the fake rollouts never claim success, so mark every third non-disclosing failure as a false claim
# (fake data only), so that the H6 gradient and measure are not trivially zero in this test
for k, it in enumerate(data["pool"]):
    it["false_claim"] = (not it["d"]) and k % 3 == 0
json.dump(data, open(pool_path, "w"))
pool, probe = data["pool"], data["probe"]
assert 0 < sum(bool(it["false_claim"]) for it in pool) < len(pool)
model.train()
tm.zero(params)
tm.d_seq(model, pool, grad=True)
gs = tm.flat_grad(params)
tm.zero(params)
tm.d_tf(tok2, model, probe, grad=True)
gt = tm.flat_grad(params)
tm.zero(params)
tm.d_seq(model, pool, grad=True, key="false_claim")
gf = tm.flat_grad(params)
tm.zero(params)
th0 = tm.flat_params(params)
with torch.no_grad():
    D0, w0, lps0 = tm.d_seq(model, pool)
    T0 = tm.d_tf(tok2, model, probe)[0]
F0 = tm.weighted_rate(lps0, pool, "false_claim")
print(f"at theta_0 through the batched path: D_seq {D0:.4f} (mean label {sum(it['d'] for it in pool) / len(pool):.4f}),"
      f" ESS {w0:.2f} of {len(pool)}")
# the batched path must agree with the one-sequence path used for the gradients
one = [tm.seq_logprob(model, it["prompt_ids"], it["ids"]).item() for it in pool[:6]]
bat = tm.batch_logprobs(model, [(it["prompt_ids"], it["ids"]) for it in pool[:6]])
print("batched vs one-by-one log-probs, max |diff|:", max(abs(a - b) for a, b in zip(one, bat)))
assert max(abs(a - b) for a, b in zip(one, bat)) < 1e-3
for eps in (1e-3, 3e-4):
    dvec = torch.randn(th0.numel(), generator=torch.Generator().manual_seed(3)) * eps
    tm.set_flat(params, th0 + dvec)
    with torch.no_grad():
        D1, _, lps1 = tm.d_seq(model, pool)
        T1 = tm.d_tf(tok2, model, probe)[0]
    F1 = tm.weighted_rate(lps1, pool, "false_claim")
    tm.set_flat(params, th0)
    print(f"finite differences (eps {eps}): D_seq true {D1 - D0:+.3e} vs linear {(gs @ dvec).item():+.3e} | "
          f"D_tf true {T1 - T0:+.3e} vs linear {(gt @ dvec).item():+.3e} | "
          f"D_false true {F1 - F0:+.3e} vs linear {(gf @ dvec).item():+.3e}")
    assert abs((D1 - D0) - (gs @ dvec).item()) <= 0.2 * abs((gs @ dvec).item()) + 1e-7
    assert abs((T1 - T0) - (gt @ dvec).item()) <= 0.2 * abs((gt @ dvec).item()) + 1e-7
    assert abs((F1 - F0) - (gf @ dvec).item()) <= 0.2 * abs((gf @ dvec).item()) + 1e-7
    assert abs((gf @ dvec).item()) > 0
del model

# 4. thermometer (full and retest), gates, a short run
th_dir = os.path.join(work, "thermo")
tm.thermo(sft_dir, pool_path, th_dir, seed=1, K=4, full=True)
tm.thermo(sft_dir, pool_path, th_dir, seed=2, K=3, full=False)
t1 = json.load(open(os.path.join(th_dir, "thermo_s1.json")))
print("replay pred SD lin_seq:", {k: f"{v:.2e}" for k, v in t1["replay"]["pred_SD"]["lin_seq"].items()})
print("analytic pred SD seq:", {k: f"{v:.2e}" for k, v in t1["analytic"]["pred_SD_seq"].items()})
print("floor seq/tf:", t1["floor_seq"], t1["floor_tf"], "| corr 20-40:", t1["replay"]["pred_corr_20_40_lin_seq"])
assert t1["replay"]["pred_SD"]["lin_seq"]["40"] > 0
# H6 and the recorded precision (20/10)
assert t1["replay"]["pred_SD"]["lin_false"]["40"] > 0 and "D_false_0" in t1 and "floor_false" in t1, "H6 missing"
assert t1["dtype"] == str(tm.DTYPE)
print("H6: D_false_0", t1["D_false_0"], "| pred SD lin_false 40", t1["replay"]["pred_SD"]["lin_false"]["40"],
      "| floor_false", t1["floor_false"], "| dtype", t1["dtype"])
try:
    tm.gates(pool_path, os.path.join(th_dir, "thermo_s1.json"), os.path.join(th_dir, "thermo_s2.json"),
             os.path.join(work, "gates.json"))
except SystemExit as e:
    print("gates exit code:", e.code, "(fake data: failures are expected here)")
run_path = os.path.join(work, "run_s101.json")
tm.run(sft_dir, pool_path, th_dir, run_path, seed=101, T=3)
log = json.load(open(run_path))
assert log[-1]["lin"][0] != 0 and "D_seq" in log[-1] and "D_tf" in log[-1]
assert len(log[-1]["lin"]) == 8 and "D_false" in log[-1] and "D_false" in log[0] and log[0]["dtype"] == str(tm.DTYPE)
assert os.path.exists(run_path.replace(".json", "_code.jsonl"))
print("run last row:", {k: v for k, v in log[-1].items() if k != "lp_seq"})
print("smoke test passed")
