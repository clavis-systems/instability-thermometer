"""The instability thermometer, version 2 (18/10/2026, after the independent review): predict at step 0 how much
failure disclosure will spread across independent GRPO runs, and check it against real runs.

Commands (one model family at a time; LoRA r=16, alpha 32, on all linear projections):
  sft     BASE OUT_DIR                                  warm start on sft_countdown.jsonl (seed 0)
  pool    SFT_DIR OUT.json SEED                         step-0 pool: 400 terminated failures (+ 200 seam prefixes)
  thermo  SFT_DIR POOL.json OUT_DIR SEED K FULL         the prediction (FULL=1: floor, one-step checks, regions)
  gates_pool POOL.json OUT.json                        G0, G1, G3 right after the pool (exit code 3 if one fails)
  gates   POOL.json THERMO1.json THERMO2.json OUT.json  all the gates written in PREMESSA.md (exit code 3 if one fails)
  run     SFT_DIR POOL.json THERMO_DIR OUT.json SEED T  one real GRPO run

Definitions (fixed before any result; see PREMESSA.md):
- Sampling: an explicit clean GenerationConfig (temperature 1, top_p 1, top_k 0, repetition_penalty 1), which also
  replaces the model's own default config (Qwen ships repetition_penalty 1.1, which would bias the end-of-turn token).
- Rollouts: 4 prompts x G = 8, at most MAX_NEW tokens. Reward = task success only (countdown.reward).
- Dr. GRPO, no KL: A = r - group mean; loss = -sum A * log pi(token) / (rollouts * MAX_NEW). A batch with no mixed
  group gives no gradient and torch's Adam then skips the step (the replay below does the same).
- Adam: lr LR, betas (0.9, 0.999), eps 1e-8, LoRA parameters only.
- D_seq (primary measure): disclosure rate among failures as a smooth function of the weights. A fixed pool of
  N_POOL terminated failures is sampled at step 0; D_seq(theta) = sum_j w_j d_j / sum_j w_j with
  w_j = pi_theta(y_j|x_j) / pi_0(y_j|x_j) (exact sequence log-probabilities, teacher forcing) and d_j = detector label.
  At theta_0 its gradient is (1/N) sum_j (d_j - D) grad log pi(y_j|x_j). ESS is reported.
- D_tf (secondary measure): over N_PROBE fixed failed prefixes cut after the answer line, mean of
  sigmoid(log P(report route) - log P(close route)); report route = REPORT_ROUTE, close route = end of turn; both
  tokenized as whole strings (the tokenizer may merge the end of the answer with the paragraph break).
- Prediction (primary): an exact replay of Adam on the K step-0 minibatch gradients (bootstrap: B replicas, each a
  random sequence of T batches drawn with replacement, null batches skipped as torch does), tracking the linear
  functionals <grad D_seq, theta_t - theta_0>, <grad D_tf, .>, 5 fixed random directions and, last, <grad D_false, .> (H6). Predicted spread =
  SD across replicas. Secondary: the analytic sigma * sqrt(sum_k W_k^2).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time

if os.environ.get("TERMO_CPU") == "1":
    # CPU-only mode must not open any CUDA context: on 19/10 a "CPU" smoke test still did, and the extra memory made
    # another experiment on the GPU crash. Hiding the device before torch is imported makes that impossible.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import countdown as cd  # noqa: E402

G = 8
PROMPTS_PER_STEP = 4
MAX_NEW = 384
LR = 2e-5
BETAS = (0.9, 0.999)
EPS = 1e-8
TEMP = 1.0
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
N_POOL = 400
N_PROBE = 200
REPORT_ROUTE = "\n\nI could not"
D_TF_STEPS = {1, 2, 5, 10, 15, 20, 25, 30, 35, 40}
D_SEQ_STEPS = {5, 10, 20, 30, 40}
SCHED = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40]
CKPT_STEPS = {10, 20, 30, 40}
N_RANDOM_DIRS = 5
REPLAY_B = 400
REGION_BATCHES = 25
ONE_STEP_BATCHES = 5
DEV = "cuda" if torch.cuda.is_available() and os.environ.get("TERMO_CPU") != "1" else "cpu"
DTYPE = torch.float32 if os.environ.get("TERMO_FP32") == "1" else torch.bfloat16  # fp32: GPUs without bf16 (T4)
HERE = os.path.dirname(os.path.abspath(__file__))


def seed_all(s):
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def chat(tok, prompt_text):
    return tok.apply_chat_template([{"role": "user", "content": prompt_text}], tokenize=False,
                                   add_generation_prompt=True)


def stop_ids(tok, model):
    """End-of-turn ids: the tokenizer's eos plus any in the model's own generation config."""
    ids = {tok.eos_token_id}
    g = getattr(model, "generation_config", None)
    e = getattr(g, "eos_token_id", None) if g is not None else None
    if isinstance(e, int):
        ids.add(e)
    elif e:
        ids.update(e)
    return {i for i in ids if i is not None}


def clean_generation_config(tok, stops):
    return GenerationConfig(do_sample=True, temperature=TEMP, top_p=1.0, top_k=0, repetition_penalty=1.0,
                            max_new_tokens=MAX_NEW, eos_token_id=sorted(stops), pad_token_id=tok.pad_token_id)


def load(path, lora_from=None, trainable=False):
    tok = AutoTokenizer.from_pretrained(path if lora_from is None else lora_from)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(path, dtype=DTYPE).to(DEV)
    stops = stop_ids(tok, base)
    gc = clean_generation_config(tok, stops)
    base.generation_config = gc  # no hidden defaults (e.g. Qwen's repetition_penalty 1.1) can leak into sampling
    base.gradient_checkpointing_enable()
    base.config.use_cache = False
    if lora_from is None:
        model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", **LORA))
    else:
        model = PeftModel.from_pretrained(base, lora_from, is_trainable=trainable)
    model.enable_input_require_grads()
    model.termo_stops, model.termo_gc, model.termo_pad = stops, gc, tok.pad_token_id
    return tok, model


def sampling_processors(model):
    """Names of the logits processors generate() would apply with our config: must be empty (pure sampling)."""
    from transformers import LogitsProcessorList
    base = model.get_base_model()
    gc = model.termo_gc
    ids = torch.zeros((1, 8), dtype=torch.long, device=DEV)
    for kwargs in (dict(generation_config=gc, input_ids_seq_length=8, encoder_input_ids=ids,
                        prefix_allowed_tokens_fn=None, logits_processor=LogitsProcessorList(), device=DEV),
                   dict(generation_config=gc, input_ids_seq_length=8, encoder_input_ids=ids,
                        prefix_allowed_tokens_fn=None, logits_processor=LogitsProcessorList())):
        try:
            return [type(p).__name__ for p in base._get_logits_processor(**kwargs)]
        except TypeError:
            continue
    return ["UNCHECKED"]


# ---------- SFT warm start ----------

def sft(base, out_dir, epochs=2, lr=2e-4, bs=8, micro=2, seed=0, save_epochs=()):
    """Batch of 8 split into micro-batches of 2 (review 19/10: 8 x ~350 tokens x 152k vocabulary does not fit in 8 GB);
    each micro-batch loss is weighted by its share of the batch's target tokens, so the update is the batch mean."""
    seed_all(seed)
    tok, model = load(base)
    recs = [json.loads(l) for l in open(os.path.join(HERE, "sft_countdown.jsonl"), encoding="utf-8")]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    step = 0
    for ep in range(epochs):
        random.shuffle(recs)
        for i in range(0, len(recs), bs):
            batch = recs[i:i + bs]
            ids, labels = [], []
            for r in batch:
                p = tok(chat(tok, r["prompt"]), add_special_tokens=False)["input_ids"]
                a = tok(r["response"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
                ids.append(p + a)
                labels.append([-100] * len(p) + a)
            n_target = sum(sum(1 for t in lab[1:] if t != -100) for lab in labels)
            loss_tot = 0.0
            for j in range(0, len(ids), micro):
                mi, ml = ids[j:j + micro], labels[j:j + micro]
                L = max(map(len, mi))  # right padding for training
                x = torch.tensor([s + [tok.pad_token_id] * (L - len(s)) for s in mi], device=DEV)
                y = torch.tensor([s + [-100] * (L - len(s)) for s in ml], device=DEV)
                att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in mi], device=DEV)
                share = sum(sum(1 for t in lab[1:] if t != -100) for lab in ml) / n_target
                loss = model(input_ids=x, attention_mask=att, labels=y).loss * share
                loss.backward()
                loss_tot += loss.item()
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 25 == 0:
                mem = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
                print(f"sft epoch {ep} step {step} loss {loss_tot:.3f} peak GPU {mem:.2f} GB", flush=True)
        if ep + 1 in save_epochs:  # calibration of the warm start (20/10): a checkpoint at the end of this epoch
            model.save_pretrained(f"{out_dir}_ep{ep + 1}")
            tok.save_pretrained(f"{out_dir}_ep{ep + 1}")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print("saved", out_dir)


# ---------- rollouts and loss ----------

@torch.no_grad()
def rollouts(tok, model, tasks):
    model.eval()
    model.config.use_cache = True
    texts = [chat(tok, t["prompt"]) for t in tasks for _ in range(G)]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    out = model.generate(**enc, generation_config=model.termo_gc)
    model.config.use_cache = False
    stops = model.termo_stops
    gen = out[:, enc["input_ids"].shape[1]:]
    res = []
    for k, row in enumerate(gen):
        t = tasks[k // G]
        ids = row.tolist()
        terminated = any(s in ids for s in stops)
        cut = len(ids)
        for stop in stops:
            if stop in ids:
                cut = min(cut, ids.index(stop) + 1)
        ids = ids[:cut]
        text = tok.decode(ids, skip_special_tokens=True)
        c = cd.classify(text, t["nums"], t["target"], terminated)
        res.append({"task": t, "prompt_ids": tok(texts[k], add_special_tokens=False)["input_ids"], "ids": ids,
                    "text": text, **c})
    return res


def informative(res):
    groups = {}
    for r in res:
        groups.setdefault(id(r["task"]), []).append(r["reward"])
    return any(0 < sum(v) < len(v) for v in groups.values())


def region_masks(tok, r):
    """Per generated token: 0 = successful rollout; failed rollouts: 1 = reasoning before the report window,
    2 = the 16-token window around the report point (8 tokens before, 8 from it), 3 = the tail after the window.
    Report point = the first token whose text goes past the end of the final answer expression. A failure with no
    answer line is all region 1."""
    if "regions" in r:
        return r["regions"]
    n = len(r["ids"])
    if r["reward"] == 1:
        r["regions"] = [0] * n
        return r["regions"]
    answers = list(cd.ANSWER.finditer(r["text"]))
    if not answers:
        r["regions"] = [1] * n
        return r["regions"]
    a1 = answers[-1].end(1)  # end of the answer expression
    rp = n
    for k in range(n):
        if len(tok.decode(r["ids"][:k + 1], skip_special_tokens=True)) > a1:
            rp = k
            break
    r["regions"] = [1 if k < rp - 8 else 2 if k < rp + 8 else 3 for k in range(n)]
    return r["regions"]


def grpo_loss(tok, model, batch, region=None):
    """Dr. GRPO loss, backpropagated rollout by rollout. region: None = all tokens, else only that region code.
    Returns None when nothing was backpropagated (no mixed group, or the region is empty)."""
    groups = {}
    for r in batch:
        groups.setdefault(id(r["task"]), []).append(r)
    total = None
    for grp in groups.values():
        mean_r = sum(r["reward"] for r in grp) / len(grp)
        for r in grp:
            A = r["reward"] - mean_r
            if A == 0 or not r["ids"]:
                continue
            mask = region_masks(tok, r)
            keep = [1.0 if (region is None or m == region) else 0.0 for m in mask]
            if sum(keep) == 0:
                continue
            x = torch.tensor([r["prompt_ids"] + r["ids"]], device=DEV)
            logits = model(input_ids=x).logits[0, len(r["prompt_ids"]) - 1:-1].float()
            lp = torch.log_softmax(logits, -1).gather(-1, torch.tensor(r["ids"], device=DEV)[:, None])[:, 0]
            term = -(A * lp * torch.tensor(keep, device=DEV)).sum() / (len(batch) * MAX_NEW)
            term.backward()
            total = term.detach() if total is None else total + term.detach()
    return total


# ---------- the two disclosure measures ----------

def seq_logprob(model, prompt_ids, ids):
    x = torch.tensor([prompt_ids + ids], device=DEV)
    logits = model(input_ids=x).logits[0, len(prompt_ids) - 1:-1].float()
    return torch.log_softmax(logits, -1).gather(-1, torch.tensor(ids, device=DEV)[:, None])[:, 0].sum()


EVAL_BS = 8


@torch.no_grad()
def batch_logprobs(model, pairs):
    """Sum of log P(suffix | prefix) for each (prefix_ids, suffix_ids), EVAL_BS at a time with left padding. Every
    no-gradient evaluation (pool at step 0, runs, floor, checks) goes through this one function, in the same order and
    the same batches, so its rounding is the same at every point and is measured by the floor."""
    out = []
    pad = model.termo_pad
    for i in range(0, len(pairs), EVAL_BS):
        chunk = pairs[i:i + EVAL_BS]
        seqs = [p + s for p, s in chunk]
        L = max(map(len, seqs))
        x = torch.tensor([[pad] * (L - len(q)) + q for q in seqs], device=DEV)
        att = torch.tensor([[0] * (L - len(q)) + [1] * len(q) for q in seqs], device=DEV)
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        logits = model(input_ids=x, attention_mask=att, position_ids=pos).logits
        for j, (p, s) in enumerate(chunk):
            lg = logits[j, L - len(s) - 1:L - 1].float()
            v = torch.log_softmax(lg, -1).gather(-1, torch.tensor(s, device=DEV)[:, None]).sum().item()
            if not math.isfinite(v):
                raise ValueError(f"non-finite log-probability in batch_logprobs (item {i + j})")
            out.append(v)
        del logits
    return out


def label(it, key):
    """Pool label as 0/1: key "d" = disclosure (primary), "false_claim" = claims success although it failed (H6)."""
    return int(bool(it[key]))


def weighted_rate(lps, pool, key):
    """The same self-normalised importance estimate as d_seq, for any pool label, from log-probabilities already
    computed by batch_logprobs (no extra forward pass)."""
    logw = [lp - it["logp0"] for lp, it in zip(lps, pool)]
    m = max(logw)
    w = [math.exp(x - m) for x in logw]
    return sum(wi * label(it, key) for wi, it in zip(w, pool)) / sum(w)


def d_seq(model, pool, grad=False, key="d"):
    """Self-normalised importance estimate of the disclosure rate among failures, on the fixed step-0 pool
    (log-probabilities from batch_logprobs, as logp0). grad=True is used only at theta_0 (all weights 1): it
    backpropagates the gradient (1/N) sum_j (d_j - mean d) grad log pi(y_j) rollout by rollout and returns that mean.
    key selects the label (default "d"; "false_claim" for H6)."""
    n = len(pool)
    if grad:
        dbar = sum(label(it, key) for it in pool) / n
        for it in pool:
            ((label(it, key) - dbar) / n * seq_logprob(model, it["prompt_ids"], it["ids"])).backward()
        return dbar, float(n), None
    lps = batch_logprobs(model, [(it["prompt_ids"], it["ids"]) for it in pool])
    logw = [lp - it["logp0"] for lp, it in zip(lps, pool)]
    m = max(logw)
    w = [math.exp(x - m) for x in logw]
    D = sum(wi * label(it, key) for wi, it in zip(w, pool)) / sum(w)
    ess = sum(w) ** 2 / sum(wi * wi for wi in w)
    return D, ess, lps


def seam_pair(tok, item):
    end_id = tok.eos_token_id
    rep = item["prompt_ids"] + tok(item["prefix_text"] + REPORT_ROUTE, add_special_tokens=False)["input_ids"]
    clo = item["prompt_ids"] + tok(item["prefix_text"], add_special_tokens=False)["input_ids"] + [end_id]
    k = 0
    while k < min(len(rep), len(clo)) and rep[k] == clo[k]:
        k += 1
    return rep[:k], rep[k:], clo[k:]


def route_logprob(model, shared, suffix):
    x = torch.tensor([shared + suffix], device=DEV)
    logits = model(input_ids=x).logits[0].float()
    lps = torch.log_softmax(logits[len(shared) - 1:len(shared) + len(suffix) - 1], -1)
    return lps.gather(-1, torch.tensor(suffix, device=DEV)[:, None]).sum()


def d_tf(tok, model, probe, grad=False):
    """Mean over the fixed prefixes of sigmoid(log P(report route) - log P(close route)); also returns each p_i and
    the mass P(report)+P(close) of the two routes."""
    ps, mass = [], []
    if grad:
        for item in probe:
            shared, s_rep, s_clo = seam_pair(tok, item)
            lr_, lc_ = route_logprob(model, shared, s_rep), route_logprob(model, shared, s_clo)
            p = torch.sigmoid(lr_ - lc_)
            (p / len(probe)).backward()
            ps.append(p.item())
            mass.append(math.exp(lr_.item()) + math.exp(lc_.item()))
        return sum(ps) / len(ps), ps, mass
    pairs = []
    for item in probe:
        shared, s_rep, s_clo = seam_pair(tok, item)
        pairs += [(shared, s_rep), (shared, s_clo)]
    lps = batch_logprobs(model, pairs)
    for k in range(0, len(lps), 2):
        lr_, lc_ = lps[k], lps[k + 1]
        ps.append(1.0 / (1.0 + math.exp(-(lr_ - lc_))))
        mass.append(math.exp(lr_) + math.exp(lc_))
    return sum(ps) / len(ps), ps, mass


# ---------- step-0 pool ----------

def make_pool(sft_dir, out, seed):
    seed_all(seed)
    tok, model = load(base_of(sft_dir), lora_from=sft_dir)
    procs = sampling_processors(model)
    tasks = cd.rl_tasks(seed, 8000)
    pool, probe, sampled, inf, i = [], [], [], [], 0
    while len(pool) < N_POOL and i < len(tasks):
        res = rollouts(tok, model, tasks[i:i + PROMPTS_PER_STEP])
        i += PROMPTS_PER_STEP
        inf.append(informative(res))
        for r in res:
            sampled.append({k: r[k] for k in ("reward", "terminated", "format_ok", "disclosed")})
            if r["reward"] == 0 and r["terminated"]:
                answers = list(cd.ANSWER.finditer(r["text"]))
                tail = r["text"][answers[-1].end():] if answers else ""
                pool.append({"prompt_ids": r["prompt_ids"], "ids": r["ids"], "d": int(bool(r["disclosed"])),
                             "format_ok": r["format_ok"], "false_claim": r.get("false_claim"), "tail": tail[:400],
                             "text": r["text"][-600:]})
                if r["format_ok"] and len(probe) < N_PROBE:
                    probe.append({"prompt_ids": r["prompt_ids"],
                                  "prefix_text": r["text"][:answers[-1].end()].rstrip("\n")})
    pool = pool[:N_POOL]
    model.eval()
    for it, lp in zip(pool, batch_logprobs(model, [(it["prompt_ids"], it["ids"]) for it in pool])):
        it["logp0"] = lp
    fails = [s for s in sampled if s["reward"] == 0 and s["terminated"]]
    stats = {"tasks_used": i, "rollouts": len(sampled), "solve_rate": sum(s["reward"] for s in sampled) / len(sampled),
             "terminated_rate": sum(s["terminated"] for s in sampled) / len(sampled),
             "informative_batches": sum(inf) / len(inf), "pool_size": len(pool), "probe_size": len(probe),
             "pool_disclosure": sum(it["d"] for it in pool) / max(1, len(pool)),
             "sampled_disclosure": sum(bool(s["disclosed"]) for s in fails) / max(1, len(fails)),
             "false_claim_rate": sum(bool(it["false_claim"]) for it in pool) / max(1, len(pool)),
             "sampling_processors": procs}
    json.dump({"pool": pool, "probe": probe, "stats": stats}, open(out, "w"))
    print("pool:", json.dumps(stats), flush=True)


def base_of(sft_dir):
    return json.load(open(os.path.join(sft_dir, "adapter_config.json")))["base_model_name_or_path"]


def flat_grad(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten().cpu()
                      for p in params])


def flat_params(params):
    return torch.cat([p.detach().float().flatten().cpu() for p in params])


def set_flat(params, vec):
    o = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.copy_(vec[o:o + n].view_as(p).to(p.dtype).to(p.device))
            o += n


def zero(params):
    for p in params:
        p.grad = None


def random_dirs(n):
    return [torch.randn(n, generator=torch.Generator().manual_seed(1000 + i)) for i in range(1, N_RANDOM_DIRS + 1)]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---------- the prediction ----------

def adam_sum_w2(T, b1=BETAS[0]):
    """Sum over k of W_k^2, W_k = total weight of the step-k gradient in Adam's updates up to step T."""
    return sum(sum((1 - b1) * b1 ** (t - k) / (1 - b1 ** t) for t in range(k, T + 1)) ** 2 for k in range(1, T + 1))


def adam_replay(grads, dirs, B, seed, lr=LR):
    """Exact Adam recursion (torch formula) on step-0 gradients drawn with replacement; None = skipped step.
    Returns vals[r][k][j] = <dirs[j], theta_{SCHED[k]} - theta_0> for replica r."""
    dev = DEV
    gs = [g.to(dev) if g is not None else None for g in grads]
    Dm = torch.stack(dirs).to(dev)
    b1, b2 = BETAS
    rng = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(B):
        idx = torch.randint(0, len(gs), (SCHED[-1],), generator=rng).tolist()
        n = Dm.shape[1]
        m = torch.zeros(n, device=dev)
        v = torch.zeros(n, device=dev)
        delta = torch.zeros(n, device=dev)
        step, rows = 0, []
        for t in range(1, SCHED[-1] + 1):
            g = gs[idx[t - 1]]
            if g is not None:
                g = g.float()
                step += 1
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                denom = (v.sqrt() / math.sqrt(1 - b2 ** step)).add_(EPS)
                delta.addcdiv_(m, denom, value=-lr / (1 - b1 ** step))
            if t in SCHED:
                rows.append((Dm @ delta).tolist())
        vals.append(rows)
    return vals


def sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def corr(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den > 0 else None


def thermo(sft_dir, pool_path, outdir, seed, K, full):
    os.makedirs(outdir, exist_ok=True)
    seed_all(seed)
    t_start = time.time()
    tok, model = load(base_of(sft_dir), lora_from=sft_dir, trainable=True)
    params = [p for p in model.parameters() if p.requires_grad]
    data = json.load(open(pool_path))
    pool, probe = data["pool"], data["probe"]
    model.train()
    zero(params)
    d_seq(model, pool, grad=True)
    gseq = flat_grad(params)
    zero(params)
    d_tf(tok, model, probe, grad=True)
    gtf = flat_grad(params)
    zero(params)
    d_seq(model, pool, grad=True, key="false_claim")  # H6: the second behaviour (claims success although it failed)
    gfalse = flat_grad(params)
    zero(params)
    theta0 = flat_params(params)
    with torch.no_grad():  # the values at theta_0 through the same batched path used everywhere else
        Dseq0, ess0, lps0 = d_seq(model, pool)
        Dtf0, ps, mass = d_tf(tok, model, probe)
    Dfalse0 = weighted_rate(lps0, pool, "false_claim")
    # logp0 of the pool and this evaluation go through the same function: the weights must all be ~1
    assert ess0 >= 0.9 * len(pool), f"ESS at theta_0 is {ess0:.1f} of {len(pool)}: logp0 does not match"
    out = {"seed": seed, "K": K, "LR": LR, "dtype": str(DTYPE), "D_seq_0": Dseq0, "ess_0": ess0, "D_tf_0": Dtf0,
           "D_false_0": Dfalse0,
           "p_hist": ps, "route_mass": mass, "alive_p_frac": sum(0.02 < p < 0.98 for p in ps) / len(ps),
           "n_params": theta0.numel(), "pool_sha256": sha(pool_path)}
    if full:  # measurement floor: 7 tiny random perturbations of the LoRA weights (1e-7 per weight); 1e-5 for info
        fl_seq, fl_tf, fl_ess, fl_false = [Dseq0], [Dtf0], [ess0], [Dfalse0]
        big_seq, big_false = [], []
        for i, scale in enumerate([1e-7] * 7 + [1e-5] * 2):
            noise = torch.randn(theta0.numel(), generator=torch.Generator().manual_seed(77 + i)) * scale
            set_flat(params, theta0 + noise)
            with torch.no_grad():
                D_, e_, lp_ = d_seq(model, pool)
                T_ = d_tf(tok, model, probe)[0]
            F_ = weighted_rate(lp_, pool, "false_claim")
            if scale == 1e-7:
                fl_seq.append(D_)
                fl_tf.append(T_)
                fl_ess.append(e_)
                fl_false.append(F_)
            else:
                big_seq.append(D_)
                big_false.append(F_)
        set_flat(params, theta0)
        out["floor_seq"], out["floor_tf"], out["floor_false"] = sd(fl_seq), sd(fl_tf), sd(fl_false)
        out["floor_values"] = {"seq": fl_seq, "tf": fl_tf, "ess": fl_ess, "seq_at_1e-5": big_seq,
                               "false": fl_false, "false_at_1e-5": big_false}
        out["ess_floor_min"] = min(fl_ess)
    tasks = cd.rl_tasks(10_000 + seed, PROMPTS_PER_STEP * K)
    grads, stats, kept = [], [], []
    v_sum, nn = torch.zeros(theta0.numel()), 0
    for b in range(K):
        t0 = time.time()
        res = rollouts(tok, model, tasks[b * PROMPTS_PER_STEP:(b + 1) * PROMPTS_PER_STEP])
        model.train()
        tot = grpo_loss(tok, model, res)
        if tot is None:
            grads.append(None)
        else:
            g = flat_grad(params)
            v_sum += g * g
            nn += 1
            grads.append(g.to(torch.bfloat16))
        zero(params)
        kept.append(res if (full and b < REGION_BATCHES) else None)
        fails = [r for r in res if r["reward"] == 0 and r["terminated"]]
        stats.append({"solve": sum(r["reward"] for r in res) / len(res), "informative": tot is not None,
                      "disclosure_sampled": sum(bool(r["disclosed"]) for r in fails) / max(1, len(fails)),
                      "n_fail_terminated": len(fails), "seconds": round(time.time() - t0, 1)})
        print(f"thermo seed {seed} batch {b + 1}/{K}: {stats[-1]}", flush=True)
    out["batches"] = stats
    out["informative_frac"] = nn / K
    # analytic (secondary): preconditioner from the non-null batches only
    P = 1.0 / ((v_sum / max(1, nn)).sqrt() + EPS)
    w_seq, w_tf, w_false = LR * P * gseq, LR * P * gtf, LR * P * gfalse
    dseq = [0.0 if g is None else -(w_seq * g.float()).sum().item() for g in grads]
    dtf = [0.0 if g is None else -(w_tf * g.float()).sum().item() for g in grads]
    dfalse = [0.0 if g is None else -(w_false * g.float()).sum().item() for g in grads]
    out["analytic"] = {"sigma_seq": sd(dseq), "sigma_tf": sd(dtf), "mu_seq": sum(dseq) / K, "mu_tf": sum(dtf) / K,
                       "pred_SD_seq": {T: sd(dseq) * math.sqrt(adam_sum_w2(T)) for T in SCHED},
                       "pred_SD_tf": {T: sd(dtf) * math.sqrt(adam_sum_w2(T)) for T in SCHED},
                       "sigma_false": sd(dfalse), "mu_false": sum(dfalse) / K,
                       "pred_SD_false": {T: sd(dfalse) * math.sqrt(adam_sum_w2(T)) for T in SCHED}}
    if full:
        # one-step checks: the exact first Adam step u = -lr * g / (|g| + eps), true change vs <grad D, u>
        checks = []
        for b in [i for i, g in enumerate(grads) if g is not None][:ONE_STEP_BATCHES]:
            g = grads[b].float()
            for mult in (1, 10):
                u = -(LR * mult) * g / (g.abs() + EPS)
                set_flat(params, theta0 + u)
                with torch.no_grad():
                    ds, _, lps_ = d_seq(model, pool)
                    dt = d_tf(tok, model, probe)[0]
                set_flat(params, theta0)
                checks.append({"batch": b, "lr_mult": mult, "true_dseq": ds - Dseq0, "pred_dseq": (gseq * u).sum().item(),
                               "true_dtf": dt - Dtf0, "pred_dtf": (gtf * u).sum().item(),
                               "true_dfalse": weighted_rate(lps_, pool, "false_claim") - Dfalse0,
                               "pred_dfalse": (gfalse * u).sum().item()})
                print("one-step check:", checks[-1], flush=True)
        out["one_step_checks"] = checks
        # regions (exploratory, H3): projections of each region's gradient on the analytic directions
        reg = {name: {"seq": [], "tf": []} for name in ("successful", "reasoning", "window16", "tail")}
        for b, res in enumerate(kept):
            if res is None or grads[b] is None:
                continue
            parts = {}
            for code, name in ((1, "reasoning"), (2, "window16"), (3, "tail")):
                if grpo_loss(tok, model, res, region=code) is None:
                    parts[name] = torch.zeros_like(theta0)
                else:
                    parts[name] = flat_grad(params)
                zero(params)
            parts["successful"] = grads[b].float() - parts["reasoning"] - parts["window16"] - parts["tail"]
            for name, gpart in parts.items():
                reg[name]["seq"].append(-(w_seq * gpart).sum().item())
                reg[name]["tf"].append(-(w_tf * gpart).sum().item())
        summ = {}
        for which in ("seq", "tf"):
            s = {name: sd(reg[name][which]) if len(reg[name][which]) > 1 else None for name in reg}
            wt = [a + b for a, b in zip(reg["window16"][which], reg["tail"][which])]
            s["window16_plus_tail"] = sd(wt) if len(wt) > 1 else None
            summ[which] = s
        out["regions"] = {"sd": summ, "values": reg}
    # free the model, then the replay (primary prediction)
    del model, params
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # the H6 direction goes LAST, so the 5 random directions and their seeds are unchanged
    dirs = [gseq, gtf] + random_dirs(theta0.numel()) + [gfalse]
    if seed == 1:
        torch.save({"gseq": gseq, "gtf": gtf, "gfalse": gfalse, "theta0": theta0},
                   os.path.join(outdir, "vettori_s1.pt"))
        torch.save(grads, os.path.join(outdir, "gradienti_s1.pt"))
    vals = adam_replay(grads, dirs, REPLAY_B, seed=500 + seed)
    names = ["lin_seq", "lin_tf"] + [f"rand{i}" for i in range(1, N_RANDOM_DIRS + 1)] + ["lin_false"]
    pred = {nm: {T: sd([vals[r][k][j] for r in range(len(vals))]) for k, T in enumerate(SCHED)}
            for j, nm in enumerate(names)}
    k20, k40 = SCHED.index(20), SCHED.index(40)
    jf = names.index("lin_false")
    out["replay"] = {"B": REPLAY_B, "pred_SD": pred,
                     "pred_mean_lin_seq_40": sum(vals[r][k40][0] for r in range(len(vals))) / len(vals),
                     "pred_corr_20_40_lin_seq": corr([vals[r][k20][0] for r in range(len(vals))],
                                                     [vals[r][k40][0] for r in range(len(vals))]),
                     # H6: how linked the two behaviours are (the two labels are nearly complementary in the pilot)
                     "pred_corr_seq_false_40": corr([vals[r][k40][0] for r in range(len(vals))],
                                                    [vals[r][k40][jf] for r in range(len(vals))])}
    out["label_corr_d_false"] = corr([label(it, "d") for it in pool], [label(it, "false_claim") for it in pool])
    out["minutes"] = round((time.time() - t_start) / 60, 1)
    path = os.path.join(outdir, f"thermo_s{seed}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("PREVISIONE:", json.dumps({"D_seq_0": Dseq0, "D_tf_0": Dtf0, "informative": out["informative_frac"],
                                     "pred_SD_lin_seq_40": pred["lin_seq"][40], "pred_SD_lin_tf_40": pred["lin_tf"][40],
                                     "analytic_seq_40": out["analytic"]["pred_SD_seq"][40],
                                     "D_false_0": Dfalse0, "pred_SD_lin_false_40": pred["lin_false"][40]}), flush=True)


# ---------- gates ----------

def estrai_code(pool_path, out_blind, out_labels, n):
    """G5 pilot: the first n failures of a pool whose text after 'Answer:' is not empty. Writes the answer line and the
    tail for an independent labeller (out_blind, no detector label) and the detector labels separately (out_labels)."""
    pool = json.load(open(pool_path))["pool"]
    blind, labels = [], []
    for k, it in enumerate(pool):
        answers = list(cd.ANSWER.finditer(it["text"]))
        if not answers or not it["tail"].strip():
            continue
        blind.append({"id": k, "answer_line": answers[-1].group(0), "tail": it["tail"]})
        labels.append({"id": k, "detector": it["d"]})
        if len(blind) == n:
            break
    with open(out_blind, "w", encoding="utf-8") as f:
        for b in blind:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    json.dump(labels, open(out_labels, "w"))
    print(f"{len(blind)} non-empty tails written; detector says 'admits' in {sum(l['detector'] for l in labels)}")


def gates_pool(pool_path, out_path):
    """The gates that only need the step-0 pool (G0, G1, G3), checked right after it is built (saves ~1.5 h)."""
    st = json.load(open(pool_path))["stats"]
    res = {"G0_pure_sampling": st["sampling_processors"] == [],
           "G1_solve_2_to_60pct": 0.02 <= st["solve_rate"] <= 0.60,
           "G3_pool_disclosure_10_to_90pct": 0.10 <= st["pool_disclosure"] <= 0.90 and st["pool_size"] == N_POOL,
           "info": st}
    ok = all(v for k, v in res.items() if k.startswith("G"))
    res["ALL_PASS"] = ok
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "info"}, indent=1))
    sys.exit(0 if ok else 3)


def gates(pool_path, t1_path, t2_path, out_path):
    st = json.load(open(pool_path))["stats"]
    t1, t2 = json.load(open(t1_path)), json.load(open(t2_path))
    pred = t1["replay"]["pred_SD"]["lin_seq"]["40"] if "40" in t1["replay"]["pred_SD"]["lin_seq"] else \
        t1["replay"]["pred_SD"]["lin_seq"][40]
    res = {
        "G0_pure_sampling": st["sampling_processors"] == [],
        "G1_solve_2_to_60pct": 0.02 <= st["solve_rate"] <= 0.60,
        "G2_informative_batches_ge_30pct": t1["informative_frac"] >= 0.30,
        "G3_pool_disclosure_10_to_90pct": 0.10 <= st["pool_disclosure"] <= 0.90 and st["pool_size"] == N_POOL,
        "G4_pred_SD_ge_5x_floor": pred >= 5 * t1["floor_seq"],
        "G6_ESS_ge_70pct_under_tiny_perturbation": t1["ess_floor_min"] >= 0.7 * N_POOL,
        "info": {"solve_rate": st["solve_rate"], "informative": t1["informative_frac"],
                 "pool_disclosure": st["pool_disclosure"], "pred_SD_lin_seq_40": pred, "floor_seq": t1["floor_seq"],
                 "D_tf_alive_frac": t1["alive_p_frac"], "retest_ratio_s2_s1": (
                     (t2["replay"]["pred_SD"]["lin_seq"].get("40") or t2["replay"]["pred_SD"]["lin_seq"].get(40))
                     / pred if pred else None),
                 "sampling_processors": st["sampling_processors"]}}
    res["D_tf_reported_as_flat"] = t1["alive_p_frac"] < 0.20
    ok = all(v for k, v in res.items() if k.startswith("G"))
    res["ALL_PASS"] = ok
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res, indent=1))
    sys.exit(0 if ok else 3)


# ---------- a real run ----------

def run(sft_dir, pool_path, thermo_dir, out, seed, T):
    seed_all(seed)
    tok, model = load(base_of(sft_dir), lora_from=sft_dir, trainable=True)
    params = [p for p in model.parameters() if p.requires_grad]
    data = json.load(open(pool_path))
    pool, probe = data["pool"], data["probe"]
    vec = torch.load(os.path.join(thermo_dir, "vettori_s1.pt"))
    theta0 = vec["theta0"]
    assert torch.allclose(theta0, flat_params(params)), "the run does not start from the thermometer's theta_0"
    dirs = [vec["gseq"], vec["gtf"]] + random_dirs(theta0.numel()) + [vec["gfalse"]]  # same order as the thermometer
    opt = torch.optim.Adam(params, lr=LR, betas=BETAS, eps=EPS)
    tasks = cd.rl_tasks(20_000 + seed, PROMPTS_PER_STEP * T)
    Dtf, _, _ = d_tf(tok, model, probe)
    Ds, ess, lps = d_seq(model, pool)
    assert ess >= 0.9 * len(pool), f"ESS at step 0 is {ess:.1f} of {len(pool)}: logp0 does not match this run"
    log = [{"step": 0, "D_tf": Dtf, "D_seq": Ds, "D_false": weighted_rate(lps, pool, "false_claim"), "ess": ess,
            "lp_seq": lps, "lin": [0.0] * len(dirs), "dtype": str(DTYPE)}]
    # non-empty tails of the failures in the last 5 steps, for the detector agreement after training (report only)
    code_path = out.replace(".json", "_code.jsonl")
    open(code_path, "w").close()
    for t in range(T):
        t0 = time.time()
        res = rollouts(tok, model, tasks[t * PROMPTS_PER_STEP:(t + 1) * PROMPTS_PER_STEP])
        model.train()
        tot = grpo_loss(tok, model, res)
        opt.step()
        zero(params)
        delta = flat_params(params) - theta0
        fails = [r for r in res if r["reward"] == 0 and r["terminated"]]
        row = {"step": t + 1, "solve": sum(r["reward"] for r in res) / len(res), "informative": tot is not None,
               "disclosure_sampled": sum(bool(r["disclosed"]) for r in fails) / max(1, len(fails)),
               "n_fail_terminated": len(fails), "lin": [float(d @ delta) for d in dirs]}
        if t + 1 > T - 5:
            with open(code_path, "a", encoding="utf-8") as f:
                for r in fails:
                    answers = list(cd.ANSWER.finditer(r["text"]))
                    if answers and r["text"][answers[-1].end():].strip():
                        f.write(json.dumps({"step": t + 1, "answer_line": answers[-1].group(0),
                                            "tail": r["text"][answers[-1].end():], "detector": bool(r["disclosed"])},
                                           ensure_ascii=False) + "\n")
        if (t + 1) in D_TF_STEPS or t + 1 == T:
            row["D_tf"] = d_tf(tok, model, probe)[0]
        if (t + 1) in D_SEQ_STEPS or t + 1 == T:
            row["D_seq"], row["ess"], row["lp_seq"] = d_seq(model, pool)
            row["D_false"] = weighted_rate(row["lp_seq"], pool, "false_claim")  # H6, from the same weights
        if (t + 1) in CKPT_STEPS:
            model.save_pretrained(out.replace(".json", f"_lora{t + 1}"))
        row["seconds"] = round(time.time() - t0, 1)
        log.append(row)
        print(f"run seed {seed} step {t + 1}/{T}: " + json.dumps({k: v for k, v in row.items() if k != "lp_seq"}),
              flush=True)
        json.dump(log, open(out, "w"))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sft":
        sft(sys.argv[2], sys.argv[3])
    elif cmd == "pool":
        make_pool(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif cmd == "thermo":
        thermo(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7] == "1")
    elif cmd == "gates_pool":
        gates_pool(sys.argv[2], sys.argv[3])
    elif cmd == "estrai_code":
        estrai_code(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    elif cmd == "gates":
        gates(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif cmd == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), int(sys.argv[7]))
