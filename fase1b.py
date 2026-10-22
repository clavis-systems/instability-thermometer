"""Phase 1b (PREMESSA_FASE1B.md, 20/10/2026): the long horizon. It imports the FROZEN termometro.py and changes only
the horizon: the replay schedule, the measurement steps and the checkpoints. Nothing else.

The horizon follows a rule written before any run (review of 20/10): estimated time of a 150-step run = 150 x the
median seconds per batch of A's thermometer (seed 1) + 8 measurements x 260 s (the T4 float32 speed test). If it is
above 3.5 hours, the horizon is 100 steps and the tests are at T = 80 and 100; otherwise 150 steps, tests at 80 and 150.

  predict A_DIR OUT.json            replay of Adam with the phase-1 step-0 gradients (gradienti_s1.pt)
  run SFT_DIR POOL A_DIR OUT SEED   one run (seeds 201-206), same code as phase 1 with the 1b measuring steps
  analyze A_DIR PRED.json RUN_DIR   the frozen criteria (writes analisi_fase1b.json and RAPPORTO_FASE1B.md)
  horizon A_DIR                     prints the horizon chosen by the rule
  test                              the analysis on synthetic runs (no torch)
"""
import glob
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW = 20           # D_samp: the training failures of the last 20 steps before T (review 20/10: 10 was too noisy)
EVAL_SECONDS = 260    # one D_seq + D_tf measurement on a T4 in float32 (speed test of 20/10)
LIMIT_HOURS = 3.5
NOISE_RULE = 0.5      # D_samp verdict only if its noise SD is at most half the predicted SD
SLOPE_BAND = 0.5      # H2b (descriptive): beyond +-0.5 from the predicted slope it is called faster / slower


def plan(T_max):
    steps = [10, 20, 40, 60, 80, 100] + ([120, 150] if T_max == 150 else [])
    return {"T_max": T_max, "sched": steps, "steps": set(steps), "ckpt": {40, 80, T_max},
            "tests": [80, T_max], "slope_T": [T for T in steps if T >= 40]}


def horizon(a_dir):
    t1 = json.load(open(os.path.join(a_dir, "thermo_s1.json")))
    per_batch = statistics.median(b["seconds"] for b in t1["batches"])
    hours_150 = (150 * per_batch + 8 * EVAL_SECONDS) / 3600
    return (150 if hours_150 <= LIMIT_HOURS else 100), {"seconds_per_batch": per_batch, "hours_150": hours_150}


def sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def verdict(R):
    if R is None or R != R or R in (float("inf"), float("-inf")):
        return "NON CALCOLABILE"
    return "PROMOSSO" if 0.5 <= R <= 2 else "INDIZIO" if 0.33 <= R <= 3 else "BOCCIATO"


def slope(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def _tm(p):
    sys.path.insert(0, HERE)
    import termometro as tm  # the frozen code
    tm.SCHED = list(p["sched"])
    tm.D_SEQ_STEPS = set(p["steps"])
    tm.D_TF_STEPS = set(p["steps"])
    tm.CKPT_STEPS = set(p["ckpt"])
    return tm


def predict(a_dir, out):
    T_max, info = horizon(a_dir)
    p = plan(T_max)
    tm = _tm(p)
    import torch
    vec = torch.load(os.path.join(a_dir, "vettori_s1.pt"))
    grads = torch.load(os.path.join(a_dir, "gradienti_s1.pt"))
    dirs = [vec["gseq"], vec["gtf"]] + tm.random_dirs(vec["theta0"].numel()) + [vec["gfalse"]]
    names = ["lin_seq", "lin_tf"] + [f"rand{i}" for i in range(1, tm.N_RANDOM_DIRS + 1)] + ["lin_false"]
    vals = tm.adam_replay(grads, dirs, tm.REPLAY_B, seed=501)  # same seed as the phase-1 thermometer seed 1
    pred = {nm: {str(T): sd([vals[r][k][j] for r in range(len(vals))]) for k, T in enumerate(p["sched"])}
            for j, nm in enumerate(names)}
    t1 = json.load(open(os.path.join(a_dir, "thermo_s1.json")))
    res = {"T_max": T_max, "horizon_rule": info, "pred_SD": pred,
           "phase1_pred_SD_lin_seq_40": t1["replay"]["pred_SD"]["lin_seq"]["40"],
           "note": "replica draws now cover T_max steps, so T=40 differs from phase 1 only by Monte Carlo noise"}
    json.dump(res, open(out, "w"), indent=1)
    print("PREVISIONE 1B:", json.dumps({"T_max": T_max, **{T: pred["lin_seq"][str(T)] for T in map(str, p["sched"])}}))


def run(sft_dir, pool, a_dir, out, seed):
    T_max, _ = horizon(a_dir)
    tm = _tm(plan(T_max))
    tm.run(sft_dir, pool, a_dir, out, int(seed), T_max)


def window(rows, T, key):
    """Per-step values of `key` over the last WINDOW steps up to T, with the number of terminated failures."""
    return [(r[key], r["n_fail_terminated"]) for r in rows if T - WINDOW < r["step"] <= T and key in r]


def d_samp(rows, T):
    """Disclosure among the terminated failures of the training batches of the last WINDOW steps (no importance
    weights), and the noise variance of that window mean estimated from the spread of the per-step rates (it includes the
    clustering of the answers in groups of 8 per task, which the binomial formula misses)."""
    w = [(x, n) for x, n in window(rows, T, "disclosure_sampled") if n > 0]
    if len(w) < 3:
        return None, None
    tot = sum(n for _, n in w)
    mean = sum(x * n for x, n in w) / tot
    rates = [x for x, _ in w]
    return mean, statistics.variance(rates) / len(rates)


def analyse(t1, pred, runs, T_max):
    p = plan(T_max)
    n = len(runs)
    out = {"n_runs": n, "T_max": T_max, "power_note": "with 6 runs the 95% CI of an SD is 0.62-2.45 times"}
    if n < 4:
        out["verdict"] = "NON CONCLUSIVO"
        out["reason"] = f"only {n} complete runs (the rule needs at least 4)"
        return out
    if n < 6:
        out["reduced_power"] = f"{n} runs instead of 6"
    floors = {"D_seq": t1.get("floor_seq", 0.0), "D_tf": t1.get("floor_tf", 0.0), "D_false": t1.get("floor_false", 0.0)}
    funcs = {"D_seq": "lin_seq", "D_tf": "lin_tf", "D_false": "lin_false"}

    def row(run, T):
        return next(r for r in run if r["step"] == T)

    def corrected(key, T):
        return math.sqrt(max(sd([row(run, T)[key] for run in runs]) ** 2 - floors[key] ** 2, 0.0))

    held = []
    for T in p["tests"]:
        pr = pred["pred_SD"]["lin_seq"][str(T)]
        sc = corrected("D_seq", T)
        R = sc / pr if pr > 0 else None
        ess = statistics.median(row(run, T)["ess"] for run in runs)
        v = verdict(R)
        if ess < 100 and v != "NON CALCOLABILE":
            v = "NON CONCLUSIVO"
        samp = [d_samp(run, T) for run in runs]
        if all(m is not None for m, _ in samp):
            noise = sum(var for _, var in samp) / n
            ssc = math.sqrt(max(sd([m for m, _ in samp]) ** 2 - noise, 0.0))
            Rs = ssc / pr if pr > 0 else None
            vs = "NON CONCLUSIVO" if math.sqrt(noise) > NOISE_RULE * pr else verdict(Rs)
        else:
            noise, ssc, Rs, vs = None, None, None, "NON CALCOLABILE"
        used = vs if v == "NON CONCLUSIVO" else v
        out[f"H1b_T{T}"] = {"measured_SD_D_seq": sd([row(run, T)["D_seq"] for run in runs]), "corrected": sc,
                            "predicted": pr, "R": R, "median_ESS": ess, "verdict": v,
                            "D_samp_SD_corrected": ssc, "D_samp_noise_var": noise, "R_samp": Rs, "verdict_samp": vs,
                            "verdict_used": used, "measure_used": "D_samp" if v == "NON CONCLUSIVO" else "D_seq",
                            "descriptive": {k: {"R": (corrected(k, T) / pred["pred_SD"][funcs[k]][str(T)]
                                                      if pred["pred_SD"][funcs[k]][str(T)] > 0 else None)}
                                            for k in ("D_tf", "D_false") if all(k in row(run, T) for run in runs)},
                            "solve_window_mean": statistics.mean(
                                statistics.mean(x for x, _ in window(run, T, "solve")) for run in runs)}
        held.append(used == "PROMOSSO")
    # how far it holds: the largest tested T such that every tested T up to it is PROMOSSO
    out["holds_up_to"] = None
    for T, ok in zip(p["tests"], held):
        if not ok:
            break
        out["holds_up_to"] = T
    # H2b (descriptive only): slope of log SD vs log T, on the steps whose median ESS is at least 100
    Ts = [T for T in p["slope_T"] if statistics.median(row(run, T)["ess"] for run in runs) >= 100]
    meas = [corrected("D_seq", T) for T in Ts]
    if len(Ts) >= 3 and min(meas) > 0:
        sm = slope([math.log(T) for T in Ts], [math.log(x) for x in meas])
        sp = slope([math.log(T) for T in Ts], [math.log(pred["pred_SD"]["lin_seq"][str(T)]) for T in Ts])
        shape = "più veloce" if sm > sp + SLOPE_BAND else "più lenta" if sm < sp - SLOPE_BAND else "come prevista"
        out["H2b"] = {"T_used": Ts, "slope_measured": sm, "slope_predicted": sp, "shape": shape, "descriptive": True}
    else:
        out["H2b"] = {"T_used": Ts, "note": "fewer than 3 steps with median ESS >= 100: not computable"}
    return out


FRASI = {  # written before any run (20/10/2026)
    "PROMOSSO": "A {T} passi la dispersione vera è {R:.2f} volte quella prevista al passo 0 (misura {m}): la previsione regge.",
    "INDIZIO": "A {T} passi la dispersione vera è {R:.2f} volte quella prevista (misura {m}): solo un indizio.",
    "BOCCIATO": "A {T} passi la dispersione vera è {R:.2f} volte quella prevista (misura {m}): la previsione non regge.",
    "NON CONCLUSIVO": "A {T} passi non si può dire: i pesi per importanza sono crollati e la misura diretta è troppo rumorosa.",
    "NON CALCOLABILE": "A {T} passi il rapporto non è calcolabile.",
}


def scrivi(a):
    righe = [f"# Termometro, fase 1b: l'orizzonte lungo ({a['n_runs']} corse, fino a {a['T_max']} passi)"]
    if "verdict" in a:
        return "\n\n".join(righe + [f"NON CONCLUSIVO: {a['reason']}."]) + "\n"
    if "reduced_power" in a:
        righe.append(f"Potenza ridotta: {a['reduced_power']}.")
    for k in sorted((x for x in a if x.startswith("H1b_T")), key=lambda x: int(x[5:])):
        h = a[k]
        R = h["R_samp"] if h["measure_used"] == "D_samp" else h["R"]
        righe.append(FRASI[h["verdict_used"]].format(T=k[5:], R=R if R is not None else float("nan"), m=h["measure_used"]))
    primo = a[f"H1b_T{min(int(x[5:]) for x in a if x.startswith('H1b_T'))}"]["verdict_used"]
    if a["holds_up_to"]:
        righe.append(f"La previsione del passo 0 regge fino a {a['holds_up_to']} passi.")
    elif primo in ("BOCCIATO", "INDIZIO"):
        righe.append("La previsione del passo 0 non regge già al primo orizzonte provato.")
    else:
        righe.append("Sull'orizzonte lungo non si può dire se la previsione regge.")
    if "shape" in a["H2b"]:
        righe.append(f"(Descrittivo) la crescita della dispersione è {a['H2b']['shape']} rispetto alla diffusione prevista "
                     f"(pendenza {a['H2b']['slope_measured']:.2f} contro {a['H2b']['slope_predicted']:.2f}).")
    return "\n\n".join(righe) + "\n"


def analyze(a_dir, pred_path, run_dir):
    t1 = json.load(open(os.path.join(a_dir, "thermo_s1.json")))
    pred = json.load(open(pred_path))
    T_max = pred["T_max"]
    files = sorted(glob.glob(os.path.join(run_dir, "run_s2[0-9][0-9].json")))
    runs, used = [], []
    for f in files:
        r = json.load(open(f))
        if len(r) == T_max + 1:
            runs.append(r)
            used.append(os.path.basename(f))
    res = analyse(t1, pred, runs, T_max)
    res["files_used"] = used
    res["files_incomplete"] = [os.path.basename(f) for f in files if os.path.basename(f) not in used]
    json.dump(res, open(os.path.join(run_dir, "analisi_fase1b.json"), "w"), indent=1)
    testo = scrivi(res)
    open(os.path.join(run_dir, "RAPPORTO_FASE1B.md"), "w", encoding="utf-8").write(testo)
    print(json.dumps(res, indent=1))
    print(testo)


def _walks(rng, n, T_max, sigma, group_noise, ess):
    b1, runs = 0.9, []
    for _ in range(n):
        m, lin, rows = 0.0, 0.0, [{"step": 0, "D_seq": 0.5, "D_tf": 0.5, "D_false": 0.3, "ess": 400}]
        for t in range(1, T_max + 1):
            m = b1 * m + (1 - b1) * rng.gauss(0, sigma)
            lin += m / (1 - b1 ** t)
            row = {"step": t, "n_fail_terminated": 30, "solve": 0.05,
                   "disclosure_sampled": min(1, max(0, 0.5 + lin + rng.gauss(0, group_noise)))}
            if t in plan(T_max)["steps"]:
                row.update(D_seq=0.5 + lin + rng.gauss(0, 0.001), D_tf=0.5 + lin, D_false=0.3 - lin, ess=ess)
            rows.append(row)
        runs.append(rows)
    return runs


def test():
    import random
    rng = random.Random(0)
    T_max, sigma = 150, 0.003
    # the "prediction" comes from 4000 INDEPENDENT walks, not from the runs being tested
    ref = _walks(rng, 4000, T_max, sigma, 0.0, 300)
    pred = {"pred_SD": {nm: {str(T): sd([next(r for r in run if r["step"] == T)["D_seq"] - 0.5 for run in ref])
                             for T in plan(T_max)["sched"]} for nm in ("lin_seq", "lin_tf", "lin_false")}}
    big = analyse({"floor_seq": 0.001}, pred, _walks(rng, 600, T_max, sigma, 0.12, 300), T_max)
    assert 0.9 < big["H1b_T150"]["R"] < 1.1 and big["H2b"]["shape"] == "come prevista", big
    assert 0.85 < big["H1b_T150"]["R_samp"] < 1.15, big["H1b_T150"]  # cluster noise estimated from the window
    low = analyse({"floor_seq": 0.001}, pred, _walks(rng, 6, T_max, sigma, 0.12, 50), T_max)
    assert low["H1b_T150"]["verdict"] == "NON CONCLUSIVO" and low["H1b_T150"]["measure_used"] == "D_samp", low
    assert analyse({}, pred, _walks(rng, 3, T_max, sigma, 0.12, 300), T_max)["verdict"] == "NON CONCLUSIVO"
    loud = analyse({"floor_seq": 0.001}, pred, _walks(rng, 6, T_max, sigma, 2.0, 50), T_max)
    assert loud["H1b_T150"]["verdict_samp"] == "NON CONCLUSIVO", loud["H1b_T150"]  # noise rule
    print(scrivi(big))
    print(scrivi(low))
    print("fase 1b analysis test passed")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "predict":
        predict(sys.argv[2], sys.argv[3])
    elif cmd == "run":
        run(*sys.argv[2:7])
    elif cmd == "analyze":
        analyze(*sys.argv[2:5])
    elif cmd == "horizon":
        print(horizon(sys.argv[2]))
    elif cmd == "test":
        test()
