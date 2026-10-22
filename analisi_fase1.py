"""Pre-registered analysis of phase 1 of the instability thermometer (written 18/10/2026, before any real data).

Usage: python -I analisi_fase1.py RESULTS_DIR      (needs thermo_s1.json, thermo_s2.json, run_s*.json)
       python -I analisi_fase1.py test             (synthetic check of the code)

Primary (H1): R = SD across runs of D_seq at T = 40, floor-corrected in quadrature, divided by the replay prediction
SD of <grad D_seq, theta_40 - theta_0>. PROMOSSO if 0.5 <= R <= 2; INDIZIO if 0.33 <= R <= 3; else BOCCIATO.
Everything else is secondary or exploratory, as written in PREMESSA.md.
"""
import glob
import json
import math
import os
import random
import sys

SCHED = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40]
NAMES = ["lin_seq", "lin_tf"] + [f"rand{i}" for i in range(1, 6)] + ["lin_false"]  # lin_false last (H6, 20/10)


def sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def corr(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


def slope(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def chi2_quantile(p, df):
    """Wilson-Hilferty approximation (within about 1% for df >= 5)."""
    z = {0.025: -1.959964, 0.975: 1.959964}[p]
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def sd_ci(s, n):
    df = n - 1
    return s * math.sqrt(df / chi2_quantile(0.975, df)), s * math.sqrt(df / chi2_quantile(0.025, df))


def verdict(R):
    if R is None or R != R or R in (float("inf"), float("-inf")):
        return "NON CALCOLABILE"
    return "PROMOSSO" if 0.5 <= R <= 2 else "INDIZIO" if 0.33 <= R <= 3 else "BOCCIATO"


def adam_sum_w2(T, b1=0.9):
    return sum(sum((1 - b1) * b1 ** (t - k) / (1 - b1 ** t) for t in range(k, T + 1)) ** 2 for k in range(1, T + 1))


def key(d, T):
    return d[str(T)] if str(T) in d else d[T]


def analyse(t1, t2, runs):
    n = len(runs)
    out = {"n_runs": n, "seeds_note": "runs are listed in seed order"}
    pred = {nm: {T: key(t1["replay"]["pred_SD"][nm], T) for T in SCHED} for nm in NAMES if nm in t1["replay"]["pred_SD"]}
    floor = t1.get("floor_seq", 0.0)
    floor_tf = t1.get("floor_tf", 0.0)

    def at(T, field, idx=None):
        vals = []
        for r in runs:
            row = r[T]
            v = row[field] if idx is None else row[field][idx]
            vals.append(v)
        return vals

    # H1 (primary)
    x40 = at(40, "D_seq")
    s40 = sd(x40)
    s40c = math.sqrt(max(s40 ** 2 - floor ** 2, 0.0))
    p40 = pred["lin_seq"][40]
    R = s40c / p40 if p40 > 0 else None
    lo, hi = sd_ci(s40c, n)
    ess40 = sorted(r[40]["ess"] for r in runs)
    med_ess = ess40[len(ess40) // 2]
    v = verdict(R)
    if med_ess < 100 and v != "NON CALCOLABILE":
        v = "NON CONCLUSIVO"  # the importance weights have collapsed: the measure, not the thermometer, is at fault
    out["H1"] = {"measured_SD_D_seq_40": s40, "floor_seq": floor, "measured_SD_corrected": s40c,
                 "measured_SD_CI95": [lo, hi], "predicted_SD_replay": p40, "R": R,
                 "R_CI95": [lo / p40, hi / p40] if p40 > 0 else None, "median_ESS_40": med_ess, "verdict": v}
    # sensitivity to the numerical noise (added 20/10, before the freeze; does not change the verdict): the floor
    # taken from the two 1e-5 perturbations (root mean square distance from D_seq_0) instead of the 1e-7 ones
    big = t1.get("floor_values", {}).get("seq_at_1e-5", [])
    if big:
        fb = math.sqrt(sum((b - t1["D_seq_0"]) ** 2 for b in big) / len(big))
        sb = math.sqrt(max(s40 ** 2 - fb ** 2, 0.0))
        Rb = sb / p40 if p40 > 0 else None
        out["H1_sensitivity_floor_1e-5"] = {"floor_1e-5": fb, "floor_ratio_1e-5_vs_1e-7": fb / floor if floor > 0 else None,
                                            "R": Rb, "verdict": verdict(Rb), "same_verdict": verdict(Rb) == verdict(R)}
    # H6 (exploratory, added 20/10 before the freeze): the second behaviour, a false claim of success, measured with
    # the same importance weights; same R rule; and whether the thermometer ranks the two behaviours right
    if all("D_false" in r[40] for r in runs) and "lin_false" in pred and "floor_false" in t1:
        xf = at(40, "D_false")
        ff = t1["floor_false"]
        sfc = math.sqrt(max(sd(xf) ** 2 - ff ** 2, 0.0))
        pf = pred["lin_false"][40]
        Rf = sfc / pf if pf > 0 else None
        vf = verdict(Rf)
        if med_ess < 100 and vf != "NON CALCOLABILE":
            vf = "NON CONCLUSIVO"
        ratio_p = pf / p40 if p40 > 0 else None
        # parity zone (review 20/10): the two labels are nearly complementary, so a predicted ratio within
        # [0.9, 1.1] means the thermometer does not distinguish them, and the ranking is not scored
        tie = ratio_p is None or 0.9 <= ratio_p <= 1.1
        pred_more = None if tie else ("false_claim" if pf > p40 else "disclosure")
        meas_more = "false_claim" if sfc > s40c else "disclosure"
        out["H6"] = {"measured_SD_D_false_40": sd(xf), "floor_false": ff, "measured_SD_corrected": sfc,
                     "predicted_SD_replay": pf, "R": Rf, "verdict": vf,
                     "predicted_ratio_false_over_disclosure": ratio_p, "ranking_scored": not tie,
                     "measured_ratio_false_over_disclosure": sfc / s40c if s40c > 0 else None,
                     "predicted_more_unstable": pred_more, "measured_more_unstable": meas_more,
                     "ranking_correct": None if tie else pred_more == meas_more,
                     "pred_corr_seq_false_40": t1["replay"].get("pred_corr_seq_false_40"),
                     "measured_corr_seq_false_40": corr(at(40, "D_seq"), xf),
                     "label_corr_d_false": t1.get("label_corr_d_false")}
    # secondary: D_tf, the exact linear functionals, the analytic prediction, the retest seed
    xtf = at(40, "D_tf")
    stf = math.sqrt(max(sd(xtf) ** 2 - floor_tf ** 2, 0.0))
    ptf = pred["lin_tf"][40]
    out["secondary"] = {
        "D_tf_40": {"measured_SD_corrected": stf, "predicted": ptf, "R": stf / ptf if ptf > 0 else None,
                    "reported_as_flat": t1.get("alive_p_frac", 1.0) < 0.20},
        "linear_functionals_40": {nm: {"measured_SD": sd(at(40, "lin", j)), "predicted": pred[nm][40],
                                       "R": sd(at(40, "lin", j)) / pred[nm][40] if pred[nm][40] > 0 else None}
                                  for j, nm in enumerate(NAMES) if nm in pred},
        "analytic_R": s40c / key(t1["analytic"]["pred_SD_seq"], 40) if key(t1["analytic"]["pred_SD_seq"], 40) > 0 else None,
        "retest_R_with_seed2": s40c / key(t2["replay"]["pred_SD"]["lin_seq"], 40),
        "retest_ratio_pred_s2_s1": key(t2["replay"]["pred_SD"]["lin_seq"], 40) / p40 if p40 > 0 else None}
    # H2: growth with T (floor-corrected D_seq at 10, 20, 30, 40; exact linear functional at 5..40)
    Ts = [10, 20, 30, 40]
    meas = [math.sqrt(max(sd(at(T, "D_seq")) ** 2 - floor ** 2, 0.0)) for T in Ts]
    sl_m = slope([math.log(T) for T in Ts], [math.log(v) for v in meas]) if min(meas) > 0 else None
    sl_p = slope([math.log(T) for T in Ts], [math.log(pred["lin_seq"][T]) for T in Ts])
    Tl = [5, 10, 15, 20, 25, 30, 35, 40]
    sl_lin = slope([math.log(T) for T in Tl], [math.log(sd(at(T, "lin", 0))) for T in Tl])
    sl_lin_p = slope([math.log(T) for T in Tl], [math.log(pred["lin_seq"][T]) for T in Tl])
    out["H2"] = {"slope_measured_D_seq": sl_m, "slope_predicted": sl_p,
                 "pass": (abs(sl_m - sl_p) <= 0.25) if sl_m is not None else False,
                 "note": None if sl_m is not None else "a floor-corrected SD is zero: slope not computable",
                 "slope_measured_lin_seq": sl_lin, "slope_predicted_lin_seq": sl_lin_p}
    # H3 (exploratory): regions
    reg = t1.get("regions", {}).get("sd", {}).get("seq", {})
    if reg:
        out["H3"] = {"sd_reasoning": reg.get("reasoning"), "sd_window16_plus_tail": reg.get("window16_plus_tail"),
                     "sd_successful": reg.get("successful"),
                     "pass": (reg.get("reasoning") or 0) > (reg.get("window16_plus_tail") or 0)}
    # H5 (exploratory): correlation between step 20 and step 40
    out["H5"] = {"measured_corr_D_seq_20_40": corr(at(20, "D_seq"), at(40, "D_seq")),
                 "measured_corr_lin_seq_20_40": corr(at(20, "lin", 0), at(40, "lin", 0)),
                 "predicted": t1["replay"]["pred_corr_20_40_lin_seq"]}
    # baseline: naive extrapolation from the spread at step 5
    s5 = math.sqrt(max(sd(at(5, "D_seq")) ** 2 - floor ** 2, 0.0))
    naive = s5 * math.sqrt(adam_sum_w2(40) / adam_sum_w2(5))
    out["baseline_naive_from_step5"] = {"prediction": naive, "R": s40c / naive if naive > 0 else None}
    # checks of the linear picture
    dD = [r[40]["D_seq"] - r[0]["D_seq"] for r in runs]
    lin = at(40, "lin", 0)
    out["linearisation"] = {"corr_true_change_vs_linear": corr(dD, lin), "slope": slope(lin, dD)}
    for mult in (1, 10):
        chk = [c for c in t1.get("one_step_checks", []) if c["lr_mult"] == mult]
        if len(chk) >= 2:
            px, ty = [c["pred_dseq"] for c in chk], [c["true_dseq"] for c in chk]
            out[f"one_step_lr_x{mult}"] = {"corr": corr(px, ty),
                                           "slope": slope(px, ty) if len(set(px)) > 1 else None}
    out["ess_40"] = sorted(r[40]["ess"] for r in runs)
    out["mean_D_seq"] = {T: sum(at(T, "D_seq")) / n for T in [5, 10, 20, 30, 40]}
    out["mean_D_seq"][0] = sum(r[0]["D_seq"] for r in runs) / n
    out["solve_mean_last5"] = sum(sum(r[t]["solve"] for t in range(36, 41)) / 5 for r in runs) / n
    return out


def main(d):
    t1 = json.load(open(os.path.join(d, "thermo_s1.json")))
    t2 = json.load(open(os.path.join(d, "thermo_s2.json")))
    files = sorted(glob.glob(os.path.join(d, "run_s*.json")))
    runs = [json.load(open(f)) for f in files]
    runs = [r for r in runs if len(r) == 41]
    res = analyse(t1, t2, runs)
    res["files"] = [os.path.basename(f) for f in files]
    json.dump(res, open(os.path.join(d, "analisi_fase1.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def test():
    """Synthetic runs from a known random walk: R must come out near 1."""
    rng = random.Random(0)
    sigma = 0.004
    pred_sd = {str(T): sigma * math.sqrt(adam_sum_w2(T)) for T in SCHED}
    pred_half = {T: v / 2 for T, v in pred_sd.items()}  # H6: the second behaviour moves half as much
    t1 = {"replay": {"pred_SD": {**{nm: pred_sd for nm in NAMES}, "lin_false": pred_half}, "pred_corr_20_40_lin_seq": 0.87},
          "floor_false": 0.0005,
          "analytic": {"pred_SD_seq": pred_sd}, "floor_seq": 0.001, "floor_tf": 0.0, "alive_p_frac": 0.5,
          "D_seq_0": 0.6, "floor_values": {"seq_at_1e-5": [0.6015, 0.5985]},
          "regions": {"sd": {"seq": {"reasoning": 2.0, "window16_plus_tail": 1.0, "successful": 0.5}}},
          "one_step_checks": [{"lr_mult": 1, "pred_dseq": 0.01 * i, "true_dseq": 0.011 * i} for i in range(1, 5)]}
    t2 = t1
    runs = []
    for _ in range(400):
        b1, m, x, step, rows = 0.9, 0.0, 0.0, 0, []
        lin = 0.0
        rows.append({"step": 0, "D_seq": 0.6, "D_tf": 0.5, "D_false": 0.4, "ess": 400, "lin": [0.0] * 8, "solve": 0.3})
        for t in range(1, 41):
            g = rng.gauss(0, sigma)
            step += 1
            m = b1 * m + (1 - b1) * g
            lin += m / (1 - b1 ** step)
            rows.append({"step": t, "D_seq": 0.6 + lin + rng.gauss(0, 0.001), "D_tf": 0.5 + lin, "ess": 380,
                         "D_false": 0.4 + lin / 2 + rng.gauss(0, 0.0005), "lin": [lin] * 7 + [lin / 2], "solve": 0.3})
        runs.append(rows)
    res = analyse(t1, t2, runs)
    print(json.dumps({k: res[k] for k in ("H1", "H2", "H5", "H6", "baseline_naive_from_step5", "linearisation")}, indent=1))
    assert 0.9 < res["H1"]["R"] < 1.1 and res["H2"]["pass"]
    h6 = res["H6"]
    assert 0.9 < h6["R"] < 1.1 and h6["ranking_scored"] and h6["ranking_correct"] and h6["predicted_more_unstable"] == "disclosure", h6
    sens = res["H1_sensitivity_floor_1e-5"]
    assert abs(sens["floor_1e-5"] - 0.0015) < 1e-9 and sens["R"] < res["H1"]["R"] and sens["same_verdict"], sens
    print("analysis test passed")


if __name__ == "__main__":
    test() if sys.argv[1] == "test" else main(sys.argv[1])
