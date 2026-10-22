"""Phase 2, short version (PREMESSA_FASE2_MODIFICA_BOZZA.md, 21/10/2026): written after the phase-1 results and before
any phase-2 data. Per family, the verdict is H1 of phase 1 at T = 10; if the median ESS at 10 is below 100 and at 5 it
is at least 100, at T = 5; otherwise NON CONCLUSIVO. Runs last 20 steps (the frozen `termometro.py run ... <seed> 20`);
step 20 is descriptive. The frozen helpers of analisi_fase1.py and analisi_fase2.py are imported, not copied.

usage: python -I fase2_corta.py analyze THERMO_DIR RUNS_DIR OUT_JSON   (one family: thermo_s1.json, run_s*.json)
       python -I fase2_corta.py families RESULTS_DIR                    (RESULTS_DIR/<fam>/analisi_fase2_corta.json
                                                                         for q15 q05 olmo1 smol17)
       python -I fase2_corta.py test
"""
import glob
import json
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analisi_fase1 import NAMES, SCHED, adam_sum_w2, corr, key, sd, sd_ci, verdict  # noqa: E402  (frozen)
import analisi_fase2  # noqa: E402  (frozen)

T_RUN = 20
STEPS = [5, 10, 20]
NEW = ["q05", "olmo1", "smol17"]  # the blind families; q15 (phase 1) is the reference, outside the verdict
ESS_MIN = 100


def at(runs, T, field, idx=None):
    vals = []
    for r in runs:
        assert r[T]["step"] == T
        vals.append(r[T][field] if idx is None else r[T][field][idx])
    return vals


def h1_at(t1, runs, T):
    """H1 of phase 1 at step T, with the true median of the ESS (statistics.median, as in phase 1b)."""
    floor = t1.get("floor_seq", 0.0)
    s = sd(at(runs, T, "D_seq"))
    sc = math.sqrt(max(s ** 2 - floor ** 2, 0.0))
    p = key(t1["replay"]["pred_SD"]["lin_seq"], T)
    R = sc / p if p > 0 else None
    lo, hi = sd_ci(sc, len(runs))
    ess = at(runs, T, "ess")
    return {"T": T, "measured_SD_D_seq": s, "floor_seq": floor, "measured_SD_corrected": sc, "measured_SD_CI95": [lo, hi],
            "predicted_SD_replay": p, "R": R, "R_CI95": [lo / p, hi / p] if p > 0 else None,
            "median_ESS": statistics.median(ess), "min_ESS": min(ess), "verdict_R_only": verdict(R)}


def test_step(per_step):
    """The rule of the amendment: 10 if its median ESS is at least 100, else 5 if its median ESS is, else none."""
    if per_step[10]["median_ESS"] >= ESS_MIN:
        return 10
    if per_step[5]["median_ESS"] >= ESS_MIN:
        return 5
    return None


def h6_at(t1, runs, T, h1, ess_ok):
    """H6 of phase 1 (second behaviour, same weights, parity zone 0.9-1.1) at step T."""
    if not (all("D_false" in r[T] for r in runs) and "lin_false" in t1["replay"]["pred_SD"] and "floor_false" in t1):
        return None
    xf = at(runs, T, "D_false")
    ff = t1["floor_false"]
    sfc = math.sqrt(max(sd(xf) ** 2 - ff ** 2, 0.0))
    pf = key(t1["replay"]["pred_SD"]["lin_false"], T)
    p, s_c = h1["predicted_SD_replay"], h1["measured_SD_corrected"]
    Rf = sfc / pf if pf > 0 else None
    vf = verdict(Rf)
    if not ess_ok and vf != "NON CALCOLABILE":
        vf = "NON CONCLUSIVO"
    ratio_p = pf / p if p > 0 else None
    tie = ratio_p is None or 0.9 <= ratio_p <= 1.1
    pred_more = None if tie else ("false_claim" if pf > p else "disclosure")
    meas_more = "false_claim" if sfc > s_c else "disclosure"
    return {"T": T, "measured_SD_corrected": sfc, "predicted_SD_replay": pf, "R": Rf, "verdict": vf,
            "predicted_ratio_false_over_disclosure": ratio_p, "ranking_scored": not tie,
            "measured_ratio_false_over_disclosure": sfc / s_c if s_c > 0 else None,
            "ranking_correct": None if tie else pred_more == meas_more,
            "measured_corr_seq_false": corr(at(runs, T, "D_seq"), xf)}


def analyse_family(t1, runs):
    per_step = {T: h1_at(t1, runs, T) for T in STEPS if all(len(r) > T for r in runs)}
    T = test_step(per_step)
    h1 = dict(per_step[T if T else 10])
    if T is None:
        h1["verdict"] = "NON CONCLUSIVO"  # the importance weights have collapsed already at step 5
    else:
        h1["verdict"] = h1["verdict_R_only"]
    h1["test_step"] = T
    out = {"n_runs": len(runs), "H1": h1, "per_step": per_step}
    Tm = T if T else 10
    # without a verdict step, H6, the sensitivity and the linear functionals are given at step 10, descriptive only
    out["secondary_step_role"] = "verdict step" if T else "step 10, descriptive only (no step with median ESS >= 100)"
    big = t1.get("floor_values", {}).get("seq_at_1e-5", [])
    if big:
        fb = math.sqrt(sum((b - t1["D_seq_0"]) ** 2 for b in big) / len(big))
        sb = math.sqrt(max(h1["measured_SD_D_seq"] ** 2 - fb ** 2, 0.0))
        Rb = sb / h1["predicted_SD_replay"] if h1["predicted_SD_replay"] > 0 else None
        out["H1_sensitivity_floor_1e-5"] = {"T": Tm, "floor_1e-5": fb, "R": Rb, "verdict_R_only": verdict(Rb),
                                            "same_verdict_R_only": verdict(Rb) == h1["verdict_R_only"]}
    out["H6"] = h6_at(t1, runs, Tm, h1, T is not None)
    pred = t1["replay"]["pred_SD"]
    out["linear_functionals"] = {"T": Tm, **{nm: {"measured_SD": sd(at(runs, Tm, "lin", j)), "predicted": key(pred[nm], Tm),
                                                  "R": sd(at(runs, Tm, "lin", j)) / key(pred[nm], Tm) if key(pred[nm], Tm) > 0 else None}
                                             for j, nm in enumerate(NAMES) if nm in pred}}
    out["mean_D_seq"] = {T_: sum(at(runs, T_, "D_seq")) / len(runs) for T_ in per_step}
    out["mean_D_seq"][0] = sum(r[0]["D_seq"] for r in runs) / len(runs)
    out["solve_mean_steps_16_20"] = sum(sum(r[t]["solve"] for t in range(16, 21)) / 5 for r in runs) / len(runs)
    return out


def load_runs(d, rows):
    files = sorted(glob.glob(os.path.join(d, "run_s*.json")))
    runs = [json.load(open(f)) for f in files]
    return [r for r in runs if len(r) >= rows], [os.path.basename(f) for f in files]


def analyze(thermo_dir, runs_dir, out_json):
    t1 = json.load(open(os.path.join(thermo_dir, "thermo_s1.json")))
    runs, files = load_runs(runs_dir, T_RUN + 1)
    res = analyse_family(t1, runs)
    res["files"] = files
    res["scomposizione"] = analisi_fase2.scomposizione(t1)
    json.dump(res, open(out_json, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("n_runs", "H1")}, indent=1))


def compare(per_family):
    """Across families, at a common step: 10 if every new family with a verdict uses 10, otherwise 5. Only the
    families (q15 included, as reference) whose median ESS at the common step is at least 100 are ranked."""
    used = [per_family[f]["H1"]["test_step"] for f in NEW if f in per_family and per_family[f]["H1"]["test_step"]]
    Tc = 10 if used and all(t == 10 for t in used) else 5
    ranked, skipped = {}, []
    for fam, a in per_family.items():
        h = a["per_step"].get(Tc) or a["per_step"].get(str(Tc))
        if h and h["median_ESS"] >= ESS_MIN:
            v = a["H1"]["verdict"] if fam in NEW else "RIFERIMENTO (non cieca)"
            ranked[fam] = {"H1": {"R": h["R"], "verdict": v, "predicted_SD_replay": h["predicted_SD_replay"],
                                  "measured_SD_corrected": h["measured_SD_corrected"]}}
        else:
            skipped.append(fam)
    res = analisi_fase2.analyse(ranked)
    res.pop("promossi", None)  # it would count q15 too: the count is the one below, on the new families only
    for fam in skipped:  # the frozen analyse would call them "stopped by a gate or missing"
        res["families"][fam] = {"status": f"not ranked: median ESS below {ESS_MIN} at step {Tc}"}
    res["common_step"] = Tc
    res["not_ranked_ESS_below_100"] = skipped
    res["verdicts_new_families"] = {f: {"verdict": per_family[f]["H1"]["verdict"], "test_step": per_family[f]["H1"]["test_step"],
                                        "R": per_family[f]["H1"]["R"]} for f in NEW if f in per_family}
    res["promosse_tra_le_3_nuove"] = f"{sum(1 for v in res['verdicts_new_families'].values() if v['verdict'] == 'PROMOSSO')}/" \
                                     f"{len(NEW)}"
    return res


def families(d):
    per = {}
    for fam in ["q15"] + NEW:
        p = os.path.join(d, fam, "analisi_fase2_corta.json")
        if os.path.exists(p):
            a = json.load(open(p))
            a["per_step"] = {int(k): v for k, v in a["per_step"].items()}
            per[fam] = a
    res = compare(per)
    res["scomposizione"] = {f: a.get("scomposizione") for f, a in per.items()}
    json.dump(res, open(os.path.join(d, "analisi_fase2_corta.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def synthetic(rng, sigma, n_runs, rows, ess_of_step):
    """Runs from a known random walk (as the frozen test of analisi_fase1.py), with a chosen ESS profile."""
    runs = []
    for _ in range(n_runs):
        m, lin, out = 0.0, 0.0, [{"step": 0, "D_seq": 0.6, "D_false": 0.4, "ess": 400, "lin": [0.0] * 8, "solve": 0.3}]
        for t in range(1, rows):
            m = 0.9 * m + 0.1 * rng.gauss(0, sigma)
            lin += m / (1 - 0.9 ** t)
            out.append({"step": t, "D_seq": 0.6 + lin + rng.gauss(0, 0.001), "D_false": 0.4 + lin / 2 + rng.gauss(0, 0.0005),
                        "ess": ess_of_step(t), "lin": [lin] * 7 + [lin / 2], "solve": 0.3})
        runs.append(out)
    return runs


def test():
    rng = random.Random(0)
    sigma = 0.004
    pred = {str(T): sigma * math.sqrt(adam_sum_w2(T)) for T in SCHED}
    t1 = {"replay": {"pred_SD": {**{nm: pred for nm in NAMES}, "lin_false": {k: v / 2 for k, v in pred.items()}}},
          "floor_seq": 0.001, "floor_false": 0.0005, "D_seq_0": 0.6, "floor_values": {"seq_at_1e-5": [0.6015, 0.5985]},
          "batches": [{"n_fail_terminated": 40}], "regions": {"sd": {"seq": {"successful": 1, "reasoning": 2, "window16": 1, "tail": 1}}}}
    # 1. healthy measure: verdict at 10, R near 1 (400 runs to make R precise)
    a = analyse_family(t1, synthetic(rng, sigma, 400, 21, lambda t: 380))
    assert a["H1"]["test_step"] == 10 and 0.9 < a["H1"]["R"] < 1.1 and a["H1"]["verdict"] == "PROMOSSO", a["H1"]
    assert a["H6"]["ranking_scored"] and a["H6"]["ranking_correct"] and 0.9 < a["H6"]["R"] < 1.1, a["H6"]
    # 2. collapsed at 10 but not at 5: verdict at 5
    b = analyse_family(t1, synthetic(rng, sigma, 400, 21, lambda t: 300 if t <= 5 else 60))
    assert b["H1"]["test_step"] == 5 and b["H1"]["T"] == 5 and b["H1"]["verdict"] == "PROMOSSO", b["H1"]
    # 3. collapsed already at 5: NON CONCLUSIVO, and H6 too
    c = analyse_family(t1, synthetic(rng, sigma, 400, 21, lambda t: 50))
    assert c["H1"]["test_step"] is None and c["H1"]["verdict"] == "NON CONCLUSIVO" and c["H6"]["verdict"] == "NON CONCLUSIVO"
    assert c["secondary_step_role"].startswith("step 10, descriptive") and a["secondary_step_role"] == "verdict step"
    # 4. a spread twice the prediction: R near 2; and the true median of the ESS
    d = analyse_family(t1, [[{**row, "D_seq": 0.6 + 2 * (row["D_seq"] - 0.6)} for row in r]
                            for r in synthetic(rng, sigma, 400, 21, lambda t: 380)])
    assert 1.8 < d["H1"]["R"] < 2.2, d["H1"]
    e = analyse_family(t1, synthetic(rng, sigma, 4, 21, lambda t: 380))
    assert e["per_step"][10]["median_ESS"] == 380
    # 5. comparison: q15 (41 rows) is ranked but not counted; one new family at 5 moves the common step to 5
    q15 = analyse_family(t1, synthetic(rng, sigma, 12, 41, lambda t: 380))
    res = compare({"q15": q15, "q05": a, "olmo1": b, "smol17": c})
    assert res["common_step"] == 5 and res["promosse_tra_le_3_nuove"] == "2/3", res
    assert "smol17" in res["not_ranked_ESS_below_100"] and res["n_families_ranked"] == 3, res
    assert res["families"]["q15"]["verdict"] == "RIFERIMENTO (non cieca)", res
    res10 = compare({"q15": q15, "q05": a, "olmo1": a})
    assert res10["common_step"] == 10 and res10["promosse_tra_le_3_nuove"] == "2/3", res10
    print(json.dumps({"caso_1": a["H1"]["verdict"], "caso_2": [b["H1"]["test_step"], b["H1"]["verdict"]],
                      "caso_3": c["H1"]["verdict"], "caso_4_R": round(d["H1"]["R"], 2), "confronto": res["common_step"]}))
    print("fase2_corta test passed")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "test":
        test()
    elif cmd == "analyze":
        analyze(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "families":
        families(sys.argv[2])
