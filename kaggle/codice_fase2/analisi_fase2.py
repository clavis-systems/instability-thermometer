"""Pre-registered analysis of phase 2 (written 19/10/2026, before any real data): the four families together.

Usage: python -I analisi_fase2.py RISULTATI_DIR     (needs <fam>/analisi_fase1.json for fam in q15 q05 olmo1 smol17;
                                                     run analisi_fase1.py on each family folder first)
       python -I analisi_fase2.py test

H4: per family, the same H1 verdict as phase 1 (R in [0.5, 2] = PROMOSSO). Across families: Spearman correlation
between predicted and measured SD of D_seq at T = 40 (descriptive only with 4 families) and the number of families
whose measured SD is ordered as predicted relative to Qwen2.5-1.5B (the phase-1 family).
A family stopped by a gate is reported as such and left out of the ranking.
"""
import json
import os
import sys

FAMILIES = ["q15", "q05", "olmo1", "smol17"]


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):  # average ranks for ties
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def analyse(per_family):
    rows, out = [], {"families": {}}
    for fam in FAMILIES:
        a = per_family.get(fam)
        if a is None:
            out["families"][fam] = {"status": "stopped by a gate or missing"}
            continue
        h1 = a["H1"]
        out["families"][fam] = {"R": h1["R"], "verdict": h1["verdict"], "predicted": h1["predicted_SD_replay"],
                                "measured": h1["measured_SD_corrected"]}
        rows.append((fam, h1["predicted_SD_replay"], h1["measured_SD_corrected"]))
    out["n_families_ranked"] = len(rows)
    if len(rows) >= 3:
        out["spearman_predicted_vs_measured"] = spearman([r[1] for r in rows], [r[2] for r in rows])
    ref = next((r for r in rows if r[0] == "q15"), None)
    if ref:
        agree = [(r[1] > ref[1]) == (r[2] > ref[2]) for r in rows if r[0] != "q15"]
        out["order_vs_q15_agree"] = f"{sum(agree)}/{len(agree)}"
    out["promossi"] = sum(1 for f in out["families"].values() if f.get("verdict") == "PROMOSSO")
    return out


def scomposizione(t1):
    """The three factors of the idea, from the step-0 thermometer of one family (defined 19/10, before data):
    exposure = mean number of terminated failed rollouts per batch; balance = D_seq_0 * (1 - D_seq_0);
    coupling = share of the predicted per-step spread that comes from the failed reasoning region (regions SD)."""
    b = t1.get("batches", [])
    exp_ = sum(x["n_fail_terminated"] for x in b) / len(b) if b else None
    bal = t1["D_seq_0"] * (1 - t1["D_seq_0"])
    reg = t1.get("regions", {}).get("sd", {}).get("seq", {})
    parts = [reg.get(k) for k in ("successful", "reasoning", "window16", "tail")]
    coup = (reg["reasoning"] / sum(x for x in parts if x)) if reg.get("reasoning") and all(x is not None for x in parts) else None
    return {"esposizione": exp_, "bilico": bal, "accoppiamento_quota_ragionamento": coup,
            "sigma_analitica": t1.get("analytic", {}).get("sigma_seq")}


def main(d):
    per, scom = {}, {}
    for fam in FAMILIES:
        p = os.path.join(d, fam, "analisi_fase1.json")
        if os.path.exists(p):
            per[fam] = json.load(open(p))
        t = os.path.join(d, fam, "thermo_s1.json")
        if os.path.exists(t):
            scom[fam] = scomposizione(json.load(open(t)))
    res = analyse(per)
    res["scomposizione"] = scom
    json.dump(res, open(os.path.join(d, "analisi_fase2.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def test():
    fake = {f: {"H1": {"R": r, "verdict": "PROMOSSO", "predicted_SD_replay": p, "measured_SD_corrected": m}}
            for f, r, p, m in (("q15", 1.0, 0.03, 0.03), ("q05", 1.2, 0.05, 0.06), ("olmo1", 0.8, 0.02, 0.016),
                               ("smol17", 1.1, 0.04, 0.044))}
    res = analyse(fake)
    print(json.dumps(res, indent=1))
    assert abs(res["spearman_predicted_vs_measured"] - 1.0) < 1e-9 and res["order_vs_q15_agree"] == "3/3"
    res2 = analyse({k: v for k, v in fake.items() if k != "olmo1"})
    assert res2["families"]["olmo1"]["status"].startswith("stopped") and res2["n_families_ranked"] == 3
    print("phase-2 analysis test passed")


if __name__ == "__main__":
    test() if sys.argv[1] == "test" else main(sys.argv[1])
