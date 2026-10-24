"""Ricalcolo indipendente H1 fase 2 (SmolLM2-1.7B) + riferimento Qwen2.5-1.5B.
Legge soltanto: thermo_s1/s2.json e run_s*.json. Non importa codice del progetto.
Uso: py -3.12 ricalcolo.py
"""
import glob
import json
import math
import os
import re
import statistics
import sys

import numpy as np
from scipy.stats import chi2

# percorso reso relativo prima della pubblicazione (24/10/2026); prima era il percorso assoluto del PC
R0 = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SMOL = os.path.join(R0, "fase2_kaggle_B", "smol17", "smol17")
SMOL_S2 = os.path.join(R0, "fase2_kaggle_A", "smol17", "thermo_s2.json")
QW_PRED = os.path.join(R0, "q15_kaggle", "q15", "thermo_s1.json")
QW_RUNS = os.path.join(R0, "q15_kaggle_B", "q15")


def carica_corse(cartella):
    """Ritorna {seme: {passo: (D_seq, ess)}} solo da run_s<numero>.json."""
    out = {}
    for p in sorted(glob.glob(os.path.join(cartella, "run_s*.json"))):
        nome = os.path.basename(p)
        m = re.fullmatch(r"run_s(\d+)\.json", nome)
        if not m:
            continue
        recs = json.load(open(p))
        d = {}
        for r in recs:
            if "D_seq" in r and r.get("D_seq") is not None:
                assert r["step"] not in d, (nome, r["step"])
                d[r["step"]] = (float(r["D_seq"]), float(r["ess"]))
        out[int(m.group(1))] = d
    return out


def sd_camp(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.sum((x - x.mean()) ** 2) / (len(x) - 1)))


def corr(sd, floor):
    return math.sqrt(max(sd * sd - floor * floor, 0.0))


def stat(corse, passo, previsto, floor, ripetibilita=None):
    vals = []
    ess = []
    for s, d in sorted(corse.items()):
        assert passo in d, (s, passo)
        vals.append(d[passo][0])
        ess.append(d[passo][1])
    n = len(vals)
    sd = sd_camp(vals)
    sdc = corr(sd, floor)
    # intervallo chi-quadro al 95% sulla SD con n-1 g.d.l.
    lo = sd * math.sqrt((n - 1) / chi2.ppf(0.975, n - 1))
    hi = sd * math.sqrt((n - 1) / chi2.ppf(0.025, n - 1))
    loc, hic = corr(lo, floor), corr(hi, floor)
    R = sdc / previsto
    ess_med = statistics.median(ess)
    return dict(n=n, sd=sd, sdc=sdc, pred=previsto, R=R, ci=(loc / previsto, hic / previsto),
                ci_ratio=(lo / sd, hi / sd), ess_med=ess_med, ess_min=min(ess),
                ess_max=max(ess))


def verdetto_R(R):
    if not math.isfinite(R):
        return "NON CALCOLABILE"
    if 0.5 <= R <= 2:
        return "PROMOSSO"
    if 1 / 3 <= R <= 3:
        return "INDIZIO"
    return "BOCCIATO"


def verdetto_passo(st):
    if st["ess_med"] < 100:
        return "NON CONCLUSIVO"
    return verdetto_R(st["R"])


def main():
    out = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        out.append(s)

    th = json.load(open(os.path.join(SMOL, "thermo_s1.json")))
    floor = th["floor_seq"]
    pred = th["replay"]["pred_SD"]["lin_seq"]
    corse = carica_corse(SMOL)
    P("SmolLM2: corse", sorted(corse), "n =", len(corse), " floor_seq =", floor)
    res = {}
    for t in (5, 10, 20):
        st = stat(corse, t, pred[str(t)], floor)
        res[t] = st
        P(f"SmolLM2 passo {t}: SD={st['sd']:.5f} SDcorr={st['sdc']:.5f} prev={st['pred']:.5f} "
          f"R={st['R']:.3f} IC95=[{st['ci'][0]:.3f}, {st['ci'][1]:.3f}] (x{st['ci_ratio'][0]:.3f}..x{st['ci_ratio'][1]:.3f}) "
          f"ESS med={st['ess_med']:.1f} min={st['ess_min']:.1f} -> H1 al passo: {verdetto_passo(st)}")
    # regola del passo
    if res[10]["ess_med"] >= 100:
        passo_v = 10
    elif res[5]["ess_med"] >= 100:
        passo_v = 5
    else:
        passo_v = None
    if passo_v is None:
        P("SmolLM2 verdetto H1 (regola passo 10, riserva 5): NON CONCLUSIVO")
    else:
        P(f"SmolLM2 verdetto H1 (regola passo 10, riserva 5): passo {passo_v} -> {verdetto_R(res[passo_v]['R'])}")

    # Qwen
    tq = json.load(open(QW_PRED))
    cq = carica_corse(QW_RUNS)
    P("Qwen2.5-1.5B: corse", sorted(cq), "n =", len(cq), " floor_seq =", tq["floor_seq"])
    resq = {}
    for t in (5, 10, 20):
        st = stat(cq, t, tq["replay"]["pred_SD"]["lin_seq"][str(t)], tq["floor_seq"])
        resq[t] = st
        P(f"Qwen passo {t}: SD={st['sd']:.5f} SDcorr={st['sdc']:.5f} prev={st['pred']:.5f} "
          f"R={st['R']:.3f} IC95=[{st['ci'][0]:.3f}, {st['ci'][1]:.3f}] "
          f"ESS med={st['ess_med']:.1f} min={st['ess_min']:.1f} -> {verdetto_passo(st)}")

    # Ordine
    P("ORDINE")
    for t in (5, 10):
        ps, pq = res[t]["pred"], resq[t]["pred"]
        ms, mq = res[t]["sdc"], resq[t]["sdc"]
        P(f"passo {t}: prevista Smol {ps:.4f} vs Qwen {pq:.4f} -> "
          f"{'Smol meno dispersa' if ps < pq else 'Smol piu dispersa'}; "
          f"misurata(corretta) Smol {ms:.4f} vs Qwen {mq:.4f} -> "
          f"{'Smol meno dispersa' if ms < mq else 'Smol piu dispersa'}; "
          f"ordine uguale: {(ps < pq) == (ms < mq)}")

    # Extra: bande a piu cifre e rapporto fra le SD misurate (test F, solo descrittivo)
    from scipy.stats import f as fdist
    for t in (5, 10, 20):
        st = res[t]
        P(f"  (extra) Smol passo {t}: R={st['R']:.5f} IC=[{st['ci'][0]:.5f}, {st['ci'][1]:.5f}] fattori={st['ci_ratio'][0]:.5f}/{st['ci_ratio'][1]:.5f}; "
          f"IC con fattori arrotondati 0.71/1.70 = [{st['R']*0.71:.3f}, {st['R']*1.70:.3f}]")
    for t in (5, 10):
        rr = res[t]["sd"] / resq[t]["sd"]
        lo = rr / math.sqrt(fdist.ppf(0.975, 11, 11))
        hi = rr * math.sqrt(fdist.ppf(0.975, 11, 11))
        P(f"  (extra) passo {t}: SD misurata Smol/Qwen = {rr:.3f}, IC95 [{lo:.3f}, {hi:.3f}] (F 11,11); previsione Smol/Qwen = {res[t]['pred']/resq[t]['pred']:.3f}")
    # Ripetibilita (seconda previsione, seme 2, K=25)
    s2 = json.load(open(SMOL_S2))
    p2 = s2["replay"]["pred_SD"]["lin_seq"]["10"]
    R2 = res[10]["sdc"] / p2
    P(f"Ripetibilita (thermo_s2) passo 10: prevista={p2:.5f} R con s2={R2:.3f}; "
      f"rapporto s2/s1 previste = {p2 / res[10]['pred']:.3f}  (pavimento s1 usato: s2 non ne ha)")
    for t in (5, 20):
        p2t = s2["replay"]["pred_SD"]["lin_seq"][str(t)]
        P(f"  (extra) passo {t}: prevista s2={p2t:.5f} R con s2={res[t]['sdc'] / p2t:.3f}")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ricalcolo_out.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
