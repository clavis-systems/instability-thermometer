import json, math, statistics as st

BASE = r"C:\Users\emari\Desktop\MIA IA LOCALE\.claude\worktrees\robotica-congela-previsione-legge-1-929f74\strumenti\termometro\risultati"
runs = [json.load(open(BASE + r"\q15_kaggle_B\q15\run_s%d.json" % s)) for s in range(101, 113)]
S1 = json.load(open(BASE + r"\q15_kaggle\q15\thermo_s1.json"))
pool = json.load(open(BASE + r"\q15_kaggle\q15\pool.json"))["pool"]
logp0 = [e["logp0"] for e in pool]
d = [e["d"] for e in pool]
f = [1 if e["false_claim"] else 0 for e in pool]
STEPS = [0, 5, 10, 20, 30, 40]

def at(r, s): return [x for x in r if x["step"] == s][0]

def recon(lp):
    z = [a - b for a, b in zip(lp, logp0)]
    m = max(z)
    w = [math.exp(x - m) for x in z]
    sw = sum(w)
    D = sum(wi * di for wi, di in zip(w, d)) / sw
    Fv = sum(wi * fi for wi, fi in zip(w, f)) / sw
    ess = sw * sw / sum(x * x for x in w)
    ws = sorted(w, reverse=True)
    top1 = ws[0] / sw
    top5 = sum(ws[:5]) / sw
    top10 = sum(ws[:10]) / sw
    # peso portato dagli esempi con d=1
    wd1 = sum(wi for wi, di in zip(w, d) if di == 1) / sw
    return D, Fv, ess, top1, top5, top10, wd1

mx = {"D": 0, "F": 0, "ess": 0, "essrel": 0}
rows = {s: [] for s in STEPS}
for i, r in enumerate(runs):
    for s in STEPS:
        x = at(r, s)
        D, Fv, ess, t1, t5, t10, wd1 = recon(x["lp_seq"])
        mx["D"] = max(mx["D"], abs(D - x["D_seq"]))
        mx["F"] = max(mx["F"], abs(Fv - x["D_false"]))
        mx["ess"] = max(mx["ess"], abs(ess - x["ess"]))
        mx["essrel"] = max(mx["essrel"], abs(ess - x["ess"]) / x["ess"])
        rows[s].append((t1, t5, t10, wd1, ess))
print("max |D_seq ricostruito - salvato| =", mx["D"])
print("max |D_false ricostruito - salvato| =", mx["F"])
print("max |ESS ricostruita - salvata| =", mx["ess"], " (relativo:", mx["essrel"], ")")
print()
print("Concentrazione del peso (media sulle 12 corse):")
print("passo | ESS media | ESS mediana | peso 1 esempio | peso top-5 (media) | min..max top-5 | peso top-10")
for s in STEPS:
    t1 = [x[0] for x in rows[s]]; t5 = [x[1] for x in rows[s]]; t10 = [x[2] for x in rows[s]]; es = [x[4] for x in rows[s]]
    print("%3d   | %7.1f   | %7.1f   | %.3f | %.3f | %.3f..%.3f | %.3f" % (s, st.mean(es), st.median(es), st.mean(t1), st.mean(t5), min(t5), max(t5), st.mean(t10)))
print("peso medio portato dagli esempi con d=1 (ammissione) al passo 40, per corsa:", [round(x[3], 3) for x in rows[40]])
print("D_seq al passo 40 per corsa vs peso su d=1:", [(round(at(r, 40)['D_seq'], 3), round(rows[40][i][3], 3)) for i, r in enumerate(runs)][:12])

# quanti esempi hanno peso 0 effettivo? numero di esempi che portano il 90% del peso al passo 40
def n_for_frac(lp, frac):
    z = [a - b for a, b in zip(lp, logp0)]
    m = max(z); w = sorted([math.exp(x - m) for x in z], reverse=True); sw = sum(w)
    c = 0
    for k, x in enumerate(w, 1):
        c += x
        if c / sw >= frac:
            return k
print("esempi necessari per il 90% del peso al passo 40 (per corsa):", [n_for_frac(at(r, 40)['lp_seq'], 0.9) for r in runs])
print("esempi necessari per il 50% del peso al passo 40 (per corsa):", [n_for_frac(at(r, 40)['lp_seq'], 0.5) for r in runs])
# log-peso: spostamento tipico
z40 = [[a - b for a, b in zip(at(r, 40)['lp_seq'], logp0)] for r in runs]
print("spread di log(w) al passo 40 (SD sugli esempi, media sulle corse):", st.mean([st.pstdev(z) for z in z40]))
print("max log w - min log w al passo 40 (media):", st.mean([max(z) - min(z) for z in z40]))

# naive: pesi di Adam, riscritto da me
def sumw2(T, b1=0.9):
    tot = 0.0
    for k in range(1, T + 1):
        c = sum((1 - b1) * b1 ** (t - k) / (1 - b1 ** t) for t in range(k, T + 1))
        tot += c * c
    return tot
sd5 = st.stdev([at(r, 5)["D_seq"] for r in runs])
fl = S1["floor_seq"]
s5c = math.sqrt(max(sd5 ** 2 - fl ** 2, 0))
naive = s5c * math.sqrt(sumw2(40) / sumw2(5))
sd40 = st.stdev([at(r, 40)["D_seq"] for r in runs])
print("\nnaive: SD5 corr=%.5f, predizione=%.5f, R=%.3f" % (s5c, naive, sd40 / naive))

# R per D_seq a ogni T contro replay lin_seq e ESS mediana
print("\nR(T) di D_seq (pavimento sottratto) contro replay lin_seq:")
for T in (5, 10, 20, 30, 40):
    v = [at(r, T)["D_seq"] for r in runs]
    s = st.stdev(v); sc = math.sqrt(max(s * s - fl * fl, 0))
    p = S1["replay"]["pred_SD"]["lin_seq"][str(T)]
    print("  T=%2d SD=%.4f  prev=%.4f  R=%.2f  ESS mediana=%.1f  ESS min=%.1f" % (T, sc, p, sc / p, st.median([at(r, T)["ess"] for r in runs]), min(at(r, T)["ess"] for r in runs)))

# R(T) senza le 3 corse con ESS<10? descrittivo: usando solo le corse con ESS>=100 al passo 20
print("\nDescrittivo: R a T=10 (ESS mediana 148) e a T=40 con solo corse con ESS>=20:")
sel = [r for r in runs if at(r, 40)["ess"] >= 20]
v = [at(r, 40)["D_seq"] for r in sel]
print("  n corse con ESS40>=20:", len(sel), " SD=", st.stdev(v), " R=", st.stdev(v) / S1["replay"]["pred_SD"]["lin_seq"]["40"])
