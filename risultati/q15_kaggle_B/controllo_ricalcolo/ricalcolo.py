import json, math, statistics as st, hashlib, itertools, sys

BASE = r"C:\Users\emari\Desktop\MIA IA LOCALE\.claude\worktrees\robotica-congela-previsione-legge-1-929f74\strumenti\termometro\risultati"
RUNS = [BASE + r"\q15_kaggle_B\q15\run_s%d.json" % s for s in range(101, 113)]
S1 = json.load(open(BASE + r"\q15_kaggle\q15\thermo_s1.json"))
S2 = json.load(open(BASE + r"\q15_kaggle\q15\thermo_s2.json"))
AN = json.load(open(BASE + r"\q15_kaggle_B\q15\analisi_fase1.json"))
runs = [json.load(open(f)) for f in RUNS]

def out(*a):
    print(*a)

# ---------------- 1. INTEGRITA ----------------
out("=== 1. INTEGRITA ===")
out("n corse:", len(runs), "righe per corsa:", [len(r) for r in runs])
out("passi ok (0..40):", all([x["step"] for x in r] == list(range(41)) for r in runs))

def has_bad(o, path=""):
    bad = []
    if isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            bad.append(path)
    elif isinstance(o, dict):
        for k, v in o.items():
            bad += has_bad(v, path + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            bad += has_bad(v, path + "[%d]" % i)
    return bad

for i, r in enumerate(runs):
    b = has_bad(r)
    if b:
        out("  NaN/inf in run", 101 + i, b[:5])
out("NaN/inf totale:", sum(len(has_bad(r)) for r in runs))
# raw text check for NaN tokens
for f in RUNS:
    t = open(f, encoding="utf8").read()
    if "NaN" in t or "Infinity" in t:
        out("  token NaN/Infinity nel testo:", f)

# cadenze
SEQ_STEPS = [0, 5, 10, 20, 30, 40]
TF_STEPS = [0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40]
for i, r in enumerate(runs):
    s_seq = [x["step"] for x in r if "D_seq" in x]
    s_tf = [x["step"] for x in r if "D_tf" in x]
    s_f = [x["step"] for x in r if "D_false" in x]
    s_ess = [x["step"] for x in r if "ess" in x]
    s_lp = [x["step"] for x in r if "lp_seq" in x]
    s_lin = [x["step"] for x in r if "lin" in x]
    if s_seq != SEQ_STEPS or s_tf != TF_STEPS or s_f != SEQ_STEPS or s_ess != SEQ_STEPS or s_lp != SEQ_STEPS:
        out("  cadenza diversa run", 101 + i, s_seq, s_tf, s_f)
    if s_lin != list(range(1, 41)):
        out("  lin: passi", 101 + i, s_lin[:3], s_lin[-3:], len(s_lin))
    for x in r:
        if "lin" in x and len(x["lin"]) != 8:
            out("  lin len !=8", 101 + i, x["step"])
        if "lp_seq" in x and len(x["lp_seq"]) != 400:
            out("  lp len !=400", 101 + i, x["step"], len(x["lp_seq"]))
out("cadenze controllate (nessun messaggio sopra = tutte coerenti)")

# identiche
dumps = [json.dumps(r, sort_keys=True) for r in runs]
ident_full = [(a + 101, b + 101) for a, b in itertools.combinations(range(12), 2) if dumps[a] == dumps[b]]
out("coppie di corse identiche (file intero):", ident_full)
# versione senza 'seconds' e passo 0
def traj_key(r, field):
    return [tuple(x[field]) if isinstance(x[field], list) else x[field] for x in r if field in x and x["step"] > 0]
for field in ["lin", "D_tf", "D_seq", "ess"]:
    keys = [str(traj_key(r, field)) for r in runs]
    ident = [(a + 101, b + 101) for a, b in itertools.combinations(range(12), 2) if keys[a] == keys[b]]
    out("  coppie identiche in traiettoria '%s' (passi>0):" % field, ident)
# solve/informative traj
keys = [str([(x.get("solve"), x.get("informative")) for x in r]) for r in runs]
out("  coppie identiche in solve/informative:", [(a + 101, b + 101) for a, b in itertools.combinations(range(12), 2) if keys[a] == keys[b]])
# coppie quasi identiche: min distanza tra lin al passo 40
def lin40(r): return r[40]["lin"]
md = []
for a, b in itertools.combinations(range(12), 2):
    d = max(abs(x - y) for x, y in zip(lin40(runs[a]), lin40(runs[b])))
    md.append((d, a + 101, b + 101))
md.sort()
out("  distanza massima (norma max) minima tra lin al passo 40 di due corse:", md[0])

# passo 0 uguali
r0 = [r[0] for r in runs]
for field in ["D_seq", "D_tf", "D_false", "ess"]:
    vals = set(x[field] for x in r0)
    out("  passo 0 %s: valori distinti = %d  (%s)" % (field, len(vals), sorted(vals)[:3]))
out("  passo 0 lp_seq identici fra corse:", all(x["lp_seq"] == r0[0]["lp_seq"] for x in r0))
out("  passo 0 chiavi:", sorted(r0[0].keys()))
pool = json.load(open(BASE + r"\q15_kaggle\q15\pool.json"))
logp0 = [e["logp0"] for e in pool["pool"]]
dlab = [e["d"] for e in pool["pool"]]
flab = [1 if e["false_claim"] else 0 for e in pool["pool"]]
out("  lp_seq passo0 == logp0 del pool? max|diff| =", max(abs(a - b) for a, b in zip(r0[0]["lp_seq"], logp0)))
out("  D_seq_0 dal pool:", sum(dlab) / 400, "D_false_0:", sum(flab) / 400)
out("  step0 D_seq/D_tf/D_false vs previsione s1:", r0[0]["D_seq"], S1["D_seq_0"], r0[0]["D_tf"], S1["D_tf_0"], r0[0]["D_false"], S1["D_false_0"])
out("  dtype nelle righe:", set(x.get("dtype") for r in runs for x in r))

# sha256 previsione
import os
sha = {}
for fn in ["thermo_s1.json", "thermo_s2.json", "gates.json", "gates_pool.json", "pool.json"]:
    sha[fn] = hashlib.sha256(open(BASE + r"\q15_kaggle\q15" + "\\" + fn, "rb").read()).hexdigest()
out("sha256 calcolati:", {k: v[:12] for k, v in sha.items()})
txt = open(BASE + r"\q15_kaggle\q15\previsione_sha256.txt", encoding="utf8").read()
for k, v in sha.items():
    out("  ", k, "corrisponde a previsione_sha256.txt:", v in txt)

# ---------------- 2. RICALCOLO H1 ----------------
out("\n=== 2. H1 ===")
def sd(v): return st.stdev(v)
def at(r, step, field): return [x[field] for x in r if x["step"] == step][0]

D40 = [at(r, 40, "D_seq") for r in runs]
ESS40 = [at(r, 40, "ess") for r in runs]
sd_raw = sd(D40)
floor = S1["floor_seq"]
sd_corr = math.sqrt(max(sd_raw ** 2 - floor ** 2, 0))
pred_lin_seq40 = S1["replay"]["pred_SD"]["lin_seq"]["40"]
R = sd_corr / pred_lin_seq40
ess_med = st.median(ESS40)
out("D_seq(40) per corsa:", [round(x, 4) for x in D40])
out("SD(n-1) =", sd_raw, " pavimento =", floor, " SD corretta =", sd_corr)
out("previsione replay lin_seq[40] =", pred_lin_seq40, " R =", R)
out("ESS(40):", [round(x, 2) for x in ESS40], " mediana =", ess_med)

def verdict(R, ess):
    if ess < 100:
        return "NON CONCLUSIVO"
    if R is None or not math.isfinite(R):
        return "NON CALCOLABILE"
    if 0.5 <= R <= 2:
        return "PROMOSSO"
    if 0.33 <= R <= 3:
        return "INDIZIO"
    return "BOCCIATO"

def verdict_range_only(R):
    if 0.5 <= R <= 2:
        return "PROMOSSO"
    if 0.33 <= R <= 3:
        return "INDIZIO"
    return "BOCCIATO"

out("verdetto (con regola ESS):", verdict(R, ess_med), " | solo per intervallo di R:", verdict_range_only(R))
# IC chi quadro 95% con df=11: chi2 quantili 0.025/0.975 df 11 (valori tabulati): 3.8157, 21.9201
lo = math.sqrt(11 / 21.9200490); hi = math.sqrt(11 / 3.8157425)
out("fattori IC95 chi2:", lo, hi, " IC95 SD misurata:", sd_corr * lo, sd_corr * hi, " IC R:", R * lo, R * hi)

# sensibilita' 1e-5
vals5 = S1["floor_values"]["seq_at_1e-5"]
d0 = S1["D_seq_0"]
floor5 = math.sqrt(sum((v - d0) ** 2 for v in vals5) / len(vals5))
sd_corr5 = math.sqrt(max(sd_raw ** 2 - floor5 ** 2, 0))
R5 = sd_corr5 / pred_lin_seq40
out("floor 1e-5 (RMS da D_seq_0) =", floor5, " rapporto vs 1e-7:", floor5 / floor, " R5 =", R5)
out("   verdetto con ESS:", verdict(R5, ess_med), " | solo R:", verdict_range_only(R5))
# floor 1e-7 ricalcolato dai valori
v7 = S1["floor_values"]["seq"]
out("   floor 1e-7: SD(8 valori) n-1 =", st.stdev(v7), " RMS da D0 (7 pert) =", math.sqrt(sum((v - d0) ** 2 for v in v7[1:]) / 7), " valore salvato", floor)
out("   R al limite 2? margine: R=%.5f, soglia 2, SD che darebbe R=2: %.6f, misurata %.6f" % (R, 2 * pred_lin_seq40, sd_corr))

# ---------------- 2b. lineari ----------------
out("\n=== 2b. FUNZIONALI LINEARI (passo 40) ===")
NAMES = ["lin_seq", "lin_tf", "rand1", "rand2", "rand3", "rand4", "rand5", "lin_false"]
lin_R = {}
for j, nm in enumerate(NAMES):
    vals = [at(r, 40, "lin")[j] for r in runs]
    s = sd(vals)
    p = S1["replay"]["pred_SD"][nm]["40"]
    lin_R[nm] = s / p
    out("  %-9s SD misurata=%.5f  prev=%.5f  R=%.4f  IC95(R)=[%.2f, %.2f]  media misurata=%.4f" % (nm, s, p, s / p, s / p * lo, s / p * hi, st.mean(vals)))
out("  media misurata lin_seq 40 =", st.mean([at(r, 40, 'lin')[0] for r in runs]), " prevista media:", S1["replay"]["pred_mean_lin_seq_40"])
seven = [lin_R[n] for n in NAMES[:7]]
out("  7 funzionali (senza lin_false): min=%.3f max=%.3f; tutti in [0.5,2]? %s" % (min(seven), max(seven), all(0.5 <= x <= 2 for x in seven)))
out("  tutti e 8: min=%.3f max=%.3f; tutti in [0.5,2]? %s" % (min(lin_R.values()), max(lin_R.values()), all(0.5 <= x <= 2 for x in lin_R.values())))
out("  media geometrica di R (7):", math.exp(sum(math.log(x) for x in seven) / 7), " mediana:", st.median(seven), " n>1:", sum(x > 1 for x in seven))
# anche a passi intermedi
out("  R delle lineari ai passi 10,20,30,40:")
for T in (10, 20, 30, 40):
    row = []
    for j, nm in enumerate(NAMES):
        vals = [at(r, T, "lin")[j] for r in runs]
        row.append(sd(vals) / S1["replay"]["pred_SD"][nm][str(T)])
    out("   T=%d" % T, [round(x, 2) for x in row])
# R con seed 2
out("  R lin_seq con previsione seed2 =", sd(D40) and sd([at(r, 40, 'lin')[0] for r in runs]) / S2["replay"]["pred_SD"]["lin_seq"]["40"])

# ---------------- 2c. H2 ----------------
out("\n=== 2c. H2 ===")
def slope(xs, ys):
    lx = [math.log(x) for x in xs]; ly = [math.log(y) for y in ys]
    mx = sum(lx) / len(lx); my = sum(ly) / len(ly)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)
Ts = [10, 20, 30, 40]
sd_seq_T = []
for T in Ts:
    v = [at(r, T, "D_seq") for r in runs]
    s = sd(v)
    sd_seq_T.append(math.sqrt(max(s * s - floor ** 2, 0)))
out("  SD corretta D_seq a T=10,20,30,40:", [round(x, 4) for x in sd_seq_T])
sl_m = slope(Ts, sd_seq_T)
pred_rep = [S1["replay"]["pred_SD"]["lin_seq"][str(T)] for T in Ts]
pred_an = [S1["analytic"]["pred_SD_seq"][str(T)] for T in Ts]
sl_rep = slope(Ts, pred_rep); sl_an = slope(Ts, pred_an)
out("  pendenza misurata D_seq = %.4f ; replay lin_seq = %.4f (diff %.3f) ; analitica = %.4f (diff %.3f)" % (sl_m, sl_rep, sl_m - sl_rep, sl_an, sl_m - sl_an))
out("  entro +-0.25 dal replay?", abs(sl_m - sl_rep) <= 0.25, " dall'analitica?", abs(sl_m - sl_an) <= 0.25)
sd_lin_T = [sd([at(r, T, "lin")[0] for r in runs]) for T in Ts]
out("  pendenza misurata lin_seq = %.4f (SD:%s)" % (slope(Ts, sd_lin_T), [round(x, 4) for x in sd_lin_T]))
# sd D_seq a 5 (non usato)
out("  SD D_seq a T=5:", sd([at(r, 5, 'D_seq') for r in runs]), " ESS mediana per passo:", {T: round(st.median([at(r, T, 'ess') for r in runs]), 1) for T in SEQ_STEPS})
# pendenza solo su T=10,20,30 se il crollo e' a 40
out("  pendenza T=10,20,30 D_seq:", slope([10, 20, 30], sd_seq_T[:3]))
out("  pendenza T=5,10,20,30,40 D_seq:", slope([5, 10, 20, 30, 40], [sd([at(r, T, 'D_seq') for r in runs]) for T in (5, 10, 20, 30, 40)]))

# ---------------- 2d. H6 ----------------
out("\n=== 2d. H6 ===")
F40 = [at(r, 40, "D_false") for r in runs]
sdF = sd(F40); floorF = S1["floor_false"]
sdFc = math.sqrt(max(sdF ** 2 - floorF ** 2, 0))
predF = S1["replay"]["pred_SD"]["lin_false"]["40"]
RF = sdFc / predF
out("  SD D_false(40) =", sdF, " corretta =", sdFc, " prev =", predF, " R_false =", RF, " verdetto:", verdict(RF, ess_med))
ratio_pred = predF / pred_lin_seq40
ratio_meas = sdFc / sd_corr
out("  rapporto previsto SD(false)/SD(amm) =", ratio_pred, " -> ordine giudicato?", not (0.9 <= ratio_pred <= 1.1), " | rapporto misurato =", ratio_meas)
def corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
out("  corr misurata D_seq(40) vs D_false(40) =", corr(D40, F40), " prevista (replay) =", S1["replay"]["pred_corr_seq_false_40"], " corr etichette =", S1["label_corr_d_false"])
out("  corr label d vs false_claim ricalcolata dal pool:", corr(dlab, flab))
out("  D_seq+D_false per corsa a 40 (somma):", [round(a + b, 3) for a, b in zip(D40, F40)])

# ---------------- 2e. H5 ----------------
out("\n=== 2e. H5 ===")
D20 = [at(r, 20, "D_seq") for r in runs]
out("  corr D_seq 20 vs 40 =", corr(D20, D40), " prevista:", S1["replay"]["pred_corr_20_40_lin_seq"])
L20 = [at(r, 20, "lin")[0] for r in runs]; L40 = [at(r, 40, "lin")[0] for r in runs]
out("  corr lin_seq 20 vs 40 =", corr(L20, L40))

# ---------------- 2f. altro ----------------
out("\n=== 2f. SECONDARIE ===")
T40 = [at(r, 40, "D_tf") for r in runs]
sT = sd(T40); fT = S1["floor_tf"]
out("  D_tf(40): SD corretta =", math.sqrt(max(sT ** 2 - fT ** 2, 0)), " prev =", S1["replay"]["pred_SD"]["lin_tf"]["40"], " R =", math.sqrt(max(sT ** 2 - fT ** 2, 0)) / S1["replay"]["pred_SD"]["lin_tf"]["40"])
out("  R analitica =", sd_corr / S1["analytic"]["pred_SD_seq"]["40"], " R seed2 =", sd_corr / S2["replay"]["pred_SD"]["lin_seq"]["40"], " rapporto prev s2/s1 =", S2["replay"]["pred_SD"]["lin_seq"]["40"] / pred_lin_seq40)
# naive
sd5 = sd([at(r, 5, "D_seq") for r in runs])
out("  SD misurata D_seq a 5 =", sd5)
# linearizzazione: variazione vera di D_seq vs lin_seq al 40
dtrue = [d - 0.485 for d in D40]
out("  linearizzazione a 40: corr(D_seq-D0, lin_seq) =", corr(dtrue, L40))
for T in (10, 20, 30, 40):
    dt = [at(r, T, "D_seq") - 0.485 for r in runs]; ll = [at(r, T, "lin")[0] for r in runs]
    ml = st.mean(ll); md_ = st.mean(dt)
    b = sum((x - ml) * (y - md_) for x, y in zip(ll, dt)) / sum((x - ml) ** 2 for x in ll)
    out("    T=%d corr=%.3f pendenza=%.3f  medianaESS=%.1f" % (T, corr(dt, ll), b, st.median([at(r, T, 'ess') for r in runs])))
out("  corr(D_false-D0, lin_false) a 40:", corr([f - 0.445 for f in F40], [at(r, 40, 'lin')[7] for r in runs]))
out("  media D_seq per passo:", {T: round(st.mean([at(r, T, 'D_seq') for r in runs]), 4) for T in SEQ_STEPS})
out("  SD D_seq per passo (grezza):", {T: round(sd([at(r, T, 'D_seq') for r in runs]), 4) for T in SEQ_STEPS})

# ---------------- confronto con analisi_fase1.json ----------------
out("\n=== 3. CONFRONTO con analisi_fase1.json ===")
def cmp(name, mine, theirs, tol=1e-6):
    flag = "OK " if abs(mine - theirs) <= tol * max(1, abs(theirs)) else "DIFF"
    out("  %s %-34s mio=%.10g  json=%.10g" % (flag, name, mine, theirs))
cmp("SD D_seq 40 grezza", sd_raw, AN["H1"]["measured_SD_D_seq_40"])
cmp("SD corretta", sd_corr, AN["H1"]["measured_SD_corrected"])
cmp("pred replay", pred_lin_seq40, AN["H1"]["predicted_SD_replay"])
cmp("R", R, AN["H1"]["R"])
cmp("ESS mediana 40", ess_med, AN["H1"]["median_ESS_40"])
out("  verdetto mio:", verdict(R, ess_med), " json:", AN["H1"]["verdict"])
cmp("floor 1e-5", floor5, AN["H1_sensitivity_floor_1e-5"]["floor_1e-5"])
cmp("R 1e-5", R5, AN["H1_sensitivity_floor_1e-5"]["R"])
out("  verdetto 1e-5 mio (con ESS):", verdict(R5, ess_med), " (solo R):", verdict_range_only(R5), " json:", AN["H1_sensitivity_floor_1e-5"]["verdict"], "same_verdict json:", AN["H1_sensitivity_floor_1e-5"]["same_verdict"])
for nm in NAMES:
    cmp("R " + nm, lin_R[nm], AN["secondary"]["linear_functionals_40"][nm]["R"])
cmp("H2 pendenza misurata", sl_m, AN["H2"]["slope_measured_D_seq"])
cmp("H2 pendenza prevista (json)", sl_an, AN["H2"]["slope_predicted"])
cmp("H2 pendenza replay lin_seq prev", sl_rep, AN["H2"]["slope_predicted_lin_seq"])
cmp("H2 pendenza misurata lin_seq", slope(Ts, sd_lin_T), AN["H2"]["slope_measured_lin_seq"])
cmp("H6 R", RF, AN["H6"]["R"])
cmp("H6 SD corretta", sdFc, AN["H6"]["measured_SD_corrected"])
cmp("H6 corr misurata", corr(D40, F40), AN["H6"]["measured_corr_seq_false_40"])
cmp("H6 rapporto previsto", ratio_pred, AN["H6"]["predicted_ratio_false_over_disclosure"])
cmp("H6 rapporto misurato", ratio_meas, AN["H6"]["measured_ratio_false_over_disclosure"])
cmp("H5 corr D_seq 20-40", corr(D20, D40), AN["H5"]["measured_corr_D_seq_20_40"])
cmp("H5 corr lin_seq 20-40", corr(L20, L40), AN["H5"]["measured_corr_lin_seq_20_40"])
cmp("D_tf R", math.sqrt(max(sT ** 2 - fT ** 2, 0)) / S1["replay"]["pred_SD"]["lin_tf"]["40"], AN["secondary"]["D_tf_40"]["R"])
cmp("analytic R", sd_corr / S1["analytic"]["pred_SD_seq"]["40"], AN["secondary"]["analytic_R"])
cmp("seed2 R", sd_corr / S2["replay"]["pred_SD"]["lin_seq"]["40"], AN["secondary"]["retest_R_with_seed2"])
cmp("naive R", sd5 and (sd_corr / (sd5 * (S1['replay']['pred_SD']['lin_seq']['40'] / S1['replay']['pred_SD']['lin_seq']['5']))), AN["baseline_naive_from_step5"]["R"], 1e-3)
cmp("linearisation corr", corr(dtrue, L40), AN["linearisation"]["corr_true_change_vs_linear"], 1e-3)
for i, e in enumerate(sorted(ESS40)):
    cmp("ess40 ordinata %d" % i, e, AN["ess_40"][i])
