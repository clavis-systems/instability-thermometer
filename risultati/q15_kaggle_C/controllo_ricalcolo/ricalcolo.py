"""Ricalcolo indipendente del verdetto della fase 1b (termometro dell'instabilita').

Scritto da zero, solo dalle regole di PREMESSA_FASE1B.md e PREMESSA.md e dai file grezzi.
Non importa e non legge nessuno script di analisi del progetto.
Uso:  python -X utf8 ricalcolo.py
Scrive:  miei_numeri.json (nella stessa cartella dello script)
I dati grezzi si leggono (sola lettura) da Q15, modificabile con la variabile d'ambiente Q15_DIR.
"""
import json
import math
import os
import statistics as st

import numpy as np

QUI = os.path.dirname(os.path.abspath(__file__))
Q15 = os.environ.get(
    "Q15_DIR",
    os.path.join(QUI, "..", "q15"),
)
SEMI = [201, 202, 203, 204, 205, 206]
PASSI_MISURA = [10, 20, 40, 60, 80, 100, 120, 150]
T_PROVA = [80, 150]
FINESTRA = 20
SOGLIA_ESS = 100.0


# ----------------------------------------------------------------- utilita
def carica(nome):
    with open(os.path.join(Q15, nome), encoding="utf-8") as f:
        return json.load(f)


def sd(x, ddof=1):
    x = list(x)
    if len(x) - ddof <= 0:
        return float("nan")
    m = sum(x) / len(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - ddof))


def sottrai_quadratura(sd_oss, rumore):
    v = sd_oss ** 2 - rumore ** 2
    return math.sqrt(v) if v > 0 else 0.0


def verdetto_R(R):
    if R is None or not math.isfinite(R):
        return "NON CALCOLABILE"
    if 0.5 <= R <= 2.0:
        return "PROMOSSO"
    if 0.33 <= R <= 3.0:
        return "INDIZIO"
    return "BOCCIATO"


def pendenza_log(T, S):
    x = np.log(np.array(T, float))
    y = np.log(np.array(S, float))
    A = np.vstack([x, np.ones_like(x)]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][0])


def gamma_inc_reg(a, x):
    """P(a, x) regolarizzata (serie), per la CDF del chi quadro."""
    if x <= 0:
        return 0.0
    n = 0
    t = 1.0 / math.gamma(a + 1)
    somma = 0.0
    while True:
        somma += t
        n += 1
        t = t * x / (a + n)
        if t < 1e-18 * somma or n > 5000:
            break
    return somma * math.exp(a * math.log(x) - x)


def chi2_cdf(x, k):
    return gamma_inc_reg(k / 2.0, x / 2.0)


def chi2_quantile(p, k):
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if chi2_cdf(mid, k) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ------------------------------------------------------------------ dati
run = {}
for s in SEMI:
    d = carica(f"run_s{s}.json")
    run[s] = {e["step"]: e for e in d}
prev = carica("previsione_1b.json")
termo = carica("thermo_s1.json")
pool = carica("pool.json")["pool"]
etichette = [p["d"] for p in pool]
false_claim = [1 if p["false_claim"] else 0 for p in pool]

out = {"nota": "ricalcolo indipendente, vedi ricalcolo.py"}

# ---------------------------------------------------------- orizzonte
hz = prev["horizon_rule"]
ore_150 = (150 * hz["seconds_per_batch"] + 8 * 260) / 3600.0
out["orizzonte"] = {
    "T_max_previsione": prev["T_max"],
    "secondi_per_lotto": hz["seconds_per_batch"],
    "ore_150_mie": ore_150,
    "ore_150_file": hz["hours_150"],
    "regola": "ore_150 > 3.5 -> orizzonte 100 (T=80,100) altrimenti 150 (T=80,150)",
    "orizzonte_applicato": 150 if ore_150 <= 3.5 else 100,
}

# ------------------------------------------------------- pavimento (fase 1)
fv = termo["floor_values"]
pav = {
    "seq_stdev_n1_su_8_valori": st.stdev(fv["seq"]),
    "seq_pstdev": st.pstdev(fv["seq"]),
    "seq_file": termo["floor_seq"],
    "tf_stdev_n1": st.stdev(fv["tf"]),
    "tf_file": termo["floor_tf"],
    "false_stdev_n1": st.stdev(fv["false"]),
    "false_file": termo["floor_false"],
    "n_valori_seq": len(fv["seq"]),
    # sensibilita (regola fase 1): scarto quadratico medio delle due perturbazioni 1e-5 da D_seq_0
    "seq_1e-5_rms_da_D0": math.sqrt(
        sum((v - termo["D_seq_0"]) ** 2 for v in fv["seq_at_1e-5"]) / len(fv["seq_at_1e-5"])
    ),
}
out["pavimento"] = pav
FLOOR = {"seq": pav["seq_stdev_n1_su_8_valori"], "tf": pav["tf_stdev_n1"], "false": pav["false_stdev_n1"]}

# ----------------------------- controllo D_seq e ESS ricalcolati dai log-prob
lp0 = {s: run[s][0]["lp_seq"] for s in SEMI}
w_info = {}
max_diff_D, max_diff_ess, max_diff_false = 0.0, 0.0, 0.0
for s in SEMI:
    for T in PASSI_MISURA:
        lp = run[s][T]["lp_seq"]
        a = np.array(lp) - np.array(lp0[s])
        w = np.exp(a - a.max())
        D = float((w * np.array(etichette)).sum() / w.sum())
        ess = float(w.sum() ** 2 / (w ** 2).sum())
        Df = float((w * np.array(false_claim)).sum() / w.sum())
        wn = w / w.sum()
        srt = np.sort(wn)[::-1]
        w_info[(s, T)] = {
            "max_quota_peso": float(srt[0]),
            "campioni_per_90pct_peso": int(np.searchsorted(np.cumsum(srt), 0.9) + 1),
            "campioni_per_50pct_peso": int(np.searchsorted(np.cumsum(srt), 0.5) + 1),
        }
        max_diff_D = max(max_diff_D, abs(D - run[s][T]["D_seq"]))
        max_diff_false = max(max_diff_false, abs(Df - run[s][T]["D_false"]))
        max_diff_ess = max(max_diff_ess, abs(ess - run[s][T]["ess"]) / run[s][T]["ess"])
out["controllo_da_lp_seq"] = {
    "max_diff_assoluta_D_seq_ricalcolato_vs_salvato": max_diff_D,
    "max_diff_assoluta_D_false_ricalcolato_vs_salvato": max_diff_false,
    "max_diff_relativa_ESS_ricalcolata_vs_salvata": max_diff_ess,
    "nota": "D_seq = sum(w d)/sum(w), w=exp(lp_t - lp_0), d da pool.json; ESS=(sum w)^2/sum w^2",
}

# ---------------------------------------------------- misure ai passi di misura
def vals(misura, T):
    return [run[s][T][misura] for s in SEMI]


tabella = {}
for T in PASSI_MISURA:
    r = {}
    D = vals("D_seq", T)
    ess = vals("ess", T)
    sd_grezza = sd(D)
    sd_corr = sottrai_quadratura(sd_grezza, FLOOR["seq"])
    pred = prev["pred_SD"]["lin_seq"][str(T)]
    R = sd_corr / pred
    r["D_seq_per_corsa"] = D
    r["D_seq_media"] = sum(D) / len(D)
    r["SD_grezza_n1"] = sd_grezza
    r["SD_con_pavimento_sottratto"] = sd_corr
    r["SD_prevista"] = pred
    r["R_seq"] = R
    r["R_seq_ddof0_solo_per_confronto"] = sottrai_quadratura(sd(D, 0), FLOOR["seq"]) / pred
    r["ESS_per_corsa"] = ess
    r["ESS_mediana_vera"] = st.median(ess)
    r["ESS_min"] = min(ess)
    r["ESS_max"] = max(ess)
    r["verdetto_solo_D_seq"] = (
        "NON CONCLUSIVO" if st.median(ess) < SOGLIA_ESS else verdetto_R(R)
    )
    # intervallo chi quadro 95% della SD misurata (5 gdl con 6 corse)
    k = len(SEMI) - 1
    lo = math.sqrt(k / chi2_quantile(0.975, k))
    hi = math.sqrt(k / chi2_quantile(0.025, k))
    r["R_intervallo_chi2_95"] = [R * lo, R * hi]
    # D_tf e D_false descrittive
    Dtf = vals("D_tf", T)
    Dfa = vals("D_false", T)
    r["R_tf_descrittivo"] = sottrai_quadratura(sd(Dtf), FLOOR["tf"]) / prev["pred_SD"]["lin_tf"][str(T)]
    r["R_false_descrittivo"] = sottrai_quadratura(sd(Dfa), FLOOR["false"]) / prev["pred_SD"]["lin_false"][str(T)]
    r["SD_tf"] = sd(Dtf)
    r["SD_false"] = sd(Dfa)
    # sensibilita: pavimento 1e-5
    r["R_seq_pavimento_1e-5"] = sottrai_quadratura(sd_grezza, pav["seq_1e-5_rms_da_D0"]) / pred
    # diagnostica: rumore atteso da sola ESS (euristica, non nelle regole)
    r["rumore_euristico_sqrt_D(1-D)/ESS_per_corsa"] = [
        math.sqrt(max(d * (1 - d), 0) / e) for d, e in zip(D, ess)
    ]
    r["vincolo_0_1_D_min_max"] = [min(D), max(D)]
    r["pesi"] = {str(s): w_info[(s, T)] for s in SEMI}
    tabella[str(T)] = r
out["per_passo"] = tabella

# ---------------------------------------------------------------- D_samp
def finestra_passi(T, variante):
    if variante == "A_T-19..T":
        return list(range(T - FINESTRA + 1, T + 1))
    if variante == "B_T-20..T-1":
        return list(range(T - FINESTRA, T))
    raise ValueError


def tassi(s, passi):
    return [run[s][p]["disclosure_sampled"] for p in passi]


def pesi_n(s, passi):
    return [run[s][p]["n_fail_terminated"] for p in passi]


def dsamp(T, variante="A_T-19..T", ddof_var=1, pooled=False):
    passi = finestra_passi(T, variante)
    medie, varianze = [], []
    for s in SEMI:
        x = tassi(s, passi)
        n = pesi_n(s, passi)
        if pooled:
            medie.append(sum(a * b for a, b in zip(x, n)) / sum(n))
        else:
            medie.append(sum(x) / len(x))
        varianze.append(sd(x, ddof_var) ** 2)
    rumore_var = sum(varianze) / len(varianze) / len(passi)
    rumore_sd = math.sqrt(rumore_var)
    sd_oss = sd(medie)
    sd_corr = sottrai_quadratura(sd_oss, rumore_sd)
    pred = prev["pred_SD"]["lin_seq"][str(T)]
    return {
        "finestra": [passi[0], passi[-1]],
        "medie_per_corsa": medie,
        "media_tra_corse": sum(medie) / len(medie),
        "SD_osservata_tra_corse": sd_oss,
        "rumore_SD_stimato": rumore_sd,
        "SD_corretta_in_quadratura": sd_corr,
        "SD_prevista": pred,
        "R_samp": sd_corr / pred,
        "R_samp_senza_correzione_rumore": sd_oss / pred,
        "soglia_regola_rumore_meta_SD_prevista": 0.5 * pred,
        "rumore_supera_meta_SD_prevista": rumore_sd > 0.5 * pred,
        "SD_per_passo_media_sulle_corse": math.sqrt(sum(varianze) / len(varianze)),
    }


campionato = {}
for T in [20, 40, 60, 80, 100, 120, 150]:
    campionato[str(T)] = dsamp(T)
out["D_samp"] = campionato
sens = {}
for T in T_PROVA:
    sens[str(T)] = {
        "B_finestra_T-20..T-1": dsamp(T, "B_T-20..T-1"),
        "varianza_ddof0": dsamp(T, ddof_var=0),
        "pooled_pesato_per_n_fail": dsamp(T, pooled=True),
    }
out["D_samp_sensibilita"] = sens

# ------------------------------------------------------ riuscita nelle finestre
riusc = {}
for T in [20] + [t for t in PASSI_MISURA if t >= 40]:
    passi = finestra_passi(T, "A_T-19..T")
    per_corsa = [sum(run[s][p]["solve"] for p in passi) / len(passi) for s in SEMI]
    riusc[str(T)] = {
        "finestra": [passi[0], passi[-1]],
        "media": sum(per_corsa) / len(per_corsa),
        "per_corsa": per_corsa,
        "quota_lotti_informativi": sum(
            1 for s in SEMI for p in passi if run[s][p]["informative"]
        ) / (len(SEMI) * len(passi)),
    }
out["riuscita_finestre"] = riusc
out["riuscita_finestre_B_T-20..T-1"] = {
    str(T): sum(run[s][p]["solve"] for s in SEMI for p in finestra_passi(T, "B_T-20..T-1")) / (len(SEMI) * FINESTRA)
    for T in T_PROVA
}

# --------------------------------------------------------------- verdetti
def verdetto_T(T, campione=None):
    r = tabella[str(T)]
    c = campione if campione is not None else campionato[str(T)]
    ess_ok = r["ESS_mediana_vera"] >= SOGLIA_ESS
    v_samp = "NON CONCLUSIVO" if c["rumore_supera_meta_SD_prevista"] else verdetto_R(c["R_samp"])
    if ess_ok:
        v, fonte = verdetto_R(r["R_seq"]), "D_seq"
    else:
        v, fonte = v_samp, "D_samp (D_seq NON CONCLUSIVO per ESS < 100)"
    return {
        "T": T,
        "ESS_mediana": r["ESS_mediana_vera"],
        "R_seq": r["R_seq"],
        "R_samp": c["R_samp"],
        "rumore_SD": c["rumore_SD_stimato"],
        "meta_SD_prevista": c["soglia_regola_rumore_meta_SD_prevista"],
        "verdetto_D_seq": r["verdetto_solo_D_seq"],
        "verdetto_D_samp": v_samp,
        "verdetto_usato": v,
        "fonte": fonte,
    }


def fin_dove_regge(verdetti):
    primo = verdetti[0]["verdetto_usato"]
    if primo == "NON CONCLUSIVO":
        return "non si può dire", None
    if primo != "PROMOSSO":
        return "non regge già al primo orizzonte", None
    ultimo = verdetti[0]["T"]
    for v in verdetti[1:]:
        if v["verdetto_usato"] == "PROMOSSO":
            ultimo = v["T"]
        else:
            break
    return f"regge fino a T={ultimo}", ultimo


verd = [verdetto_T(T) for T in T_PROVA]
frase, T_reg = fin_dove_regge(verd)
out["verdetti"] = verd
out["fin_dove_regge"] = {"frase": frase, "T": T_reg}

# sensibilita del verdetto alle varianti di D_samp
sens_verd = {}
for nome, kw in [
    ("B_finestra_T-20..T-1", dict(variante="B_T-20..T-1")),
    ("varianza_ddof0", dict(ddof_var=0)),
    ("pooled", dict(pooled=True)),
]:
    vv = [verdetto_T(T, dsamp(T, **kw)) for T in T_PROVA]
    sens_verd[nome] = {
        "R_samp": {str(v["T"]): v["R_samp"] for v in vv},
        "verdetti": {str(v["T"]): v["verdetto_usato"] for v in vv},
        "fin_dove_regge": fin_dove_regge(vv)[0],
    }
out["verdetti_sensibilita_D_samp"] = sens_verd

# ------------------------------------------------------------------- H2b
passi_h2 = [T for T in PASSI_MISURA if T >= 40 and tabella[str(T)]["ESS_mediana_vera"] >= SOGLIA_ESS]
h2 = {"passi_usati": passi_h2}
if len(passi_h2) >= 3:
    sd_mis = [tabella[str(T)]["SD_con_pavimento_sottratto"] for T in passi_h2]
    sd_pre = [tabella[str(T)]["SD_prevista"] for T in passi_h2]
    if all(v > 0 for v in sd_mis):
        pm = pendenza_log(passi_h2, sd_mis)
        pp = pendenza_log(passi_h2, sd_pre)
        diff = pm - pp
        h2.update(
            {
                "pendenza_misurata": pm,
                "pendenza_prevista": pp,
                "differenza": diff,
                "esito": "più veloce" if diff > 0.5 else ("più lenta" if diff < -0.5 else "come prevista"),
            }
        )
else:
    h2["esito"] = "non calcolabile (meno di 3 passi con ESS mediana >= 100)"
# varianti descrittive (non nelle regole)
tutti40 = [T for T in PASSI_MISURA if T >= 40]
sdm = [tabella[str(T)]["SD_con_pavimento_sottratto"] for T in tutti40]
sdp = [tabella[str(T)]["SD_prevista"] for T in tutti40]
h2["variante_tutti_i_passi_da_40_ignorando_ESS"] = {
    "passi": tutti40,
    "pendenza_misurata": pendenza_log(tutti40, sdm) if all(v > 0 for v in sdm) else None,
    "pendenza_prevista": pendenza_log(tutti40, sdp),
}
sd_ds = [campionato[str(T)]["SD_corretta_in_quadratura"] for T in tutti40]
if all(v > 0 for v in sd_ds):
    h2["variante_D_samp_SD_corretta_da_40"] = {
        "passi": tutti40,
        "pendenza_misurata": pendenza_log(tutti40, sd_ds),
        "pendenza_prevista": pendenza_log(tutti40, sdp),
    }
else:
    h2["variante_D_samp_SD_corretta_da_40"] = {"nota": "qualche SD corretta e' zero: log non definito", "SD": sd_ds}
out["H2b"] = h2

# ---------------------------------------------- diagnostiche per le riserve
diag = {}
diag["medie_D_seq_tra_corse"] = {str(T): tabella[str(T)]["D_seq_media"] for T in PASSI_MISURA}
diag["medie_D_samp_tra_corse"] = {T: campionato[T]["media_tra_corse"] for T in campionato}
corr = {}
for T in T_PROVA:
    a = np.array(tabella[str(T)]["D_seq_per_corsa"])
    b = np.array(campionato[str(T)]["medie_per_corsa"])
    corr[str(T)] = float(np.corrcoef(a, b)[0, 1])
diag["correlazione_tra_corse_D_seq_vs_D_samp"] = corr
a = np.array(tabella["80"]["D_seq_per_corsa"]); b = np.array(tabella["150"]["D_seq_per_corsa"])
diag["corr_D_seq_80_150"] = float(np.corrcoef(a, b)[0, 1])
a = np.array(campionato["80"]["medie_per_corsa"]); b = np.array(campionato["150"]["medie_per_corsa"])
diag["corr_D_samp_80_150"] = float(np.corrcoef(a, b)[0, 1])
diag["ESS_mediana_per_T"] = {str(T): tabella[str(T)]["ESS_mediana_vera"] for T in PASSI_MISURA}
diag["rapporto_SD_samp_su_SD_seq"] = {
    str(T): (
        campionato[str(T)]["SD_corretta_in_quadratura"] / tabella[str(T)]["SD_con_pavimento_sottratto"]
        if tabella[str(T)]["SD_con_pavimento_sottratto"] > 0 else None
    )
    for T in [20, 40, 60, 80, 100, 120, 150]
}
# jackknife leave-one-out
jk = {}
for T in T_PROVA:
    D = tabella[str(T)]["D_seq_per_corsa"]
    m = campionato[str(T)]["medie_per_corsa"]
    pred = prev["pred_SD"]["lin_seq"][str(T)]
    jr, js = [], []
    for i in range(len(SEMI)):
        Dx = [v for j, v in enumerate(D) if j != i]
        jr.append(sottrai_quadratura(sd(Dx), FLOOR["seq"]) / pred)
        passi = finestra_passi(T, "A_T-19..T")
        var_ok = [sd(tassi(s, passi)) ** 2 for j, s in enumerate(SEMI) if j != i]
        rum = math.sqrt(sum(var_ok) / len(var_ok) / len(passi))
        mx = [v for j, v in enumerate(m) if j != i]
        js.append(sottrai_quadratura(sd(mx), rum) / pred)
    jk[str(T)] = {"R_seq_senza_una_corsa": jr, "R_samp_senza_una_corsa": js}
diag["leave_one_out"] = jk
# bootstrap sulle corse
rng = np.random.default_rng(12345)
bs = {}
for T in T_PROVA:
    D = np.array(tabella[str(T)]["D_seq_per_corsa"])
    pred = prev["pred_SD"]["lin_seq"][str(T)]
    Rb = []
    for _ in range(20000):
        x = rng.choice(D, size=len(D), replace=True)
        if np.std(x) == 0:
            continue
        Rb.append(sottrai_quadratura(float(np.std(x, ddof=1)), FLOOR["seq"]) / pred)
    bs[str(T)] = {"p2.5": float(np.percentile(Rb, 2.5)), "p50": float(np.percentile(Rb, 50)), "p97.5": float(np.percentile(Rb, 97.5))}
diag["bootstrap_R_seq_con_6_corse"] = bs


def p_prom(k_err):
    lo, hi = (0.5 / k_err) ** 2 * 5, (2.0 / k_err) ** 2 * 5
    return chi2_cdf(hi, 5) - chi2_cdf(lo, 5)


def p_boc(k_err):
    lo, hi = (0.33 / k_err) ** 2 * 5, (3.0 / k_err) ** 2 * 5
    return chi2_cdf(lo, 5) + (1 - chi2_cdf(hi, 5))


diag["potenza_ricontrollata"] = {
    "P_PROMOSSO_se_previsione_giusta": p_prom(1.0),
    "P_PROMOSSO_con_errore_2x_sopra": p_prom(2.0),
    "P_PROMOSSO_con_errore_2x_sotto": p_prom(0.5),
    "P_BOCCIATO_con_errore_3x_sopra": p_boc(3.0),
    "P_BOCCIATO_con_errore_4x_sopra": p_boc(4.0),
    "P_BOCCIATO_con_errore_3x_sotto": p_boc(1 / 3),
    "intervallo_95_SD_moltiplicatori": [
        math.sqrt(5 / chi2_quantile(0.975, 5)),
        math.sqrt(5 / chi2_quantile(0.025, 5)),
    ],
}
sim = np.random.default_rng(777)
if len(passi_h2) >= 3:
    Tt = np.log(np.array(passi_h2, float))
    pre = np.array([prev["pred_SD"]["lin_seq"][str(T)] for T in passi_h2])
    sl = []
    for _ in range(20000):
        z = np.sqrt(sim.chisquare(5, len(Tt)) / 5)
        sl.append(np.polyfit(Tt, np.log(z), 1)[0])
    diag["H2b_errore_pendenza_se_T_indipendenti"] = float(np.std(sl))
    sl2 = []
    v = np.concatenate([[pre[0] ** 2], np.diff(pre ** 2)])
    for _ in range(4000):
        inc = sim.normal(0, np.sqrt(np.clip(v, 0, None)), size=(6, len(pre)))
        x = np.cumsum(inc, axis=1)
        s_ = x.std(axis=0, ddof=1)
        sl2.append(np.polyfit(Tt, np.log(s_), 1)[0])
    diag["H2b_errore_pendenza_se_diffusione_correlata"] = float(np.std(sl2))
# --- intervalli chi quadro anche per R_samp
kk = len(SEMI) - 1
lo_f = math.sqrt(kk / chi2_quantile(0.975, kk))
hi_f = math.sqrt(kk / chi2_quantile(0.025, kk))
diag["R_samp_intervallo_chi2_95"] = {
    str(T): [campionato[str(T)]["R_samp"] * lo_f, campionato[str(T)]["R_samp"] * hi_f] for T in T_PROVA
}
# --- il rumore di D_samp: tendenza dentro la finestra e autocorrelazione dei residui
aut = {}
for T in T_PROVA:
    passi = finestra_passi(T, "A_T-19..T")
    righe = []
    for s in SEMI:
        x = np.array(tassi(s, passi))
        t = np.arange(len(x), dtype=float)
        pend, inter = np.polyfit(t, x, 1)
        res = x - (pend * t + inter)
        ac = float(np.corrcoef(x[:-1] - x.mean(), x[1:] - x.mean())[0, 1])
        righe.append({"semi": s, "var_totale": float(x.var(ddof=1)), "var_residua_dopo_tendenza": float(res.var(ddof=2)),
                      "autocorr_lag1": ac, "pendenza_per_passo": float(pend)})
    aut[str(T)] = righe
diag["rumore_D_samp_tendenza_e_autocorrelazione"] = aut
# --- scala logit di D_samp (descrittiva, nessuna previsione scritta in logit)
def logit(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


diag["SD_logit_medie_D_samp"] = {
    T: sd([logit(v) for v in campionato[T]["medie_per_corsa"]]) for T in campionato
}
diag["corse_vicine_ai_limiti_D_samp_entro_0.12"] = {
    T: sum(1 for v in campionato[T]["medie_per_corsa"] if v < 0.12 or v > 0.88) for T in campionato
}
# --- n di fallimenti terminati per passo (denominatore dei tassi)
diag["n_fail_terminated_medio_per_passo"] = {
    str(T): float(np.mean([run[s][p]["n_fail_terminated"] for s in SEMI for p in finestra_passi(T, "A_T-19..T")]))
    for T in T_PROVA
}
# --- spostamento della media: D_seq (pool riponderato) contro D_samp (rollout veri)
diag["media_D_seq_meno_media_D_samp"] = {
    str(T): tabella[str(T)]["D_seq_media"] - campionato[str(T)]["media_tra_corse"] for T in [20, 40, 60, 80, 100, 120, 150]
}
out["diagnostiche"] = diag


# --------------------------------------------------------------- stampa
def f(x, n=4):
    return "nan" if x is None else f"{x:.{n}f}"


print("== ORIZZONTE ==", out["orizzonte"])
print("== PAVIMENTO ==", pav)
print("== CONTROLLO da lp_seq ==", out["controllo_da_lp_seq"])
print()
print("T   SDgr   SDcorr  SDprev  R_seq  ESSmed  ESSmin  ESSmax  verd_D_seq   R_tf   R_false")
for T in PASSI_MISURA:
    r = tabella[str(T)]
    print(f"{T:3d} {f(r['SD_grezza_n1'])} {f(r['SD_con_pavimento_sottratto'])} {f(r['SD_prevista'])} {f(r['R_seq'],3)} "
          f"{f(r['ESS_mediana_vera'],1):>7} {f(r['ESS_min'],1):>7} {f(r['ESS_max'],1):>7}  {r['verdetto_solo_D_seq']:<15} "
          f"{f(r['R_tf_descrittivo'],3)} {f(r['R_false_descrittivo'],3)}")
print()
print("D_samp (finestra A = T-19..T)")
print("T   media   SDoss   rumSD   SDcorr  SDprev  R_samp  rum>meta?")
for T in campionato:
    c = campionato[T]
    print(f"{T:>3} {f(c['media_tra_corse'])} {f(c['SD_osservata_tra_corse'])} {f(c['rumore_SD_stimato'])} "
          f"{f(c['SD_corretta_in_quadratura'])} {f(c['SD_prevista'])} {f(c['R_samp'],3)}  {c['rumore_supera_meta_SD_prevista']}")
print()
for v in verd:
    print(v)
print("FIN DOVE REGGE:", frase)
print("H2b:", {k: v for k, v in h2.items() if k != 'variante_D_samp_SD_corretta_da_40'})
print("H2b variante D_samp:", h2.get("variante_D_samp_SD_corretta_da_40"))
print("RIUSCITA:", {k: v["media"] for k, v in riusc.items()})
print("RIUSCITA B:", out["riuscita_finestre_B_T-20..T-1"])
print("SENSIBILITA D_samp:", json.dumps(sens_verd, ensure_ascii=False))
print("POTENZA:", diag["potenza_ricontrollata"])
print("BOOTSTRAP:", bs)
print("LOO:", jk)
print("CORR D_seq vs D_samp:", corr, " D_seq 80-150:", diag["corr_D_seq_80_150"], " D_samp 80-150:", diag["corr_D_samp_80_150"])
print("errore pendenza H2b:", diag.get("H2b_errore_pendenza_se_T_indipendenti"), diag.get("H2b_errore_pendenza_se_diffusione_correlata"))
print("R_samp chi2 CI:", diag["R_samp_intervallo_chi2_95"])
print("SD logit:", diag["SD_logit_medie_D_samp"])
print("vicine ai limiti:", diag["corse_vicine_ai_limiti_D_samp_entro_0.12"])
print("D_seq - D_samp:", diag["media_D_seq_meno_media_D_samp"])
for T, righe in aut.items():
    print("rumore D_samp T=", T, [(round(r['var_totale'], 4), round(r['var_residua_dopo_tendenza'], 4), round(r['autocorr_lag1'], 2)) for r in righe])

with open(os.path.join(QUI, "miei_numeri.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=2)
print("\nSalvato miei_numeri.json")
