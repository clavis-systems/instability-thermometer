"""Confronto dei miei numeri (miei_numeri.json) con quelli ufficiali (analisi_fase1b.json). Solo lettura."""
import json, os, math
QUI = os.path.dirname(os.path.abspath(__file__))
Q15 = os.path.join(QUI, "..", "q15")
m = json.load(open(os.path.join(QUI, "miei_numeri.json"), encoding="utf-8"))
u = json.load(open(os.path.join(Q15, "analisi_fase1b.json"), encoding="utf-8"))
righe = []
for T in ("80", "150"):
    k = f"H1b_T{T}"
    pt = m["per_passo"][T]
    ds = m["D_samp"][T]
    dp = m["D_samp_sensibilita"][T]["pooled_pesato_per_n_fail"]
    ref = [
        ("SD D_seq grezza", pt["SD_grezza_n1"], u[k]["measured_SD_D_seq"]),
        ("SD D_seq con pavimento", pt["SD_con_pavimento_sottratto"], u[k]["corrected"]),
        ("SD prevista", pt["SD_prevista"], u[k]["predicted"]),
        ("R (D_seq)", pt["R_seq"], u[k]["R"]),
        ("ESS mediana", pt["ESS_mediana_vera"], u[k]["median_ESS"]),
        ("rumore D_samp (varianza)", ds["rumore_SD_stimato"] ** 2, u[k]["D_samp_noise_var"]),
        ("SD D_samp corretta [media semplice]", ds["SD_corretta_in_quadratura"], u[k]["D_samp_SD_corrected"]),
        ("SD D_samp corretta [media pesata n_fail]", dp["SD_corretta_in_quadratura"], u[k]["D_samp_SD_corrected"]),
        ("R_samp [media semplice]", ds["R_samp"], u[k]["R_samp"]),
        ("R_samp [media pesata n_fail]", dp["R_samp"], u[k]["R_samp"]),
        ("R D_tf descrittivo", pt["R_tf_descrittivo"], u[k]["descriptive"]["D_tf"]["R"]),
        ("R D_false descrittivo", pt["R_false_descrittivo"], u[k]["descriptive"]["D_false"]["R"]),
        ("riuscita media finestra", m["riuscita_finestre"][T]["media"], u[k]["solve_window_mean"]),
    ]
    for nome, a, b in ref:
        righe.append({"T": int(T), "grandezza": nome, "mio": a, "ufficiale": b, "diff_rel": (a - b) / b if b else None})
    v = [x for x in m["verdetti"] if x["T"] == int(T)][0]
    righe.append({"T": int(T), "grandezza": "verdetto D_seq", "mio": v["verdetto_D_seq"], "ufficiale": u[k]["verdict"]})
    righe.append({"T": int(T), "grandezza": "verdetto D_samp", "mio": v["verdetto_D_samp"], "ufficiale": u[k]["verdict_samp"]})
    righe.append({"T": int(T), "grandezza": "verdetto usato", "mio": v["verdetto_usato"], "ufficiale": u[k]["verdict_used"]})
righe.append({"T": None, "grandezza": "fin dove regge", "mio": m["fin_dove_regge"]["T"], "ufficiale": u["holds_up_to"]})
righe.append({"T": None, "grandezza": "H2b", "mio": m["H2b"]["esito"], "ufficiale": u["H2b"].get("note")})
json.dump(righe, open(os.path.join(QUI, "confronto.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for r in righe:
    if "diff_rel" in r and isinstance(r["mio"], (int, float)) and isinstance(r["ufficiale"], (int, float)):
        print(f"T={r['T']} {r['grandezza']:<42} mio={r['mio']:.6g}  uff={r['ufficiale']:.6g}  diff_rel={r['diff_rel']*100 if r['diff_rel'] is not None else float('nan'):+.3g}%")
    else:
        print(f"T={r['T']} {r['grandezza']:<42} mio={r['mio']}  uff={r['ufficiale']}")

# supplemento: D_samp con media pesata a tutti i T, e D_tf / D_false a T=100..150
def carica(n): return json.load(open(os.path.join(Q15, n), encoding="utf-8"))
SEMI=[201,202,203,204,205,206]
run={s:{e["step"]:e for e in carica(f"run_s{s}.json")} for s in SEMI}
prev=carica("previsione_1b.json")
import statistics as st
print("\nR_samp con media pesata (come ufficiale), tutti i T:")
for T in [20,40,60,80,100,120,150]:
    medie=[];var=[]
    for s in SEMI:
        w=[(run[s][p]["disclosure_sampled"],run[s][p]["n_fail_terminated"]) for p in range(T-19,T+1)]
        tot=sum(n for _,n in w); medie.append(sum(x*n for x,n in w)/tot); var.append(st.variance([x for x,_ in w])/len(w))
    noise=sum(var)/len(var); sdo=st.stdev(medie); sc=math.sqrt(max(sdo**2-noise,0)); pr=prev["pred_SD"]["lin_seq"][str(T)]
    print(f"  T={T:3d} R_samp={sc/pr:.3f}  SD_corr={sc:.4f}  SD_oss={sdo:.4f}  rumoreSD={math.sqrt(noise):.4f}  0.5*pred={0.5*pr:.4f}")
print("\nD_tf e D_false per corsa:")
for T in [100,120,150]:
    print(" T",T,"D_tf",[round(run[s][T]["D_tf"],3) for s in SEMI],"D_false",[round(run[s][T]["D_false"],3) for s in SEMI])
