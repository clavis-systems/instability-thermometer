"""Turns analisi_fase1.json into a short report in Italian and English with sentences written BEFORE any real data
(19/10/2026). The only thing the data choose is WHICH sentence is printed and the numbers inside it.

Usage: python -I rapporto_fase1.py RESULTS_DIR      (writes RAPPORTO_FASE1.md next to analisi_fase1.json)
       python -I rapporto_fase1.py test
"""
import json
import os
import sys

FRASI = {
    "PROMOSSO": (
        "Il termometro è promosso: la dispersione misurata tra le 12 corse ({mis:.4f}) sta entro un fattore 2 da quella "
        "prevista al passo 0 ({pre:.4f}); R = {R:.2f}, intervallo al 95% da {lo:.2f} a {hi:.2f}.",
        "The thermometer passes: the measured between-run spread over 12 runs ({mis:.4f}) is within a factor of 2 of the "
        "step-0 prediction ({pre:.4f}); R = {R:.2f}, 95% interval {lo:.2f} to {hi:.2f}."),
    "INDIZIO": (
        "Solo un indizio: la dispersione misurata ({mis:.4f}) e quella prevista ({pre:.4f}) differiscono più di 2 volte "
        "ma meno di 3 (R = {R:.2f}, intervallo da {lo:.2f} a {hi:.2f}). Il termometro coglie l'ordine di grandezza, non "
        "il valore.",
        "Only a hint: measured ({mis:.4f}) and predicted ({pre:.4f}) spread differ by more than 2x but less than 3x "
        "(R = {R:.2f}, interval {lo:.2f} to {hi:.2f}). The thermometer gets the order of magnitude, not the value."),
    "BOCCIATO": (
        "Il termometro è bocciato: la dispersione misurata ({mis:.4f}) e quella prevista ({pre:.4f}) differiscono più "
        "di 3 volte (R = {R:.2f}). Lo scriviamo come risultato negativo.",
        "The thermometer fails: measured ({mis:.4f}) and predicted ({pre:.4f}) spread differ by more than 3x "
        "(R = {R:.2f}). We report it as a negative result."),
    "NON CONCLUSIVO": (
        "Non conclusivo: al passo 40 i pesi della misura si sono concentrati su pochi esempi (ESS mediana sotto 100), "
        "quindi la misura non è affidabile e il confronto non si può fare. Non è un verdetto sul termometro.",
        "Inconclusive: by step 40 the importance weights of the measure have collapsed onto few examples (median ESS "
        "below 100), so the measure is unreliable and the comparison cannot be made. This is not a verdict on the "
        "thermometer."),
    "NON CALCOLABILE": (
        "Il verdetto non è calcolabile (la previsione è zero o mancano dati): vedi i dettagli.",
        "The verdict cannot be computed (zero prediction or missing data): see the details."),
}
LIN = ("Sui funzionali lineari dei pesi, che non hanno rumore di misura né non linearità, R va da {a:.2f} a {b:.2f}: "
       "{giudizio}.",
       "On the linear functionals of the weights, with no measurement noise and no nonlinearity, R ranges from "
       "{a:.2f} to {b:.2f}: {judgement}.")
H2 = ("Crescita nel tempo: pendenza misurata {m:.2f} contro {p:.2f} prevista ({esito}).",
      "Growth over time: measured slope {m:.2f} against {p:.2f} predicted ({outcome}).")
H3 = ("Zone (esplorativo): il ragionamento fallito pesa {r} della finestra intorno al punto dell'ammissione più la "
      "coda, {esito} quanto trovato nell'articolo.",
      "Regions (exploratory): failed reasoning weighs {r} the window around the admission point plus the tail, "
      "{outcome} the paper's finding.")
BASE = ("Confronto con la scorciatoia ingenua (dispersione al passo 5 allungata): R = {R:.2f}.",
        "Comparison with the naive shortcut (spread at step 5, extrapolated): R = {R:.2f}.")


def scrivi(a):
    h1 = a["H1"]
    lo, hi = h1["R_CI95"] if h1.get("R_CI95") else (float("nan"), float("nan"))
    v = h1["verdict"]
    it, en = [f"# Termometro, fase 1: verdetto ({a['n_runs']} corse)"], [f"# Thermometer, phase 1: verdict ({a['n_runs']} runs)"]
    args = dict(mis=h1["measured_SD_corrected"], pre=h1["predicted_SD_replay"], R=h1["R"] or float("nan"), lo=lo, hi=hi)
    it.append(FRASI[v][0].format(**args))
    en.append(FRASI[v][1].format(**args))
    sens = a.get("H1_sensitivity_floor_1e-5")  # sentences added 20/10, before the freeze
    if sens and sens.get("R") is not None:
        if sens["same_verdict"]:
            it.append(f"Con il pavimento del rumore misurato a perturbazioni 100 volte più grandi (1e-5) R = "
                      f"{sens['R']:.2f}: il verdetto non cambia.")
            en.append(f"With the noise floor measured at 100 times larger perturbations (1e-5), R = {sens['R']:.2f}: "
                      f"the verdict does not change.")
        else:
            it.insert(1, f"**Attenzione:** con il pavimento del rumore misurato a perturbazioni 100 volte più grandi "
                         f"(1e-5) R = {sens['R']:.2f} e il verdetto sarebbe {sens['verdict']}: il rumore numerico conta.")
            en.insert(1, f"**Warning:** with the noise floor measured at 100 times larger perturbations (1e-5), "
                         f"R = {sens['R']:.2f} and the verdict would be {sens['verdict']}: numerical noise matters.")
    Rs = [x["R"] for nm, x in a["secondary"]["linear_functionals_40"].items() if x["R"] is not None and nm != "lin_false"]
    if Rs:
        ok = all(0.5 <= r <= 2 for r in Rs)
        it.append(LIN[0].format(a=min(Rs), b=max(Rs), giudizio="la teoria diffusione + Adam regge" if ok else
                                "la teoria diffusione + Adam da sola non basta"))
        en.append(LIN[1].format(a=min(Rs), b=max(Rs), judgement="the diffusion + Adam theory holds" if ok else
                                "the diffusion + Adam theory alone is not enough"))
    h6 = a.get("H6")  # sentences added 20/10, before the freeze
    if h6 and h6.get("R") is not None:
        nomi_it = {"disclosure": "l'ammissione dell'errore", "false_claim": "la falsa dichiarazione di riuscita"}
        nomi_en = {"disclosure": "failure disclosure", "false_claim": "the false claim of success"}
        it.append(f"**H6 (esplorativa), il secondo comportamento** (dire di aver risolto quando non è vero): "
                  f"R = {h6['R']:.2f} ({h6['verdict']}).")
        en.append(f"**H6 (exploratory), the second behaviour** (claiming success when it failed): R = {h6['R']:.2f} "
                  f"({h6['verdict']}).")
        if not h6.get("ranking_scored", True):
            it.append("Le due dispersioni previste erano quasi uguali (rapporto tra 0,9 e 1,1): il termometro non le "
                      "distingue, e l'ordine non si giudica. Le due etichette sono quasi complementari.")
            en.append("The two predicted spreads were almost equal (ratio between 0.9 and 1.1): the thermometer does not "
                      "tell them apart, so the ranking is not scored. The two labels are nearly complementary.")
        else:
            it.append(f"Il termometro aveva previsto che il comportamento più instabile fosse "
                      f"{nomi_it[h6['predicted_more_unstable']]}; nelle corse è stato "
                      f"{nomi_it[h6['measured_more_unstable']]}: "
                      + ("previsione giusta." if h6["ranking_correct"] else "previsione sbagliata."))
            en.append(f"The thermometer predicted that the more unstable behaviour would be "
                      f"{nomi_en[h6['predicted_more_unstable']]}; in the runs it was "
                      f"{nomi_en[h6['measured_more_unstable']]}: "
                      + ("the prediction was right." if h6["ranking_correct"] else "the prediction was wrong."))
    h2 = a["H2"]
    it.append(H2[0].format(m=h2["slope_measured_D_seq"], p=h2["slope_predicted"], esito="regge" if h2["pass"] else "non regge"))
    en.append(H2[1].format(m=h2["slope_measured_D_seq"], p=h2["slope_predicted"], outcome="holds" if h2["pass"] else "does not hold"))
    if "H3" in a and a["H3"].get("sd_reasoning") is not None:
        r = a["H3"]["sd_reasoning"] / a["H3"]["sd_window16_plus_tail"] if a["H3"]["sd_window16_plus_tail"] else float("inf")
        it.append(H3[0].format(r=f"{r:.1f} volte", esito="come" if a["H3"]["pass"] else "al contrario di"))
        en.append(H3[1].format(r=f"{r:.1f} times", outcome="as in" if a["H3"]["pass"] else "contrary to"))
    if a.get("baseline_naive_from_step5", {}).get("R") is not None:
        it.append(BASE[0].format(R=a["baseline_naive_from_step5"]["R"]))
        en.append(BASE[1].format(R=a["baseline_naive_from_step5"]["R"]))
    return "\n\n".join(it) + "\n\n---\n\n" + "\n\n".join(en) + "\n"


def main(d):
    a = json.load(open(os.path.join(d, "analisi_fase1.json")))
    testo = scrivi(a)
    open(os.path.join(d, "RAPPORTO_FASE1.md"), "w", encoding="utf-8").write(testo)
    print(testo)


def test():
    fake = {"n_runs": 12, "H1": {"measured_SD_corrected": 0.031, "predicted_SD_replay": 0.028, "R": 1.1,
                                 "R_CI95": [0.78, 1.87], "verdict": "PROMOSSO"},
            "secondary": {"linear_functionals_40": {"lin_seq": {"R": 1.05}, "rand1": {"R": 0.9}}},
            "H2": {"slope_measured_D_seq": 0.47, "slope_predicted": 0.45, "pass": True},
            "H3": {"sd_reasoning": 2.0, "sd_window16_plus_tail": 1.0, "pass": True},
            "baseline_naive_from_step5": {"R": 1.2}}
    print(scrivi(fake))
    fake["H6"] = {"R": 0.8, "verdict": "PROMOSSO", "predicted_more_unstable": "false_claim", "ranking_scored": True,
                  "measured_more_unstable": "disclosure", "ranking_correct": False}
    assert "previsione sbagliata" in scrivi(fake)
    fake["H6"].update(ranking_scored=False, predicted_more_unstable=None, ranking_correct=None)
    assert "non si giudica" in scrivi(fake)
    fake["secondary"]["linear_functionals_40"]["lin_false"] = {"R": 0.3}
    assert "regge" in scrivi(fake) and "non basta" not in scrivi(fake)
    fake["H1_sensitivity_floor_1e-5"] = {"R": 1.0, "verdict": "PROMOSSO", "same_verdict": True}
    assert "non cambia" in scrivi(fake)
    fake["H1_sensitivity_floor_1e-5"] = {"R": 0.45, "verdict": "INDIZIO", "same_verdict": False}
    t = scrivi(fake)
    assert t.split("\n\n")[1].startswith("**Attenzione:**"), t
    for v in ("INDIZIO", "BOCCIATO", "NON CONCLUSIVO", "NON CALCOLABILE"):
        fake["H1"]["verdict"] = v
        assert scrivi(fake)
    print("rapporto test passed")


if __name__ == "__main__":
    test() if sys.argv[1] == "test" else main(sys.argv[1])
