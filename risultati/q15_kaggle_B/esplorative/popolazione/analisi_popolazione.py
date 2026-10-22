"""Analisi secondaria, scritta il 20/10/2026 alle 23:05 PRIMA di vedere le corse della fase 1 del termometro.

Domanda (negazione dell'ipotesi implicita A10 dell'articolo arXiv 2609.33220): il tasso di ammissione misurato sui PROPRI
fallimenti di ogni corsa (la misura dell'articolo) si sparge tra le corse più del tasso misurato su un pool FISSO di
fallimenti del passo 0 (D_seq)? Se sì, una parte dello spread dell'articolo è un cambio di popolazione (ogni corsa
fallisce su problemi diversi), non un cambio di comportamento.

Dati: i file run_s*.json della fase 1 (12 corse da 40 passi). Per ogni corsa:
- D_seq al passo 40 (pool fisso, 400 fallimenti del passo 0);
- D_camp = ammissioni / fallimenti terminati sommati sugli ultimi 5 passi (36-40), dalle righe "disclosure_sampled" e
  "n_fail_terminated".
Il rumore binomiale di D_camp (media di p(1-p)/n sulle corse) si toglie in quadratura dalla sua varianza tra corse.

Regola (fissata ora): rapporto Q = SD_corretta(D_camp) / SD(D_seq).
- Q >= 1,25: "la popolazione conta" (lo spread sui propri fallimenti supera quello a popolazione fissa);
- Q <= 1,00: "la popolazione non conta";
- in mezzo: "non conclusivo".
Con 12 corse l'intervallo di Q è largo: si riporta un intervallo bootstrap al 90% sulle corse, e il verdetto vale solo
se l'intervallo non attraversa la soglia usata. Si riporta anche la correlazione tra D_seq e D_camp sulle corse.

Uso: python -I analisi_popolazione.py CARTELLA_CORSE | test
"""
import glob
import json
import math
import os
import random
import sys

ULTIMI = 5
T = 40


def sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def corr(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


def misure(log):
    """Da una corsa (lista di righe per passo) a (D_seq a T, D_camp sugli ultimi passi, n fallimenti usati)."""
    righe = {r["step"]: r for r in log}
    d_seq = righe[T]["D_seq"]
    amm, n = 0.0, 0
    for t in range(T - ULTIMI + 1, T + 1):
        r = righe[t]
        k = r.get("n_fail_terminated", 0)
        amm += r.get("disclosure_sampled", 0.0) * k
        n += k
    return d_seq, (amm / n if n else float("nan")), n


def q_di(coppie):
    seq = [c[0] for c in coppie]
    camp = [c[1] for c in coppie]
    rumore = sum(c[1] * (1 - c[1]) / c[2] for c in coppie if c[2]) / len(coppie)
    var_corr = max(sd(camp) ** 2 - rumore, 0.0)
    return math.sqrt(var_corr) / sd(seq) if sd(seq) > 0 else float("nan"), rumore


def analizza(coppie, seme=20261020):
    q, rumore = q_di(coppie)
    rng = random.Random(seme)
    boot = []
    for _ in range(2000):
        camp = [coppie[rng.randrange(len(coppie))] for _ in coppie]
        qb, _ = q_di(camp)
        if qb == qb:
            boot.append(qb)
    boot.sort()
    lo, hi = boot[int(0.05 * len(boot))], boot[int(0.95 * len(boot)) - 1]
    if q >= 1.25 and lo >= 1.25:
        verdetto = "la popolazione conta"
    elif q <= 1.0 and hi <= 1.0:
        verdetto = "la popolazione non conta"
    else:
        verdetto = "non conclusivo"
    return {"n_corse": len(coppie), "SD_D_seq": sd([c[0] for c in coppie]), "SD_D_camp_grezza": sd([c[1] for c in coppie]),
            "rumore_binomiale_var": rumore, "Q": q, "Q_boot90": [lo, hi],
            "corr_seq_camp": corr([c[0] for c in coppie], [c[1] for c in coppie]),
            "fallimenti_per_corsa": [c[2] for c in coppie], "verdetto": verdetto}


def test():
    """Due mondi finti: popolazione fissa (Q circa 1) e popolazione che deriva (Q ben sopra 1,25)."""
    rng = random.Random(7)
    for deriva, nome in [(0.0, "senza deriva"), (0.12, "con deriva")]:
        coppie = []
        for _ in range(12):
            d_seq = min(max(rng.gauss(0.5, 0.10), 0.01), 0.99)
            p_camp = min(max(d_seq + rng.gauss(0, deriva), 0.01), 0.99)
            n = 150
            k = sum(1 for _ in range(n) if rng.random() < p_camp)
            coppie.append((d_seq, k / n, n))
        r = analizza(coppie)
        print(nome, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "fallimenti_per_corsa"}))
    print("test ok")


def main(cartella):
    file = sorted(glob.glob(os.path.join(cartella, "run_s*.json")))
    coppie = []
    for f in file:
        log = json.load(open(f))
        if len(log) != T + 1:
            print("corsa incompleta, saltata:", os.path.basename(f))
            continue
        coppie.append(misure(log))
    if len(coppie) < 6:
        print("meno di 6 corse complete: nessun verdetto")
        return
    r = analizza(coppie)
    print(json.dumps(r, indent=1))
    json.dump(r, open(os.path.join(cartella, "analisi_popolazione.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "test":
        test()
    else:
        main(sys.argv[1])
