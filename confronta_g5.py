"""G5 pilot: agreement between the detector (countdown.DISCLOSE) and the independent labeller.

Pass if the agreement is at least 95% over exactly 100 non-empty tails (PREMESSA.md, table of gates). Disagreements are
listed with their text so the only allowed action (fix the detector, redo the pilot on another 100 tails) can be
decided. Written before the labels exist.

usage: python confronta_g5.py FOLDER [THRESHOLD]   (threshold 0.95 for G5; 0.90 for the H6 detector, PREMESSA)
"""
import json
import sys
from pathlib import Path

N_ATTESI = 100
SOGLIA = 0.95


def main(cartella: str, soglia: float = SOGLIA) -> int:
    d = Path(cartella)
    code = [json.loads(r) for r in (d / "code_cieche.jsonl").read_text(encoding="utf-8").splitlines() if r.strip()]
    rilevatore = {x["id"]: int(bool(x["detector"])) for x in json.loads((d / "etichette_rilevatore.json").read_text())}
    indip = json.loads((d / "etichette_indipendenti.json").read_text(encoding="utf-8"))
    per_id = {x["id"]: x for x in indip}

    ids = [c["id"] for c in code]
    assert len(ids) == len(set(ids)), "id ripetuti in code_cieche.jsonl"
    assert set(per_id) == set(ids) and len(indip) == len(ids), "le etichette indipendenti non coprono esattamente i casi"
    assert all(per_id[i]["etichetta"] in (0, 1) for i in ids), "etichetta diversa da 0/1"

    accordi = sum(rilevatore[i] == per_id[i]["etichetta"] for i in ids)
    n = len(ids)
    quota = accordi / n
    disaccordi = [{"id": i, "rilevatore": rilevatore[i], "indipendente": per_id[i]["etichetta"],
                   "dubbio": per_id[i].get("dubbio", False), "motivo": per_id[i].get("motivo", ""),
                   "answer_line": c["answer_line"], "tail": c["tail"][:600]}
                  for c in code for i in [c["id"]] if rilevatore[i] != per_id[i]["etichetta"]]
    tabella = {f"rilevatore {a} / indipendente {b}": sum(rilevatore[i] == a and per_id[i]["etichetta"] == b for i in ids)
               for a in (0, 1) for b in (0, 1)}
    esito = {"casi": n, "casi_attesi": N_ATTESI, "accordi": accordi, "quota_accordo": round(quota, 4),
             "soglia": soglia, "PASSA": n == N_ATTESI and quota >= soglia, "tabella": tabella,
             "dubbi_indipendente": sum(bool(per_id[i].get("dubbio")) for i in ids), "disaccordi": disaccordi}
    (d / "esito_g5.json").write_text(json.dumps(esito, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in esito.items() if k != "disaccordi"}, indent=1, ensure_ascii=False))
    print(f"{len(disaccordi)} disaccordi scritti in {d / 'esito_g5.json'}")
    return 0 if esito["PASSA"] else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else SOGLIA))
