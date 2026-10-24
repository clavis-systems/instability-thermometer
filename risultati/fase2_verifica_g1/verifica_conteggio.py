"""Verifica del conteggio della riuscita al passo 0 (fase 2, q05 e olmo1 entrambe 8/416).

Per ogni famiglia: le 400 risposte del pool sono quelle contate come fallimenti terminati. Si decodifica la domanda di
ognuna (prompt_ids), si ricavano numeri e bersaglio, e (1) si controlla che il compito sia uno dei 52 di rl_tasks(7)
nell'ordine giusto; (2) si rifà il punteggio della risposta (testo = ultimi 600 caratteri, dove sta la riga Answer) con
la funzione congelata countdown.reward: deve essere 0; (3) per ogni compito si contano le risposte nel pool: 8 meno
questo numero = riuscite + non terminate (+ quelle tagliate dal tetto di 400 nell'ultimo lotto).

uso: python verifica_conteggio.py TERMOMETRO_DIR FAM1 POOL1 TOK1 [FAM2 POOL2 TOK2 ...]
"""
import json
import re
import sys
from collections import Counter, OrderedDict

from tokenizers import Tokenizer

sys.path.insert(0, sys.argv[1])
import countdown as cd  # noqa: E402

TASKS = cd.rl_tasks(7, 8000)
PROMPT = re.compile(r"Using the numbers \[([0-9, ]+)\], write an expression that equals (\d+)\.")
G = 8
mancanti = {}
for k in range(2, len(sys.argv), 3):
    fam, pool_path, tok_path = sys.argv[k:k + 3]
    pool = json.load(open(pool_path, encoding="utf-8"))
    tok = Tokenizer.from_file(tok_path)
    per_task = OrderedDict()
    rifatti = Counter()
    decodifiche = {}
    for it in pool["pool"]:
        key = tuple(it["prompt_ids"])
        if key not in decodifiche:
            m = PROMPT.search(tok.decode(it["prompt_ids"], skip_special_tokens=False))
            assert m, "domanda non riconosciuta"
            decodifiche[key] = ([int(x) for x in m.group(1).split(",")], int(m.group(2)))
        nums, target = decodifiche[key]
        per_task[(tuple(nums), target)] = per_task.get((tuple(nums), target), 0) + 1
        r = cd.classify(it["text"], nums, target, True)["reward"]
        rifatti[r] += 1
    ordine = list(per_task)
    attesi = [(tuple(t["nums"]), t["target"]) for t in TASKS[:len(ordine)]]
    stessi = ordine == attesi
    miss = {i: G - per_task[t] for i, t in enumerate(ordine) if per_task[t] < G}
    mancanti[fam] = miss
    print(f"{fam}: {len(pool['pool'])} fallimenti nel pool, {len(ordine)} compiti, ordine = rl_tasks(7): {stessi}")
    print(f"   punteggio rifatto con countdown.reward congelato: {dict(rifatti)} (atteso: tutti 0)")
    print(f"   compiti con risposte mancanti (indice: quante): {miss} -> totale {sum(miss.values())}")
    print(f"   stats del notebook: {pool['stats']}")
if len(mancanti) == 2:
    a, b = mancanti.values()
    print("compiti con mancanti in comune:", sorted(set(a) & set(b)), "| solo nel primo:", sorted(set(a) - set(b)),
          "| solo nel secondo:", sorted(set(b) - set(a)))
