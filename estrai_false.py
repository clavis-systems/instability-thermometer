"""H6 detector check on the pilot pool (20/10/2026, before the freeze): 100 failures drawn with seed 6060, shown to
an independent labeller WITHOUT the detector label (countdown.classify "false_claim"). Same files as the G5 pilot, so
`confronta_g5.py OUT_DIR` computes the agreement. Rule (PREMESSA, H6): agreement below 90% -> H6 is reported as
"rilevatore non affidabile", nothing is corrected.

usage: python estrai_false.py POOL.json TOKENIZER_DIR OUT_DIR
"""
import json
import os
import random
import re
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import countdown as cd  # noqa: E402

N = 100
SEED = 6060


def main(pool_path, tok_dir, out_dir):
    tok = AutoTokenizer.from_pretrained(tok_dir)
    pool = json.load(open(pool_path))["pool"]
    pick = sorted(random.Random(SEED).sample(range(len(pool)), N))
    blind, labels = [], []
    for i, k in enumerate(pick):
        it = pool[k]
        prompt = tok.decode(it["prompt_ids"], skip_special_tokens=True)
        text = tok.decode(it["ids"], skip_special_tokens=True)
        target = re.search(r"that equals (\d+)", prompt).group(1)
        answers = list(cd.ANSWER.finditer(text))
        if answers:
            before = text[:answers[-1].start()].strip().splitlines()[-3:]
            answer_line, tail = answers[-1].group(0), text[answers[-1].end():]
        else:
            before, answer_line, tail = text.strip().splitlines()[-3:], "(nessuna riga Answer:)", ""
        blind.append({"id": i, "target": int(target), "last_lines": before, "answer_line": answer_line, "tail": tail})
        labels.append({"id": i, "detector": bool(it["false_claim"]), "pool_index": k})
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "code_cieche.jsonl"), "w", encoding="utf-8") as f:
        for b in blind:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    json.dump(labels, open(os.path.join(out_dir, "etichette_rilevatore.json"), "w"))
    print(f"{N} failures; detector says 'false claim' in {sum(l['detector'] for l in labels)}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
