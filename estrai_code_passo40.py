"""Detector agreement AFTER training (report only, no correction): 100 non-empty tails drawn with seed 4040 from the
failures of steps 36-40 of every run (run_s*_code.jsonl, written by `termometro.py run`). Same files as the G5 pilot,
so `confronta_g5.py OUT_DIR` gives the agreement with an independent labeller (instructions: SPEC_G5_ETICHETTE.md).
Written before any run (20/10/2026).

usage: python estrai_code_passo40.py RUN_DIR OUT_DIR
"""
import glob
import json
import os
import random
import sys

N = 100
SEED = 4040


def main(run_dir, out_dir):
    items = []
    for f in sorted(glob.glob(os.path.join(run_dir, "run_s*_code.jsonl"))):
        seed = os.path.basename(f)[len("run_s"):-len("_code.jsonl")]
        for k, line in enumerate(open(f, encoding="utf-8")):
            if line.strip():
                items.append({"origin": f"s{seed}_{k}", **json.loads(line)})
    if len(items) < N:
        print(f"only {len(items)} non-empty tails: all of them are used")
    pick = random.Random(SEED).sample(items, min(N, len(items)))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "code_cieche.jsonl"), "w", encoding="utf-8") as f:
        for i, it in enumerate(pick):
            f.write(json.dumps({"id": i, "answer_line": it["answer_line"], "tail": it["tail"]}, ensure_ascii=False) + "\n")
    json.dump([{"id": i, "detector": it["detector"]} for i, it in enumerate(pick)],
              open(os.path.join(out_dir, "etichette_rilevatore.json"), "w"))
    json.dump([{"id": i, "origin": it["origin"], "step": it["step"]} for i, it in enumerate(pick)],
              open(os.path.join(out_dir, "origine.json"), "w"))
    print(f"{len(pick)} tails from {len(items)}; detector says 'admits' in {sum(it['detector'] for it in pick)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
