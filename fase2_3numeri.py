"""Phase 2, families stopped at gate G1 (PREMESSA_FASE2_3NUMERI.md, 22/10/2026): Countdown with 3 numbers.

The one fix allowed by the frozen rules for G1 below threshold ("compiti con 3 numeri"), decided by Emanuele on 22/10 for q05 and olmo1 (and smol17 if it stops at G1).
It changes ONLY countdown.make_instance: 3 numbers instead of 4, same range (1-50), same targets (10-200, chosen among
the reachable ones for solvable tasks). Then it runs the FROZEN termometro.py unchanged, with the same command line, so
every task it generates (step-0 sample, failure pool, thermometer gradients, runs) has 3 numbers. The warm start stays
the one of 20/10 of each family (trained on 4-number data): retraining it would be a second change.

  python fase2_3numeri.py <any termometro.py command and arguments>
  python fase2_3numeri.py test      checks on CPU, without torch
"""
import os
import random
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# the frozen code: this folder at home; on Kaggle the notebook passes the frozen dataset folder in TERMO_CODICE
CODICE = os.environ.get("TERMO_CODICE", HERE)
sys.path.insert(0, CODICE)
import countdown  # noqa: E402  (the frozen module; termometro.py will get this same object from sys.modules)

N_NUMBERS = 3
_FROZEN_MAKE_INSTANCE = countdown.make_instance


def make_instance_3(rng: random.Random, solvable: bool):
    """Same as the frozen countdown.make_instance, with range(N_NUMBERS) instead of range(4)."""
    while True:
        nums = [rng.randint(1, 50) for _ in range(N_NUMBERS)]
        reach = countdown.reachable(nums)
        pool = [t for t in range(10, 201) if (t in reach) == solvable]
        if pool:
            t = rng.choice(pool)
            return nums, t, reach.get(t)


countdown.make_instance = make_instance_3


def test():
    # 1. the frozen function and the new one differ only in the number of numbers
    tasks3 = countdown.rl_tasks(10_000 + 1, 300)
    assert all(len(t["nums"]) == 3 for t in tasks3), "not all tasks have 3 numbers"
    assert all(1 <= n <= 50 for t in tasks3 for n in t["nums"])
    assert all(10 <= t["target"] <= 200 for t in tasks3)
    # 2. every RL task is solvable, and its known solution gets reward 1 from the frozen scorer
    rng = random.Random(5)
    ok = 0
    for _ in range(100):
        nums, t, sol = make_instance_3(rng, True)
        assert sol is not None and t in countdown.reachable(nums)
        ok += countdown.reward(sol, nums, t)
    assert ok == 100, f"frozen reward accepts only {ok}/100 known solutions"
    # 3. unsolvable instances exist too (used only by sft_data, which no family retrains)
    nums, t, sol = make_instance_3(random.Random(6), False)
    assert sol is None and t not in countdown.reachable(nums)
    # 4. the prompt names exactly the 3 numbers
    assert all(str(n) in t["prompt"] for t in tasks3[:50] for n in t["nums"])
    # 5. with the frozen function the same seed gives 4 numbers (the patch is what changes it)
    countdown.make_instance = _FROZEN_MAKE_INSTANCE
    assert all(len(t["nums"]) == 4 for t in countdown.rl_tasks(10_000 + 1, 3))
    countdown.make_instance = make_instance_3
    # 6. how many targets are reachable with 3 numbers vs 4 (descriptive)
    r = random.Random(7)
    def frac(k, n):
        tot = 0
        for _ in range(n):
            ns = [r.randint(1, 50) for _ in range(k)]
            tot += sum(1 for x in range(10, 201) if x in countdown.reachable(ns)) / 191
        return tot / n
    print(f"OK: 300 tasks with 3 numbers, 100/100 known solutions accepted; reachable targets 10-200: "
          f"3 numbers {frac(3, 200):.1%}, 4 numbers {frac(4, 5):.1%} (5 draws only)")


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        test()
    else:
        # la partenza resta quella a 4 numeri: con 3 numeri anche sft_data cambierebbe, quindi qui l'SFT non si fa
        assert sys.argv[1:2] != ["sft"], "fase2_3numeri.py non riaddestra la partenza (PREMESSA_FASE2_3NUMERI.md)"
        sys.argv[0] = os.path.join(CODICE, "termometro.py")
        runpy.run_path(sys.argv[0], run_name="__main__")
