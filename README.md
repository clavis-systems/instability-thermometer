# Can you tell before training how much GRPO runs will disagree? A pre-registered test

*Emanuele Rizzan, independent researcher, October 2026.*

> **Status (24 October 2026):** phase 1 is finished and **inconclusive**. Phase 1b (the long horizon) is finished:
> by its frozen rule the prediction **holds at steps 80 and 150**, but only with the measure that does not use the
> weights and with 6 runs, so I read it as a hint, not a confirmation. Phase 2 (other model families) is finished:
> two of the three new families stopped at a gate, and the one left, SmolLM2-1.7B, is **promoted at the edge** (R =
> 1.98 at step 10). But the thermometer's own repeat gave a prediction about 1.7 times as large for this family, so
> the prediction itself is quite uncertain (see "Phase 2").

## The question

"When Do Models Admit They Are Wrong? Failure Disclosure Is Unstable Under Reinforcement Learning" (Feng et al.,
Stanford University and Anthropic, [arXiv 2609.33220](https://arxiv.org/abs/2609.33220)) trains Qwen2.5-1.5B on
Countdown with GRPO and watches *failure disclosure*: how often a model says so when it could not solve the task.
Disclosure is not rewarded, and it ends up very different from run to run with the same recipe. Among the future
directions (Section 7), the authors ask how to "predict which auxiliary behaviors will be weakly constrained before
launching large training runs".

I tried one answer, an "instability thermometer", and I pre-registered the test.

## The idea

If disclosure is not rewarded, GRPO moves it mostly through the noise of the updates. So at step 0 I measure that
noise and simulate where it takes the model:

- I take K = 50 GRPO gradients at step 0, all from the same starting model.
- I replay Adam exactly over them: 400 simulated runs, each drawing its gradients from those 50, for 40 steps.
- I project every simulated run on the gradient of a disclosure measure. The spread across the 400 simulated runs is
  the **predicted** between-run standard deviation (SD).
- The disclosure measure (`D_seq`) is a self-normalised importance estimate on a fixed pool of 400 failures sampled
  at step 0. It has no sampling noise, and its numerical floor can be measured.

## The test (phase 1)

- **Model and task:** Qwen2.5-1.5B-Instruct with a LoRA warm start (1,000 examples: 865 solutions and 135 admitted
  failures on unreachable targets, the same split as the paper's warm start, Appendix A.3), my own Countdown with four
  numbers. Step-0 solve rate 4% (18 of 448
  answers in the step-0 sample).
- **Training:** Dr. GRPO without KL, 4 problems × 8 answers per step, reward = success only, LoRA, learning rate
  2e-5, **12 independent runs of 40 steps**, on Kaggle (2 × T4, float32).
- **Frozen before any data:** gates, thresholds and even the sentences of the verdict
  (`PREMESSA.md`, `CONGELATA.txt`).
- **Prediction before the runs:** the hashes of the prediction were committed to git, and the runs notebook checked
  the prediction against them before the first run (see "Provenance" for what each piece of evidence shows).
- **Rule:** R = measured SD / predicted SD at step 40.
  - "Promoted" if 0.5 ≤ R ≤ 2.
  - "Inconclusive" if the median effective sample size (ESS) of the measure is below 100, because then the measure
    itself is not reliable.

## The result: inconclusive

**At step 40, R = 1.99 (95% CI 1.41–3.38), but the median ESS was 15.** The importance weights had collapsed. On
average, the 5 heaviest of the 400 failures carried 45% of the weight. The four runs with the lowest ESS all had values
of 0.63 or more, against a mean of 0.45 (correlation between log ESS and the measure: −0.74).

**Not part of the pre-registered test**, computed after the data:

| step | measured SD | predicted SD | R | median ESS |
|---|---|---|---|---|
| 5 | 0.024 | 0.034 | 0.69 | 228 |
| 10 | 0.040 | 0.052 | 0.78 | 148 |
| 20 | 0.117 | 0.074 | 1.57 | 44 |
| 40 | 0.206 | 0.104 | 1.99 | 15 |

These numbers are descriptive, not a pre-registered confirmation. R is 0.69 at step 5 and 0.78 at step 10, while the
median ESS is still above 100. At steps 20 and 40 the ESS is below the pre-registered threshold, and R grows. One
possible explanation is that the measure degrades as the ESS falls. These data cannot tell whether the prediction
itself holds.

The ESS column is the true median over the 12 runs (independent recomputation). The frozen analysis uses the upper
median, which gives 15.7 at step 40 instead of 15.1.

**The other results:**

- **Linear functionals of the weights**, which have no measurement noise, at step 40: R between 0.80 and 1.71 for
  the eight functionals (the two measure gradients, five random directions, the false-claim gradient). That is the
  same order as predicted. Their geometric mean is 1.12, or 1.17 without the false-claim functional. They are not
  independent of each other, so this is not eight confirmations. Only one of them (R 1.71) has a 95% interval that
  excludes 1.
- **Growth over time (pre-registered): failed.** The measured log-log slope was 1.22 against 0.51 predicted. Most of
  that slope comes from steps where the measure had already collapsed.
- **Where the noise comes from (exploratory):** at step 0, the noise of the disclosure gradient comes 3.5 times more
  from the failed reasoning than from the window around the admission plus the tail. This is in line with the paper.
- **A second behaviour (exploratory),** claiming success after failing: inconclusive for the same ESS reason. The two
  predicted spreads were almost equal, so the thermometer could not rank them, and that ranking is not scored.
- **A check that does not use the weights (exploratory, written before the data by a separate session):** the
  disclosure rate among each run's own failures over steps 36–40 has a between-run SD of about 0.07, after removing
  binomial noise. The prediction for the fixed-pool measure at step 40 was 0.104, but the two are not the same
  quantity:
  - it measures a different population, because each run fails on different problems;
  - its noise estimate ignores that answers come in groups of 8 per problem.

  So I only report it (`risultati/q15_kaggle_B/esplorative/popolazione/ESITO.md`).

**A practical lesson.** If you use a fixed-pool importance measure like mine to track a behaviour during RL, in this
setup it degrades after about 20 steps. Report its ESS.

## Checks

- **Detector of disclosure:** 100/100 agreement with an independent labeller before the freeze, and 100/100 again at
  steps 36–40. Both samples are non-empty tails, which are all the warm-start sentence, so these checks test false
  positives only. I declared this before the data.
- **Detector of false claims (second behaviour):** 99/100 agreement before the freeze.
- **Two independent checks of the verdict, both "confirmed with reservations":**
  - Codex read everything (read-only, it could not run code).
  - A separate Claude agent with a clean context recomputed every number from the raw files with its own code
    (`risultati/q15_kaggle_B/controllo_ricalcolo/`). All numbers matched.
- **What the reviewers corrected in my reading:**
  - My frozen report says "the diffusion + Adam theory holds" for the linear functionals. That is too strong: "the
    same order as predicted for these functionals" is what the data say. The frozen report stays as it was written.
  - "This is not a verdict on the thermometer" is a hypothesis I wrote before the data, not a result.
  - The frozen code takes the upper median of the ESS (15.68) instead of the true median (15.13). This changes nothing.

## Phase 1b, the long horizon: it holds by the frozen rule, but it is only a hint

Phase 1b was frozen before any phase-1 result (`PREMESSA_FASE1B.md`, `FASE1B_CONGELATA.txt`): 6 new runs of 150 steps
(seeds 201-206), the same step-0 prediction extended to 150 steps, tested at steps 80 and 150. Because the importance
weights could collapse again, the plan added a second measure that does not use them, `D_samp`: the disclosure rate
among each run's own finished failures in the 20 steps before the tested step, with its noise estimated from the data
and removed. The rule: if the median ESS of `D_seq` is below 100, the verdict at that step uses `D_samp`.

The frozen report (`risultati/q15_kaggle_C/q15/RAPPORTO_FASE1B.md`, in Italian) gives the noise-corrected spread of
`D_samp` across the 6 runs: 0.65 times the predicted one at step 80 and 1.49 times at step 150. So, by the rule, "the
step-0 prediction holds up to 150 steps".

| step | predicted SD | `D_samp` SD, noise removed | R | median ESS of `D_seq` | verdict |
|---|---|---|---|---|---|
| 80 | 0.145 | 0.094 | 0.65 | 13 | promoted, with `D_samp` |
| 150 | 0.188 | 0.280 | 1.49 | 13 | promoted, with `D_samp` |

A separate Claude agent with a clean context recomputed everything from the raw files with its own code: same
numbers, same verdicts, "confirmed with reservations" (`risultati/q15_kaggle_C/controllo_ricalcolo/`). I share its
reservations, and they are why I call this a hint:

- **Only 6 runs.** The 95% interval of R is 0.40-1.59 at step 80 and 0.93-3.65 at step 150. At step 80, leaving out
  one run (seed 205 or 206) would turn the verdict into "hint". At step 150 one run (seed 204) drives most of the
  spread. Here "holds" means "not refuted by a large error", which is all the plan said 6 runs can show.
- **The verdict rests only on `D_samp`.** `D_seq` collapsed again (median ESS 13 at both steps), so it is
  inconclusive. The two measures do not see the same thing: they move together until about step 60 and then separate.
  At step 150 `D_seq` is close to 0 in every run, while `D_samp` averages 0.39. `D_samp` also looks at fewer and fewer
  failures: finished failures per step fall from 29 at step 20 to about 10 at step 150. And the prediction was built
  for `D_seq`, not for `D_samp`.
- **The growth does not follow the prediction** (computed after the data, not part of the rule). The spread of
  `D_samp` grows about 10 times from step 40 to step 150, against 1.7 times predicted (log-log slope about 1.7 against
  0.4). At step 40, R for `D_samp` would be 0.24. The two tested steps fall where the measured curve crosses the
  predicted band.

So, by the frozen rule, the step-0 prediction gets the size of the spread about right at steps 80 and 150. In the
post-data look at `D_samp`, the spread grows much faster in these 6 runs than predicted. With 6 runs, all of this is a
hint.

## Phase 2, other families: only one could be tested

**The plan.** Phase 2 asks the same question of three new families: Qwen2.5-0.5B, OLMo-2-1B and SmolLM2-1.7B. It was
planned before phase 1 as the same test at step 40. **After** seeing phase 1 and **before** any phase-2 data, I moved
it to step 10, with step 5 as fallback, because that is where the measure still works (`PREMESSA_FASE2_MODIFICA*.md`).
Runs last 20 steps. I froze this on 22 October, after phase 1b and before any phase-2 run (`FASE2_CONGELATA.txt`).
Qwen2.5-1.5B is out of the verdict, because I had already seen its step-10 value; it stays as the reference for the
order of the families. I call the test confirmatory on the three new families. Codex argued it should be called
exploratory; its objection is written next to my answer.

**Two families stopped at gate G1** (step-0 solve rate between 2% and 60%):
- Qwen2.5-0.5B and OLMo-2-1B both solved 8 of 416 answers (1.9%) at step 0 (`risultati/q05_kaggle_A2/`,
  `risultati/olmo1_kaggle_A2/`). I checked that the identical count is a coincidence and not a counting bug
  (`risultati/fase2_verifica_g1/`).
- The one fix allowed by the rules, written before the data, was Countdown with 3 numbers, once
  (`PREMESSA_FASE2_3NUMERI.md`, frozen in `FASE2_3NUMERI_CONGELATA.txt` before any 3-number data). Both families then
  solved 0 of 416 (`risultati/q05_3n_kaggle_A2/`, `risultati/olmo1_3n_kaggle_A2/`). For Qwen an independent check
  found 0 right answers in 400, and every one of the 50 tasks can be solved. So both families are out of phase 2.
  The warm start stayed the one trained on 4-number tasks, so the 3-number tasks are a different regime, reported
  apart and never compared directly with the 4-number results.
- I had assumed that 3 numbers would be easier. For these two models it was the opposite (0 of 416 against 8 of 416).
  My guess is that with fewer numbers fewer random tries hit the target, but I did not test it.

**SmolLM2-1.7B passed every gate.** One deviation was declared before its data (D2, `DEVIAZIONI_FASE2.md`): the
download of the model ran out of disk, because the model's repository weighs 22.7 GB, almost all of it in an `onnx/`
folder, so the notebooks skip that folder. The model file and its SHA-256 are the same.

12 runs of 20 steps (`risultati/fase2_kaggle_A/`, `risultati/fase2_kaggle_B/`). The frozen analysis
(`fase2_corta.py`) ran in the notebook, and I redid it at home on the same files: the result is identical. A separate
AI agent then recomputed these numbers with its own code and got the same values; only the upper ends of the
intervals come out 0.01 lower with an exact chi-square factor (`risultati/fase2_confronto/ricalcolo_indipendente/`).

| step | measured SD | predicted SD | R | 95% CI of R | median ESS |
|---|---|---|---|---|---|
| 5 | 0.038 | 0.018 | 2.07 | 1.47–3.52 | 286 |
| **10 (the test)** | **0.052** | **0.026** | **1.98** | **1.40–3.37** | **181** |
| 20 (descriptive) | 0.072 | 0.039 | 1.86 | 1.31–3.16 | 119 |

- **Verdict at step 10: promoted** (0.5 ≤ R ≤ 2), but at the edge. At all three steps the measured spread is about
  twice the predicted one, and the interval goes past 2. With the noise floor at 1e-5 the verdict is the same.
- **The thermometer's own repeat.** The plan has a second thermometer as a repeatability check (seed 2, K = 25
  instead of 50; `thermo_s2.json` next to `thermo_s1.json`). For Qwen2.5-1.5B the two agree (0.052 and 0.046 at step
  10). For SmolLM2 they do not: the repeat predicts 0.046 instead of 0.026, about 1.7 times as much, which would give
  R = 1.14. The frozen rule uses the first thermometer, so the verdict stays as above, but for this family the
  prediction itself is quite uncertain.
- **Second behaviour (false claims of success, H6):** R = 1.80, also promoted. The ratio of the two spreads (false
  claims over disclosure) was predicted at 1.00 and measured at 0.91. The prediction falls in the frozen parity zone
  (0.9–1.1), so which behaviour moves more is not scored.
- **Count over the three new families:** **1 of 3 promoted**. By the frozen rule a missing family counts as not
  promoted.
- **Order against Qwen2.5-1.5B at step 10 (H4b, descriptive):** the thermometer predicted a smaller spread for
  SmolLM2 than for Qwen2.5-1.5B (0.026 against 0.052). The measured spread was larger (0.052 against 0.040). So by
  the frozen rule the order is wrong: 0 of 1. This is weak: the measured spreads overlap (95% intervals 0.037–0.089
  and 0.028–0.069; their ratio is 1.30, 95% interval 0.70–2.42, in
  `risultati/fase2_confronto/ricalcolo_indipendente/ricalcolo_out.txt`), and with the repeat for both families the
  two predictions are level (0.046 and 0.046). With two families a correlation means nothing, and the frozen code
  does not give one (`risultati/fase2_confronto/`).
- **The three factors of the prediction (descriptive),** Qwen2.5-1.5B against SmolLM2: exposure (failed, finished
  answers per batch at step 0) 30.0 and 30.3; balance (D × (1 − D) at step 0) 0.25 and 0.24; coupling (the share of
  the predicted spread that comes from the failed reasoning) 0.45 and 0.34.
- **Detector check (G5):** an independent labeller, a separate AI agent that did not see the detector, labelled 100
  non-empty tails from steps 16–20. They agree on 100 of 100. But all 100 tails were admissions ("I could not reach
  N exactly..."), so this check says nothing about tails that do not admit.

**How I read it.** In the one new family I could test, the frozen prediction stays within the factor of 2, but it is
about 2 times too small and, by the frozen rule, puts the two families in the wrong order (weakly, see above). The
thermometer's own repeat would have predicted about 1.7 times as much, close to what happened. So the main thing this
phase shows is that, for SmolLM2, the step-0 prediction is itself quite uncertain. One family and 12 runs are a hint,
not a result. This test does not show that the thermometer can rank families by how unstable they are.

## What comes next

Phase 2 was the last phase I planned. No new runs are planned yet. The obvious next test is in "What I did not do".

## What I did not do

- 40 steps in phase 1, at most 150 in phase 1b, 20 in phase 2 (tested at step 10); the paper's main runs are 300
  steps.
- LoRA with learning rate 2e-5 (the paper uses 1e-6), my own Countdown and warm start, one model family in phase 1,
  and only one new family that could be tested in phase 2.
- 12 runs in phase 1: the test catches errors of about 3 times or more, and cannot certify a factor of 2. 6 runs in
  phase 1b: it has less power, and catches an error of 3 times only about 42% of the time.
- No anchoring experiment. The paper's failure-conditional reference anchoring is the obvious next test: the same
  step-0 measurement could predict how much the anchoring should shrink the spread.

## How this was made

- **This is my project.** I asked the question and set the rules: write everything down before the data, and have
  every result checked independently.
- **How the work was split.** I directed the work and checked the results. An AI coding agent (Claude Code) wrote
  the code, ran the experiments and drafted this text. Codex, another AI agent, reviewed plans, code and results. I
  decide what is published and I answer for it.
- **The internal records are in Italian**, because that is how I work. `PREMESSA` means pre-registration.

## Provenance

The work was done in a private repository. Its commits carry my personal e-mail address, which I do not want to
publish, so I publish the files and the list of commits (`PROVENANCE.md`) instead of the history.

- **Content hashes:** `CONGELATA.txt`, `FASE1B_CONGELATA.txt`, `FASE2_CONGELATA.txt`, `FASE2_3NUMERI_CONGELATA.txt`
  and `kaggle/MANIFEST_*.txt` give
  the SHA-256 of every frozen file. Anyone can check them with `sha256sum`. The files keep their original paths so that the hashes still
  apply (on Windows, check the files with LF line endings, as in git).
- **Two frozen files got dated additions later**: `PREMESSA.md` and `kaggle/crea_notebook_fase1.py` (deviation 1, a
  technical one, appended on 20 October and written at the end of `PREMESSA.md`). Their frozen versions are in
  `congelati/979d472/`, and those match the hashes in `CONGELATA.txt`.
- **Not in this repository:** the two large prediction files `vettori_s1.pt` and `gradienti_s1.pt`, for phase 1 and
  for SmolLM2 in phase 2. Their hashes are in `risultati/q15_kaggle/q15/previsione_sha256.txt` and
  `risultati/fase2_kaggle_A/smol17/previsione_sha256.txt`.
- **Order of events:** the runs notebook as it was launched (`kaggle/push_B/fase1_B_corse.ipynb`) contains the hashes
  of the prediction. The notebook could only check the prediction against hashes that existed before it ran.
- **Third-party timestamps:** the Kaggle notebooks (prediction, runs, phase 1b) keep their versions with server
  times: [prediction](https://www.kaggle.com/code/emanuelerizzan/termometro-fase-1-previsione),
  [runs](https://www.kaggle.com/code/emanuelerizzan/termometro-fase-1-corse) and
  [phase 1b](https://www.kaggle.com/code/emanuelerizzan/termometro-fase-1b). The frozen code they read is the Kaggle dataset
  [termometro-codice-congelato](https://www.kaggle.com/datasets/emanuelerizzan/termometro-codice-congelato), and for
  phase 1b also [termometro-fase1b-codice](https://www.kaggle.com/datasets/emanuelerizzan/termometro-fase1b-codice).
- **Phase 2:** the Kaggle notebooks keep their versions with server times too:
  [Qwen2.5-0.5B](https://www.kaggle.com/code/emanuelerizzan/termometro-fase-2-q05-previsione),
  [OLMo-2-1B](https://www.kaggle.com/code/emanuelerizzan/termometro-fase-2-olmo1-previsione),
  [SmolLM2 prediction](https://www.kaggle.com/code/emanuelerizzan/termometro-fase-2-smol17-previsione),
  [SmolLM2 runs](https://www.kaggle.com/code/emanuelerizzan/termometro-fase-2-smol17-corse), and the 3-number fix for
  [Qwen2.5-0.5B](https://www.kaggle.com/code/emanuelerizzan/termometro-fase-2-q05-3n-previsione) and
  [OLMo-2-1B](https://www.kaggle.com/code/emanuelerizzan/termometro-fase-2-olmo1-3n-previsione). All six read
  termometro-codice-congelato and
  [termometro-partenze-fase-2](https://www.kaggle.com/datasets/emanuelerizzan/termometro-partenze-fase-2) (the warm
  starts of the three families); the SmolLM2 runs also read
  [termometro-fase2-codice](https://www.kaggle.com/datasets/emanuelerizzan/termometro-fase2-codice), and the two
  3-number notebooks also read
  [termometro-fase2-3numeri](https://www.kaggle.com/datasets/emanuelerizzan/termometro-fase2-3numeri). The hashes of the SmolLM2 prediction were committed to git before the runs, and the
  runs notebook as it was launched (`kaggle/push_B2_smol17/fase2_B_smol17.ipynb`) contains them.

## Files

| Path | What is inside |
|---|---|
| `PREMESSA*.md`, `CONGELATA.txt`, `FASE1B_CONGELATA.txt`, `FASE2_CONGELATA.txt`, `FASE2_3NUMERI_CONGELATA.txt`, `SPEC_*.md` | the pre-registrations, freeze records and labelling instructions |
| `DEVIAZIONI_FASE2.md` | the deviation declared in phase 2 (D2, the model download) |
| `termometro.py`, `countdown.py` | the thermometer, the GRPO runs, the task (with an exact solver) |
| `analisi_fase1.py`, `rapporto_fase1.py`, `fase1b.py`, `analisi_fase2.py`, `fase2_corta.py`, `fase2_3numeri.py` | the analyses and report sentences, written before the data |
| `prova_*.py`, `taratura/`, `confronta_g5.py`, `estrai_*.py` | smoke tests, calibration, detector checks |
| `kaggle/` | the notebook generators, the notebooks, the notebooks as launched (`push_*`), the manifests; never launched: the `B` notebooks of the two families stopped at G1, and the `*_smol17_3n` notebooks (SmolLM2 passed G1); for SmolLM2 the notebooks that ran are the `_d2` copies (deviation D2: the first launch of the frozen one stopped on a full disk) |
| `risultati/q15_kaggle/` | the prediction (notebook A) |
| `risultati/q15_kaggle_B/` | the 12 runs, the frozen analysis and report, the step-40 detector check, the two independent checks, the exploratory analysis |
| `risultati/q15_kaggle_C/` | phase 1b: the 6 runs, the extended prediction, the frozen analysis and report, the independent recomputation |
| `risultati/q05_kaggle_A2/`, `risultati/olmo1_kaggle_A2/`, `risultati/fase2_verifica_g1/` | phase 2: the two families stopped at gate G1, and the check of their identical count |
| `risultati/q05_3n_kaggle_A2/`, `risultati/olmo1_3n_kaggle_A2/` | phase 2 with 3 numbers: the same two families, stopped at G1 again |
| `risultati/fase2_kaggle_A/`, `risultati/fase2_kaggle_B/` | phase 2, SmolLM2-1.7B: the prediction, the 12 runs, the frozen analysis, the detector check (G5) |
| `risultati/fase2_confronto/` | phase 2: the count over the three new families and the order against Qwen2.5-1.5B |
| `risultati/pilota*/`, `risultati/taratura/` | the checks made before the freeze |
| `revisioni/` | Codex's reviews of the phase-2 plan, the 3-number fix, the code and this README, with their prompts (Italian) |
| `sft_countdown.jsonl` | the 1,000 warm-start examples |
| `congelati/979d472/` | the frozen versions of the two files that got dated additions later |
| `PROVENANCE.md`, `SHA256SUMS.txt` | the list of private commits, and the SHA-256 of every file here |

**To redo the phase-1 analysis** (Python 3, standard library only):

```
mkdir check && cp risultati/q15_kaggle/q15/thermo_s*.json risultati/q15_kaggle_B/q15/run_s*.json check/
python analisi_fase1.py check
python rapporto_fase1.py check
```

This gives the same `analisi_fase1.json` and `RAPPORTO_FASE1.md` as in `risultati/q15_kaggle_B/q15/`. I checked this
on 21 October; on Windows only the line endings differ.

**To redo the phase-1b analysis:**

```
mkdir check1b && cp risultati/q15_kaggle_C/q15/run_s20*.json check1b/
python fase1b.py analyze risultati/q15_kaggle/q15 risultati/q15_kaggle_C/q15/previsione_1b.json check1b
```

This gives the same `analisi_fase1b.json` and `RAPPORTO_FASE1B.md` as in `risultati/q15_kaggle_C/q15/` (checked on
22 October).

**To redo the phase-2 analysis and the comparison of the families:**

```
mkdir -p check2/q15 check2/smol17
python fase2_corta.py analyze risultati/fase2_kaggle_B/smol17/smol17 risultati/fase2_kaggle_B/smol17/smol17 check2/smol17/analisi_fase2_corta.json
python fase2_corta.py analyze risultati/q15_kaggle/q15 risultati/q15_kaggle_B/q15 check2/q15/analisi_fase2_corta.json
python fase2_corta.py families check2
```

The first command gives the same `analisi_fase2_corta.json` as in `risultati/fase2_kaggle_B/smol17/smol17/`, and the
last one the same as in `risultati/fase2_confronto/`.

**To redo the runs** you need a GPU and the warm start. The LoRA weights of my warm start are not in this repository,
but they are on Kaggle: [termometro-partenza-q15](https://www.kaggle.com/datasets/emanuelerizzan/termometro-partenza-q15) (the notebooks
check their SHA-256). `termometro.py sft` trains a new one from `sft_countdown.jsonl`, but mine was trained at home in
bf16 and a new one will not be bit-identical.

## License

Apache-2.0.
