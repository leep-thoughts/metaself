# metaself: state of the project (2026-11-13)

Meta-evaluation of **self-preference bias (SPB) measures** for LLM judges: do published measures
respond only to a judge's true self-lean s, or also to confounds (answer quality, judge accuracy,
difficulty, pair effects, …)? Two tools: a **simulator** with known knobs, and **real data**
(public releases + our own runs on open models) to check what the simulator gets right.

Companion docs: `CLAUDE.md` (working context), `docs/CONFOUND_PIPELINE.md` (plan item 5 design),
`infer/README.md` (how to run inference), `PLAN.md` (original synthetic-only plan, historical).

---

## 1. Framework

### Knobs (simulator)
| Knob | Meaning | Where |
|---|---|---|
| p | how often the judge's own answer is truly better (pairwise) / passes (rubric) | `synth/oracle.py`, `synth/difficulty.py`, `synth/rubric.py` |
| a | judge accuracy (base, at s = 0) | same |
| s | true self-lean (log-odds toward own answer) — the target | same |
| β | difficulty coupling: judge evaluates better where it solves the item | `synth/difficulty.py` |
| γ | **generator-pair lean**: judge favours the stronger generator's answer, b = γ·logit(p_pair) | `synth/difficulty.py` (`gamma`, `pair_p`) |
| ρ | **consensus errors**: shares of correct answers that look wrong / wrong answers that look right to all judges | `synth/difficulty.py` (`rho`) |
| h, leniency, q | rubric: halo toward strong generators, "met"-leniency, others' pass rate | `synth/rubric.py` |
| σ_e | score noise (continuous scores, Yang) | `synth/continuous.py` |

### Requirements for a good measure (R1–R4)
| | Requirement | Test |
|---|---|---|
| R1 | insensitive to confounds | dM/dp, dM/da ≈ 0 at s = 0 |
| R2 | zero without bias | M(s = 0) ≈ 0 |
| R3 | responds to bias | dM/ds > 0 |
| R4 | same response everywhere | spread of dM/ds over the (p, a) grid |

### Measures
| Setting | Measures | File |
|---|---|---|
| Pairwise | Panickssery (win rate − 0.5), equal opportunity (TPR+FPR−1), HSPP (FPR), Roytburg (matched proxy), win-rate gap, Yang et al. 2026 | `measures/*.py` |
| Rubric | score gap, calibrated gap, MRSPB, HSPP-Rub (diff and ratio), EO-Rub, matched_fail, matched_both (ours) | `measures/rubric.py` |

---

## 2. Data

| Source | What | Use | Location |
|---|---|---|---|
| Chen et al. 2025 | 11 judges × 7 opponents, MATH500 / MMLU / MBPP+, answers + oracle + verdicts | pairwise real data | `data/llm-sp` |
| Roytburg et al. 2026 | judge-swap release: J judging proxies K vs R (null pairs) | null / injection tests, slopes | `data/roytburg` |
| Yang et al. 2026 | equal-quality pair measure (paper only) | measure + our replication | `measures/yang.py`, `infer/yang_data.py` |
| Pombal et al. 2026 | rubric SPB: IFEval conversion, HealthBench subset, grader prompts | rubric setting | `data/pombal-simple-evals` |
| **Our runs** | 14 models (8 on Aleph, 6 self-hosted) as generators and judges on IFEval (541 prompts, 834 criteria) and HealthBench (100 prompts, 1,157 criteria); β-calibration pairs; Yang quality scores | real rubric data, calibration | `infer/data/` (git-ignored) |

Models: Aleph — gemma-4-31b, gemma-4-26b-a4b, qwen38-27b, qwen35-122b, gpt-oss-120b, gpt-oss-20b,
command-r-7b, aya-expanse-8b (model defaults, reasoning on, as Pombal). GPU (vLLM) — Chen's judges
qwen-2.5-{3,7,14,32,72}b, llama-3.x-{3,8,70}b, gemma-2-{9,27}b; OLMo-2-32B and Command-R+ as Yang raters.

---

## 3. Findings

### 3.1 Simulator (pairwise)
- With only p, a, s: EO and Roytburg pass R1–R2; Panickssery and win-rate gap move with p; HSPP moves with a.
- Adding difficulty β: EO picks up a p-slope (items J wins are easier → judged better); Roytburg stays exactly 0 at every β.
- Yang et al.: main leak is score noise σ_e in the equal-quality filter (`yang_sweep.py`).

### 3.2 Real pairwise data: why measures move with p and accuracy
Real slopes (grid reweighting, median over judges, `real/slope_gap.py`): EO 0.68 (p) / 1.15 (a);
HSPP 0.79 / 0.06; Panickssery 0.18 / 1.01; Roytburg −0.13 / −0.07. The base simulator predicted ≈ 0.

| Candidate cause | Test | Verdict |
|---|---|---|
| Difficulty β | `real/beta_fit.py`: 11 judges grade other models' pairs on problems they solve vs fail | **Real but too small**: median β 0.79 (Qwen-2.5 0.9–3.0, survives difficulty control; Gemma-2 mostly "easy items"; Llama ≈ 0). Even β = 3 gives EO p-slope 0.12 vs 0.68 |
| **Generator-pair lean** γ | `real/pair_check.py`: remove each pair's lean from real votes, rerun the grid | **Explains the p-slopes**: EO 0.68 → −0.04, HSPP 0.79 → 0.18, Pan 0.18 → −0.05. Replacing it by γ·logit(p) (γ = 0.48) restores them (EO 0.77). Lean doesn't change judging strength (L vs p ≈ 0) |
| Item-level shared lean | `real/item_check.py`: shuffle / remove shared item residuals | Residuals strongly shared (corr 0.61) but **not** the accuracy-slope cause (EO-a stays 1.26–1.83) |
| **Consensus errors** ρ | confidence quartiles + cross-judge consensus | **Explains the accuracy slope**: higher confidence raises TPR (.56 → .73) but FPR stays .46, because 41% of confidently judged wrong answers are confidently accepted by all judges. A sharper judge rejects detectable errors harder and accepts convincing ones harder |
| Shared errors, ties, family affinity | `real/slope_gap.py` | ruled out / small |

Simulator with β + γ + ρ, slopes computed like the real grid (`synth/pair_mixture.py`):

| Slope | Real | β only | + γ | + ρ | β + γ + ρ |
|---|---|---|---|---|---|
| EO, p | 0.68 | 0.10 | 0.89 | −0.06 | **0.51** |
| EO, a | 1.15 | −0.08 | −0.52 | 0.83 | **0.53** |
| HSPP, p | 0.79 | 0.17 | 1.08 | 0.05 | **0.77** |
| HSPP, a | 0.06 | −1.07 | −1.41 | −0.33 | −0.58 |
| Pan, p / a | 0.18 / 1.01 | 0.21 / 0.58 | 0.45 / 0.44 | 0.05 / 0.87 | **0.24 / 0.77** |

→ Signs and most magnitudes now match; HSPP's accuracy slope is still off. Roytburg's matched
measure stays blind to γ and ρ in the simulator (its proxy shares the item and pair strength).

### 3.3 Rubric setting
Simulator (`rubric_sweep.py`): score gap always confounded by p (slope 0.70); β confounds HSPP-Rub
(0.19) and its ratio (0.81) but not the matched measures; halo h confounds everything, including
matched (0.46–0.55) → needs a second judge.

IFEval, 14 judges × 14 generators, 162k verdicts (`real/rubric_real.py`, 95% bootstrap CI over prompts):

| Judge | p own | q others | HSPP-Rub | matched_fail | Halo (third parties) | **DiD (halo-controlled)** |
|---|---|---|---|---|---|---|
| qwen38-27b | .94 | .85 | .42 | .33 [.17, .45] | .17 [.10, .25] | **.16 [.05, .26]** |
| gpt-oss-20b | .87 | .86 | .18 | .12 [.04, .21] | .03 | **.09 [.01, .17]** |
| gpt-oss-120b | .91 | .86 | .16 | .13 [.07, .29] | .05 | **.08 [.01, .16]** |
| qwen35-122b | .94 | .85 | .41 | .23 [.06, .34] | .16 [.10, .23] | .08 [−.02, .18] |
| llama-3.1-70b | .88 | .86 | .07 | .05 | −.01 | **.06 [.00, .12]** |
| qwen-2.5-32b | .85 | .85 | .15 | .08 [.02, .12] | .02 | **.06 [.02, .11]** |
| qwen-2.5-7b | .79 | .85 | .11 | .05 | −.01 | **.05 [.02, .09]** |
| gemma-4-26b / 31b | .97 / .96 | .85 | .28 / .30 | .12 / .10 | .08 / .09 | .04 / .01 (CI incl. 0) |
| gemma-2-27b / 9b, llama-3.1-8b | | | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 |
| command-r-7b, aya-8b | | | ≈ 0 | ≈ 0 | −.06 / −.03 | .05 / .03 (spurious: these judges ignore quality, see below) |

- **Self-preference survives the halo control** for Qwen3-27B, gpt-oss (both sizes), Qwen-2.5, Llama-70B (small).
- **Gemma-4's apparent bias is halo** (everyone favours its failed answers); half of Qwen3.x's is halo.
- DiD assumes third parties react to halo like J does; judges that ignore quality (Aya says "met" to all;
  Llama-3.1-8B accuracy 0.52) get spurious DiD → pipeline stage S5 (sensitivity weighting).
- Judging quality: reasoning models 0.89–0.93 accuracy; Llama-3.1-8B is far too strict (leniency −0.41).

### 3.4 Yang et al. on real data
Quality scores done: gpt-oss-120b (AUC 0.94 / 0.76 / 0.87 on MATH / MMLU / MBPP+), OLMo-2-32B (weak:
0.73 / 0.58 / 0.67). Command-R+ running (2,500 / 8,100 before a vLLM crash; fixed, resumed).

---

## 4. Status by plan item

| Plan item | Status | Done | Open |
|---|---|---|---|
| 1. Address difficulty (toy sim) | ✅ largely done | β model + calibration; γ and ρ found on real data, added to the sim; sim reproduces most real slopes | HSPP accuracy slope; ρ measured on confident items only |
| 2. New measures | 🟡 | Yang implemented + swept; rubric matched measures; halo-controlled DiD | Yang on real data (waits Command-R+); pairwise DiD; our own estimator (pipeline S7) |
| 3. Rubric settings | 🟡 | sim; IFEval complete with 14 judges + halo control | HealthBench running (Aleph generation, job 2325677); GPU models on HealthBench (user submits); committee labels |
| 4. Concrete suggestions | ⚪ | candidate takeaways below | write up after the pipeline ablation (V5) |
| 5. Dataset / pipeline | 🟠 design | `docs/CONFOUND_PIPELINE.md`: stages S1–S8, validation V1–V7 | decisions in §5 of that doc; build |

### Candidate takeaways (item 4)
1. Don't compare a judge's self-judgments across opponents of different strength: judges lean toward
   the stronger generator (γ ≈ 0.5 per logit), which alone produces EO/HSPP's p-slopes.
2. Use matched comparisons (same item, same correctness, similar generator strength): Roytburg-style
   proxies are blind to the pair lean and to consensus errors.
3. Raw "accuracy" is not one knob: confident judges are also confidently wrong on convincing errors
   (~40% of confidently judged wrong answers); measures built on FPR (HSPP) misread this.
4. Rubric: never use the raw score gap; use matched failures **and** a third-party difference to remove
   halo. Without it, Gemma-4 looks biased and isn't.
5. Check the judge first: near-chance or "always met" judges give meaningless (or spuriously positive) SPB.

---

## 5. Running and queued work

| Job | What | State |
|---|---|---|
| 2323799 rate-cmdrplus | Command-R+ quality scores (Yang) | running (resumed from 2,500) |
| 2325677 hb-aleph | HealthBench: 8 Aleph models generate, then judge | running |
| — | HealthBench on 6 GPU models | ready: `RUBRIC_DATASET=healthbench bash infer/ifeval_gpu_submit.sh` (after Aleph generation) |
| — | Yang pairs → judging → analysis | after Command-R+: `python -m infer.yang_data pairs` → `bash infer/yang_judges_submit.sh` → `python -m real.yang_real` |

## 6. Infrastructure
- Clusters: Vulcan (L40S 48 GB, Aleph) and Narval (A100 40 GB, no compute internet); per-cluster settings
  in `infer/clusters/`; models that don't fit are skipped (`infer/cluster.py`).
- vLLM 0.25.0 from the Alliance wheelhouse; async scheduling off for pipeline-parallel runs (vLLM bug).
- All inference steps resumable; outputs in `infer/data/` (git-ignored), logs in `logs/` (git-ignored).

## 7. Caveats
- Real slopes come from reweighting existing data (grid); they mix pair, difficulty and confidence channels.
- β uses "J solved x" (binary) as a proxy for competence c and third-party pairs; assumed to carry over to self-judgment.
- ρ is estimated from cross-judge consensus on confidently judged items; part of it may be oracle noise (MMLU).
- DiD assumes homogeneous halo response across judges.
- IFEval GPU judges use each model's default sampling; Aleph judges run with reasoning on (Pombal's settings).

## 8. File map (new since PLAN.md)
| Area | Files |
|---|---|
| Simulator | `synth/difficulty.py` (β, γ, ρ), `synth/pair_mixture.py` (grid-style slopes), `synth/rubric.py`, `synth/continuous.py` |
| Sweeps | `difficulty_sweep.py`, `difficulty_gap.py`, `yang_sweep.py`, `rubric_sweep.py` |
| Real analyses | `real/slope_gap.py`, `real/pair_check.py`, `real/item_check.py`, `real/beta_fit.py`, `real/rubric_real.py`, `real/rubric_did.py`, `real/yang_real.py` |
| Inference | `infer/ifeval_rubric.py` (IFEval + HealthBench), `infer/beta_data.py`, `infer/yang_data.py`, `infer/judge.py`, `infer/aleph.py`, submit scripts `infer/*_submit.sh` |
