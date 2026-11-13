# Confound-controlled self-preference: pipeline design (draft, 2026-11-13)

Plan item 5: "These measures don't work; here is how to make it work in practice." Not naive,
not contrived, and validated. Status: **design only**; building blocks exist (listed per stage).

## 1. What the pipeline must survive

Every confound below is measured on real data, not assumed.

| Confound | What it does | Evidence (file) | Size |
|---|---|---|---|
| Quality gap p | Own answer wins more → raw win rate / score gap rises | simulator, all measures | score gap p-slope 0.70 (rubric sim) |
| **Generator-pair lean** γ | Judge favours the *stronger generator's* answer, right or wrong | `real/pair_check.py` | γ ≈ 0.5 per logit(p); explains all p-slopes of EO/HSPP (0.68 → −0.04 when removed) |
| **Consensus errors** ρ | Some wrong answers look right (and vice versa) to *every* judge | `real/item_check.py` | 41% of confidently judged wrong-A items accepted by other judges (MMLU 31%, MATH/MBPP+ ~10% overall); residual corr across judges 0.61 |
| Difficulty coupling β | Judge evaluates better where it solves the problem itself | `real/beta_fit.py` | median 0.79; Qwen-2.5 1–3; Llama ≈ 0 |
| Halo h (rubric) | All judges lean "met" for strong generators' failed answers | `real/rubric_did.py` | third-party halo up to 0.17 (Qwen3.x), 0.09 (Gemma-4) |
| Position bias | Slot A favoured | Chen/Roytburg both orders | cancels with both orders |
| Family affinity | Judge favours its family | `real/slope_gap.py` §7 | ±0.014: negligible |
| Judge insensitivity | Judges that ignore content (Aya says "met" to all) distort comparisons | `real/rubric_real.py` | accuracy ≈ chance for 3 small judges |

## 2. Stages

| Stage | What | Removes | Built already |
|---|---|---|---|
| **S1 Units** | Item x, J's answer, ≥ 3 reference answers spanning strength, oracle label (checker/tests; committee where no checker) | — | Chen/Roytburg (pairwise), IFEval + HealthBench (rubric) |
| **S2 Match on correctness** | Compare J's answer only with answers of the *same* oracle label on the same item (Roytburg proxy; rubric `matched_fail`) | p, β | `measures/roytburg.py`, `measures/rubric.py` |
| **S3 Match on strength** | Proxy/comparison generators chosen with pair win rate close to J's (|logit p_K − logit p_J| small) | γ (pair lean) | not yet (filter on unit p from `real/pair_check.units`) |
| **S4 Third-party difference** | Same comparison graded by ≥ 3 judges outside J's family and the comparison generators' families; DiD = J's gap − their gap | halo h, consensus errors ρ, shared errors | `real/rubric_did.py` (rubric); pairwise version not yet |
| **S5 Sensitivity weighting** | Weight third-party judges by similarity of sensitivity to J (accuracy, slope of verdict on quality); drop judges at chance | spurious DiD from insensitive judges (Aya, Command-R) | not yet |
| **S6 Filter** | Drop non-verdict outputs (verdict_mass < 0.5), same-family pairs; flag oracle–consensus conflicts (ρ items) and report with/without | oracle noise | partly (`MIN_MASS`, family filters) |
| **S7 Estimate** | Log-odds model per judge: l = item FE + s·own + γ·pair + halo_gen + e; bootstrap over items | — | pieces in `real/pair_check.py`, `real/rubric_did.py` |
| **S8 Calibrate** | Null set (cross-family pairs, s = 0) → offset; injected shift δ on J's own-answer log-odds → slope; report s in log-odds units | residual bias, scale | `real/null_injection.py` |

## 3. Validation

| # | Test | Pass criterion | Data |
|---|---|---|---|
| V1 | **Simulator, full confound model** (β, γ, ρ, h, position; `synth/difficulty.py` + `synth/rubric.py`) | R1: dM/dp, dM/da ≈ 0; R2: M(s=0) ≈ 0; R3: dM/ds > 0; R4: same response across the grid | sim |
| V2 | **Real placebo**: assign "own" to a model that is NOT the judge (third-party pairs) | estimates centred at 0; 5% false-positive rate at α = 0.05 | Chen/Roytburg nulls, IFEval |
| V3 | **Real injection**: add δ to J's log-odds on own answers | recovered δ, slope 1, at every p and accuracy | same |
| V4 | **Grid slopes on real data** (`real/slope_gap.py` grid) | pipeline estimate flat in p and accuracy (where raw EO/HSPP slope 0.7 / 1.1) | Roytburg |
| V5 | **Ablation**: drop one stage at a time | each confound leaks exactly when its stage is removed → "why each step" table | sim + real |
| V6 | **Consistency**: same judge, different datasets / formats (IFEval vs HealthBench; pairwise vs rubric) | estimates agree within CI | all |
| V7 | **Budget**: number of third-party judges and references needed | CI width and bias vs count | resampling |

## 4. Deliverables

1. `pipeline/` module: S1–S8 as functions over a common unit table (judge, generator, item, oracle, verdict).
2. Released matched dataset: units + verdicts from 14 judges on IFEval/HealthBench and Chen/Roytburg pairs, with strength-matched comparison sets and third-party verdicts.
3. "Why each step" table (V5) + recommendations (plan item 4).

## 5. Decisions for the user

- Start with pairwise (Roytburg data, all confounds measured) or rubric (cleaner labels, our own runs)?
- S4 needs third-party verdicts on the *same* comparisons: pairwise needs new GPU judging (J judging K vs R for many K); rubric already has it.
- Whether consensus-error items (ρ) count as judge errors or oracle errors (MMLU's 31% suggests some oracle/label noise).
