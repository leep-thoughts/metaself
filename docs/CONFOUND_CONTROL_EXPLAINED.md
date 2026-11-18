# Measuring self-preference fairly: a plain-language guide (2026-11-17)

This explains, without jargon, what we tested last and what came out. Technical version:
`docs/CONFOUND_PIPELINE.md` §6. Code: `real/rubric_placebo.py`, `real/rubric_control.py`, `real/pair_control.py`.

---

## 1. The question

**Self-preference**: a model acting as a judge grades *its own* answers more kindly than it should.

To see it, we compare two numbers:

```
how kindly the judge grades ITS OWN answers
                 vs
how kindly the judge grades OTHER models' answers
```

If the first is bigger, we call it self-preference. The problem: the first can be bigger for
**innocent reasons** too. Those innocent reasons are *confounds*.

## 2. The innocent reasons (confounds)

| Confound | Everyday picture | Why it fakes self-preference |
|---|---|---|
| **Quality gap** | A strong student grading their own essay *and* weaker students' essays | Their essay really is better, so a higher grade is deserved, not biased |
| **Halo** | A teacher who knows a student is brilliant overlooks that student's mistakes | Strong models' mistakes get a pass from **every** judge, not just from themselves |
| **Pair lean** (pairwise only) | When comparing two answers, judges side with the answer from the better-known model | Same as halo, in head-to-head comparisons |

All three have the same root: **the judge's own answers come from a strong model, and strong models
get treated better by everyone.** That is not self-preference.

## 3. How we test a measure: the "placebo" test

We can't know a judge's true self-preference. But we can create situations where it is **certainly zero**:

> Pretend the judge wrote some *other* model's answers. Run the measure as if those were "its own".

The judge didn't write them, so a good measure should read **0**. This is like a placebo pill in a drug
trial: any "effect" it shows is fake.

We do this for every judge × every other model (152 combinations on IFEval) and check two things:

| Check | Good measure | Bad measure |
|---|---|---|
| Average over placebos | ≈ 0 | clearly above or below 0 |
| Does it grow when the pretend-"own" model is stronger? | No (flat) | Yes: it's really measuring strength |

The second check is the important one. We report it as a **slope**: 0 is perfect; 0.6 means the
measure rises noticeably with the model's strength.

## 4. What we found with the usual approach

With the usual comparison ("own" answers vs **all** other models' answers), **every** rubric measure
fails: they all rise with the strength of the pretend-"own" model.

| Measure | Slope (want 0) |
|---|---|
| Raw score gap | .76 |
| Pombal's HSPP-Rub | 1.03 |
| Pombal's headline ratio | 6.0 (and averages +.28 when it should be 0) |
| Same-criterion comparison (`matched_fail`) | .63 |

So these measures mostly measure "how strong is this model", not "does the judge favour itself".

## 5. Two fixes that work

### Fix A: compare with models of the **same strength** ("strength-matched comparison")

Instead of comparing the judge's answers with *all* other models, compare only with models that are
**about as good** (pass rate within ±4 points).

```
Usual:     judge's own answers (pass rate 94%)  vs  everyone (average 85%)    ← unfair
Fix A:     judge's own answers (pass rate 94%)  vs  models at 90–98% only    ← fair
```

Why it works: equally strong models get the same quality boost and the same halo, so those cancel out.
Only the judge's special treatment of *itself* remains.

### Fix B: ask **another judge** ("two-judge difference")

Have a second, unrelated judge grade the same answers, and subtract:

```
Judge J's gap on its own answers      = self-preference + halo
Judge K's gap on the same answers     =                   halo
-------------------------------------------------------------
Difference                            = self-preference
```

Why it works: halo is felt by every judge; self-preference only by J.

### Results

| Measure | Usual comparison | Fix A | Fix B | A + B together |
|---|---|---|---|---|
| Same-criterion comparison (`matched_fail`) | .63 ❌ | **−.007 ✅** | **−.004 ✅** | **−.05 ✅** |
| Raw score gap | .76 ❌ | .09 🟡 | | |
| Pombal's HSPP-Rub | 1.03 ❌ | .28 ❌ | | |
| Pombal's ratio | 6.0 ❌ | 1.2 ❌ | | |

(Numbers are slopes; 0 = passes the placebo test.)

- **Fix A and Fix B each make the same-criterion comparison pass.** Using both together also passes,
  and detects real self-preference slightly better (real cases stand out 0.97 vs 0.86 standard
  deviations from the placebos).
- **Pombal's measures still fail** even with Fix A: their formula divides by how often the judge's own
  answers fail, which itself depends on the model.

### When to use which

| | Fix A: same-strength comparison | Fix B: second judge |
|---|---|---|
| Needs | other models of similar quality | other judges grading the same answers |
| Cost | throws away comparisons with unequal models (left with ~2–3 comparison models) | extra judging runs |
| Breaks if | the halo comes from something other than strength (e.g. style) | judges react to halo differently (e.g. a judge that says "met" to everything) |
| Recommendation | use both when possible: each covers the other's weak spot | |

## 6. The same idea for head-to-head (pairwise) judging

In pairwise data (Roytburg's), a judge compares answer K against a reference answer R. Roytburg's
measure already compares K with *other* models' answers on the same question that are equally
right or wrong. We checked whether that's enough.

**Surprise:** Roytburg's earlier "passes the null test" result was partly automatic. Averaged over all
models, the measure is 0 *by its arithmetic*, whatever the judge does. Looking at each model
separately (the honest placebo), it **does** favour stronger models:

| Comparison answers | Tracks strength? (correlation, want 0) | Data kept |
|---|---|---|
| All other models (Roytburg's version) | .54 ❌ | 100% |
| Only models within a small strength band | .39 → .28 | 89% → 72% |
| Only models of nearly equal strength | **−.08 ✅** | 46% |

So Fix A works for pairwise too, but only with a tight match, which costs about half the data. The
size of the leak (.12) is what the measured pair lean predicts, which supports the explanation.

*Strength here is measured on a stretched scale (log-odds) because the pair lean works that way:
going from 90% to 95% wins is a much bigger step than from 50% to 55%.*

## 7. Takeaways

1. **Don't compare a model's own answers with everybody else's.** Compare with models of the
   same strength, or subtract what a second judge does. Otherwise you are mostly measuring strength.
2. **Same-question matching alone is not enough.** It fixes difficulty, not halo.
3. **Pombal's rubric measures (HSPP-Rub and its ratio) fail** even after matching; the ratio is the
   worst (reads +.28 when the truth is 0).
4. **Roytburg's pairwise measure needs strength-matched comparison answers too**; its usual null test
   hides the problem.
5. **Always run a placebo test** before trusting a self-preference number.

## 8. Limits

- Few equally strong comparison models exist for the very best models (Gemma-4, Qwen3.x): 20 of 152
  placebo combinations had none within ±4 points.
- Tight matching throws away data, so estimates get noisier.
- Tested on IFEval (rubric) and Roytburg's math/multiple-choice/code data (pairwise) only so far;
  HealthBench pending.

## Glossary

| Term | Meaning |
|---|---|
| Judge | The model doing the grading |
| Generator | The model that wrote the answer being graded |
| Rubric / criterion | A checklist item ("uses exactly 3 bullet points"); the judge says met / not met |
| Pairwise | The judge sees two answers and picks the better one |
| Pass rate | Share of criteria a model's answers truly meet (from IFEval's automatic checker) |
| Placebo | A pretend "own" model the judge didn't write; true self-preference 0 |
| Slope | How much a measure rises as the pretend-"own" model gets stronger; 0 = good |
| Halo | Everyone going easy on a strong model's mistakes |
| Pair lean | Judges favouring the stronger model's answer in head-to-head comparisons |
| Strength-matched | Comparing only with models of about the same quality |
| Two-judge difference | A judge's gap minus an unrelated judge's gap on the same answers |
