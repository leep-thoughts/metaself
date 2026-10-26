# Meta-evaluation of self-preference bias measures (synthetic data)

Goal: evaluate existing *measures* of self-preference bias using a fully synthetic dataset
with independent control over (1) generator quality, (2) judge/evaluator accuracy, and
(3) judge-to-generator (J→G) bias.

## 0. Framing decision (default assumption — flag if wrong)

- Fully numeric/statistical simulation. "Generators" and "judges" are parameterized
  score/decision functions, not text models or LLM calls. This is what makes the three
  control knobs cleanly settable and identifiable against ground truth.
- Primary observation unit is **pairwise comparison** (judge picks a winner between two
  generators' outputs on an item) — the format nearly all published self-preference bias
  measures are built around. Pointwise absolute scoring can be added as a secondary mode
  later.

## 1. Statistical model

- **Items**: `i = 1..N`, each with a difficulty/offset `d_i` (can start at 0).
- **Generators**: `g = 1..G`, each with a latent quality `θ_g`. Oracle quality of g's
  response to item i:

  ```
  O(g, i) = θ_g + d_i + ε_gi
  ```

  `ε_gi` is idiosyncratic response-level noise, controlling how "spread out" a generator's
  own outputs are.

- **Judges**: `j = 1..J`. Each judge has:
  - accuracy parameter `α_j` — how sharply its pairwise decision tracks the true oracle
    gap. Bradley-Terry / Thurstone-style link:

    ```
    P(j prefers g1 over g2 | i) = σ( α_j · (O(g1,i) − O(g2,i)) + β_j · (S(j,g1) − S(j,g2)) )
    ```

    where `S(j,g) ∈ {0,1}` is "is generator g the one judge j is self-affiliated with" and
    `β_j` is j's **self-preference bias magnitude**.
  - `α_j = 0` → random judge; `α_j → ∞` → oracle-perfect judge. This single parameter is
    the "evaluator accuracy" knob.
  - `β_j = 0` → unbiased judge (null case, needed for false-positive testing). `β_j > 0`
    → biased.
- **J-to-G affiliation** `S(j,g)`: start with one-to-one (judge j is literally "the same
  model as" generator g=j, G=J), but design `S` as its own matrix so partial/family-level
  affiliation (judge biased toward a *family* of generators, not just itself) can be
  tested later without changing the rest of the pipeline.

This one model gives all three knobs: `θ_g` = generator quality, `α_j` = judge accuracy,
`β_j` = J→G bias — fully known, fully independent, fully controllable.

## 2. Module breakdown

- **`synth/oracle.py`** — generator quality model. Samples/sets `θ_g`, `d_i`, `ε_gi`,
  exposes `oracle_score(g, i)` and `oracle_pairwise_prob(g1, g2, i)`. Foundational, small —
  good candidate to write by hand since every downstream module depends on its contract.

- **`synth/judges.py`** — judge model. Given `α_j`, `β_j`, `S`, produces pairwise decisions
  (sampled Bernoulli from the logistic model above) or continuous scores. Highest-value to
  control directly: the bias injection mechanism is the thing being validated against, and
  subtle bugs here invalidate everything downstream silently.

- **`synth/sampler.py`** — orchestration: given a scenario config (N, G, J, distributions
  for θ/α/β, affiliation structure), draws a full dataset of
  `(item, g1, g2, judge, decision)` rows plus a parallel "ground truth" table (`θ_g`,
  `α_j`, `β_j` per row). Mechanical glue — good generation candidate.

- **`measures/`** — one file per existing self-preference bias measure to test (win-rate
  gap, residualized/controlled-for-quality regression estimator, Bradley-Terry-with-bias
  MLE, etc.), all implementing a common `estimate(data) -> {(j,g): β_hat}` interface. Good
  generation candidates once specific measures are named — "existing self-preference bias
  measures" needs to resolve to specific formulas.

- **`evaluation/metrics.py`** — the meta-evaluation itself: for each measure, compare
  `β_hat` to true `β` across repeated draws → bias, variance, MSE, correlation, null
  false-positive rate (when all `β_j=0`), power curves (as `β_j` grows), and critically:
  **confound robustness** — does `β_hat` stay accurate as the generator quality gap
  `θ_g1 − θ_g2` grows, independent of judge accuracy `α_j`? That axis is the actual
  research question (can a measure separate "self is truly better" from "judge is biased
  toward self"), so this file deserves care — probably worth co-writing.

- **`scenarios/`** — factorial grid of configs (vary quality spread, accuracy range, bias
  range, N items, J judges) + a runner that sweeps the grid, Monte-Carlo repeats each cell,
  and dumps results to `$SCRATCH` (pure CPU/numpy, no GPU needed — cheap enough to
  prototype interactively, but should go through `sbatch` once the grid/repeat count
  scales up).

## 3. Validation sanity checks (build in early)

- **Null calibration**: with all `β_j=0`, every measure's mean `β_hat` ≈ 0 and FPR ≈
  nominal.
- **Recovery at the limits**: `α_j→∞` (perfect judge) should make bias trivially
  identifiable; `α_j→0` (random judge) should make all measures degrade gracefully, not
  blow up.
- **Confound stress test**: fix `β_j`, sweep `θ_g1−θ_g2` from 0 to large — a good measure's
  `β_hat` should stay flat; a bad one will show `β_hat` drifting with the quality gap
  (the exact failure mode these measures are prone to in the real literature).

## 4. Suggested build order

1. `oracle.py` + unit tests (write by hand — contract-critical, small).
2. `judges.py` + unit tests, verify empirical pairwise win-rates match the logistic model
   in a large-N limit (write or tightly review).
3. `sampler.py` (generate, then review).
4. Implement 1–2 known measures by hand first to validate the harness before generating
   the rest.
5. `evaluation/metrics.py` meta-metrics (co-write — this is where the actual claims come
   from).
6. Scenario grid + Slurm array sweep once the single-scenario path is validated.

## 5. Phase 2 (done) — replicating published measures

Replicated three published self-preference-bias measures against the `G_A`/`G_B`/`J_A`
sandbox, using closed-form derivations from a provided notebook survey.

**Model correction**: the notebook derives its measures against a probability-additive
lean (`P(vote=A) = accuracy ± self_lean`), which has a real flaw — at high `accuracy`
there's little probability headroom left for a positive lean, so `accuracy` and
`self_lean` stop being independently controllable (breaking one of this project's three
founding design goals). Fixed by moving the lean into log-odds space instead:

```
P(vote=A | x) = sigmoid( z(x)*logit(accuracy) + self_lean )
```

(`z(x) = +1` if the oracle says A won, `-1` if B did). `self_lean=0` exactly reproduces
the old meaning of `accuracy`; `self_lean` is now valid for any real number, with no
saturation. Implemented in `synth/judge.py` (`vote_probability`,
`simulate_judge_votes` — replaces the old override-based judge entirely).

**Measures implemented** (`L = logit(accuracy)`, `p = P(oracle says A wins)`):
- `measures/win_rate_gap.py` — idealized reference baseline (uses known `accuracy`),
  inverts the logit exactly via the `D_A`/`D_B` split. Not one of the three published
  measures; a "what if you knew everything" contrast point.
- `measures/panickssery.py` — naive measure, `mean(vote=A) - 0.5`, ignores the oracle
  entirely. `E[Bias] = p·σ(L+self_lean) + (1-p)·σ(-L+self_lean) - 0.5`; nonzero at
  `self_lean=0` whenever `p≠0.5` or `accuracy≠0.5` — the quality/accuracy confound this
  measure is known to suffer from, reproduced on purpose.
- `measures/equal_opportunity.py` — `TPR - TNR` on the oracle-verdict split.
  `E[Bias] = σ(L+self_lean) - σ(L-self_lean)`, exactly `0` at `self_lean=0` for *any*
  `accuracy`/`p` — the oracle split cancels the quality confound that wrecks
  Panickssery's measure.
- **Deferred**: Evaluator Quality Baseline (Roytburg et al.) — needs a proxy set
  `K(x) ⊆ S\{A,B}` of other generator systems, which the sandbox doesn't have yet.
  Natural next step: extend to multiple generators (see §1's generalization note),
  simulate each proxy against B with the same oracle/judge machinery, then apply the
  aggregation formula.

**Demos**: `run_demo.py` (single idealized measure, quick look) and
`run_demo_measures.py` (all three measures side by side, each with its Monte Carlo
estimate next to its theoretical expectation computed from the configured parameters).

**Verified** (via Slurm job, convergence checked at N=8k/50k/300k): all three measures'
Monte Carlo estimates converge tightly to their derived theoretical expectations;
`vote_probability` stays valid for large `self_lean` (e.g. 1.5) at high `accuracy` (e.g.
0.9) — exactly the regime that broke the old additive-probability model.

## 6. Phase 3 (done) — simple meta-evaluation grid sweep

Goal: turn the three knobs (quality `p`, judge `accuracy`, true `self_lean`) and see
how `panickssery` and `equal_opportunity` react.

**Methodology correction, before writing any code**: the first draft of this phase
planned to sweep the *closed-form theoretical expectations* derived in Phase 2 directly
(adding `theoretical_bias(...)` functions to each measure module). On review, that's
wrong — those closed forms are our own derivation of what the measures do under this
model, not the measures themselves; evaluating them would mean evaluating our
dissection, not the published equations. Corrected approach, implemented in
`meta_eval_sweep.py`: turn the knobs by varying how the *synthetic dataset itself* is
generated (`p` → `theta_a - theta_b` via the existing `theta_gap_from_winrate` helper,
`accuracy`, `self_lean`), then call the real, unmodified `estimate_bias` functions from
`measures/panickssery.py` and `measures/equal_opportunity.py` on each generated
dataset. No changes were made to either measure module.

**Grid parameters now live in `sweep_config.py`** (pulled out of `meta_eval_sweep.py`
so the knobs can be edited without touching the sweep logic):
`P_VALUES = ACCURACIES = [0.02, 0.05, 0.1, 0.2, ..., 0.9, 0.95, 0.98]` (13 values each),
`SELF_LEANS = [-2.0, -1.5, -1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]`
(13 values) — 2197 grid points, `N_ITEMS=2000`, `N_REPEATS=10` independent datasets per
point. `p` and `accuracy` stop just short of the true 0/1 endpoints on purpose: both
get converted to log-odds via `logit`, and `logit(0)`/`logit(1)` are mathematically
undefined (Python raises `ZeroDivisionError` on the underlying `1.0/0.0` before even
reaching `log`) — `0.02`/`0.98` is as close as the pipeline can go without crashing.
Widened progressively from an initial 6x6x7=252-point grid once the run cost was
confirmed negligible (pure stdlib, well under the job's time budget) — the wider,
now-symmetric range shows the group-size effect at *both* tails (small `|D_A|` near
`p=0.02`, small `|D_B|` near `p=0.98`), not just one. Seeding: one `MASTER_SEED`
feeds a single RNG that spawns a fresh sub-seed per draw (`getrandbits(63)`, same
pattern as numpy's `SeedSequence.spawn()`) — reproducible from one number, collision
risk across the ~44k spawned seeds is still astronomically small.

**Sensitivity summary**: for each knob, hold the other two fixed and measure how much
the *mean* estimate (across repeats) moves as that one knob sweeps — `self_lean` is the
wanted signal; `p` and `accuracy` are confounds a trustworthy measure should score near
zero on. Reported as one number per measure per knob.

### Group size vs. variance: two different questions, both worth answering

`equal_opportunity` splits the data into two piles: "items where A truly won" (`D_A`)
and "items where B truly won" (`D_B`), computes one rate from each, and subtracts them.

**Group size** answers: *how much data is each pile actually built from?* If `A` wins
95% of the time, then out of 2000 items, about 1900 land in `D_A` and only ~100 in
`D_B`. That's just arithmetic (`N·p`, `N·(1-p)`) — no simulation needed, known before
drawing a single sample.

**Variance (spread across repeats)** answers a different question: *given that one pile
is small, does that actually make the final number unreliable?* A small pile doesn't
automatically wreck every statistic computed from it — how much it matters depends on
the specific formula. The only way to see this directly, without inventing a new
formula ourselves, is to actually run the measure many times under identical conditions
and watch how much its answer bounces around run to run. Bounces a lot → the small pile
is really hurting reliability. Stays stable → it isn't.

**Why report both**: group size is free and instant — a flag for "this grid point might
be dicey" before running anything. Variance is the actual proof — it confirms or
disproves that the flagged points are dicey in practice. Analogy: group size is
checking how few people voted in a district before trusting a poll from it; variance is
running that same poll many times and seeing how much the result swings. The first
tells you where to be suspicious; the second tells you whether you were right to be.

`meta_eval_sweep.py` reports both: group sizes (`|D_A|`, `|D_B|`) alongside every grid
point's estimate, and a variance-vs-group-size table that fixes `accuracy`/`self_lean`
at one representative setting and sweeps `p`, showing each measure's mean and std
across repeats next to the group sizes at that point.

## 7. HSPP — third measure added to the roster

`measures/hspp.py` replicates **harmful self-preference propensity (HSPP)**, Eq. 5 of
Chen et al. 2025, "Do LLM Evaluators Prefer Themselves for a Reason?"
(arXiv:2504.03846):

```
HSPP = P(judge votes for itself | the OTHER response is actually correct)
     = (# items where oracle says B is right AND judge still votes for A)
       / (# items where oracle says B is right)
```

The paper restricts to a "differential" subset where exactly one response is correct;
our oracle always picks exactly one winner per item, so every item already satisfies
that condition — no separate filter needed. Same wiring as the other two measures: a
real `estimate_bias(oracle_truth, judge_decisions)` call on freshly generated data, no
closed-form shortcut.

**Why it was added — this is the "no baseline" property.** A conversation about
`equal_opportunity` raised a concrete question: isolate the items where the *other*
model is actually better — even with zero true self-preference bias, won't the judge
still pick itself sometimes, purely from inaccuracy? Yes — that rate is exactly
`1 - TNR`, and it's real: nonzero whenever `accuracy < 1`. `equal_opportunity` hides
that floor because its `TPR - TNR` contrast nets it against the mirror-image mistake on
the *other* oracle-split (`1 - TPR`, wrongly voting against yourself when you're
actually right) — same accuracy parameter, same magnitude, opposite sign, cancels
exactly at `self_lean = 0`. **HSPP is that raw floor, reported on its own, with nothing
subtracted off.** It's a one-sided rate, not a contrast — which is the published
critique of measures like this: there's no built-in baseline of 0 under pure
inaccuracy, so a non-biased-but-imperfect judge still reports a nonzero HSPP.

Derivation (`L = logit(accuracy)`): `E[HSPP] = sigmoid(self_lean - L) = 1 - TNR`. At
`self_lean = 0`, `E[HSPP] = 1 - accuracy` — independent of `p`, but strongly and
*directly* confounded by `accuracy`, by construction. **Confirmed in the Slurm run**:
sensitivity-to-accuracy for HSPP is `0.9349` (near the full possible range), versus
`0.0236` for sensitivity-to-`p` — the sharpest accuracy confound of the three measures,
exactly the "no baseline" property stated above, not a bug in the replication.

Wired into `meta_eval_sweep.py` (`run_one`, the results dict, the sensitivity summary,
and the variance-vs-group-size table) and into the dashboard (third categorical slot,
aqua — `--series-hspp` — added to every chart, plus a third heatmap panel). The
heatmap panel is the clearest illustration: `panickssery` and `equal_opportunity` read
flat gray at `self_lean=0` (mostly) while `hspp`'s panel visibly tracks the accuracy
axis even with zero true bias, by design.

**Not implemented from the same paper**: SPR (self-preference ratio, Eq. 2) — it's
`mean(judge_decisions)` exactly, i.e. `panickssery.estimate_bias(...) + 0.5`, so no new
code was needed. LSPR (legitimate self-preference ratio, Eq. 4) — precision of
self-votes (`P(oracle=A | vote=A)`), a different conditioning than anything currently
implemented — left out since it wasn't asked for; flag if useful later.
