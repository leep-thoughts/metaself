"""
Meta-evaluation sweep: turn the three knobs (true quality `p`, judge `accuracy`,
true `self_lean`) by varying how the SYNTHETIC DATASET ITSELF is generated, then run
the real published measures (panickssery, equal_opportunity, hspp) on each generated
dataset, unmodified. See PLAN.md Phase 3 for why this replaced an earlier draft that
swept closed-form formulas directly instead of generated data.

Reports two different things, on purpose -- see PLAN.md's "group size vs. variance"
section for the full explanation of why both are needed:
  - group sizes (|D_A|, |D_B|): free, deterministic, flags which grid points put a
    measure's estimate on a small subgroup.
  - variance (spread across repeats): the actual proof of whether that flag matters in
    practice, obtained by running the real estimator many times and watching it move.

Run with: python3 meta_eval_sweep.py
"""

import random

from synth.oracle import draw_oracle_truth, theta_gap_from_winrate
from synth.judge import simulate_judge_votes
from measures import panickssery, equal_opportunity, hspp
from sweep_config import (
    P_VALUES,
    ACCURACIES,
    SELF_LEANS,
    N_ITEMS,
    N_REPEATS,
    MASTER_SEED,
    REP_ACCURACY,
    REP_SELF_LEAN,
)

# Grid/run knobs now live in sweep_config.py -- edit that file to change the swept
# ranges, item count, or repeat count without touching the logic below.

# One master RNG spawns a fresh sub-seed for every individual draw_oracle_truth /
# simulate_judge_votes call in the whole sweep (len(P_VALUES)*len(ACCURACIES)*
# len(SELF_LEANS)*N_REPEATS*2 draws -- tens of thousands at current grid sizes). Same
# pattern as numpy's SeedSequence.spawn(): fully reproducible from MASTER_SEED alone,
# with a technically nonzero but astronomically small collision chance at this scale
# against a 2**63 seed space.
_seed_rng = random.Random(MASTER_SEED)


def fresh_seed() -> int:
    return _seed_rng.getrandbits(63)


def mean(xs):
    return sum(xs) / len(xs)


def stdev(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def run_one(p: float, accuracy: float, self_lean: float):
    """One repeat at one grid point: generate a fresh dataset, run both measures on it
    exactly as implemented in measures/, unmodified."""
    theta_gap = theta_gap_from_winrate(p)
    oracle_truth = draw_oracle_truth(N_ITEMS, theta_gap, 0.0, seed=fresh_seed())
    judge_decisions = simulate_judge_votes(oracle_truth, accuracy, self_lean, seed=fresh_seed())

    pan = panickssery.estimate_bias(judge_decisions)
    eo = equal_opportunity.estimate_bias(oracle_truth, judge_decisions)
    hs = hspp.estimate_bias(oracle_truth, judge_decisions)

    n_a = sum(oracle_truth)
    n_b = len(oracle_truth) - n_a
    return pan, eo, hs, n_a, n_b


# ---------------------------------------------------------------------------
# Run the full grid: N_REPEATS independent datasets per grid point.
# ---------------------------------------------------------------------------
results = {}  # (p, accuracy, self_lean) -> summary dict

for p in P_VALUES:
    for accuracy in ACCURACIES:
        for self_lean in SELF_LEANS:
            pan_vals, eo_vals, hs_vals = [], [], []
            n_a, n_b = None, None
            for _ in range(N_REPEATS):
                pan, eo, hs, n_a, n_b = run_one(p, accuracy, self_lean)
                pan_vals.append(pan)
                eo_vals.append(eo)
                hs_vals.append(hs)
            results[(p, accuracy, self_lean)] = {
                "pan_mean": mean(pan_vals),
                "pan_std": stdev(pan_vals),
                "eo_mean": mean(eo_vals),
                "eo_std": stdev(eo_vals),
                "hspp_mean": mean(hs_vals),
                "hspp_std": stdev(hs_vals),
                # group sizes are random (binomial around N*p / N*(1-p)) but stable
                # enough across repeats that the last repeat's counts are a fine
                # representative sample -- not worth re-reporting all N_REPEATS of them.
                "n_a": n_a,
                "n_b": n_b,
            }

# ---------------------------------------------------------------------------
# Sensitivity summary: for each knob, hold the other two fixed and measure how much
# the MEAN estimate moves across that knob's swept values (range = max - min),
# averaged over all held-fixed combinations. self_lean is the wanted signal; p and
# accuracy are confounds -- a trustworthy measure should score low on those two.
# ---------------------------------------------------------------------------


def sensitivity_to_self_lean(key):
    ranges = []
    for p in P_VALUES:
        for accuracy in ACCURACIES:
            vals = [results[(p, accuracy, sl)][key] for sl in SELF_LEANS]
            ranges.append(max(vals) - min(vals))
    return mean(ranges)


def sensitivity_to_p(key):
    ranges = []
    for accuracy in ACCURACIES:
        for self_lean in SELF_LEANS:
            vals = [results[(p, accuracy, self_lean)][key] for p in P_VALUES]
            ranges.append(max(vals) - min(vals))
    return mean(ranges)


def sensitivity_to_accuracy(key):
    ranges = []
    for p in P_VALUES:
        for self_lean in SELF_LEANS:
            vals = [results[(p, accuracy, self_lean)][key] for accuracy in ACCURACIES]
            ranges.append(max(vals) - min(vals))
    return mean(ranges)


print("=" * 72)
print("SENSITIVITY SUMMARY -- average range of the MEAN estimate per knob")
print("(self_lean = wanted signal; p and accuracy = confounds, want these near 0)")
print("=" * 72)
print(f"{'knob':<24} {'panickssery':>15} {'equal_opportunity':>20} {'hspp':>10}")
print(
    f"{'self_lean (wanted)':<24} {sensitivity_to_self_lean('pan_mean'):>15.4f} "
    f"{sensitivity_to_self_lean('eo_mean'):>20.4f} {sensitivity_to_self_lean('hspp_mean'):>10.4f}"
)
print(
    f"{'p / quality (confound)':<24} {sensitivity_to_p('pan_mean'):>15.4f} "
    f"{sensitivity_to_p('eo_mean'):>20.4f} {sensitivity_to_p('hspp_mean'):>10.4f}"
)
print(
    f"{'accuracy (confound)':<24} {sensitivity_to_accuracy('pan_mean'):>15.4f} "
    f"{sensitivity_to_accuracy('eo_mean'):>20.4f} {sensitivity_to_accuracy('hspp_mean'):>10.4f}"
)
print()

# ---------------------------------------------------------------------------
# Variance vs. group size: fix accuracy and self_lean at one representative setting
# (REP_ACCURACY, REP_SELF_LEAN, from sweep_config.py), sweep p, and show group sizes
# alongside each measure's mean/std across repeats.
# ---------------------------------------------------------------------------
print("=" * 72)
print(f"VARIANCE vs. GROUP SIZE (accuracy={REP_ACCURACY}, self_lean={REP_SELF_LEAN}, sweeping p)")
print("=" * 72)
print(
    f"{'p':>6} {'|D_A|':>8} {'|D_B|':>8} {'eo_mean':>10} {'eo_std':>10} "
    f"{'pan_mean':>10} {'pan_std':>10} {'hspp_mean':>10} {'hspp_std':>10}"
)
for p in P_VALUES:
    r = results[(p, REP_ACCURACY, REP_SELF_LEAN)]
    print(
        f"{p:>6.2f} {r['n_a']:>8} {r['n_b']:>8} {r['eo_mean']:>10.4f} {r['eo_std']:>10.4f} "
        f"{r['pan_mean']:>10.4f} {r['pan_std']:>10.4f} {r['hspp_mean']:>10.4f} {r['hspp_std']:>10.4f}"
    )

# ---------------------------------------------------------------------------
# Optional: dump the full grid to JSON for downstream plotting. Not part of the
# core measure logic -- just a flat re-shaping of `results` for a charting tool.
# ---------------------------------------------------------------------------
import json
import sys

if "--dump-json" in sys.argv:
    out_path = sys.argv[sys.argv.index("--dump-json") + 1]
    rows = [
        {"p": p, "accuracy": accuracy, "self_lean": self_lean, **r}
        for (p, accuracy, self_lean), r in results.items()
    ]
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": {
                    "P_VALUES": P_VALUES,
                    "ACCURACIES": ACCURACIES,
                    "SELF_LEANS": SELF_LEANS,
                    "N_ITEMS": N_ITEMS,
                    "N_REPEATS": N_REPEATS,
                    "REP_ACCURACY": REP_ACCURACY,
                    "REP_SELF_LEAN": REP_SELF_LEAN,
                },
                "rows": rows,
            },
            f,
        )
    print(f"\nWrote full grid ({len(rows)} rows) to {out_path}")
