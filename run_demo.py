"""
Minimal end-to-end demo of the synthetic self-preference-bias pipeline:

    generators (theta_a, theta_b)
        -> oracle (who's actually better, per item)
            -> judge (accuracy + self-preference lean, in log-odds space)
                -> measure (tries to recover self_lean from what it observed)

Run with:  python3 run_demo.py
(Pure Python stdlib, no numpy/venv needed, instant to run.)

Edit the CONFIG block below and re-run to explore different settings. See PLAN.md for
the full write-up of the model. For a side-by-side comparison of multiple published
bias measures (not just this one idealized reference), see run_demo_measures.py.
"""

from synth.oracle import (
    draw_oracle_truth,
    empirical_win_rate,
    theta_gap_from_winrate,
    sigmoid,
)
from synth.judge import simulate_judge_votes
from measures.win_rate_gap import estimate_bias

# ---------------------------------------------------------------------------
# CONFIG -- change these and re-run to explore
# ---------------------------------------------------------------------------
N_ITEMS = 2000

# Generator quality: set directly as a theta gap ...
THETA_A = 1.0
THETA_B = 0.0
# ... or derive the gap from a target win-rate instead (uncomment to use):
# gap = theta_gap_from_winrate(0.8)  # "I want A to truly win 80% of the time"
# THETA_A, THETA_B = gap, 0.0

ACCURACY = 0.75  # judge's skill: 0.5 = random, 1.0 = perfect
SELF_LEAN = 0.2  # judge's self-preference toward A, in LOG-ODDS units (any real number)

# Two *different* seeds for the two independent random sources (oracle truth, judge's
# vote sampling) -- Random.Random(seed) with the same seed reused for both would make
# their "independent" draws secretly identical.
SEED_ORACLE = 0
SEED_JUDGE = 1
# ---------------------------------------------------------------------------

oracle_truth = draw_oracle_truth(N_ITEMS, THETA_A, THETA_B, seed=SEED_ORACLE)
judge_decisions = simulate_judge_votes(oracle_truth, ACCURACY, SELF_LEAN, seed=SEED_JUDGE)
self_lean_hat = estimate_bias(oracle_truth, judge_decisions, ACCURACY)

true_win_rate = sigmoid(THETA_A - THETA_B)
observed_pick_a_rate = sum(judge_decisions) / len(judge_decisions)

print(f"n_items                      = {N_ITEMS}")
print(f"theta_a - theta_b            = {THETA_A - THETA_B:.3f}")
print(f"  -> theoretical A win-rate  = {true_win_rate:.3f}")
print(f"  -> empirical  A win-rate   = {empirical_win_rate(oracle_truth):.3f}  (this sample)")
print(f"accuracy        (configured) = {ACCURACY:.3f}")
print(f"self_lean       (configured) = {SELF_LEAN:.3f}  (log-odds units)")
print(f"judge's observed pick-A rate = {observed_pick_a_rate:.3f}")
print(f"self_lean       (RECOVERED)  = {self_lean_hat:.3f}")
