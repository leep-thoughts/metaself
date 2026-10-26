"""
Side-by-side comparison of the three self-preference-bias measures against the
synthetic A-vs-B data, with each measure's THEORETICAL expectation printed next to its
Monte Carlo estimate -- so you can check the code against its own derived math, not
just whether the numbers look plausible.

See PLAN.md for the full derivations. Quick recap (L = logit(accuracy),
p = P(oracle says A wins) = sigmoid(theta_a - theta_b)):

  win_rate_gap (idealized, uses known accuracy): recovers self_lean directly.
  panickssery   (naive, ignores the oracle):
      E[Bias] = p*sigmoid(L+self_lean) + (1-p)*sigmoid(-L+self_lean) - 0.5
  equal_opportunity (splits by the oracle's verdict):
      E[Bias] = sigmoid(L+self_lean) - sigmoid(L-self_lean)

Run with:  python3 run_demo_measures.py
"""

from synth.oracle import draw_oracle_truth, sigmoid, logit
from synth.judge import simulate_judge_votes
from measures import win_rate_gap, panickssery, equal_opportunity

# ---------------------------------------------------------------------------
# CONFIG -- change these and re-run to explore
# ---------------------------------------------------------------------------
N_ITEMS = 5000

THETA_A = 1.0
THETA_B = 0.0

ACCURACY = 0.75
SELF_LEAN = 0.2  # log-odds units

SEED_ORACLE = 0
SEED_JUDGE = 1
# ---------------------------------------------------------------------------

oracle_truth = draw_oracle_truth(N_ITEMS, THETA_A, THETA_B, seed=SEED_ORACLE)
judge_decisions = simulate_judge_votes(oracle_truth, ACCURACY, SELF_LEAN, seed=SEED_JUDGE)

p = sigmoid(THETA_A - THETA_B)
L = logit(ACCURACY)

# --- theoretical expectations, computed directly from the configured parameters -----
panickssery_expected = p * sigmoid(L + SELF_LEAN) + (1 - p) * sigmoid(-L + SELF_LEAN) - 0.5
equal_opportunity_expected = sigmoid(L + SELF_LEAN) - sigmoid(L - SELF_LEAN)

# --- Monte Carlo estimates from the simulated sample --------------------------------
win_rate_gap_hat = win_rate_gap.estimate_bias(oracle_truth, judge_decisions, ACCURACY)
panickssery_hat = panickssery.estimate_bias(judge_decisions)
equal_opportunity_hat = equal_opportunity.estimate_bias(oracle_truth, judge_decisions)

print(f"n_items                       = {N_ITEMS}")
print(f"true quality win-rate p       = {p:.3f}")
print(f"accuracy                      = {ACCURACY:.3f}")
print(f"self_lean       (configured) = {SELF_LEAN:.3f}  (log-odds units, ground truth)")
print()
print(f"{'measure':<22} {'estimate':>10} {'theory':>10}")
print(f"{'win_rate_gap (ideal)':<22} {win_rate_gap_hat:>10.3f} {SELF_LEAN:>10.3f}")
print(f"{'panickssery':<22} {panickssery_hat:>10.3f} {panickssery_expected:>10.3f}")
print(f"{'equal_opportunity':<22} {equal_opportunity_hat:>10.3f} {equal_opportunity_expected:>10.3f}")
print()
print("Note: panickssery's theory column is NOT expected to match self_lean unless")
print("p=0.5 or accuracy=0.5 -- that mismatch is the quality/accuracy confound it's")
print("known to suffer from, reproduced here on purpose.")
