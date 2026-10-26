"""
The "oracle" is the ground-truth referee: for any two generators, it knows the true
winner. We never let the judge or the bias measure see theta_a/theta_b directly -- they
only ever see the oracle's sampled win/lose outcomes, exactly like a real-world setup
where nobody has direct access to "true quality", only to comparison outcomes.

Model: Bradley-Terry. Each generator has an intrinsic strength (theta_a, theta_b). The
probability that A is judged the true winner on a given item is sigmoid(theta_a -
theta_b) -- a bigger gap means a more lopsided true win-rate.
"""

import math
import random


def sigmoid(x: float) -> float:
    """Logistic sigmoid: squashes any real number into a probability in (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    """
    Inverse of sigmoid: turns a probability back into a real-valued "gap".
    Lets you think in terms of "I want A to win 80% of the time" and get the
    equivalent theta_a - theta_b, instead of picking thetas by feel.
    """
    return math.log(p / (1.0 - p))


def theta_gap_from_winrate(p_a_wins: float) -> float:
    """
    Convenience: given a desired long-run win-rate for A over B, return the
    theta_a - theta_b gap that produces it under the Bradley-Terry oracle below.
    e.g. theta_gap_from_winrate(0.8) -> gap such that A wins ~80% of comparisons.
    """
    return logit(p_a_wins)


def draw_oracle_truth(n_items: int, theta_a: float, theta_b: float, seed: int = 0):
    """
    Simulates the oracle comparing generator A against generator B on n_items
    independent items.

    On each item, P(A is the true winner) = sigmoid(theta_a - theta_b), sampled
    independently per item. The *observed* fraction of items where A wins will only
    approximately equal sigmoid(theta_a - theta_b) -- see empirical_win_rate() below
    to check how close a given sample lands, given sampling noise.

    Returns: list[bool] of length n_items. True means "A is the true winner on this item".
    """
    rng = random.Random(seed)
    p_a_wins = sigmoid(theta_a - theta_b)
    return [rng.random() < p_a_wins for _ in range(n_items)]


def empirical_win_rate(oracle_truth) -> float:
    """
    What fraction of this drawn sample did A actually win?
    Compare against sigmoid(theta_a - theta_b) to see sampling noise at your chosen
    n_items -- the two should get closer as n_items grows.
    """
    return sum(oracle_truth) / len(oracle_truth)
