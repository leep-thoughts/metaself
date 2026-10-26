"""
Judge model: how likely is the judge to vote for A, given what the oracle actually
thinks and the judge's own skill/bias?

Model: P(vote=A | x) = sigmoid( z(x)*logit(accuracy) + self_lean )
  - z(x) = +1 if the oracle says A is the true winner on item x, else -1.
  - logit(accuracy) is the oracle-correctness signal in log-odds space: when
    self_lean=0, sigmoid(z*logit(accuracy)) is exactly `accuracy` when the oracle says A
    won, and exactly `1-accuracy` when the oracle says B won -- same meaning `accuracy`
    had before.
  - self_lean is an extra LOG-ODDS shift representing self-preference bias. Because it's
    added inside the sigmoid rather than to a raw probability, it can be any real number
    without ever pushing the resulting probability outside [0, 1] -- there's no
    "saturation" problem where a very accurate judge (accuracy near 1) has no room left
    for a positive bias to express itself. An earlier version of this model added the
    bias directly to the probability (accuracy + self_lean), which broke exactly that
    way: at high accuracy there was barely any probability "headroom" left, so bias and
    accuracy weren't truly independent knobs anymore. Working in log-odds space fixes
    this -- the same reason Bradley-Terry (synth/oracle.py) works in log-odds space too.
"""

import random

from synth.oracle import sigmoid, logit


def vote_probability(oracle_says_a_wins: bool, accuracy: float, self_lean: float) -> float:
    """
    P(judge votes for A on this item), given what the oracle truly decided.

    No validity check is needed on self_lean here (unlike a probability-space bias) --
    sigmoid() maps *any* real number into (0, 1), so this is always a well-formed
    probability no matter how large self_lean is.
    """
    z = 1.0 if oracle_says_a_wins else -1.0
    return sigmoid(z * logit(accuracy) + self_lean)


def simulate_judge_votes(oracle_truth, accuracy: float, self_lean: float, seed: int = 0):
    """
    Samples one judge vote per item.

    Returns: list[bool], same length as oracle_truth. True = judge voted for A.
    """
    rng = random.Random(seed)
    votes = []
    for truth in oracle_truth:
        p_vote_a = vote_probability(truth, accuracy, self_lean)
        votes.append(rng.random() < p_vote_a)
    return votes
