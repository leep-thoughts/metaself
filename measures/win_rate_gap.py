"""
Idealized reference baseline -- NOT one of the three published measures being
replicated. This one is given the judge's true `accuracy`, which none of the published
measures below assume; it exists purely as a "what could you recover if you knew
everything" contrast point against measures that have to work from data alone.

Model it's inverting (see synth/judge.py):
    P(vote=A | oracle says A won) = sigmoid(L + self_lean)   =: TPR
    P(vote=A | oracle says B won) = sigmoid(-L + self_lean)  =: FPR
  where L = logit(accuracy).

Each of those is a one-to-one (sigmoid) function of self_lean, so each can be inverted
exactly via logit (sigmoid's inverse):
    logit(TPR) = L + self_lean   =>  self_lean = logit(TPR) - L
    logit(FPR) = -L + self_lean  =>  self_lean = logit(FPR) + L

With finite data, TPR/FPR are only estimated (TPR_hat, FPR_hat), so these two give two
independent estimates of the same self_lean -- averaging them uses both halves of the
data instead of just one.
"""

from synth.oracle import logit


def estimate_bias(oracle_truth, judge_decisions, accuracy: float) -> float:
    """
    Estimate self_lean from observed data, given the true accuracy.

    Splits items by the oracle's verdict (same D_A / D_B split used by
    measures/equal_opportunity.py), computes the empirical TPR and FPR, inverts each
    through logit, and averages the two resulting estimates.
    """
    L = logit(accuracy)

    a_wins_votes = [vote for truth, vote in zip(oracle_truth, judge_decisions) if truth]
    b_wins_votes = [vote for truth, vote in zip(oracle_truth, judge_decisions) if not truth]

    tpr_hat = sum(a_wins_votes) / len(a_wins_votes)
    fpr_hat = sum(b_wins_votes) / len(b_wins_votes)

    # logit(0) and logit(1) are undefined (the data would have to show a *perfectly*
    # one-sided vote rate on one side of the split) -- not expected at a few thousand
    # items with accuracy/self_lean away from the extremes, so not guarded against here.
    self_lean_from_tpr = logit(tpr_hat) - L
    self_lean_from_fpr = logit(fpr_hat) + L

    return (self_lean_from_tpr + self_lean_from_fpr) / 2
