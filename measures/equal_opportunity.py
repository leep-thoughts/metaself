"""
Equal Opportunity (Wataoka et al.).

Bias = TPR - TNR, where "positive" means "oracle says A is the true winner":
    TPR = P(judge votes A | oracle says A won)   -- rate of voting A correctly
    TNR = P(judge votes B | oracle says B won)   -- rate of voting B correctly

Equivalently (matching the notebook's phrasing): legitimate self-preference
(TPR, voting for A when A genuinely deserves it) minus (1 - illegitimate self-preference)
(TNR is "not wrongly voting for A when B deserves it").

Splitting by the oracle's own verdict is exactly what Panickssery's measure (which
ignores the oracle entirely) doesn't do -- that's the point of this measure.

Derivation (see synth/judge.py; L = logit(accuracy)):
    TPR = sigmoid(L + self_lean)
    TNR = 1 - P(vote=A | oracle says B won) = 1 - sigmoid(-L + self_lean) = sigmoid(L - self_lean)
    E[Bias] = sigmoid(L + self_lean) - sigmoid(L - self_lean)

Two properties worth knowing:
  1. At self_lean = 0, E[Bias] = sigmoid(L) - sigmoid(L) = 0 EXACTLY, for any accuracy
     and any quality gap p. Conditioning on the oracle's verdict cancels the
     quality/difficulty confound that wrecks Panickssery's measure -- no trace of p
     appears in E[Bias] at all, at any self_lean.
  2. For small self_lean, sigmoid(L+self_lean) - sigmoid(L-self_lean) is approximately
     linear: E[Bias] ~= 2 * accuracy * (1-accuracy) * self_lean (since the sigmoid's
     derivative at L is exactly accuracy*(1-accuracy)). This gives a quick sense of how
     much of a given log-odds lean actually shows up on this probability-scale measure,
     without needing the exact nonlinear form.
"""


def estimate_bias(oracle_truth, judge_decisions) -> float:
    a_wins_votes = [vote for truth, vote in zip(oracle_truth, judge_decisions) if truth]
    b_wins_votes = [vote for truth, vote in zip(oracle_truth, judge_decisions) if not truth]

    tpr_hat = sum(a_wins_votes) / len(a_wins_votes)
    tnr_hat = 1 - (sum(b_wins_votes) / len(b_wins_votes))

    return tpr_hat - tnr_hat
