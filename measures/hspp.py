"""
HSPP (harmful self-preference propensity), from Chen et al. 2025, "Do LLM Evaluators
Prefer Themselves for a Reason?" (arXiv:2504.03846), Eq. 5.

HSPP = P(judge votes for itself | the OTHER response is actually the correct one)
     = (# items where oracle says B is right AND judge still votes for A)
       / (# items where oracle says B is right)

The paper restricts to a "differential" subset D_diff where exactly one of the two
responses is correct. Our oracle (synth/oracle.py) always picks exactly one winner per
item, so every item already satisfies that condition -- there's no separate filter to
apply here, unlike the paper's setting where a response's correctness is judged
independently of the comparison.

Derivation (L = logit(accuracy), see synth/judge.py):
    HSPP = P(vote=A | oracle says B) = sigmoid(self_lean - L) = 1 - TNR

NOTE -- this is the measure that motivated adding it to the roster: unlike
equal_opportunity's TPR - TNR, HSPP is a ONE-SIDED rate, not a contrast. It never
subtracts off the mirror-image mistake (judge wrongly voting against itself when it's
actually right). At self_lean = 0, HSPP = 1 - accuracy -- strictly positive whenever
accuracy < 1, even with zero genuine self-preference bias. A judge that is simply not
perfectly accurate will always show a nonzero HSPP, and that floor rises as accuracy
falls. See PLAN.md's HSPP section.
"""


def estimate_bias(oracle_truth, judge_decisions) -> float:
    other_correct_votes = [vote for truth, vote in zip(oracle_truth, judge_decisions) if not truth]
    return sum(other_correct_votes) / len(other_correct_votes)
