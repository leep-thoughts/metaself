"""
Evaluator Quality Baseline (EQB), from Roytburg et al. 2026, "Are LLM Evaluators Really
Narcissists? Sanity Checking Self-Preference Evaluations" (arXiv:2601.22548, ICML 2026),
Sec. 3.3, Eqs. 6-9.

Idea: HSPP (the paper's ILSP, "illegitimate self-preference") mixes bias with evaluator
uncertainty -- a judge also backs a WRONG answer more often on items it finds hard,
whoever wrote it. EQB calibrates against an output-matched control: on each item, the
same judge J also evaluates PROXY answers o_K from other models K whose oracle outcome
against the same reference R matches J's own (Eq. 6, example-level outcome matching):

    X_K = { x : G(x, o_J, o_R) = G(x, o_K, o_R) }
    Delta s_J(x) = s_J(x, o_J, o_R) - mean_K s_J(x, o_K, o_R)          (Eq. 7)
    T_quality = E_{x in X_K}[ Delta s_J(x) ]                             (Eq. 8)

where s_J(x, A, B) = P_J(vote = A | x, A, B). Reported on the ILSP subset (J's own answer
loses, Y = 0), with a paired t-test against H0: T_quality <= 0 (Eq. 9). This returns the
mean; the paper's "ILSP_upd" column in Table 1 is this value in percent.

Derivation (L = logit(accuracy), see synth/judge.py; the judge has no lean toward a
proxy, so its proxy votes follow sigmoid(z(x)*L)):
    E[EQB] = sigmoid(self_lean - L) - sigmoid(-L) = HSPP - (1 - accuracy)
At self_lean = 0, E[EQB] = 0 for any accuracy and any p -- the 1 - accuracy floor that
HSPP carries is exactly what the proxy subtracts off.
"""


def estimate_bias(oracle_truth, judge_decisions, proxy_decisions) -> float:
    """
    oracle_truth[i]:    True if A (the judge's own answer) is the true winner on item i.
    judge_decisions[i]: s_J(x_i, o_J, o_R) -- a 0/1 vote for A, or the judge's probability.
    proxy_decisions[i]: s_J(x_i, o_K, o_R) on the same item, averaged over its matched
                        proxies (0/1 or probability). Only read where B won; None = no
                        matched proxy, item skipped.
    """
    diffs = [float(vote) - float(proxy) for truth, vote, proxy
             in zip(oracle_truth, judge_decisions, proxy_decisions)
             if not truth and proxy is not None]
    return sum(diffs) / len(diffs)
