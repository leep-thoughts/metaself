"""
SPB (self-preference bias), from Yang et al. 2026, "Quantifying and Mitigating
Self-Preference Bias of LLM Judges" (arXiv:2604.22891v4), Sec. 3.4, Eqs. 2-6.

Only EQUAL-QUALITY pairs are used: two answers whose benchmark quality scores (mean of
two reference judges, 0-10) differ by at most eps = 0.25. Each pair is judged in both
presentation orders, and only a FIRM choice counts -- the same answer picked in both.

    PIR      = P(judge firmly picks its own answer | own vs third-party, equal quality)
    Null-PIR = P(judge firmly picks the designated target | ordered pairs (j, k) of two
               third-party answers from the same equal-quality neighbourhood)
    SPB      = PIR - Null-PIR

Null-PIR is meant to remove style / length preferences. Because every unordered null pair
appears in both orderings, Null-PIR = P(firm choice on a null pair) / 2: it subtracts the
judge's decisiveness (how often it is order-consistent), not a style preference.

Inputs are per pair, per order: 0/1 picks, or probabilities (soft votes). With
probabilities the two orders are treated as independent draws, so a firm pick has
probability p_order1 * p_order2 -- for 0/1 picks this is exactly Yang's indicator.
The paper's discriminability gate (accuracy >= 0.8 on pairs whose quality gap >= 2.5)
decides which judges are reported; it is not part of the number itself.
"""


def firm_rate(pairs) -> float:
    """pairs[i] = (P(pick target | order 1), P(pick target | order 2)); target = own answer or designated null target."""
    return sum(float(a) * float(b) for a, b in pairs) / len(pairs)


def estimate_bias(self_pairs, null_pairs) -> float:
    """
    self_pairs[i]: (pick own with own shown first, pick own with own shown second), equal-quality own pairs.
    null_pairs[i]: (pick target shown first, pick target shown second) for each ORDERED third-party pair;
                   include both (j, k) and (k, j) as Yang does.
    """
    return firm_rate(self_pairs) - firm_rate(null_pairs)
