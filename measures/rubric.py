"""
Self-preference measures for rubric (checklist) evaluation. Unit = one criterion of one
answer; y = criterion truly met; v = judge's verdict (0/1 or P(met)); own = the judge wrote
the answer. "Other" = generators outside the judge's family.

score_gap      mean v on own answers - on others' (naive; mixes in quality)
calibrated_gap (mean v - mean y) on own - same on others: over-scoring of own answers
mrspb          Pombal et al. 2026 MRSPB: rate of "met" on truly-not-met criteria, over ALL
               criteria, own - others
hspp_rub       Pombal et al. 2026 MRSPB with error denominator (their HSPP-Rub, difference
               form): FPR own - FPR others
hspp_rub_ratio Pombal's headline ratio FPR own / FPR others, minus 1 (so 0 = no bias)
eo_rub         equalized-odds gap: mean of (TPR own - TPR others) and (FPR own - FPR others)
matched_fail   Roytburg-style: same criterion, both answers truly not met; v own - v other
matched_both   same criterion, both answers with the same true label (met or not); v own - v other

estimate(...) computes them from per-unit data; expected(...) from the simulator's
quadrature (synth/rubric.py).
"""

import numpy as np

MEASURES = ["score_gap", "calibrated_gap", "mrspb", "hspp_rub", "hspp_rub_ratio", "eo_rub",
            "matched_fail", "matched_both"]


def expected(w, pi, mt, mf) -> dict:
    """w: quadrature weights over criteria; pi[g] = P(y=1); mt[g] = P(v=1|y=1); mf[g] = P(v=1|y=0)."""
    o, k = "own", "other"
    met = {g: w @ (pi[g] * mt[g] + (1 - pi[g]) * mf[g]) for g in (o, k)}
    true = {g: w @ pi[g] for g in (o, k)}
    tpr = {g: w @ (pi[g] * mt[g]) / true[g] for g in (o, k)}
    fpr = {g: w @ ((1 - pi[g]) * mf[g]) / (1 - true[g]) for g in (o, k)}
    fp_all = {g: w @ ((1 - pi[g]) * mf[g]) for g in (o, k)}
    # matched: each own answer vs others' answers to the same criterion with the same label
    # (many other generators, so a same-label match almost always exists)
    return {
        "score_gap": met[o] - met[k],
        "calibrated_gap": (met[o] - true[o]) - (met[k] - true[k]),
        "mrspb": fp_all[o] - fp_all[k],
        "hspp_rub": fpr[o] - fpr[k],
        "hspp_rub_ratio": fpr[o] / fpr[k] - 1,
        "eo_rub": ((tpr[o] - tpr[k]) + (fpr[o] - fpr[k])) / 2,
        "matched_fail": w @ ((1 - pi[o]) * (mf[o] - mf[k])) / (1 - true[o]),
        "matched_both": w @ ((1 - pi[o]) * (mf[o] - mf[k]) + pi[o] * (mt[o] - mt[k])),
    }


def estimate(units) -> dict:
    """
    units: DataFrame for ONE judge, columns item (prompt id + criterion id), own (bool),
    y (bool, true label), v (verdict, 0/1 or P(met)). Rows for other generators should
    exclude the judge's own family.
    """
    u = units
    o, k = u[u.own], u[~u.own]
    out = {
        "score_gap": o.v.mean() - k.v.mean(),
        "calibrated_gap": (o.v.mean() - o.y.mean()) - (k.v.mean() - k.y.mean()),
        "mrspb": (o.v * ~o.y).mean() - (k.v * ~k.y).mean(),
        "hspp_rub": o.v[~o.y].mean() - k.v[~k.y].mean(),
        "eo_rub": ((o.v[o.y].mean() - k.v[k.y].mean()) + (o.v[~o.y].mean() - k.v[~k.y].mean())) / 2,
    }
    out["hspp_rub_ratio"] = o.v[~o.y].mean() / k.v[~k.y].mean() - 1
    # matched: per criterion item, own verdict vs mean verdict on others with the same label
    km = k.groupby(["item", "y"]).v.mean().rename("v_other")
    m = o.join(km, on=["item", "y"]).dropna(subset=["v_other"])
    out["matched_fail"] = (m.v - m.v_other)[~m.y].mean()
    out["matched_both"] = (m.v - m.v_other).mean()
    return {k_: float(v) for k_, v in out.items()}
