"""
How far can actual accuracy move on real data with p and true self_lean held fixed?

Data: the cross-family null pairs (real/null_injection.py), true s ~ 0. Items are binned by
a covariate that never looks at the judge's own votes, and whole bins are reweighted, never
individual labels or votes. For each judge a linear program picks the bin weights that
minimize / maximize actual accuracy subject to weighted p = the judge's natural p, with no
bin above MAX_BIN_WEIGHT.

Covariates (all leave-this-judge-out):
    pair_acc     other judges' accuracy on this exact answer pair (uses the oracle)
    problem_acc  other judges' accuracy on this problem, any answer pair (uses the oracle)
    confidence   other judges' mean |log-odds| on this exact answer pair (label-free)
    random       placebo: how much range the optimizer finds in pure noise

Run with: python -m real.tune_accuracy
"""

import os

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from measures import panickssery, equal_opportunity, hspp, roytburg
from real.null_injection import load, sigmoid

N_BINS = 10
MAX_BIN_WEIGHT = 0.3
SEED = 0
COVARIATES = ("pair_acc", "problem_acc", "confidence", "random")
MEASURES = ("panickssery", "equal_opportunity", "hspp", "roytburg")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tune_accuracy_results.csv")


def _leave_judge_out(df, value, keys):
    """Mean of `value` over the other judges, per group in keys. NaN if no other judge."""
    per = df.groupby(keys + ["judge"])[value].agg(["sum", "count"])
    tot = per.groupby(level=list(range(len(keys)))).transform("sum")
    out = (tot["sum"] - per["sum"]) / (tot["count"] - per["count"]).replace(0, np.nan)
    idx = pd.MultiIndex.from_frame(df[keys + ["judge"]])
    return out.reindex(idx).to_numpy()


def add_covariates(df, rng) -> pd.DataFrame:
    df = df.copy()
    df["q"] = sigmoid(df.l)
    df["acc"] = np.where(df.a_correct, df.q, 1 - df.q)
    df["abs_l"] = df.l.abs()
    pair = ["task", "opponent", "proxy", "problem_idx"]
    df["pair_acc"] = _leave_judge_out(df, "acc", pair)
    df["problem_acc"] = _leave_judge_out(df, "acc", ["task", "problem_idx"])
    df["confidence"] = _leave_judge_out(df, "abs_l", pair)
    df["random"] = rng.random(len(df))
    return df


def weighted_stats(g, w) -> dict:
    """The four measures under row weights w (sums to 1). Unit weights match measures/*.py."""
    w = w / w.sum()
    t, q, qp = g.a_correct.to_numpy(), g.q.to_numpy(), g.q_proxy.to_numpy()
    wm = lambda x, mask: np.sum(w[mask] * x[mask]) / np.sum(w[mask])
    tpr, fp = wm(q, t), wm(q, ~t)
    matched = ~t & ~np.isnan(qp)
    return {"p": np.sum(w * t),
            "actual_accuracy": np.sum(w * g.acc.to_numpy()),
            "panickssery": np.sum(w * q) - 0.5,
            "equal_opportunity": tpr - (1 - fp),
            "hspp": fp,
            "roytburg": wm(q - qp, matched) if w[matched].sum() > 0 else np.nan,
            "ess": 1 / np.sum(w ** 2)}


def bin_weights(g, covariate):
    """(row bin index, bin p, bin accuracy, bin size)."""
    b = pd.qcut(g[covariate].rank(method="first"), N_BINS, labels=False).to_numpy()
    grouped = g.groupby(b)
    return b, grouped.a_correct.mean().to_numpy(), grouped.acc.mean().to_numpy(), grouped.size().to_numpy()


def extreme(bin_p, bin_acc, target_p, sign):
    """Bin weights minimizing sign * accuracy with weighted p = target_p."""
    r = linprog(sign * bin_acc, A_eq=[np.ones(N_BINS), bin_p], b_eq=[1, target_p],
                bounds=(0, MAX_BIN_WEIGHT))
    return r.x if r.success else None


def run(df) -> pd.DataFrame:
    rows = []
    for judge, g in df.groupby("judge"):
        for cov in COVARIATES:
            h = g.dropna(subset=[cov]).reset_index(drop=True)
            b, bp, ba, n = bin_weights(h, cov)
            base = {"judge": judge, "covariate": cov, "n_pairs": len(h),
                    "r_cov_acc": np.corrcoef(h[cov], h.acc)[0, 1],
                    "r_cov_label": np.corrcoef(h[cov], h.a_correct)[0, 1]}
            for name, w in (("natural", n / n.sum()), ("low", extreme(bp, ba, h.a_correct.mean(), 1)),
                            ("high", extreme(bp, ba, h.a_correct.mean(), -1))):
                if w is None:
                    continue
                rows.append({**base, "point": name, **weighted_stats(h, (w / n)[b])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = add_covariates(load(), np.random.default_rng(SEED))
    # unit weights must reproduce the real measure code
    g = df[df.judge == df.judge.iloc[0]].reset_index(drop=True)
    st = weighted_stats(g, np.ones(len(g)))
    t, v, px = list(g.a_correct), list(g.q), [None if np.isnan(x) else x for x in g.q_proxy]
    for m, val in (("panickssery", panickssery.estimate_bias(v)), ("equal_opportunity", equal_opportunity.estimate_bias(t, v)),
                   ("hspp", hspp.estimate_bias(t, v)), ("roytburg", roytburg.estimate_bias(t, v, px))):
        assert abs(st[m] - val) < 1e-9, (m, st[m], val)

    res = run(df)
    res.to_csv(OUT_PATH, index=False)
    print(f"{len(df)} pairs; wrote {OUT_PATH}")
    wide = res.pivot_table(index=["covariate", "judge"], columns="point",
                           values=["actual_accuracy", "ess", "p"] + list(MEASURES))
    print("\n== Accuracy range at the judge's natural p")
    a = wide["actual_accuracy"][["low", "natural", "high"]].copy()
    a["range"] = a.high - a.low
    a["ess_low"], a["ess_high"], a["n"] = wide["ess"]["low"], wide["ess"]["high"], res[res.point == "natural"].set_index(["covariate", "judge"]).n_pairs
    print(a.round(3).to_string())
    print("\n== Covariate correlations, mean across judges")
    print(res[res.point == "natural"].groupby("covariate")[["r_cov_acc", "r_cov_label"]].mean().round(3).to_string())
    print("\n== Mean across judges")
    print(res.groupby(["covariate", "point"])[["p", "actual_accuracy"] + list(MEASURES)].mean().round(3).to_string())
