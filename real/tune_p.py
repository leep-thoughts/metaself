"""
How far can p move on real data with actual accuracy and true self_lean held fixed?

Data: the cross-family null pairs, true s ~ 0. p is moved through WHO wrote the answers:
items are grouped by generator pair (task, proxy K, reference R), and whole groups are
reweighted, never individual labels or votes. A group's p comes from other problems than
the ones scored, so no item's own label decides its weight.

Each repeat splits problems in half. On the fit half, groups are sorted by p into N_BINS
bins, and a linear program picks bin weights minimizing / maximizing p subject to weighted
accuracy = the judge's natural accuracy (no bin above MAX_BIN_WEIGHT). On the held-out half
those weights are scored.

Methods:
    generator  A is always K, as in the data
    flip       also allow naming R as A for a whole generator pair (l -> -l). J wrote
               neither answer, so which one is "A" is arbitrary here. Per-item accuracy is
               unchanged. Only valid on null data, where A is not really the judge's own.
    placebo    groups get random covariates: how much range the method finds in noise

Run with: python -m real.tune_p
"""

import os

import numpy as np
import pandas as pd

from measures import panickssery, equal_opportunity
from real.null_injection import load
from real.tune_accuracy import N_BINS, MEASURES, add_covariates, extreme, weighted_stats

N_REPS = 20
GROUP = ["task", "proxy", "opponent"]
TASKS = ["math500", "mmlu", "mbpp-plus"]
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tune_p_results.csv")


def flipped(df) -> pd.DataFrame:
    """Same judgments with R named as A. Roytburg's matched proxies become the other
    references R' on the same (judge, K, problem): R' vs K, with the same correctness."""
    out = df.copy()
    out["a_correct"] = ~df.a_correct
    out["l"] = -df.l
    out["q"] = 1 - df.q
    keys = [df[c] for c in ("task", "judge", "proxy", "problem_idx")]
    total = out.q.groupby(keys).transform("sum")
    count = out.q.groupby(keys).transform("count")
    out["q_proxy"] = np.where(count > 1, (total - out.q) / (count - 1).clip(lower=1), np.nan)
    return out


def assign_bins(train_counts, group_p, rng):
    """Whole groups into N_BINS bins by p, about equal rows each. Random order within ties."""
    keys = list(train_counts.index)
    order = sorted(keys, key=lambda k: (group_p[k], rng.random()))
    cum = np.cumsum([train_counts[k] for k in order]) - 1
    return {k: min(int(N_BINS * c / cum[-1]), N_BINS - 1) for k, c in zip(order, cum)}


def tune_judge(fit, test, group_p, target_acc, rng):
    """Low / high p weights fit on one half, scored on the other. Group key = GROUP + flip."""
    keys = GROUP + ["flip"]
    counts = fit.groupby(keys).size()
    bins = assign_bins(counts, group_p, rng)
    b_fit = np.array([bins[k] for k in fit[keys].itertuples(index=False, name=None)])
    grouped = fit.groupby(b_fit)
    if grouped.ngroups != N_BINS:
        return []
    bp, ba = grouped.a_correct.mean().to_numpy(), grouped.acc.mean().to_numpy()
    test = test[[k in bins for k in test[keys].itertuples(index=False, name=None)]].reset_index(drop=True)
    b_test = np.array([bins[k] for k in test[keys].itertuples(index=False, name=None)])
    n = np.bincount(b_test, minlength=N_BINS)
    rows = []
    for point, sign in (("low", 1), ("high", -1)):
        # minimize / maximize p with accuracy held at the natural level
        w = extreme_p(bp, ba, target_acc, sign)
        if w is None:
            continue
        w = np.where(n > 0, w, 0)
        row_w = (w / np.maximum(n, 1))[b_test]
        st = weighted_stats(test, row_w)
        shares = {t: row_w[test.task == t].sum() / row_w.sum() for t in TASKS}
        rows.append({"point": point, **st, **shares})
    return rows


def extreme_p(bin_p, bin_acc, target_acc, sign):
    # same LP as tune_accuracy.extreme with the roles of p and accuracy swapped
    return extreme(bin_acc, bin_p, target_acc, sign)


def run(df) -> pd.DataFrame:
    natural = df.assign(flip=False)
    both = pd.concat([natural, flipped(df).assign(flip=True)], ignore_index=True)
    items = df.drop_duplicates(GROUP + ["problem_idx"])
    problems = df[["task", "problem_idx"]].drop_duplicates()
    out = []
    for rep in range(N_REPS):
        rng = np.random.default_rng(rep)
        fold = problems.assign(fold=rng.integers(2, size=len(problems)))
        fit_items = items.merge(fold, on=["task", "problem_idx"]).query("fold == 0")
        p0 = fit_items.groupby(GROUP).a_correct.mean()
        group_p = {(*k, False): v for k, v in p0.items()} | {(*k, True): 1 - v for k, v in p0.items()}
        shuffled = dict(zip(group_p, rng.permutation(list(group_p.values()))))
        for method, data, gp in (("generator", natural, group_p), ("flip", both, group_p),
                                 ("placebo", natural, shuffled)):
            d = data.merge(fold, on=["task", "problem_idx"])
            for judge, g in d.groupby("judge"):
                fit, test = g[g.fold == 0], g[g.fold == 1]
                fit = fit[[k in gp for k in fit[GROUP + ["flip"]].itertuples(index=False, name=None)]].reset_index(drop=True)
                target = natural_acc = fit[~fit.flip].acc.mean()
                base = {"rep": rep, "method": method, "judge": judge, "natural_p": test[~test.flip].a_correct.mean(),
                        "natural_acc": natural_acc}
                out += [{**base, **r} for r in tune_judge(fit, test.reset_index(drop=True), gp, target, rng)]
    return pd.DataFrame(out)


if __name__ == "__main__":
    df = add_covariates(load(), np.random.default_rng(0))
    # a full flip must exactly negate panickssery and equal opportunity
    g = df[df.judge == df.judge.iloc[0]]
    f = flipped(g)
    assert abs(panickssery.estimate_bias(list(f.q)) + panickssery.estimate_bias(list(g.q))) < 1e-9
    assert abs(equal_opportunity.estimate_bias(list(f.a_correct), list(f.q))
               + equal_opportunity.estimate_bias(list(g.a_correct), list(g.q))) < 1e-9

    res = run(df)
    res.to_csv(OUT_PATH, index=False)
    print(f"{len(df)} pairs; wrote {OUT_PATH}")
    m = res.groupby(["method", "judge", "point"])[["p", "actual_accuracy"]].mean().unstack("point")
    t = pd.DataFrame({"p_low": m["p"]["low"], "p_high": m["p"]["high"],
                      "acc_gap": m["actual_accuracy"]["high"] - m["actual_accuracy"]["low"]})
    t["natural_p"] = res.groupby(["method", "judge"]).natural_p.mean()
    t["natural_acc"] = res.groupby(["method", "judge"]).natural_acc.mean()
    print(t.round(3).to_string())
    print("\n== Mean across judges (held-out)")
    print(res.groupby(["method", "point"])[["p", "actual_accuracy"] + list(MEASURES) + TASKS + ["ess"]].mean().round(3).T.to_string())
