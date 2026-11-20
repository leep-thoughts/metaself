"""
Evaluation-accuracy confound for the rubric-based measures, the fixes and the waved-through rate
(null test and form-B injection), IFEval or HealthBench (RUBRIC_DATASET).

Evaluation accuracy is varied WITHIN a judge: rubric items are split into thirds by a covariate fixed
before the judge's own verdicts are looked at, namely how often judges outside the judge's family grade
that item correctly (over all generators). On hard thirds the judge is less accurate. For every
null-test pairing (judge J, target G outside J's family), comparison set and third we compute:
    - J's evaluation accuracy: balanced accuracy (mean of TPR and TNR) on the comparison models'
      answers in that third
    - the generation-quality gap: target's pass rate minus the comparison models', in that third
    - every measure (null test) and its response to form-B injection (pi = 0.1)
Then, within each pairing (demeaned), OLS of the measure on [evaluation accuracy, generation-quality
gap]: the accuracy coefficient is the slope with evaluation accuracy holding generation quality fixed
(want 0). Fix B (two-judge difference) is included for IFEval only.

Run with: [RUBRIC_DATASET=healthbench] python -m real.rubric_accuracy
          writes real/rubric_accuracy_results[_healthbench].csv
"""

import os

import numpy as np
import pandas as pd

from infer.ifeval_rubric import DATASET, family
from measures.rubric import MEASURES
from real.rubric_control import DELTA, estimate_w
from real.rubric_real import units

N_BINS = 3
PI = 0.1
MIN_FAILED = 10     # failed items per third for the waved-through rate
MIN_REJECT = 0.3
MIN_FPR_RATIO = 0.05  # Pombal's ratio only where the comparison failures are accepted at least this often
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "rubric_accuracy_results" + ("" if DATASET == "ifeval" else f"_{DATASET}") + ".csv")


def item_difficulty(df):
    """Per judge J: share of correct verdicts on each rubric item by judges outside J's family."""
    df = df.assign(ok=((df.v > 0.5) == df.y).astype(float), jfam=df.judge.map(family))
    by = df.groupby(["item", "jfam"]).ok.agg(["sum", "count"])
    tot = by.groupby(level="item").sum()
    out = {}
    for J in df.judge.unique():
        f = family(J)
        own = by.xs(f, level="jfam") if f in by.index.get_level_values("jfam") else None
        s, c = tot["sum"].copy(), tot["count"].copy()
        if own is not None:
            s = s.sub(own["sum"], fill_value=0)
            c = c.sub(own["count"], fill_value=0)
        out[J] = (s / c.replace(0, np.nan)).dropna()
    return out


def balanced_acc(v, y):
    pos, neg = y, ~y
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    return 0.5 * ((v[pos] > 0.5).mean() + (v[neg] <= 0.5).mean())


def waved(d):
    f = d[~d.y]
    own = f[f.own].set_index("item").v
    oth = f[~f.own].groupby("item").v.mean()
    m = pd.concat([own.rename("o"), oth.rename("k")], axis=1).dropna()
    if len(m) < MIN_FAILED:
        return np.nan, np.nan, np.nan
    rej = (1 - m.k).mean()
    if rej < MIN_REJECT:                       # as real/rubric_waved.py: unstable denominator
        return np.nan, np.nan, rej
    return (m.o - m.k).mean() / rej, (1 - m.o).mean() / rej, rej


def did(v_bin, gens, gfam, J, G, ok_fn):
    """Two-judge difference (Fix B) for one pairing on the given failed units (pivot judge columns)."""
    own = v_bin[gens == G]
    if own.empty:
        return np.nan
    it = own.index.get_level_values("item")
    per_k = []
    for K in v_bin.columns:
        if family(K) in (family(J), family(G)) or K == J:
            continue
        ok = ok_fn(K)
        comp = v_bin[np.isin(gens, ok)]
        if comp.empty:
            continue
        cm = comp.groupby(level="item")[[J, K]].mean()
        dd = (own[J].to_numpy() - cm[J].reindex(it).to_numpy()) - (own[K].to_numpy() - cm[K].reindex(it).to_numpy())
        if np.isfinite(dd).any():
            per_k.append(np.nanmean(dd))
    return float(np.mean(per_k)) if per_k else np.nan


def main():
    df = units()
    rate = df.groupby("generator").y.mean()
    diff = item_difficulty(df)
    if DATASET == "ifeval":
        piv = df.pivot_table(index=["generator", "key", "item"], columns="judge", values="v")
        yy = df.drop_duplicates(["generator", "item"]).set_index(["generator", "key", "item"]).y.reindex(piv.index)
        piv = piv[~yy.to_numpy()]
    rows = []
    for J, dj in df.groupby("judge"):
        dif = diff[J]
        bins = pd.qcut(dif.rank(method="first"), N_BINS, labels=False)
        for b in range(N_BINS):
            items = set(bins.index[bins == b])
            db = dj[dj.item.isin(items)]
            for G in rate.index:
                if family(G) == family(J):
                    continue
                elig = [g for g in rate.index if family(g) not in (family(J), family(G))]
                fixa = [g for g in elig if abs(rate[g] - rate[G]) <= DELTA]
                for name, ok in (("usual", elig), ("Fix A", fixa)):
                    if not ok:
                        continue
                    d = db[(db.generator == G) | db.generator.isin(ok)].copy()
                    d["own"] = d.generator == G
                    d["w"] = 1.0
                    o, k = d[d.own], d[~d.own]
                    if (~o.y).sum() < 5 or k.empty:
                        continue
                    base_v = d.v.to_numpy()
                    m0 = estimate_w(d)
                    d["v"] = np.where(d.own, base_v + PI * (1 - base_v), base_v)
                    m1 = estimate_w(d)
                    d["v"] = base_v
                    if (k.v[~k.y]).mean() < MIN_FPR_RATIO:      # ratio's denominator too small
                        m0["hspp_rub_ratio"] = m1["hspp_rub_ratio"] = np.nan
                    row = {"judge": J, "target": G, "set": name, "bin": b,
                           "eval_acc": balanced_acc(k.v.to_numpy(), k.y.to_numpy()),
                           "gen_gap": o.y.mean() - k.y.mean()}
                    for m in MEASURES:
                        row[m] = m0[m]
                        row[m + "_resp"] = (m1[m] - m0[m]) / PI
                    if name == "Fix A":
                        row["waved_through"], row["waved_through_resp"], row["reject"] = waved(d)
                    rows.append(row)
                if DATASET == "ifeval":
                    vb = piv[piv.index.get_level_values("item").isin(items)]
                    gens = vb.index.get_level_values("generator").to_numpy()
                    gfam = None
                    for name, filt in (("Fix B", None), ("Fix A + B", DELTA)):
                        ok_fn = lambda K, filt=filt: [g for g in rate.index if family(g) not in (family(J), family(K), family(G))
                                                      and (filt is None or abs(rate[g] - rate[G]) <= filt)]
                        val = did(vb, gens, gfam, J, G, ok_fn)
                        base = [r for r in rows if r["judge"] == J and r["target"] == G and r["bin"] == b and r["set"] == "usual"]
                        if base and np.isfinite(val):
                            rows.append({"judge": J, "target": G, "set": name, "bin": b, "eval_acc": base[0]["eval_acc"],
                                         "gen_gap": base[0]["gen_gap"], "matched_fail": val})
    r = pd.DataFrame(rows)
    r.to_csv(OUT, index=False)
    summary(r)


def within_slope(x, cols):
    """OLS within pairing (judge, target, set): coefficient of each measure on eval_acc and gen_gap."""
    out = []
    for (s, m), _ in [((s, m), None) for s in x.set.unique() for m in cols]:
        d = x[(x.set == s)].replace([np.inf, -np.inf], np.nan).dropna(subset=[m, "eval_acc", "gen_gap"])
        g = d.groupby(["judge", "target"])
        d = d[g[m].transform("size") >= 2]
        if len(d) < 10:
            continue
        g = d.groupby(["judge", "target"])
        X = np.c_[d.eval_acc - g.eval_acc.transform("mean"), d.gen_gap - g.gen_gap.transform("mean")]
        y = d[m] - g[m].transform("mean")
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        acc_range = (d.groupby(["judge", "target"]).eval_acc.agg(lambda z: z.max() - z.min())).mean()
        out.append({"set": s, "measure": m, "n_points": len(d), "slope_eval_acc": b[0], "slope_gen_gap": b[1],
                    "mean within-pairing acc range": acc_range,
                    "implied change over that range": b[0] * acc_range})
    return pd.DataFrame(out)


def summary(r):
    cols = [m for m in MEASURES + ["waved_through"] if m in r]
    print("NULL TEST: slope with evaluation accuracy (within pairing, holding generation quality; want 0)")
    print(within_slope(r, cols).round(3).to_string(index=False))
    rc = [m + "_resp" for m in cols if m + "_resp" in r]
    print("\nINJECTION (form B): slope of the share read with evaluation accuracy (want 0)")
    print(within_slope(r, rc).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
