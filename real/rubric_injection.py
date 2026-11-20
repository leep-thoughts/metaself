"""
Injection test for the rubric-based measures and the two fixes (IFEval), on top of the null test
(real/rubric_placebo.py, real/rubric_control.py).

Null-test pairings: judge J, target G outside J's family (true self-preference 0). We inject a known
self-preference: J's verdicts on G's answers only are shifted by delta in log-odds,
    v' = sigmoid(logit(v) + delta),
everything else unchanged (other judges' verdicts, J's verdicts on other answers). Only assumption:
self-preference acts as a log-odds shift (the logistic model).

Per measure and comparison set we report the response dM/d(delta) at delta = 0 (central difference
over +-0.5) and check
    R3  it responds (mean response > 0)
    R4  it responds the same for every target: spread of the response across (judge, target), and
        its correlation with the target's strength gap (pass rate - comparison models' pass rate)
    linearity  response from +-1 vs +-0.5 (ratio near 1 = linear in delta)
The measures live on the probability scale, so they do not move by delta itself; what matters is that
the response does not depend on which target (or how strong) it is.

Comparison sets: all (usual), Fix A (strength-matched, |pass-rate diff| <= .04);
Fix B (two-judge difference, all comparison models) and A + B, for matched_fail.

Form B (python -m real.rubric_injection waive): instead of a log-odds shift, the judge waves the
target's answers through with probability pi, at random (independent of item and answer):
    v' = v + pi * (1 - v)
Works for near-certain judges too (a sure "not met", v ~ 0, becomes v ~ pi). Response = dM/d(pi) from
pi = 0.1 (and 0.2 for the linearity check).

Run with: python -m real.rubric_injection [shift|waive]   writes real/rubric_injection_results[_waive].csv
"""

import os
import sys

import numpy as np
import pandas as pd

from infer.ifeval_rubric import DATASET, family
from measures.rubric import MEASURES
from real.rubric_control import DELTA, estimate_w
from real.rubric_real import units

FORM = sys.argv[1] if len(sys.argv) > 1 else "shift"
CLIP = 1e-4
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   ("rubric_injection_results" + ("" if FORM == "shift" else f"_{FORM}") + ("" if DATASET == "ifeval" else f"_{DATASET}") + ".csv"))
# injection grid and response: (lo, hi) for the response, (lo2, hi2) for the linearity check
if FORM == "shift":
    DELTAS = [-1.0, -0.5, 0.5, 1.0]
    R1, R2 = (-0.5, 0.5), (-1.0, 1.0)
else:
    DELTAS = [0.0, 0.1, 0.2]
    R1, R2 = (0.0, 0.1), (0.0, 0.2)


def shift(v, delta):
    if FORM == "waive":                       # waved through with probability delta (= pi)
        return v + delta * (1 - v)
    v = np.clip(v, CLIP, 1 - CLIP)
    return 1 / (1 + np.exp(-(np.log(v / (1 - v)) + delta)))


def response(f):
    """f(delta) -> value; returns (response at R1, linearity = response at R2 / response at R1)."""
    r1 = (f(R1[1]) - f(R1[0])) / (R1[1] - R1[0])
    r2 = (f(R2[1]) - f(R2[0])) / (R2[1] - R2[0])
    return r1, (r2 / r1 if r1 else np.nan)


def single_judge(df, rate):
    rows = []
    df = df.assign(gfam=df.generator.map(family))
    for J, dj in df.groupby("judge"):
        for G in rate.index:
            if family(G) == family(J):
                continue
            elig = [g for g in rate.index if family(g) not in (family(J), family(G))]
            diff = rate[elig] - rate[G]
            gap = rate[G] - rate[elig].mean()
            for name, w in (("all", pd.Series(1.0, index=elig)), ("Fix A", (diff.abs() <= DELTA).astype(float))):
                if w.sum() == 0:
                    continue
                d = dj[(dj.generator == G) | dj.generator.isin(w.index[w > 0])].copy()
                d["own"] = d.generator == G
                d["w"] = np.where(d.own, 1.0, d.generator.map(w).fillna(0.0))
                if (~d[d.own].y).sum() < 10:
                    continue
                base_v = d.v.to_numpy()
                est = {}
                for dl in DELTAS:
                    d["v"] = np.where(d.own, shift(base_v, dl), base_v)
                    est[dl] = estimate_w(d)
                for m in MEASURES:
                    r, lin = response(lambda dl: est[dl][m])
                    rows.append({"set": name, "measure": m, "judge": J, "target": G, "gap": gap,
                                 "response": r, "linearity": lin})
    return rows


def two_judge(df, rate, delta_match):
    """Fix B (delta_match None) and A + B: shift only J's column on the target's units."""
    v = df.pivot_table(index=["generator", "key", "item"], columns="judge", values="v")
    y = df.drop_duplicates(["generator", "item"]).set_index(["generator", "key", "item"]).y.reindex(v.index)
    v = v[~y.to_numpy()]
    gens = v.index.get_level_values("generator")
    rows = []
    for J in v.columns:
        for G in rate.index:
            if family(G) == family(J):
                continue
            own = v[gens == G]
            it = own.index.get_level_values("item")
            elig_all = [g for g in rate.index if family(g) not in (family(J), family(G))]
            gap = rate[G] - rate[elig_all].mean()
            res = {}
            for dl in DELTAS:
                per_k = []
                for K in v.columns:
                    if family(K) in (family(J), family(G)):
                        continue
                    ok = [g for g in rate.index if family(g) not in (family(J), family(K), family(G))
                          and (delta_match is None or abs(rate[g] - rate[G]) <= delta_match)]
                    if not ok:
                        continue
                    cm = v[gens.isin(ok)].groupby(level="item")[[J, K]].mean()
                    oj = shift(own[J].to_numpy(), dl)
                    dd = (oj - cm[J].reindex(it).to_numpy()) - (own[K].to_numpy() - cm[K].reindex(it).to_numpy())
                    if np.isfinite(dd).any():
                        per_k.append(np.nanmean(dd))
                if per_k:
                    res[dl] = float(np.mean(per_k))
            if len(res) == len(DELTAS):
                r, lin = response(lambda dl: res[dl])
                rows.append({"set": "Fix B" if delta_match is None else "Fix A + B", "measure": "matched_fail",
                             "judge": J, "target": G, "gap": gap, "response": r, "linearity": lin})
    return rows


def main():
    df = units()
    rate = df.groupby("generator").y.mean()
    rows = single_judge(df, rate)
    if DATASET == "ifeval":   # Fix B needs labels shared by both judges
        with np.errstate(all="ignore"):
            rows += two_judge(df, rate, None) + two_judge(df, rate, DELTA)
    r = pd.DataFrame(rows)
    r.to_csv(OUT, index=False)
    summ = []
    for (s, m), x in r.groupby(["set", "measure"]):
        x = x.dropna(subset=["response"])
        mean = x.response.mean()
        summ.append({"measure": m, "set": s, "n": len(x), "mean response": mean,
                     "relative spread (sd/mean)": x.response.std() / abs(mean) if mean else np.nan,
                     "corr with target strength": np.corrcoef(x.gap, x.response)[0, 1],
                     "share responding > 0": (x.response > 0).mean(),
                     "linearity (±1 vs ±0.5)": x.linearity.median()})
    s = pd.DataFrame(summ).sort_values(["measure", "set"])
    print(s.round(3).to_string(index=False))
    # judge vs target: share of the response's variance between judges; within-judge spread and
    # correlation with target strength
    dec = []
    for (st, m), x in r.groupby(["set", "measure"]):
        x = x.dropna(subset=["response"])
        within = x.response - x.groupby("judge").response.transform("mean")
        gw = x.gap - x.groupby("judge").gap.transform("mean")
        dec.append({"measure": m, "set": st, "share of variance from judge": 1 - within.var() / x.response.var(),
                    "within-judge sd/mean": within.std() / x.response.mean(),
                    "within-judge corr with strength": np.corrcoef(gw, within)[0, 1]})
    print(); print(pd.DataFrame(dec).sort_values(["measure", "set"]).round(3).to_string(index=False))
    print(); print(r[(r.measure == "matched_fail") & (r.set == "all")].groupby("judge").response
                   .agg(["mean", "std"]).sort_values("mean").round(3).to_string())


if __name__ == "__main__":
    main()
