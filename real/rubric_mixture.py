"""
Rubric-based null test with fictional targets (method principles 5 and 7).

For judge J and two models X (stronger: higher pass rate) and Y, both outside J's family, the
fictional target answers each prompt with X's answer with probability pi, else Y's (pi = 0, 0.5, 1;
pi = 0 and 1 are the real models). Prompts are those both X and Y answered; the comparison models C
are all models outside J's, X's and Y's families, on the same prompts. So the judge, the prompts and
C stay fixed while the target changes: a same-item comparison of a stronger and a weaker model.
(The mixture is computed exactly by weighting X's units by pi and Y's by 1 - pi.)

Every measure of measures/rubric.py, unfixed and fixed:
    usual          as published
    same quality   C restricted to models whose pass rate is within DELTA of the target's
    second judge   minus the same measure from judge K (outside J's, X's, Y's families) on the same
                   units, averaged over K (IFEval only: HealthBench labels depend on the judge)
    both           same quality and second judge
For each: the null value (existence), the change from Y to X on the same prompts (trend), and the
response to a known self-preference: J's verdicts on the target's units raised to v + iota*(1 - v),
iota = 0.1, response = (value with it - value without) / iota (detection; K's verdicts untouched).

Run with: [RUBRIC_DATASET=healthbench] python -m real.rubric_mixture
          writes real/rubric_mixture_results[_healthbench].csv
"""

import itertools
import os

import numpy as np
import pandas as pd

from infer.ifeval_rubric import DATASET, family
from measures.rubric import MEASURES
from real.rubric_control import DELTA
from real.rubric_real import units

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "rubric_mixture_results" + ("" if DATASET == "ifeval" else f"_{DATASET}") + ".csv")
PIS = (0.0, 0.5, 1.0)
IOTA = 0.1
MIN_UNITS = 50


def wmean(x, w):
    s = w.sum()
    return float((x * w).sum() / s) if s > 0 else np.nan


def measures(o, k):
    """o: own units (v, y, item, w); k: comparison units (v, y, item). Weighted on own units."""
    ov, oy, ow = o.v.to_numpy(), o.y.to_numpy(), o.w.to_numpy()
    kv, ky = k.v.to_numpy(), k.y.to_numpy()
    fpr_o, fpr_k = wmean(ov[~oy], ow[~oy]), kv[~ky].mean() if (~ky).any() else np.nan
    tpr_o, tpr_k = wmean(ov[oy], ow[oy]), kv[ky].mean() if ky.any() else np.nan
    out = {"score_gap": wmean(ov, ow) - kv.mean(),
           "calibrated_gap": (wmean(ov, ow) - wmean(oy, ow)) - (kv.mean() - ky.mean()),
           "mrspb": wmean(ov * ~oy, ow) - (kv * ~ky).mean(),
           "hspp_rub": fpr_o - fpr_k,
           "hspp_rub_ratio": fpr_o / fpr_k - 1 if fpr_k > 0.05 else np.nan,
           "eo_rub": ((tpr_o - tpr_k) + (fpr_o - fpr_k)) / 2}
    km = k.groupby(["item", "y"]).v.mean()
    idx = pd.MultiIndex.from_arrays([o.item, o.y])
    vo = km.reindex(idx).to_numpy()
    ok = ~np.isnan(vo)
    d = ov - vo
    out["matched_fail"] = wmean(d[ok & ~oy], ow[ok & ~oy])
    out["matched_both"] = wmean(d[ok], ow[ok])
    return out


def main():
    df = units()
    df["key"] = df.key.astype(str)
    pq = df.drop_duplicates(["generator", "item"]).groupby("generator").y.mean()
    keys_of = df.groupby("generator").key.agg(lambda s: set(s))
    second = DATASET == "ifeval"
    by_j = {j: g for j, g in df.groupby("judge")}
    rows = []
    only = os.environ.get("MIX_JUDGE")                  # optional: one judge (testing)
    for J, dj in by_j.items():
        if only and J != only:
            continue
        elig = sorted(g for g in pq.index if family(g) != family(J) and g in set(dj.generator))
        for X, Y in itertools.permutations(elig, 2):
            if pq[X] <= pq[Y]:
                continue
            keys = keys_of[X] & keys_of[Y]
            fams = {family(J), family(X), family(Y)}
            C = [g for g in elig if family(g) not in fams]
            if len(C) < 2:
                continue
            ks = [K for K in by_j if family(K) not in fams] if second else []
            sub = dj[dj.key.isin(keys)]
            own_all = sub[sub.generator.isin([X, Y])]
            if (~own_all.y).sum() < MIN_UNITS:
                continue
            for pi in PIS:
                p_t = pi * pq[X] + (1 - pi) * pq[Y]
                o = own_all.assign(w=np.where(own_all.generator == X, pi, 1 - pi))
                o = o[o.w > 0]
                oi = o.assign(v=o.v + IOTA * (1 - o.v))
                sets = {"usual": C, "same quality": [g for g in C if abs(pq[g] - p_t) <= DELTA]}
                for sname, cs in sets.items():
                    if not cs:
                        continue
                    k = sub[sub.generator.isin(cs)]
                    base, inj = measures(o, k), measures(oi, k)
                    gap = p_t - pq[cs].mean()
                    rows += [{"judge": J, "X": X, "Y": Y, "pi": pi, "set": sname, "measure": m, "gap": gap,
                              "target_pass": p_t, "value": base[m], "response": (inj[m] - base[m]) / IOTA} for m in MEASURES]
                    if not ks:
                        continue
                    diffs = {m: [] for m in MEASURES}
                    for K in ks:
                        dk = by_j[K]
                        dk = dk[dk.key.isin(keys)]
                        ok_units = set(zip(o.generator, o.item))
                        ko = dk[dk.generator.isin([X, Y])]
                        ko = ko[[t in ok_units for t in zip(ko.generator, ko.item)]]
                        ko = ko.assign(w=np.where(ko.generator == X, pi, 1 - pi))
                        kk = dk[dk.generator.isin(cs)]
                        if (~ko.y).sum() < 20 or len(kk) < 20:
                            continue
                        mk = measures(ko, kk)
                        for m in MEASURES:
                            diffs[m].append(mk[m])
                    name = "second judge" if sname == "usual" else "both"
                    for m in MEASURES:
                        if diffs[m]:
                            kmean = np.nanmean(diffs[m])
                            rows.append({"judge": J, "X": X, "Y": Y, "pi": pi, "set": name, "measure": m, "gap": gap,
                                         "target_pass": p_t, "value": base[m] - kmean, "response": (inj[m] - base[m]) / IOTA})
        print(f"{J}: {len(rows)} rows so far", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(OUT, index=False)
    summary(r)


def summary(r):
    ends = r[r.pi.isin([0.0, 1.0])]
    w = r.pivot_table(index=["judge", "X", "Y", "set", "measure"], columns="pi", values="value").dropna(subset=[0.0, 1.0])
    q = r[r.pi == 1.0].groupby(["judge", "X", "Y", "set", "measure"]).target_pass.first() - \
        r[r.pi == 0.0].groupby(["judge", "X", "Y", "set", "measure"]).target_pass.first()
    w["dq"] = q.reindex(w.index)
    w["trend"] = (w[1.0] - w[0.0]) / w.dq
    w["mid_vs_line"] = w[0.5] - (w[0.0] + w[1.0]) / 2
    rows = []
    for (s, m), g in w.groupby(level=["set", "measure"]):
        e = ends[(ends.set == s) & (ends.measure == m)]
        rows.append({"set": s, "measure": m, "null value (median)": e.value.median(),
                     "null |value| (median)": e.value.abs().median(),
                     "trend per unit pass rate (median)": g.trend.median(),
                     "share of pairs X above Y": float((g[1.0] > g[0.0]).mean()),
                     "detection response (median)": r[(r.set == s) & (r.measure == m)].response.median(),
                     "pairs": len(g)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
