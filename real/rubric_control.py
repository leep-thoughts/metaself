"""
Confound control by choosing the comparison set (pipeline stage S3): subsample or reweight the
"other" generators toward the "own" generator's quality, then rerun the placebo null test
(real/rubric_placebo.py) on IFEval.

Why the comparison set: own and others answer the SAME criteria, so reweighting items cannot
close the quality gap p - q (it is generator ability, not item mix); reweighting units on their
own label would overwrite the labels. What can be chosen without touching labels or verdicts
is WHICH generators serve as "others", by their overall pass rate (a generator-level covariate).

Comparison sets (generators outside the judge's and the own generator's families):
    all        every eligible generator (= rubric_placebo)
    subsample  only those with |pass rate - own pass rate| <= DELTA
    reweight   weight exp(-0.5 * (diff / BAND)^2), normalised (soft version)
If halo grows with generator strength, strength-matched others also share the own generator's halo.

Per comparison set and measure: placebo mean (want 0), slope on the placebo's ORIGINAL quality
gap p - q_all (want 0: the measure no longer tracks how strong the "own" generator is), placebo
sd, and how far the true self assignments sit above the placebo (power).

Run with: python -m real.rubric_control     writes real/rubric_control_results.csv
"""

import os

import numpy as np
import pandas as pd

from infer.ifeval_rubric import DATASET, family
from measures.rubric import MEASURES
from real.rubric_real import units

DELTA, BAND = 0.04, 0.03
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rubric_control_results" + ("" if DATASET == "ifeval" else f"_{DATASET}") + ".csv")


def wmean(x, w):
    return float(np.sum(x * w) / np.sum(w)) if np.sum(w) > 0 else np.nan


def estimate_w(d):
    """measures/rubric.estimate with weights w on the 'other' units (own units weight 1)."""
    o, k = d[d.own], d[~d.own]
    ov, oy, kv, ky, kw = o.v.to_numpy(), o.y.to_numpy(), k.v.to_numpy(), k.y.to_numpy(), k.w.to_numpy()
    out = {
        "score_gap": ov.mean() - wmean(kv, kw),
        "calibrated_gap": (ov.mean() - oy.mean()) - (wmean(kv, kw) - wmean(ky, kw)),
        "mrspb": (ov * ~oy).mean() - wmean(kv * ~ky, kw),
        "hspp_rub": ov[~oy].mean() - wmean(kv[~ky], kw[~ky]),
        "eo_rub": ((ov[oy].mean() - wmean(kv[ky], kw[ky])) + (ov[~oy].mean() - wmean(kv[~ky], kw[~ky]))) / 2,
    }
    out["hspp_rub_ratio"] = ov[~oy].mean() / wmean(kv[~ky], kw[~ky]) - 1
    km = k.assign(vw=k.v * k.w).groupby(["item", "y"])[["vw", "w"]].sum()
    km = (km.vw / km.w).rename("v_other")
    m = o.join(km, on=["item", "y"]).dropna(subset=["v_other"])
    out["matched_fail"] = float((m.v - m.v_other)[~m.y].mean())
    out["matched_both"] = float((m.v - m.v_other).mean())
    return out


def main():
    df = units()
    df["gfam"] = df.generator.map(family)
    rate = df.groupby("generator").y.mean()   # pass rate (HealthBench: averaged over judge-specific labels)
    rows = []
    for K, dk in df.groupby("judge"):
        fk = family(K)
        for G in rate.index:
            fg = family(G)
            if fg == fk and G != K:
                continue
            elig = [g for g in rate.index if family(g) not in (fk, fg)]
            diff = rate[elig] - rate[G]
            sets = {"all": pd.Series(1.0, index=elig),
                    "subsample": (diff.abs() <= DELTA).astype(float),
                    "reweight": np.exp(-0.5 * (diff / BAND) ** 2)}
            q_all = rate[elig].mean()
            for name, w in sets.items():
                if w.sum() < 1e-6:
                    continue
                d = dk[(dk.generator == G) | dk.generator.isin(w.index[w > 1e-6])].copy()
                d["own"] = d.generator == G
                d["w"] = np.where(d.own, 1.0, d.generator.map(w).fillna(0.0))
                if (~d[d.own].y).sum() < 10:
                    continue
                rows.append({"set": name, "judge": K, "generator": G, "self": G == K,
                             "gap_all": rate[G] - q_all, "gap_used": rate[G] - wmean(rate[w.index].to_numpy(), w.to_numpy()),
                             "n_eff_others": w.sum() ** 2 / (w ** 2).sum(), **estimate_w(d)})
    r = pd.DataFrame(rows)
    r.to_csv(OUT, index=False)
    summ = []
    for (name, m), _ in pd.DataFrame([(s, m) for s in r.set.unique() for m in MEASURES]).groupby([0, 1]):
        rs = r[r.set == name]
        pl, se = rs[~rs.self].dropna(subset=[m]), rs[rs.self].dropna(subset=[m])
        summ.append({"set": name, "measure": m, "n_placebo": len(pl),
                     "placebo_mean": pl[m].mean(), "placebo_sd": pl[m].std(),
                     "slope_on_gap": np.polyfit(pl.gap_all, pl[m], 1)[0],
                     "corr_gap": np.corrcoef(pl.gap_all, pl[m])[0, 1],
                     "self_vs_placebo_sd": (se[m].mean() - pl[m].mean()) / pl[m].std()})
    s = pd.DataFrame(summ)
    gaps = r.groupby("set").apply(lambda x: pd.Series({"mean |gap used|": x.gap_used.abs().mean(),
                                                      "n_eff others": x.n_eff_others.mean()}))
    print(gaps.round(3).to_string())
    for m in MEASURES:
        print(f"\n{m}"); print(s[s.measure == m].drop(columns="measure").round(3).to_string(index=False))


if __name__ == "__main__":
    main()


def did_matched(delta=DELTA):
    """Both controls together: two-judge difference (real/rubric_did.py) where the comparison
    generators are ALSO strength-matched to the own generator (|pass rate diff| <= delta).
    Same placebo design as real/rubric_placebo.did_placebo; delta=None = all comparison generators."""
    df = units()
    rate = df.drop_duplicates(["generator", "item"]).groupby("generator").y.mean()
    v = df.pivot_table(index=["generator", "key", "item"], columns="judge", values="v")
    y = df.drop_duplicates(["generator", "item"]).set_index(["generator", "key", "item"]).y.reindex(v.index)
    v = v[~y.to_numpy()]
    gens = v.index.get_level_values("generator")
    rows = []
    for J in v.columns:
        for G in rate.index:
            if family(G) == family(J) and G != J:
                continue
            own = v[gens == G]
            it = own.index.get_level_values("item")
            per_k = []
            for K in v.columns:
                if family(K) in (family(J), family(G)):
                    continue
                ok = [g for g in rate.index if family(g) not in (family(J), family(K), family(G))
                      and (delta is None or abs(rate[g] - rate[G]) <= delta)]
                if not ok:
                    continue
                cm = v[gens.isin(ok)].groupby(level="item")[[J, K]].mean()
                d = (own[J].to_numpy() - cm[J].reindex(it).to_numpy()) - (own[K].to_numpy() - cm[K].reindex(it).to_numpy())
                if np.isfinite(d).any():
                    per_k.append(np.nanmean(d))
            if per_k:
                others = [g for g in rate.index if family(g) not in (family(J), family(G))]
                rows.append({"judge": J, "generator": G, "self": G == J, "did": float(np.mean(per_k)),
                             "gap_all": rate[G] - rate[others].mean(), "n_third_party": len(per_k)})
    r = pd.DataFrame(rows)
    pl, se = r[~r.self], r[r.self]
    return {"n_placebo": len(pl), "placebo_mean": pl.did.mean(), "placebo_sd": pl.did.std(),
            "slope_on_gap": np.polyfit(pl.gap_all, pl.did, 1)[0], "corr_gap": np.corrcoef(pl.gap_all, pl.did)[0, 1],
            "self_vs_placebo_sd": (se.did.mean() - pl.did.mean()) / pl.did.std(), "n_self": len(se)}, r
