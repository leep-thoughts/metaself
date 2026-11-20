"""
Real-data null test for the RUBRIC measures (IFEval): placebo "self" assignments.

For every judge K and generator G from a different family, treat G's answers as K's "own"
(true self-preference 0) and compute every measure of measures/rubric.py; "others" = generators
outside both K's and G's families. A good measure reads ~0 here (R2) and does not move with
the placebo's quality gap p - q (R1). Compare with the true self assignments (K = G).

Run with: python -m real.rubric_placebo     writes real/rubric_placebo_results.csv
"""

import os

import numpy as np
import pandas as pd

from infer.ifeval_rubric import DATASET, family
from measures.rubric import MEASURES, estimate
from real.rubric_real import units

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rubric_placebo_results" + ("" if DATASET == "ifeval" else f"_{DATASET}") + ".csv")


def main():
    df = units()
    df["gfam"] = df.generator.map(family)
    rows = []
    for K, dk in df.groupby("judge"):
        fk = family(K)
        for G in df.generator.unique():
            fg = family(G)
            if fg == fk and G != K:
                continue
            d = dk[(dk.generator == G) | ~dk.gfam.isin([fk, fg])].copy()
            d["own"] = d.generator == G
            if d.own.sum() < 50:
                continue
            o, k = d[d.own], d[~d.own]
            rows.append({"judge": K, "generator": G, "self": G == K, "p": o.y.mean(), "q": k.y.mean(),
                         "n_fail": int((~o.y).sum()), **estimate(d)})
    r = pd.DataFrame(rows)
    r["gap"] = r.p - r.q
    r.to_csv(OUT, index=False)
    pl, se = r[~r.self], r[r.self]
    out = []
    for m in MEASURES:
        x = pl[["gap", m]].dropna()
        slope = np.polyfit(x.gap, x[m], 1)[0]
        out.append({"measure": m, "placebo_mean": pl[m].mean(), "placebo_sd": pl[m].std(),
                    "placebo_|mean|>0.05": (pl[m].abs() > 0.05).mean(),
                    "slope_on_quality_gap": slope, "corr_with_gap": x.corr().iloc[0, 1],
                    "self_mean": se[m].mean(), "self_minus_placebo_in_sd": (se[m].mean() - pl[m].mean()) / pl[m].std()})
    print(f"{len(pl)} placebo pairs, {len(se)} true-self pairs")
    print(pd.DataFrame(out).round(3).to_string(index=False))



def did_placebo():
    """Two-judge difference (real/rubric_did.py) under the same placebo: own = G, third parties
    K outside the families of J and G, comparison generators outside J's, K's and G's families."""
    df = units()
    v = df.pivot_table(index=["generator", "key", "item"], columns="judge", values="v")
    y = df.drop_duplicates(["generator", "item"]).set_index(["generator", "key", "item"]).y.reindex(v.index)
    v = v[~y.to_numpy()]
    gens = v.index.get_level_values("generator")
    gfam = pd.Index(gens.map(family))
    items_all = v.index.get_level_values("item")
    pq = df.groupby("generator").y.mean()
    rows = []
    for J in v.columns:
        for G in sorted(set(gens)):
            if family(G) == family(J) and G != J:
                continue
            own = v[gens == G]
            if len(own) < 20:
                continue
            terms = []
            for K in v.columns:
                if family(K) in (family(J), family(G)):
                    continue
                comp = v[~gfam.isin([family(J), family(K), family(G)])]
                cm = comp.groupby(level="item")[[J, K]].mean()
                it = own.index.get_level_values("item")
                tJ = own[J].to_numpy() - cm[J].reindex(it).to_numpy()
                tK = own[K].to_numpy() - cm[K].reindex(it).to_numpy()
                terms.append(tJ - tK)
            if not terms:
                continue
            did = np.nanmean(np.nanmean(np.vstack(terms), axis=0))
            others = [g for g in pq.index if family(g) not in (family(J), family(G))]
            rows.append({"judge": J, "generator": G, "self": G == J, "did": did, "gap": pq[G] - pq[others].mean()})
    r = pd.DataFrame(rows)
    pl, se = r[~r.self], r[r.self]
    x = pl.dropna()
    print(f"\ntwo-judge difference: {len(pl)} placebo pairs; placebo mean {pl.did.mean():.3f}, sd {pl.did.std():.3f}, "
          f"|.|>0.05 {np.mean(pl.did.abs() > 0.05):.2f}, slope on gap {np.polyfit(x.gap, x.did, 1)[0]:.3f}, "
          f"corr {x[['gap', 'did']].corr().iloc[0, 1]:.2f}; self mean {se.did.mean():.3f} "
          f"({(se.did.mean() - pl.did.mean()) / pl.did.std():.2f} sd above placebo)")
    r.to_csv(OUT.replace(".csv", "_did.csv"), index=False)


if __name__ == "__main__":
    main()
    if DATASET == "ifeval":   # labels shared by both judges needed; HealthBench labels depend on the judge
        did_placebo()
