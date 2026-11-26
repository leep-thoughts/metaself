"""
Halo control for rubric self-preference: difference-in-differences with third-party judges.

matched_fail (measures/rubric.py) compares judge J's verdicts on its own truly-failed criteria
with its verdicts on other generators' failures of the same criterion. A generator halo
(any judge leans "met" for J's answers because they look stronger) inflates it too
(synth/rubric.py, knob h). Third-party judges K see the same halo but have no self-preference
for J, so:

    term_X(u) = v_X(J's unit u) - mean v_X(other generators, same criterion, also failed)
    DiD(u)    = term_J(u) - mean over K of term_K(u)

for every criterion u that J's own answer truly fails. K ranges over judges outside J's family;
the comparison generators are outside both J's and K's families (so neither judge grades a
relative). Halo cancels if all judges react to it alike; J's self-preference stays.
Also reported: halo_K = mean term_K (how much third parties favour J's failed answers).

Bootstrap over prompts. Labels: IFEval checker (RUBRIC_DATASET=ifeval).
Run with: python -m real.rubric_did        writes real/rubric_did_results.csv

hspp mode: the same two-judge difference for HSPP-rubric and HSPP-ratio in the null test, with the
SAME comparison models for both judges. For judge J, target G (outside J's family, or G = J) and
second judge K outside J's and G's families, comparison models C = outside J's, K's, G's families:
    HSPP-rubric_X = FPR_X(G) - FPR_X(C)        HSPP-ratio_X = FPR_X(G) / FPR_X(C) - 1
(FPR = mean verdict on truly failed criteria, on the criteria both J and K graded), then
J's value minus K's, averaged over K. Comparison sets: all of C, or only models in C whose pass
rate is within DELTA of G's (same-quality fix). Slope on G's pass rate minus that of all models
outside J's and G's families (want 0); self rows (G = J) for power.
Run with: python -m real.rubric_did hspp   writes real/rubric_did_hspp_results.csv
"""

import os

import numpy as np
import pandas as pd

from infer.ifeval_rubric import family
from real.rubric_real import units

N_BOOT = 1000
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rubric_did_results.csv")


def terms(df):
    """Per (J, K, own failed unit): term_J and term_K."""
    v = df.pivot_table(index=["generator", "key", "item"], columns="judge", values="v")
    y = df.drop_duplicates(["generator", "item"]).set_index(["generator", "key", "item"]).y.reindex(v.index)
    v = v[~y.to_numpy()]                                   # truly failed units only
    gens = v.index.get_level_values("generator")
    gfam = pd.Index(gens.map(family))
    judges = list(v.columns)
    rows = []
    for J in judges:
        if J not in set(gens):
            continue
        own = v[gens == J]
        for K in judges:
            if family(K) == family(J):
                continue
            comp = v[~gfam.isin([family(J), family(K)])]
            cm = comp.groupby(level="item")[[J, K]].mean()
            o = own[[J, K]].copy()
            items = own.index.get_level_values("item")
            o["cJ"], o["cK"] = cm[J].reindex(items).to_numpy(), cm[K].reindex(items).to_numpy()
            o = o.dropna()
            rows.append(pd.DataFrame({"J": J, "K": K, "key": o.index.get_level_values("key"),
                                      "item": o.index.get_level_values("item"),
                                      "tJ": o[J] - o.cJ, "tK": o[K] - o.cK}))
    return pd.concat(rows, ignore_index=True)


def main():
    from infer.ifeval_rubric import DATASET
    if DATASET != "ifeval":
        raise SystemExit("rubric_did: IFEval only for now (HealthBench labels depend on the judge's family)")
    df = units()
    t = terms(df)
    # per J and unit: average the K's (each unit counts once)
    u = t.groupby(["J", "key", "item"]).agg(tJ=("tJ", "mean"), tK=("tK", "mean"), nK=("K", "nunique")).reset_index()
    u["did"] = u.tJ - u.tK
    rng = np.random.default_rng(0)
    rows = []
    for J, d in u.groupby("J"):
        per_key = d.groupby("key").agg(s_did=("did", "sum"), s_j=("tJ", "sum"), s_k=("tK", "sum"), n=("did", "size"))
        keys = per_key.index.to_numpy()
        boots = []
        for _ in range(N_BOOT):
            b = per_key.loc[rng.choice(keys, len(keys))]
            boots.append((b.s_did.sum() / b.n.sum(), b.s_k.sum() / b.n.sum()))
        bd, bk = np.array(boots).T
        rows.append({"judge": J, "n_units": len(d), "n_third_party": int(t[t.J == J].K.nunique()),
                     "matched_fail_same_comparison": d.tJ.mean(), "halo_third_party": d.tK.mean(),
                     "halo_lo": np.quantile(bk, .025), "halo_hi": np.quantile(bk, .975),
                     "did": d.did.mean(), "did_lo": np.quantile(bd, .025), "did_hi": np.quantile(bd, .975)})
    res = pd.DataFrame(rows).sort_values("did", ascending=False)
    res.to_csv(OUT, index=False)
    print(res.round(3).to_string(index=False))


def hspp_null():
    from real.rubric_control import DELTA
    df = units()
    pq = df.groupby("generator").y.mean()
    v = df.pivot_table(index=["generator", "key", "item"], columns="judge", values="v")
    y = df.drop_duplicates(["generator", "item"]).set_index(["generator", "key", "item"]).y.reindex(v.index)
    v = v[~y.to_numpy()]                                   # truly failed criteria only
    gens = v.index.get_level_values("generator")
    gfam = pd.Index(gens.map(family))
    rows = []
    for J in v.columns:
        for G in sorted(set(gens)):
            if family(G) == family(J) and G != J:
                continue
            own = gens == G
            if own.sum() < 20:
                continue
            others = [g for g in pq.index if family(g) not in (family(J), family(G))]
            gap = pq[G] - pq[others].mean()
            for cset in ("all", "same quality"):
                res = {"rubric": [], "ratio": []}
                for K in v.columns:
                    if family(K) in (family(J), family(G)):
                        continue
                    comp = ~gfam.isin([family(J), family(K), family(G)])
                    if cset == "same quality":
                        near = [g for g in pq.index if abs(pq[g] - pq[G]) <= DELTA]
                        comp = comp & gens.isin(near)
                    sub = v.loc[own | comp, [J, K]].dropna()
                    so = sub.index.get_level_values("generator") == G
                    if so.sum() < 20 or (~so).sum() < 20:
                        continue
                    f = {X: (sub[X][so].mean(), sub[X][~so].mean()) for X in (J, K)}
                    res["rubric"].append((f[J][0] - f[J][1]) - (f[K][0] - f[K][1]))
                    if min(f[J][1], f[K][1]) > 0.05:
                        res["ratio"].append((f[J][0] / f[J][1]) - (f[K][0] / f[K][1]))
                if res["rubric"]:
                    rows.append({"judge": J, "generator": G, "self": G == J, "set": cset, "gap": gap,
                                 "n_second_judges": len(res["rubric"]),
                                 "hspp_rubric_did": np.mean(res["rubric"]),
                                 "hspp_ratio_did": np.mean(res["ratio"]) if res["ratio"] else np.nan})
    r = pd.DataFrame(rows)
    r.to_csv(OUT.replace(".csv", "_hspp.csv").replace("rubric_did_results_hspp", "rubric_did_hspp_results"), index=False)
    for (cset), d in r.groupby("set"):
        pl, se = d[~d.self], d[d.self]
        for m in ("hspp_rubric_did", "hspp_ratio_did"):
            x = pl[["gap", m, "judge"]].dropna()
            per = [np.polyfit(t.gap, t[m], 1)[0] for _, t in x.groupby("judge") if len(t) >= 4]
            print(f"{cset:12s} {m:16s} null mean {x[m].mean():+.3f} sd {x[m].std():.3f} | slope on gap: pooled "
                  f"{np.polyfit(x.gap, x[m], 1)[0]:+.3f}, median per judge {np.median(per):+.3f} | self mean "
                  f"{se[m].mean():+.3f} ({len(x)} null pairs)")


if __name__ == "__main__":
    import sys
    hspp_null() if sys.argv[1:] == ["hspp"] else main()
