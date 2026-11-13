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


if __name__ == "__main__":
    main()
