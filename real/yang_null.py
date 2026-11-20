"""
Null test and injection for Yang et al.'s measure on real judgments (Chen's tasks).

Yang's measure for judge J: SPB = PIR - Null-PIR, where PIR is the rate at which J firmly picks its own
answer (both orders) in equal-quality pairs, and Null-PIR the same rate for a target answer in pairs of
two other models. Here we use only J's null pairs (J wrote neither answer, so true self-preference is 0):

    null test   each model G outside J's family takes the "self" slot in turn:
                PIR_G  = firm rate for G's answer in null pairs that include G
                null   = firm rate in null pairs without G
                SPB_G  = PIR_G - null       (want ~0, and no link to G's generation quality)
    injection   form B: J waves G's answer through with probability pi in the pairs with G,
                p -> p + pi (1 - p) in each order; response = dSPB_G / d pi (want the same for every
                judge and every G)

Generation quality of G: its accuracy over Chen's problems (all tasks), on the log-odds scale.

Run with: python -m real.yang_null       writes real/yang_null_results.csv
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from real import final_answers
from real.roytburg import family

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "infer", "data")
OUT = os.path.join(HERE, "yang_null_results.csv")
MIN_MASS = 0.5
PIS = [0.0, 0.1]
MIN_PAIRS = 30


def null_pairs(judge):
    """Ordered null pairs: target, other, p1 = P(pick target | shown first), p2 = | shown second."""
    d = pd.DataFrame([json.loads(l) for l in open(os.path.join(DATA, f"verdicts_{judge}.jsonl"))])
    d = d[d.kind == "null"]
    d["pa"] = d.p_A / (d.p_A + d.p_B).replace(0, np.nan)
    g = d.pivot_table(index=["id", "target", "other", "task", "problem_idx"], columns="game", values=["pa", "verdict_mass"])
    g = g[(g["verdict_mass"][1] >= MIN_MASS) & (g["verdict_mass"][2] >= MIN_MASS)].dropna()
    pa1, pa2 = g["pa"][1], g["pa"][2]
    first = pd.DataFrame({"p1": pa1, "p2": 1 - pa2}).reset_index()
    second = pd.DataFrame({"p1": pa2, "p2": 1 - pa1}).reset_index()
    second = second.rename(columns={"target": "other", "other": "target"})
    return pd.concat([first, second], ignore_index=True)


def main():
    fa = final_answers.load()
    acc = fa.groupby("model").correct.mean().clip(0.01, 0.99)
    quality = np.log(acc / (1 - acc))
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA, "verdicts_*.jsonl"))):
        J = os.path.basename(path)[len("verdicts_"):-len(".jsonl")]
        if J not in quality.index:
            continue
        v = null_pairs(J)
        if v.empty:
            continue
        for G in sorted(set(v.target)):
            if family(G) == family(J):
                continue
            with_g = v[v.target == G]
            without = v[(v.target != G) & (v.other != G)]
            if len(with_g) < MIN_PAIRS or len(without) < MIN_PAIRS:
                continue
            base = (without.p1 * without.p2).mean()
            spb = {pi: ((with_g.p1 + pi * (1 - with_g.p1)) * (with_g.p2 + pi * (1 - with_g.p2))).mean() - base
                   for pi in PIS}
            rows.append({"judge": J, "target": G, "quality": quality[G], "n_pairs": len(with_g),
                         "spb_null": spb[0.0], "response": (spb[0.1] - spb[0.0]) / 0.1})
    r = pd.DataFrame(rows)
    r["quality_rel"] = r.quality - r.groupby("judge").quality.transform("mean")
    r.to_csv(OUT, index=False)
    within = r.response - r.groupby("judge").response.transform("mean")
    print(f"{len(r)} judge-target combinations, {r.judge.nunique()} judges")
    print(f"null test: mean {r.spb_null.mean():+.3f}, sd {r.spb_null.std():.3f}, slope on target quality "
          f"{np.polyfit(r.quality, r.spb_null, 1)[0]:+.3f} (corr {np.corrcoef(r.quality, r.spb_null)[0, 1]:+.2f}); "
          f"within judge corr {np.corrcoef(r.quality_rel, r.spb_null - r.groupby('judge').spb_null.transform('mean'))[0, 1]:+.2f}")
    print(f"injection (form B): response mean {r.response.mean():.3f}, sd {r.response.std():.3f}, share from judge "
          f"{1 - within.var() / r.response.var():.2f}, within-judge corr with target quality "
          f"{np.corrcoef(r.quality_rel, within)[0, 1]:+.2f}")
    print(r.groupby("judge").agg(null=("spb_null", "mean"), null_sd=("spb_null", "std"), response=("response", "mean"),
                                 targets=("target", "size")).round(3).to_string())


if __name__ == "__main__":
    main()
