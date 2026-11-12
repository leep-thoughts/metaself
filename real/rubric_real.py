"""
Rubric self-preference measures on IFEval (measures/rubric.py), per judge, with a bootstrap
over prompts. Inputs from infer/ifeval_rubric.py: labels.jsonl (IFEval checker, strict) and
verdicts_<judge>.jsonl (P(criterion met) from Pombal's SR prompt).

For judge J: own = answers J generated; others = generators outside J's family. Verdicts with
verdict_mass < MIN_MASS (the true/false token got little probability) are dropped.
Also reported per judge: p (own answers' true pass rate), q (others'), accuracy, leniency
(mean verdict - true pass rate over all graded answers).

Run with: python -m real.rubric_real        writes real/rubric_real_results.csv
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from infer.ifeval_rubric import OUT as DATA, family
from measures.rubric import MEASURES, estimate

MIN_MASS = 0.5
N_BOOT = 500
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rubric_real_results.csv")


def units() -> pd.DataFrame:
    lab = {}
    for r in map(json.loads, open(os.path.join(DATA, "labels.jsonl"))):
        for i, f in enumerate(r["follow"]):
            lab[(r["key"], r["model"], i)] = f
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA, "verdicts_*.jsonl"))):
        for r in map(json.loads, open(path)):
            y = lab.get((r["key"], r["generator"], r["criterion"]))
            if y is None or r["v"] is None or (r["verdict_mass"] is not None and r["verdict_mass"] < MIN_MASS):
                continue
            rows.append({"judge": r["judge"], "generator": r["generator"], "key": r["key"],
                         "item": f"{r['key']}|{r['criterion']}", "y": bool(y), "v": float(r["v"])})
    # a resubmitted job can repeat a verdict: keep the first per (judge, generator, item)
    return pd.DataFrame(rows).drop_duplicates(["judge", "generator", "item"])


def judge_units(df, judge):
    d = df[df.judge == judge]
    fam = family(judge)
    d = d[(d.generator == judge) | (d.generator.map(family) != fam)].copy()
    d["own"] = d.generator == judge
    return d


def main():
    df = units()
    print(f"{len(df)} verdicts, judges {sorted(df.judge.unique())}")
    rng = np.random.default_rng(0)
    rows = []
    for judge in sorted(df.judge.unique()):
        d = judge_units(df, judge)
        if not d.own.any() or d.own.all():
            continue
        est = estimate(d)
        keys = d.key.unique()
        boots = []
        for _ in range(N_BOOT):
            pick = pd.Series(rng.choice(keys, len(keys))).value_counts()
            b = d.set_index("key").join(pick.rename("w"), how="inner")
            b = b.loc[b.index.repeat(b.w)].reset_index()
            boots.append(estimate(b))
        bd = pd.DataFrame(boots)
        o, k = d[d.own], d[~d.own]
        rows.append({"judge": judge, "n_own": int(len(o)), "n_other": int(len(k)),
                     "p": o.y.mean(), "q": k.y.mean(),
                     "accuracy": (d.v.round() == d.y).mean(),
                     "leniency": d.v.mean() - d.y.mean(),
                     **est, **{m + "_lo": bd[m].quantile(.025) for m in MEASURES},
                     **{m + "_hi": bd[m].quantile(.975) for m in MEASURES}})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS, index=False)
    print(res[["judge", "n_own", "p", "q", "accuracy", "leniency"] + MEASURES].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
