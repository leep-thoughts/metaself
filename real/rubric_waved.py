"""
Waved-through rate: a normalised rubric-based self-preference measure, tested with the null test
and form-B injection on IFEval.

On truly failed rubric items, comparing judge J's verdicts on the target's answers (o) with its
verdicts on other models' answers to the same items (k, strength-matched comparison models = Fix A):
    waved-through rate = mean(o - k) / mean(1 - k)
"of the failures J rejects from comparable models, the share it lets through for the target".
Under form-B injection (target's answers waved through with probability pi: o -> o + pi (1 - o))
it should move by exactly pi when J judges the target's failures as it judges the others'.

The denominator mean(1 - k) is J's rejection rate on the comparison models' failures; lenient judges
(rarely reject) make it small and the measure noisy, so pairings below MIN_REJECT are dropped.

Tests per MIN_REJECT:
    null test   target never the judge's family: values ~0, no slope on target strength
    injection   response to pi = mean(1 - o) / mean(1 - k): want 1, not depending on judge or strength
Bootstrap over prompts for the null-test mean of each judge would come next if needed.

Run with: python -m real.rubric_waved       writes real/rubric_waved_results.csv
"""

import os

import numpy as np
import pandas as pd

from infer.ifeval_rubric import DATASET, family
from real.rubric_control import DELTA
from real.rubric_real import units

MIN_REJECTS = [0.0, 0.1, 0.2, 0.3, 0.5]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rubric_waved_results" + ("" if DATASET == "ifeval" else f"_{DATASET}") + ".csv")


def pairings(df, rate, delta=DELTA):
    f = df[~df.y]
    rows = []
    for J, dj in f.groupby("judge"):
        for G in rate.index:
            if family(G) == family(J):
                continue
            elig = [g for g in rate.index if family(g) not in (family(J), family(G))]
            ok = [g for g in elig if delta is None or abs(rate[g] - rate[G]) <= delta]
            if not ok:
                continue
            own = dj[dj.generator == G].set_index("item").v
            oth = dj[dj.generator.isin(ok)].groupby("item").v.mean()
            m = pd.concat([own.rename("o"), oth.rename("k")], axis=1).dropna()
            if len(m) < 10:
                continue
            reject = (1 - m.k).mean()
            rows.append({"judge": J, "target": G, "gap": rate[G] - rate[elig].mean(), "n_items": len(m),
                         "reject_rate": reject, "waved": (m.o - m.k).mean() / reject,
                         "response": (1 - m.o).mean() / reject})
    return pd.DataFrame(rows)


def summary(x):
    within = x.response - x.groupby("judge").response.transform("mean")
    gw = x.gap - x.groupby("judge").gap.transform("mean")
    return {"pairings": len(x), "judges": x.judge.nunique(),
            "null mean": x.waved.mean(), "null sd": x.waved.std(),
            "null slope on strength": np.polyfit(x.gap, x.waved, 1)[0],
            "null corr with strength": np.corrcoef(x.gap, x.waved)[0, 1],
            "response mean (want 1)": x.response.mean(), "response sd": x.response.std(),
            "response share from judge": 1 - within.var() / x.response.var(),
            "response within-judge corr with strength": np.corrcoef(gw, within)[0, 1]}


def main():
    df = units()
    rate = df.groupby("generator").y.mean()
    r = pairings(df, rate)
    r.to_csv(OUT, index=False)
    s = pd.DataFrame({f"min reject {t}": summary(r[r.reject_rate >= t]) for t in MIN_REJECTS}).T
    print(s.round(3).to_string())
    print("\nrejection rate on comparison models' failures, by judge (mean over targets):")
    print(r.groupby("judge").agg(reject=("reject_rate", "mean"), null=("waved", "mean"),
                                 null_sd=("waved", "std"), response=("response", "mean"))
          .sort_values("reject").round(2).to_string())


if __name__ == "__main__":
    main()
