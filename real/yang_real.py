"""
STEP 5 of Yang et al.'s measure on Chen's data (measures/yang.py): per judge PIR, Null-PIR,
SPB with a bootstrap over problems.

Inputs (infer/yang_data.py, infer/yang_judges_submit.sh):
    infer/data/verdicts_<judge>.jsonl   the judge with Yang's A/B prompt, both orders, on
                                        kind=self (own vs opponent) and kind=null (two other
                                        models) equal-quality pairs
    infer/data/yang_self_chen.jsonl     the same self pairs with Chen's released A/T/B verdicts
                                        (cross-check, column pir_chen)

Votes are soft: first-token probabilities, renormalized over A/B. A firm pick has
probability P(order 1) * P(order 2). Each null row (unordered pair) gives both ordered pairs,
target j and target k. Rows where the verdict letters got < MIN_MASS of the first-token
probability in either order are dropped. For the Chen cross-check, judgments with tie
probability > MAX_TIE in either order are dropped (A/B renormalized from a few percent is noise).

Run with: python -m real.yang_real          writes real/yang_real_results.csv
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from measures import yang

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "infer", "data")
OUT = os.path.join(HERE, "yang_real_results.csv")
MIN_MASS = 0.5
MAX_TIE = 0.5
N_BOOT = 1000


def verdict_pairs(judge) -> pd.DataFrame:
    """One row per ORDERED (target, other) pair: p1 = P(pick target | shown first), p2 = | shown second."""
    d = pd.DataFrame([json.loads(l) for l in open(os.path.join(DATA, f"verdicts_{judge}.jsonl"))])
    d["pa"] = d.p_A / (d.p_A + d.p_B).replace(0, np.nan)
    g = d.pivot_table(index=["id", "kind", "task", "problem_idx", "gap"], columns="game", values=["pa", "verdict_mass"])
    g = g[(g["verdict_mass"][1] >= MIN_MASS) & (g["verdict_mass"][2] >= MIN_MASS)].dropna()
    pa1, pa2 = g["pa"][1], g["pa"][2]          # game 1: target in A; game 2: target in B
    first = pd.DataFrame({"p1": pa1, "p2": 1 - pa2}).reset_index()
    second = pd.DataFrame({"p1": pa2, "p2": 1 - pa1}).reset_index()   # the other answer as target
    second = second[second.kind == "null"]     # self pairs: only the judge's own answer is the target
    out = pd.concat([first, second], ignore_index=True)
    out["problem_key"] = out.task + "|" + out.problem_idx.astype(str)
    return out


def _boot(s, n, rng):
    probs = np.union1d(s.problem_key.unique(), n.problem_key.unique())
    reps = []
    for _ in range(N_BOOT):
        pick = pd.Series(rng.choice(probs, len(probs), replace=True)).value_counts()
        ws, wn = s.problem_key.map(pick).fillna(0).to_numpy(), n.problem_key.map(pick).fillna(0).to_numpy()
        if ws.sum() and wn.sum():
            reps.append(np.average(s.p1 * s.p2, weights=ws) - np.average(n.p1 * n.p2, weights=wn))
    return np.percentile(reps, [2.5, 97.5])


def main():
    chen = pd.read_json(os.path.join(DATA, "yang_self_chen.jsonl"), lines=True)
    chen = chen[(chen.tie_1 <= MAX_TIE) & (chen.tie_2 <= MAX_TIE)]
    rng = np.random.default_rng(0)
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA, "verdicts_*.jsonl"))):
        judge = os.path.basename(path)[len("verdicts_"):-len(".jsonl")]
        v = verdict_pairs(judge)
        s, n = v[v.kind == "self"], v[v.kind == "null"]
        if s.empty or n.empty:
            continue
        pir = yang.firm_rate(list(zip(s.p1, s.p2)))
        null = yang.firm_rate(list(zip(n.p1, n.p2)))
        lo, hi = _boot(s, n, rng)
        c = chen[chen.judge == judge]
        rows.append({"judge": judge, "n_self": len(s), "n_null_ordered": len(n), "pir": pir, "null_pir": null,
                     "spb": pir - null, "spb_lo": lo, "spb_hi": hi,
                     "null_firm_rate": 2 * null,        # P(order-consistent choice) on null pairs
                     "self_mean_gap": s.gap.mean(),     # mean score gap own - other inside "equal" pairs
                     "pir_chen": yang.firm_rate(list(zip(c.p_own_1, c.p_own_2))) if len(c) else np.nan,
                     "n_chen": len(c)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
