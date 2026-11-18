"""
Pairwise version of the strength-matched comparison set (pipeline stage S3), on Roytburg's
null pairs: judge J compares proxy K's answer (A) with reference R's answer (B); J wrote neither,
so true self-preference is 0.

Roytburg's measure for a candidate K: on each item, J's P(vote K) minus the mean P(vote K') over
the OTHER proxies K' on the same item (same opponent R, same correctness as K). Pooled over all
K, it is 0 by construction (each proxy is compared with the leave-one-out mean of the others).
So, as a real placebo test, we keep each K separate ("pretend K is J's own model"):
    value(J, K) = mean over items of [ q_K - mean q_K' ]
and ask whether it tracks K's strength relative to its comparison proxies (want: no).

Comparison proxies K':
    all        every other proxy on the item (Roytburg)
    matched    only proxies whose pair win rate vs R is within DELTA of K's, on the logit scale
               (pair win rate p_u from real/pair_check.units: the driver of the pair lean gamma)

Run with: python -m real.pair_control     writes real/pair_control_results.csv
"""

import os

import numpy as np
import pandas as pd

from real import null_injection
from real.pair_check import UNIT, units

DELTAS = [None, 0.5, 0.25, 0.1]   # max |logit p_K - logit p_K'| (the pair lean is linear in logit p)
ITEM = ["task", "judge", "opponent", "problem_idx"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pair_control_results.csv")


def main():
    d = null_injection.load()
    d["q"] = null_injection.sigmoid(d.l)
    u, _ = units(d)
    p = u.set_index(UNIT).p
    p = np.log(p / (1 - p))                                    # strength on the logit scale
    d["p_k"] = p.reindex(pd.MultiIndex.from_frame(d[UNIT])).to_numpy()
    d = d.dropna(subset=["p_k"])
    # all pairs (K, K') on the same item
    pairs = d[ITEM + ["proxy", "q", "p_k"]].merge(d[ITEM + ["proxy", "q", "p_k"]], on=ITEM, suffixes=("", "_o"))
    pairs = pairs[pairs.proxy != pairs.proxy_o]
    out, summ = [], []
    for delta in DELTAS:
        pr = pairs if delta is None else pairs[(pairs.p_k - pairs.p_k_o).abs() <= delta]
        item = pr.groupby(ITEM + ["proxy"]).agg(q=("q", "first"), q_o=("q_o", "mean"),
                                                 p_k=("p_k", "first"), p_o=("p_k_o", "mean")).reset_index()
        item["diff"] = item.q - item.q_o
        jk = item.groupby(["judge", "proxy"]).agg(value=("diff", "mean"), gap=("p_k", "mean"),
                                                  gap_o=("p_o", "mean"), n=("diff", "size")).reset_index()
        jk["gap"] = jk.gap - jk.gap_o
        jk = jk[jk.n >= 30]
        name = "all" if delta is None else f"matched ±{delta} logit"
        jk["set"] = name
        out.append(jk)
        w = jk.n
        slope = np.polyfit(jk.gap, jk.value, 1, w=np.sqrt(w))[0]
        summ.append({"set": name, "n (judge, proxy)": len(jk), "items": int(w.sum()),
                     "mean |logit gap|": np.average(jk.gap.abs(), weights=w),
                     "placebo mean": np.average(jk.value, weights=w), "placebo sd": jk.value.std(),
                     "slope on logit gap": slope, "corr": np.corrcoef(jk.gap, jk.value)[0, 1]})
    pd.concat(out).to_csv(OUT, index=False)
    print(pd.DataFrame(summ).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
