"""
What is the lean that the win-rate rule leaves out? (Pairwise, Roytburg's null pairs.)

Per unit u = (judge, task, opponent, proxy): r_u = b_u - b_model_u (real/pair_check.units), i.e.
the lean minus gamma * logit(win rate). Split-half reliability of r is ~0.78
(real/confound_checks.py), so ~78% of its variance is stable, not noise.

Candidate factors (within judge x task, weighted by the unit's vote count):
    length     mean log(length of A's answer / length of B's answer) on the unit's problems
               (texts from Chen's files: real/chen.py, text=True)
    opponent   which model wrote B
    proxy      which model wrote A (the target model)
    family     judge family x target-model family (only cross-family pairs exist)
Reported: R^2 of each block alone, of all blocks, and the drop in R^2 when each block is left out.
Then: does each block's fitted part line up with the target model's generation quality (share of
its answers that are right on the task)? Only a part that lines up can create slopes.
Also the same for the full lean b_u (not only what the rule leaves).

Run with: python -m real.lean_rest       writes real/lean_rest_results.json
"""

import json
import os

import numpy as np
import pandas as pd

from real import chen, null_injection
from real.pair_check import FE, UNIT, units
from real.roytburg import family

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lean_rest_results.json")


def lengths(d):
    c = chen.load(text=True)
    c["problem_idx"] = c.groupby(["task", "judge", "opponent"]).cumcount()
    c["len_a"] = np.log1p(c.own_answer.fillna("").str.len())
    c["len_b"] = np.log1p(c.other_answer.fillna("").str.len())
    c = c.rename(columns={"judge": "proxy"})[["task", "proxy", "opponent", "problem_idx", "len_a", "len_b", "own_correct"]]
    d = d.assign(problem_idx=d.problem_idx.astype(int)).merge(c, on=["task", "proxy", "opponent", "problem_idx"], how="left")
    quality = c.groupby(["task", "proxy"]).own_correct.mean().rename("quality")
    return d, quality


def demean(x, by, w):
    return x - (x * w).groupby(by).transform("sum") / w.groupby(by).transform("sum")


def r2(y, X, w):
    if X.shape[1] == 0:
        return 0.0, np.zeros(len(y))
    sw = np.sqrt(w)
    beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    fit = X @ beta
    return 1 - np.sum(w * (y - fit) ** 2) / np.sum(w * y ** 2), fit


def analyse(u, target):
    by = [u[c] for c in FE]
    w = u.n.to_numpy(float)
    y = demean(u[target], by, u.n).to_numpy()
    blocks = {
        "length": u[["length"]].astype(float),
        "opponent": pd.get_dummies(u.opponent, dtype=float),
        "proxy": pd.get_dummies(u.proxy, dtype=float),
        "family": pd.get_dummies(u.judge.map(family) + "x" + u.proxy.map(family), dtype=float),
    }
    blocks = {k: np.column_stack([demean(v[c], by, u.n) for c in v.columns]) for k, v in blocks.items()}
    full, _ = r2(y, np.hstack(list(blocks.values())), w)
    q = demean(u.quality, by, u.n).to_numpy()
    out = {"r2_all": full, "blocks": {}}
    for k, X in blocks.items():
        alone, fit = r2(y, X, w)
        rest = np.hstack([v for kk, v in blocks.items() if kk != k])
        without, _ = r2(y, rest, w)
        out["blocks"][k] = {"r2_alone": alone, "r2_drop_when_left_out": full - without,
                            "corr_fit_with_quality": float(np.corrcoef(fit, q)[0, 1]) if np.std(fit) > 0 else 0.0}
    out["corr_target_with_quality"] = float(np.corrcoef(y, q)[0, 1])
    return out


def main():
    d = null_injection.load()
    u, gamma = units(d)
    d, quality = lengths(d)
    ln = (d.len_a - d.len_b).groupby([d[c] for c in UNIT]).mean().rename("length")
    u = u.merge(ln.reset_index(), on=UNIT).merge(quality.reset_index(), on=["task", "proxy"])
    u["rest"] = u.b - u.b_model
    out = {"units": len(u), "rest": analyse(u, "rest"), "lean": analyse(u, "b")}
    # how length relates to the lean directly (units' lean per unit of log length ratio)
    by = [u[c] for c in FE]
    x, y = demean(u.length, by, u.n), demean(u.b, by, u.n)
    out["lean_per_log_length_ratio"] = float(np.sum(u.n * x * y) / np.sum(u.n * x * x))
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    for k in ("rest", "lean"):
        print(f"{k}: R2 all {out[k]['r2_all']:.2f}; corr with generation quality {out[k]['corr_target_with_quality']:+.2f}")
        for b, v in out[k]["blocks"].items():
            print(f"   {b:9s} alone {v['r2_alone']:.2f}  unique {v['r2_drop_when_left_out']:.2f}  "
                  f"fit vs generation quality {v['corr_fit_with_quality']:+.2f}")
    print(f"lean per unit log length ratio: {out['lean_per_log_length_ratio']:+.3f}")


if __name__ == "__main__":
    main()
