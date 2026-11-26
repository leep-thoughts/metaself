"""
Evaluation-accuracy confound (pairwise, Roytburg's null pairs): does the slope come from evaluation
accuracy itself or from WHICH questions are kept when accuracy is raised?

1. covariates   The LP grid (real/slope_gap.py) raises accuracy by reweighting bins of a covariate
                ("confidence": other judges' mean |log-odds| on the same answer pair). Rerun the
                same grid with the bins built from other covariates (all leave-this-judge-out):
                    confidence   (default)       problem_acc   other judges' accuracy on the problem
                    pair_acc     other judges' accuracy on this exact answer pair
                    solve_rate   share of the other generators that solve the problem (difficulty)
                    random       placebo
                If the accuracy slopes change with the covariate, the question mix drives them.
2. same_items   No reweighting at all: two judges on the SAME rows (task, opponent, proxy, problem;
                cross-family for both). p is then identical, and only the judge differs. Per judge
                pair: difference in each measure vs difference in accuracy; pooled slope through
                the origin (weights = shared rows), bootstrap over problems.

Run with: python -m real.accuracy_mix       writes real/accuracy_mix_results.json
"""

import itertools
import json
import os

import numpy as np
import pandas as pd

from real import null_injection
from real import slope_gap as sg
from real.pair_check import N_REPS
from real.tune_accuracy import MEASURES, weighted_stats

sg.WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", sg.WORKERS))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accuracy_mix_results.json")
COVS = ("confidence", "problem_acc", "pair_acc", "solve_rate", "random")
ROW = ["task", "opponent", "proxy", "problem_idx"]
N_BOOT = 300


def covariates():
    df = sg.prepare()
    out = {}
    for cov in COVS:
        x = df.dropna(subset=[cov]).copy()
        x["confidence"] = x[cov]                        # the grid bins on this column
        out[cov] = sg.grid_summary(sg.run_grid(x, n_reps=N_REPS))["median"]
        print(f"{cov:12s} " + ", ".join(f"{k} {v:+.2f}" for k, v in out[cov].items()), flush=True)
    return out


def same_items():
    d = null_injection.load()
    d["q"] = null_injection.sigmoid(d.l)
    d["acc"] = np.where(d.a_correct, d.q, 1 - d.q)
    by_j = {j: g.set_index(ROW) for j, g in d.groupby("judge")}
    pairs = []
    for j1, j2 in itertools.combinations(sorted(by_j), 2):
        common = by_j[j1].index.intersection(by_j[j2].index)
        if len(common) < 200:
            continue
        a, b = by_j[j1].loc[common].reset_index(), by_j[j2].loc[common].reset_index()
        pairs.append((j1, j2, a, b, list(zip(a.task, a.problem_idx))))
    rng = np.random.default_rng(0)
    probs = d[["task", "problem_idx"]].drop_duplicates().to_numpy()

    def slopes(sample=None):
        num, den = {m: 0.0 for m in MEASURES}, 0.0
        for j1, j2, a, b, key in pairs:
            w = np.ones(len(a))
            if sample is not None:
                w = np.array([sample.get(k, 0) for k in key], dtype=float)
                if w.sum() == 0:
                    continue
            sa, sb = weighted_stats(a, w), weighted_stats(b, w)
            da = sa["actual_accuracy"] - sb["actual_accuracy"]
            n = w.sum()
            for m in MEASURES:
                num[m] += n * da * (sa[m] - sb[m])
            den += n * da * da
        return {m: num[m] / den for m in MEASURES}

    est = slopes()
    boots = []
    for _ in range(N_BOOT):
        pick = probs[rng.integers(len(probs), size=len(probs))]
        cnt = pd.Series(1, index=pd.MultiIndex.from_arrays([pick[:, 0], pick[:, 1]])).groupby(level=[0, 1]).sum()
        boots.append(slopes(cnt.to_dict()))
    b = pd.DataFrame(boots)
    out = {m: {"slope": est[m], "lo": b[m].quantile(.025), "hi": b[m].quantile(.975)} for m in MEASURES}
    out["judge_pairs"] = len(pairs)
    out["rows_per_pair_median"] = float(np.median([len(p[2]) for p in pairs]))
    print("same items, between judges:", {m: f"{v['slope']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}]"
                                          for m, v in out.items() if isinstance(v, dict)}, flush=True)
    return out


def main():
    out = {"same_items": same_items()}
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    out["covariates"] = covariates()
    json.dump(out, open(OUT, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
