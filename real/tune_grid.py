"""
Which (p, actual accuracy) combinations can real data reach, per judge, with task mix fixed?

Data: the cross-family null pairs, true s ~ 0. Each judge's items fall in cells:
task x generator-pair p bin x other judges' confidence bin. Only whole cells are reweighted,
never individual labels or votes. Each repeat splits problems in half; on the fit half a
linear program finds the weights closest to natural (L1) that hit a target (p, accuracy),
with each task keeping its natural share and no cell above MAX_UPWEIGHT x natural. The
weights are then scored on the held-out half.

A target is reachable if it is feasible in at least MIN_FEASIBLE of repeats and the mean
held-out p and accuracy are both within TOLERANCE of it.

Run with: python -m real.tune_grid
"""

import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from real.null_injection import load
from real.tune_accuracy import MEASURES, add_covariates, weighted_stats
from real.tune_p import GROUP, TASKS

N_REPS = 20
N_P_BINS = 5
N_CONF_BINS = 5
MAX_UPWEIGHT = 5
MIN_FEASIBLE = 0.8
TOLERANCE = 0.02
MIN_ESS = 100
P_GRID = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]
ACC_OFFSETS = [round(x, 3) for x in np.arange(-0.15, 0.151, 0.025)]
WORKERS = 2
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "tune_grid_results.csv")


def cell_index(fit, test, group_p, rng):
    """Same cell labels on both halves; rows without a fit-half generator pair get -1."""
    fit_c, test_c = np.zeros(len(fit), dtype=int) - 1, np.zeros(len(test), dtype=int) - 1
    for t_i, task in enumerate(TASKS):
        f, h = (fit.task == task).to_numpy(), (test.task == task).to_numpy()
        counts = fit[f].groupby(GROUP).size()
        order = sorted(counts.index, key=lambda k: (group_p[k], rng.random()))
        cum = np.cumsum(counts.loc[order].to_numpy()) - 1
        p_bin = {k: min(int(N_P_BINS * c / cum[-1]), N_P_BINS - 1) for k, c in zip(order, cum)}
        edges = np.quantile(fit.confidence[f], np.linspace(0, 1, N_CONF_BINS + 1))[1:-1]
        for frame, mask, out in ((fit, f, fit_c), (test, h, test_c)):
            keys = frame.loc[mask, GROUP].itertuples(index=False, name=None)
            pb = np.array([p_bin.get(k, -1) for k in keys])
            cb = np.searchsorted(edges, frame.confidence[mask].to_numpy())
            out[mask] = np.where(pb >= 0, t_i * N_P_BINS * N_CONF_BINS + pb * N_CONF_BINS + cb, -1)
    return fit_c, test_c


def solve(w0, cp, ca, ct, target_p, target_acc, shares):
    """Cell weights closest to w0 (L1) hitting the targets. None if infeasible."""
    n = len(w0)
    c = np.r_[np.zeros(n), np.ones(n)]
    I = np.eye(n)
    A_ub = np.block([[I, -I], [-I, -I]])
    b_ub = np.r_[w0, -w0]
    task_rows = [np.r_[(ct == t).astype(float), np.zeros(n)] for t in range(len(TASKS))]
    A_eq = np.array(task_rows + [np.r_[cp, np.zeros(n)], np.r_[ca, np.zeros(n)]])
    b_eq = np.r_[shares, target_p, target_acc]
    bounds = [(0, MAX_UPWEIGHT * x) for x in w0] + [(0, None)] * n
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return r.x[:n] if r.success else None


def run_judge_rep(args):
    judge, rep, g, group_p = args
    rng = np.random.default_rng(rep)
    fit, test = g[g.fold == 0].reset_index(drop=True), g[g.fold == 1].reset_index(drop=True)
    fit_c, test_c = cell_index(fit, test, group_p, rng)
    fit, test = fit[fit_c >= 0].reset_index(drop=True), test[test_c >= 0].reset_index(drop=True)
    fit_c, test_c = fit_c[fit_c >= 0], test_c[test_c >= 0]
    cells, inv = np.unique(fit_c, return_inverse=True)
    n = np.bincount(inv)
    w0 = n / n.sum()
    cp = np.bincount(inv, fit.a_correct.to_numpy()) / n
    ca = np.bincount(inv, fit.acc.to_numpy()) / n
    ct = cells // (N_P_BINS * N_CONF_BINS)
    shares = np.bincount(ct, w0, minlength=len(TASKS))
    natural_acc = fit.acc.mean()
    n_cells = len(TASKS) * N_P_BINS * N_CONF_BINS
    n_test = np.zeros(n_cells)
    np.add.at(n_test, test_c, 1)
    rows = []
    for tp in P_GRID:
        for off in ACC_OFFSETS:
            ta = natural_acc + off
            base = {"judge": judge, "rep": rep, "target_p": tp, "acc_offset": off, "target_acc": ta,
                    "natural_p": fit.a_correct.mean(), "natural_acc": natural_acc}
            w = solve(w0, cp, ca, ct, tp, ta, shares)
            if w is None:
                rows.append({**base, "feasible": False})
                continue
            cell_w = np.zeros(n_cells)
            cell_w[cells] = w
            # keep each task's share on the held-out half, where some cells can be empty
            row_w = np.where(n_test[test_c] > 0, cell_w[test_c] / n_test[test_c], 0.0)
            task_i = test_c // (N_P_BINS * N_CONF_BINS)
            for t in range(len(TASKS)):
                m = task_i == t
                if row_w[m].sum() > 0:
                    row_w[m] *= shares[t] / row_w[m].sum()
            rows.append({**base, "feasible": True, **weighted_stats(test, row_w),
                         "l1_from_natural": np.abs(w - w0).sum(),
                         **{f"share_{t}": row_w[task_i == i].sum() / row_w.sum() for i, t in enumerate(TASKS)}})
    return rows


def summarize(runs) -> pd.DataFrame:
    keys = ["judge", "target_p", "acc_offset"]
    s = runs.groupby(keys).agg(feasible=("feasible", "mean"), target_acc=("target_acc", "mean"),
                               natural_p=("natural_p", "mean"), natural_acc=("natural_acc", "mean"))
    ok = runs[runs.feasible]
    vals = ["p", "actual_accuracy", "ess", "l1_from_natural"] + list(MEASURES)
    s = s.join(ok.groupby(keys)[vals].mean()).join(
        ok.groupby(keys)[["p", "actual_accuracy"]].std().rename(columns=lambda c: c + "_sd")).reset_index()
    s["reachable"] = ((s.feasible >= MIN_FEASIBLE) & ((s.p - s.target_p).abs() <= TOLERANCE)
                      & ((s.actual_accuracy - s.target_acc).abs() <= TOLERANCE) & (s.ess >= MIN_ESS))
    return s


def jobs(df, n_reps):
    items = df.drop_duplicates(GROUP + ["problem_idx"])
    problems = df[["task", "problem_idx"]].drop_duplicates()
    for rep in range(n_reps):
        rng = np.random.default_rng(1000 + rep)
        fold = problems.assign(fold=rng.integers(2, size=len(problems)))
        fit_items = items.merge(fold, on=["task", "problem_idx"]).query("fold == 0")
        group_p = fit_items.groupby(GROUP).a_correct.mean().to_dict()
        d = df.merge(fold, on=["task", "problem_idx"])
        for judge, g in d.groupby("judge"):
            yield judge, rep, g[KEEP + ["fold"]], group_p


KEEP = GROUP + ["problem_idx", "a_correct", "q", "q_proxy", "acc", "confidence"]


if __name__ == "__main__":
    import sys, time
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else N_REPS
    df = add_covariates(load(), np.random.default_rng(0)).dropna(subset=["confidence"])
    t0 = time.time()
    with Pool(WORKERS) as pool:
        runs = pd.DataFrame([r for rows in pool.imap_unordered(run_judge_rep, jobs(df, n_reps)) for r in rows])
    print(f"{len(df)} pairs, {time.time() - t0:.0f}s")
    res = summarize(runs)
    res.to_csv(OUT_PATH, index=False)
    r = res[res.reachable]
    print(r.groupby("judge").agg(n_points=("reachable", "size"), p_min=("target_p", "min"), p_max=("target_p", "max"),
                                 acc_min=("target_acc", "min"), acc_max=("target_acc", "max"),
                                 natural_p=("natural_p", "first"), natural_acc=("natural_acc", "first"),
                                 ess=("ess", "mean")).round(3).to_string())
