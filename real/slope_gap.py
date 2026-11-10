"""
Why the difficulty coupling (beta) explains only part of the real-data slopes. One test per
candidate cause, all on the cross-family null pairs (real/null_injection.py, true s ~ 0).

1. Shared errors      Is a wrong A preferred more when it is the judge's OWN wrong answer?
                      (B-won rows, MATH/MMLU, final answers from real/final_answers.py)
                      + grid slopes without those rows, vs a random-drop placebo.
2/3. What moves p     Regress TPR, FPR, EO, HSPP, Panickssery on (p, accuracy) across three
                      kinds of natural variation, each with fixed effects for the rest:
                        opponent    between opponents          FE judge x task
                        proxy       between proxies, same opp. FE judge x task x opponent
                        difficulty  between problem-difficulty FE judge x task x opponent x proxy
                                    strata, same generator pair
                      Difficulty = solve rate of the problem among the generators other than
                      J, K and R (never this pair's labels). The grid's p moves mostly
                      through the first two; beta in the simulator is only the third.
                      + how far p can move at all with the opponent mix fixed (LP grid).
5. Ties / position    Grid slopes on rows with low tie probability, and on rows whose
                      preference has the same sign in both orders.
7. Family affinity    Within the same item (judge, opponent, problem), how J's preference for
                      proxy K depends on K's family: residual after item means and proxy-
                      family means (the judge x proxy-family interaction).

Grid slope = per judge, OLS of each measure on achieved (p, actual accuracy) over that judge's
reachable targets (as real/tune_grid_analysis.py), coarser cells (3 p bins x 4 confidence
bins) than tune_grid; median over judges with >= 3 levels of both knobs.

Run with: python -m real.slope_gap        writes real/slope_gap_results.json
"""

import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from real import final_answers
from real.null_injection import load
from real.roytburg import family
from real.tune_accuracy import MEASURES, add_covariates, weighted_stats
from real.tune_p import GROUP

N_REPS = 10
N_P_BINS, N_CONF_BINS = 3, 4
MAX_UPWEIGHT, MIN_FEASIBLE, TOLERANCE = 5, 0.8, 0.02
MIN_ESS = 100
TIE_MAX = 0.2
N_DIFF_BINS = 5
P_GRID = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]
ACC_OFFSETS = [round(x, 3) for x in np.arange(-0.15, 0.151, 0.025)]
WORKERS = 2
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "slope_gap_results.json")
KEEP = GROUP + ["problem_idx", "a_correct", "q", "q_proxy", "acc", "confidence"]


# ------------------------------------------------------------------ data
def prepare() -> pd.DataFrame:
    df = add_covariates(load(), np.random.default_rng(0)).dropna(subset=["confidence"]).reset_index(drop=True)
    fa = final_answers.load()
    ans = fa.set_index(["task", "model", "problem_idx"]).answer.to_dict()
    j = [ans.get(k) for k in zip(df.task, df.judge, df.problem_idx)]
    k = [ans.get(k) for k in zip(df.task, df.proxy, df.problem_idx)]
    known = np.array([a is not None and b is not None for a, b in zip(j, k)])
    df["shared_error"] = ~df.a_correct & known & np.array([a == b for a, b in zip(j, k)])
    df["error_known"] = ~df.a_correct & known
    df["tie_max"] = df[["tie_1", "tie_2"]].max(axis=1)
    df["consistent"] = np.sign(df.l_1) == np.sign(df.l_2)
    # problem difficulty: solve rate among generators other than J, K, R
    corr = fa.set_index(["task", "model", "problem_idx"]).correct.astype(float).to_dict()
    tot = fa.groupby(["task", "problem_idx"]).correct.agg(["sum", "count"])
    s, c = tot["sum"].to_dict(), tot["count"].to_dict()
    key = list(zip(df.task, df.problem_idx))
    own = sum(np.array([corr.get((t, m, i), 0.0) for t, m, i in zip(df.task, df[col], df.problem_idx)])
              for col in ("judge", "proxy", "opponent"))
    df["solve_rate"] = (np.array([s[x] for x in key]) - own) / (np.array([c[x] for x in key]) - 3)
    return df


# ------------------------------------------------------------------ grid (LP reweighting)
def _cells(fit, test, nest, group_p, rng):
    nests = sorted(set(fit[nest].itertuples(index=False, name=None)))
    per = N_P_BINS * N_CONF_BINS
    out = [np.full(len(fit), -1), np.full(len(test), -1)]
    for i, n in enumerate(nests):
        f = (fit[nest] == pd.Series(n, index=nest)).all(axis=1).to_numpy()
        h = (test[nest] == pd.Series(n, index=nest)).all(axis=1).to_numpy()
        counts = fit[f].groupby(GROUP).size()
        order = sorted(counts.index, key=lambda k: (group_p[k], rng.random()))
        cum = np.cumsum(counts.loc[order].to_numpy()) - 1
        p_bin = {k: min(int(N_P_BINS * c / max(cum[-1], 1)), N_P_BINS - 1) for k, c in zip(order, cum)}
        edges = np.quantile(fit.confidence[f], np.linspace(0, 1, N_CONF_BINS + 1))[1:-1]
        for frame, mask, o in ((fit, f, out[0]), (test, h, out[1])):
            keys = frame.loc[mask, GROUP].itertuples(index=False, name=None)
            pb = np.array([p_bin.get(k, -1) for k in keys], dtype=int)
            cb = np.searchsorted(edges, frame.confidence[mask].to_numpy())
            o[mask] = np.where(pb >= 0, i * per + pb * N_CONF_BINS + cb, -1)
    return out[0], out[1], len(nests)


def _solve(w0, cp, ca, cn, n_nests, target_p, target_acc, shares):
    n = len(w0)
    I = np.eye(n)
    rows = [np.r_[(cn == t).astype(float), np.zeros(n)] for t in range(n_nests)]
    A_eq = np.array(rows + [np.r_[cp, np.zeros(n)], np.r_[ca, np.zeros(n)]])
    r = linprog(np.r_[np.zeros(n), np.ones(n)], A_ub=np.block([[I, -I], [-I, -I]]), b_ub=np.r_[w0, -w0],
                A_eq=A_eq, b_eq=np.r_[shares, target_p, target_acc],
                bounds=[(0, MAX_UPWEIGHT * x) for x in w0] + [(0, None)] * n, method="highs")
    return r.x[:n] if r.success else None


def run_judge_rep(args):
    judge, rep, g, group_p, nest, p_grid, acc_offsets = args
    rng = np.random.default_rng(rep)
    fit, test = g[g.fold == 0].reset_index(drop=True), g[g.fold == 1].reset_index(drop=True)
    fit_c, test_c, n_nests = _cells(fit, test, nest, group_p, rng)
    fit, test = fit[fit_c >= 0].reset_index(drop=True), test[test_c >= 0].reset_index(drop=True)
    fit_c, test_c = fit_c[fit_c >= 0], test_c[test_c >= 0]
    if len(fit) < 50 or len(test) < 50:
        return []
    per = N_P_BINS * N_CONF_BINS
    cells, inv = np.unique(fit_c, return_inverse=True)
    n = np.bincount(inv)
    w0 = n / n.sum()
    cp = np.bincount(inv, fit.a_correct.to_numpy()) / n
    ca = np.bincount(inv, fit.acc.to_numpy()) / n
    cn = cells // per
    shares = np.bincount(cn, w0, minlength=n_nests)
    natural_acc = fit.acc.mean()
    n_cells = n_nests * per
    n_test = np.zeros(n_cells)
    np.add.at(n_test, test_c, 1)
    rows = []
    for tp in p_grid:
        for off in acc_offsets:
            base = {"judge": judge, "rep": rep, "target_p": tp, "acc_offset": off, "target_acc": natural_acc + off}
            w = _solve(w0, cp, ca, cn, n_nests, tp, natural_acc + off, shares)
            if w is None:
                rows.append({**base, "feasible": False})
                continue
            cell_w = np.zeros(n_cells)
            cell_w[cells] = w
            row_w = np.where(n_test[test_c] > 0, cell_w[test_c] / np.maximum(n_test[test_c], 1), 0.0)
            nest_i = test_c // per
            for t in range(n_nests):
                m = nest_i == t
                if row_w[m].sum() > 0:
                    row_w[m] *= shares[t] / row_w[m].sum()
            if row_w.sum() == 0:
                rows.append({**base, "feasible": False})
                continue
            rows.append({**base, "feasible": True, **weighted_stats(test, row_w)})
    return rows


def _jobs(df, nest, n_reps, p_grid, acc_offsets):
    items = df.drop_duplicates(GROUP + ["problem_idx"])
    problems = df[["task", "problem_idx"]].drop_duplicates()
    for rep in range(n_reps):
        rng = np.random.default_rng(1000 + rep)
        fold = problems.assign(fold=rng.integers(2, size=len(problems)))
        group_p = items.merge(fold, on=["task", "problem_idx"]).query("fold == 0").groupby(GROUP).a_correct.mean().to_dict()
        d = df.merge(fold, on=["task", "problem_idx"])
        for judge, g in d.groupby("judge"):
            yield judge, rep, g[KEEP + ["fold"]], group_p, nest, p_grid, acc_offsets


def reachable(runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["judge", "target_p", "acc_offset"]
    s = runs.groupby(keys).agg(feasible=("feasible", "mean"), target_acc=("target_acc", "mean"))
    ok = runs[runs.feasible]
    s = s.join(ok.groupby(keys)[["p", "actual_accuracy", "ess"] + list(MEASURES)].mean()).reset_index()
    s["reachable"] = ((s.feasible >= MIN_FEASIBLE) & ((s.p - s.target_p).abs() <= TOLERANCE)
                      & ((s.actual_accuracy - s.target_acc).abs() <= TOLERANCE) & (s.ess >= MIN_ESS))
    return s[s.reachable]


def grid_slopes(r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for j, g in r.groupby("judge"):
        if g.target_p.nunique() < 3 or g.acc_offset.nunique() < 3:
            continue
        X = np.c_[np.ones(len(g)), g.p, g.actual_accuracy]
        d = {"judge": j, "n_points": len(g)}
        for m in MEASURES:
            ok = g[m].notna().to_numpy()
            b = np.linalg.lstsq(X[ok], g[m].to_numpy()[ok], rcond=None)[0]
            d[m + "_p"], d[m + "_a"] = b[1], b[2]
        rows.append(d)
    return pd.DataFrame(rows)


def run_grid(df, nest=("task",), n_reps=N_REPS, p_grid=P_GRID, acc_offsets=ACC_OFFSETS):
    with Pool(WORKERS) as pool:
        runs = pd.DataFrame([r for rows in pool.imap_unordered(
            run_judge_rep, _jobs(df.reset_index(drop=True), list(nest), n_reps, p_grid, acc_offsets)) for r in rows])
    return reachable(runs) if len(runs) and runs.feasible.any() else pd.DataFrame()


def grid_summary(r: pd.DataFrame) -> dict:
    sl = grid_slopes(r) if len(r) else pd.DataFrame()
    cols = [m + k for m in MEASURES for k in ("_p", "_a")]
    return {"n_judges": int(len(sl)), "median": sl[cols].median().round(3).to_dict() if len(sl) else {},
            "per_judge": sl.round(3).to_dict(orient="records")}


# ------------------------------------------------------------------ 1. shared errors
def shared_errors(df) -> dict:
    b = df[df.error_known].copy()
    b["kind"] = np.where(b.shared_error, "shared", "different")
    by = b.groupby(["task", "kind"]).q.agg(["mean", "size"]).round(3).reset_index().to_dict(orient="records")
    key = ["task", "judge", "opponent", "problem_idx"]
    w = b.groupby(key + ["kind"]).q.mean().unstack().dropna()
    diff = w["shared"] - w["different"]
    boot = np.random.default_rng(0)
    bs = [diff.sample(len(diff), replace=True, random_state=int(boot.integers(1e9))).mean() for _ in range(2000)]
    # how the share of shared errors among B-won rows moves with p across generator pairs
    g = df[~df.a_correct & df.task.isin(["math500", "mmlu"])].groupby(["judge", "task", "opponent"]).agg(
        shared=("shared_error", "mean"), n=("q", "size"))
    p = df.groupby(["judge", "task", "opponent"]).a_correct.mean().rename("p")
    g = g.join(p).dropna()
    return {"fpr_by_kind": by,
            "within_item": {"n_items": int(len(diff)), "mean_diff": round(float(diff.mean()), 3),
                            "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                            "per_judge": diff.groupby(level="judge").mean().round(3).to_dict()},
            "share_of_known_errors_shared": round(float(b.shared_error.mean()), 3),
            "corr_shared_share_vs_p": round(float(np.corrcoef(g.shared, g.p)[0, 1]), 3)}


# ------------------------------------------------------------------ 2/3. what moves p
def _wls(d, y, x, fe, w):
    d = d.dropna(subset=[y] + x)
    d = d[d[w] > 0]
    dummies = pd.get_dummies(d[fe].astype(str).agg("|".join, axis=1)).to_numpy(float)
    X = np.hstack([d[x].to_numpy(float), dummies])
    W = np.sqrt(d[w].to_numpy(float))
    b, *_ = np.linalg.lstsq(X * W[:, None], d[y].to_numpy(float) * W, rcond=None)
    # cluster-robust-ish spread: refit on judge halves is overkill; report point estimates
    return {k: round(float(v), 3) for k, v in zip(x, b[:len(x)])}


def _unit_stats(df, keys):
    d = df.assign(aq=np.where(df.a_correct, df.q, np.nan), bq=np.where(~df.a_correct, df.q, np.nan))
    u = d.groupby(keys).agg(n=("q", "size"), p=("a_correct", "mean"), acc=("acc", "mean"),
                            tpr=("aq", "mean"), fpr=("bq", "mean"), na=("aq", "count"), nb=("bq", "count"),
                            pan=("q", "mean")).reset_index()
    u["eo"] = u.tpr + u.fpr - 1
    u["hspp"] = u.fpr
    u["panickssery"] = u.pan - 0.5
    u["w_eo"] = 1 / (1 / u.na.clip(lower=1) + 1 / u.nb.clip(lower=1)) * ((u.na > 0) & (u.nb > 0))
    return u


def what_moves_p(df) -> dict:
    df = df.copy()
    df["diff_bin"] = df.groupby("task").solve_rate.transform(
        lambda x: pd.qcut(x.rank(method="first"), N_DIFF_BINS, labels=False))
    designs = {
        "opponent":   (["judge", "task", "opponent"], ["judge", "task"]),
        "proxy":      (["judge", "task", "opponent", "proxy"], ["judge", "task", "opponent"]),
        "difficulty": (["judge", "task", "opponent", "proxy", "diff_bin"], ["judge", "task", "opponent", "proxy"]),
    }
    out = {}
    for name, (unit, fe) in designs.items():
        u = _unit_stats(df, unit)
        # spread of p that this design uses: SD of p around its fixed-effect means, weighted
        resid = u.p - u.groupby(fe).p.transform("mean")
        out[name] = {"units": int(len(u)), "p_sd_within": round(float(np.sqrt(np.average(resid ** 2, weights=u.n))), 3),
                     "tpr": _wls(u, "tpr", ["p", "acc"], fe, "na"),
                     "fpr_hspp": _wls(u, "fpr", ["p", "acc"], fe, "nb"),
                     "equal_opportunity": _wls(u, "eo", ["p", "acc"], fe, "w_eo"),
                     "panickssery": _wls(u, "panickssery", ["p", "acc"], fe, "n")}
    opp = _unit_stats(df, ["opponent"]).sort_values("p")
    out["by_opponent"] = opp[["opponent", "p", "tpr", "fpr", "eo"]].round(3).to_dict(orient="records")
    return out


def opponent_fixed_reach(df) -> dict:
    """With the opponent mix fixed (cells nested in task x opponent), how far can p move?"""
    r = run_grid(df, ("task", "opponent"), n_reps=4, p_grid=[round(x, 3) for x in np.arange(0.6, 0.97, 0.02)],
                 acc_offsets=[round(x, 3) for x in np.arange(-0.08, 0.081, 0.02)])
    rng = r.groupby("judge").agg(p_range=("p", lambda x: x.max() - x.min()),
                                 acc_range=("actual_accuracy", lambda x: x.max() - x.min()))
    base = run_grid(df, ("task",), n_reps=4)
    rb = base.groupby("judge").agg(p_range=("p", lambda x: x.max() - x.min()),
                                   acc_range=("actual_accuracy", lambda x: x.max() - x.min()))
    return {"fixed_opponent_median": rng.median().round(3).to_dict(), "free_median": rb.median().round(3).to_dict()}


# ------------------------------------------------------------------ 7. family affinity
def family_affinity() -> dict:
    """Judge-family x proxy-family interaction in J's preference for K, within item."""
    from real.roytburg import load_proxy_pairs
    from real.null_injection import sigmoid
    d = load_proxy_pairs()
    d["q"] = sigmoid(d.l)
    d["jf"], d["kf"], d["rf"] = d.judge.map(family), d.proxy.map(family), d.opponent.map(family)
    d = d[d.jf != d.rf]                         # keep R outside J's family; R is shared within item anyway
    item = ["task", "judge", "opponent", "problem_idx"]
    d = d[d.groupby(item).kf.transform("nunique") > 1]
    d["resid"] = d.q - d.groupby(item).q.transform("mean")
    # remove what every judge feels about a proxy family (answer quality/style common to all)
    d["resid2"] = d.resid - d.groupby(["task", "kf"]).resid.transform("mean")
    m = d.groupby(["jf", "kf"]).resid2.agg(["mean", "size"]).round(3)
    same = d[d.jf == d.kf].resid2.mean()
    cross = d[d.jf != d.kf].groupby(["jf", "kf"]).resid2.mean()
    # per (judge, proxy) offsets among cross-family pairs vs the proxy's p: does null drift with p?
    c = d[d.jf != d.kf]
    jk = c.groupby(["judge", "proxy"]).resid.mean()
    pk = c.groupby(["judge", "proxy"]).a_correct.mean()
    return {"matrix": [{"judge_family": a, "proxy_family": b, "mean": float(r["mean"]), "n": int(r["size"])}
                       for (a, b), r in m.iterrows()],
            "same_family_mean": round(float(same), 3),
            "cross_family_abs_mean": round(float(cross.abs().mean()), 3),
            "cross_family_offset_vs_proxy_p_corr": round(float(np.corrcoef(jk, pk.loc[jk.index])[0, 1]), 3)}


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    import time
    t0 = time.time()
    df = prepare()
    out = {"n_rows": int(len(df))}
    out["shared_errors"] = shared_errors(df)
    print("shared errors:", out["shared_errors"]["within_item"], f"{time.time() - t0:.0f}s")
    out["what_moves_p"] = what_moves_p(df)
    for k in ("opponent", "proxy", "difficulty"):
        print(k, out["what_moves_p"][k])
    out["family"] = family_affinity()
    print("family:", {k: v for k, v in out["family"].items() if k != "matrix"})
    out["opponent_fixed_reach"] = opponent_fixed_reach(df)
    print("reach:", out["opponent_fixed_reach"], f"{time.time() - t0:.0f}s")

    # placebo: drop as many random B-won MATH/MMLU rows as there are shared-error rows (5 draws)
    bwon_mm = np.flatnonzero((~df.a_correct & df.task.isin(["math500", "mmlu"])).to_numpy())
    configs = {"base": df, "no_shared": df[~df.shared_error],
               "low_tie": df[df.tie_max <= TIE_MAX], "consistent": df[df.consistent]}
    for seed in range(1, 6):
        drop = np.zeros(len(df), bool)
        drop[np.random.default_rng(seed).choice(bwon_mm, int(df.shared_error.sum()), replace=False)] = True
        configs[f"placebo_{seed}"] = df[~drop]
    out["grid"] = {}
    for name, d in configs.items():
        out["grid"][name] = {"rows": int(len(d)), **grid_summary(run_grid(d))}
        print(f"grid {name:12s} rows={len(d):6d} judges={out['grid'][name]['n_judges']} "
              f"{out['grid'][name]['median']}  {time.time() - t0:.0f}s")
    # medians over the judges every config could estimate, so rows are comparable
    per = {k: pd.DataFrame(v["per_judge"]).set_index("judge") for k, v in out["grid"].items()}
    common = sorted(set.intersection(*[set(p.index) for p in per.values()]))
    cols = [m + k for m in MEASURES for k in ("_p", "_a")]
    out["grid_common"] = {"judges": common,
                          **{k: per[k].loc[common, cols].median().round(3).to_dict() for k in per if not k.startswith("placebo")}}
    pl = pd.DataFrame([per[k].loc[common, cols].median() for k in per if k.startswith("placebo")])
    out["grid_common"]["placebo_mean"] = pl.mean().round(3).to_dict()
    out["grid_common"]["placebo_sd"] = pl.std().round(3).to_dict()
    print("common judges:", common)
    for k, v in out["grid_common"].items():
        if k != "judges":
            print(f"  {k:13s}", v)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT_PATH)
