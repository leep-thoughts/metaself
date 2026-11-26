"""
Null-test values over natural units (method principle 4): one dot per judge x target (x comparison
set), x = the quality gap as a model property, y = the measure on ALL of that unit's rows. No rows
are dropped or reweighted to move x.

Per measure: existence = the fitted value at gap 0, trend = the slope; per judge (OLS of y on x),
then the median over judges; 95% intervals from a two-way bootstrap that resamples judges and
targets together.

Pairwise (Roytburg's null pairs, real/null_injection.load):
    unit = judge x target (tasks and opponents pooled as the data has them); x = mean over the unit's rows
    of logit(target's solve rate on the task) - logit(opponent's), from Chen's answers; y = panickssery, equal_opportunity, HSPP with soft votes.
    Caveat built into the data: Roytburg keeps a target's row only where its correctness equals the
    judge's own on that problem, so "A wrong" rows are the problems the judge itself got wrong.
    roytburg per target (real/pair_control_results.csv, set "all") and Yang (real/yang_null_results.csv).
Rubric (real/rubric_placebo_results[_healthbench].csv; fixes from rubric_control and rubric_did hspp):
    unit = judge x target, x = target's pass rate minus the comparison models'.

Run with: python -m real.natural_scatter     writes real/natural_scatter_results.json (+ _points.json)
"""

import json
import os

import numpy as np
import pandas as pd

from infer.yang_data import TASKS, load_answers
from real import null_injection

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "natural_scatter_results.json")
N_BOOT = 500
MIN_SIDE = 10


def fit(df):
    """Median over judges of the per-judge intercept (value at x = 0) and slope."""
    a, b = [], []
    for _, g in df.groupby("judge"):
        if g.x.nunique() < 3:
            continue
        s, i = np.polyfit(g.x, g.y, 1)
        a.append(i)
        b.append(s)
    return (np.median(a), np.median(b), len(a)) if a else (np.nan, np.nan, 0)


def summarise(df, target_col="target"):
    df = df.dropna(subset=["x", "y"])
    a, b, n = fit(df)
    judges, targets = list(df.judge.unique()), list(df[target_col].unique())
    cell = {k: (g.x.to_numpy(), g.y.to_numpy()) for k, g in df.groupby(["judge", target_col])}
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(N_BOOT):
        ts = rng.choice(len(targets), len(targets))
        ia, ib = [], []
        for jdx in rng.choice(len(judges), len(judges)):
            parts = [cell[(judges[jdx], targets[t])] for t in ts if (judges[jdx], targets[t]) in cell]
            if not parts:
                continue
            x = np.concatenate([p[0] for p in parts])
            y = np.concatenate([p[1] for p in parts])
            if len(np.unique(x)) < 3:
                continue
            s_, i_ = np.polyfit(x, y, 1)
            ia.append(i_)
            ib.append(s_)
        if ia:
            boots.append((np.median(ia), np.median(ib)))
    bo = np.array(boots)
    return {"at_gap_0": a, "at_gap_0_ci": np.nanpercentile(bo[:, 0], [2.5, 97.5]).tolist(),
            "slope": b, "slope_ci": np.nanpercentile(bo[:, 1], [2.5, 97.5]).tolist(),
            "judges": n, "dots": len(df), "mean_y": float(df.y.mean())}


def pairwise_cells():
    d = null_injection.load()
    d["q"] = null_injection.sigmoid(d.l)
    lg = lambda p: np.log(p / (1 - p))
    q = {}
    for t in TASKS:
        a = pd.Series({k: v[2] for k, v in load_answers(t).items()})
        for m, s in a.groupby(level=0).mean().items():
            q[(t, m)] = float(np.clip(s, 0.02, 0.98))
    d["xg"] = [lg(q[(t, k)]) - lg(q[(t, r)]) for t, k, r in zip(d.task, d.proxy, d.opponent)]
    rows = []
    for (j, k), g in d.groupby(["judge", "proxy"]):         # tasks and opponents pooled as the data has them
        right, wrong = g[g.a_correct], g[~g.a_correct]
        if len(right) < MIN_SIDE or len(wrong) < MIN_SIDE:
            continue
        rows.append({"judge": j, "target": k, "x": g.xg.mean(),
                     "panickssery": g.q.mean() - 0.5,
                     "equal_opportunity": right.q.mean() - (1 - wrong.q.mean()),
                     "hspp": wrong.q.mean(), "n": len(g)})
    return pd.DataFrame(rows)


def main():
    out, points = {"pairwise": {}, "rubric": {}}, {}
    cells = pairwise_cells()
    for m in ("panickssery", "equal_opportunity", "hspp"):
        out["pairwise"][m] = summarise(cells.assign(y=cells[m]))
        points[f"pairwise_{m}"] = cells[["judge", "x", m]].round(4).values.tolist()
    pc = pd.read_csv(os.path.join(HERE, "pair_control_results.csv"))
    pc = pc[pc.set == "all"].rename(columns={"proxy": "target", "gap": "x", "value": "y"})
    out["pairwise"]["roytburg"] = summarise(pc)
    points["pairwise_roytburg"] = pc[["judge", "x", "y"]].round(4).values.tolist()
    y = pd.read_csv(os.path.join(HERE, "yang_null_results.csv")).rename(columns={"quality_rel": "x", "spb_null": "y"})
    out["pairwise"]["yang"] = summarise(y)
    for ds, suf in (("ifeval", ""), ("healthbench", "_healthbench")):
        pl = pd.read_csv(os.path.join(HERE, f"rubric_placebo_results{suf}.csv"))
        pl = pl[~pl.self].rename(columns={"generator": "target", "gap": "x"})
        res = {}
        for m in ("score_gap", "calibrated_gap", "mrspb", "hspp_rub", "hspp_rub_ratio", "eo_rub", "matched_fail", "matched_both"):
            res[m] = summarise(pl.assign(y=pl[m]))
            if ds == "ifeval" and m in ("hspp_rub", "matched_fail"):
                points[f"rubric_{m}"] = pl[["judge", "x", m]].round(4).values.tolist()
        rc = pd.read_csv(os.path.join(HERE, f"rubric_control_results{suf}.csv"))
        rc = rc[(~rc.self) & (rc.set == "subsample")].rename(columns={"generator": "target", "gap_all": "x"})
        res["hspp_rub | same quality"] = summarise(rc.assign(y=rc.hspp_rub))
        if ds == "ifeval":
            dd = pd.read_csv(os.path.join(HERE, "rubric_did_hspp_results.csv"))
            dd = dd[~dd.self].rename(columns={"generator": "target", "gap": "x"})
            for s, name in (("all", "second judge"), ("same quality", "both")):
                res[f"hspp_rub | {name}"] = summarise(dd[dd.set == s].assign(y=dd.hspp_rubric_did))
        out["rubric"][ds] = res
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    json.dump(points, open(OUT.replace(".json", "_points.json"), "w"))
    for setting, block in out.items():
        items = block.items() if setting == "pairwise" else [(f"{ds}:{m}", v) for ds, r in block.items() for m, v in r.items()]
        for m, v in items:
            print(f"{setting:8s} {m:32s} at gap 0 {v['at_gap_0']:+.3f} [{v['at_gap_0_ci'][0]:+.3f}, {v['at_gap_0_ci'][1]:+.3f}]  "
                  f"slope {v['slope']:+.3f} [{v['slope_ci'][0]:+.3f}, {v['slope_ci'][1]:+.3f}]  ({v['judges']} judges, {v['dots']} dots)")


if __name__ == "__main__":
    main()
