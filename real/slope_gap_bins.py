"""
Binned scatter data for check 3 of real/slope_gap.py ("what moves p"), for the beta page.

For each design (p moving between opponents / between proxies / between difficulty strata
with the generator pair fixed): units as in slope_gap.what_moves_p; the measure and the knob
are residualized on the design's fixed effects and the other knob (Frisch-Waugh-Lovell, same
weights), grand means added back. The OLS slope through the binned points then equals the
regression coefficient reported in the table.

    EO   vs p          (accuracy partialled out)
    HSPP vs accuracy   (p partialled out)

Run with: python -m real.slope_gap_bins      writes real/slope_gap_bins.json
"""

import json
import os

import numpy as np
import pandas as pd

from real.slope_gap import N_DIFF_BINS, _unit_stats, prepare

N_BINS = 10
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slope_gap_bins.json")
DESIGNS = {
    "opponent":   (["judge", "task", "opponent"], ["judge", "task"]),
    "proxy":      (["judge", "task", "opponent", "proxy"], ["judge", "task", "opponent"]),
    "difficulty": (["judge", "task", "opponent", "proxy", "diff_bin"], ["judge", "task", "opponent", "proxy"]),
}


def _resid(d, y, controls, fe, w):
    X = np.hstack([pd.get_dummies(d[fe].astype(str).agg("|".join, axis=1)).to_numpy(float),
                   d[controls].to_numpy(float)])
    W = np.sqrt(d[w].to_numpy(float))
    b, *_ = np.linalg.lstsq(X * W[:, None], d[y].to_numpy(float) * W, rcond=None)
    r = d[y].to_numpy(float) - X @ b
    return r + np.average(d[y], weights=d[w])


def binned(u, y, x, other, fe, w):
    d = u.dropna(subset=[y, x, other])
    d = d[d[w] > 0].reset_index(drop=True)
    xr, yr = _resid(d, x, [other], fe, w), _resid(d, y, [other], fe, w)
    wt = d[w].to_numpy(float)
    order = np.argsort(xr)
    cum = np.cumsum(wt[order]) / wt.sum()
    bins = np.minimum((cum * N_BINS).astype(int), N_BINS - 1)
    pts = []
    for b in range(N_BINS):
        idx = order[bins == b]
        if len(idx):
            pts.append({"x": float(np.average(xr[idx], weights=wt[idx])), "y": float(np.average(yr[idx], weights=wt[idx])),
                        "n": int(len(idx))})
    slope = float(np.polyfit(xr, yr, 1, w=np.sqrt(wt))[0])
    return {"points": pts, "slope": slope, "x_mean": float(np.average(xr, weights=wt)),
            "y_mean": float(np.average(yr, weights=wt)), "units": int(len(d))}


if __name__ == "__main__":
    df = prepare()
    df["diff_bin"] = df.groupby("task").solve_rate.transform(
        lambda x: pd.qcut(x.rank(method="first"), N_DIFF_BINS, labels=False))
    out = {}
    for name, (unit, fe) in DESIGNS.items():
        u = _unit_stats(df, unit)
        out[name] = {"eo_vs_p": binned(u, "eo", "p", "acc", fe, "w_eo"),
                     "hspp_vs_acc": binned(u, "fpr", "acc", "p", fe, "nb")}
        print(name, {k: round(v["slope"], 3) for k, v in out[name].items()})
    json.dump(out, open(OUT, "w"))
    print("wrote", OUT)
