"""
Simulator side of "why beta explains only part of the real slopes" (difficulty_sweep.py
extended), and the data for the beta explainer page.

4. Difficulty spread / shape / larger beta: slopes at the real reference point over
   tau x beta x shape (normal, bimodal).
6. Comparison mismatch: simulated slopes computed the way the real ones are -- OLS of the
   measure on (p, accuracy) over each judge's reachable grid points
   (real/tune_grid_results.csv), median over the judges with >= 3 levels of both knobs.

Writes difficulty_gap_results.json (curves for the page + both tables).
Run with: python difficulty_gap.py
"""

import json

import numpy as np
import pandas as pd

from synth.difficulty import expected_measures

MEASURES = ["panickssery", "equal_opportunity", "hspp", "roytburg"]
BETAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
TAUS = [0.5, 1.0, 2.0, 3.0]
SHAPES = ["normal", "bimodal"]
REF = {"p": 0.86, "a": 0.64}
P_CURVE = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
A_CURVE = [round(x, 2) for x in np.arange(0.55, 0.951, 0.025)]
S_CURVE = [round(x, 2) for x in np.arange(-2, 2.01, 0.25)]
EPS = 1e-3


def slopes_at(p, a, beta, tau, shape):
    out = {}
    for knob in ("p", "a", "s"):
        d = {k: EPS * (k == knob) for k in ("p", "a", "s")}
        hi = expected_measures(p + d["p"], a + d["a"], d["s"], beta, tau, shape)
        lo = expected_measures(p - d["p"], a - d["a"], -d["s"], beta, tau, shape)
        out[knob] = {m: (hi[m] - lo[m]) / (2 * EPS) for m in MEASURES}
    return out


def real_points():
    r = pd.read_csv("real/tune_grid_results.csv")
    r = r[r.reachable]
    keep = [j for j, g in r.groupby("judge") if g.target_p.nunique() >= 3 and g.acc_offset.nunique() >= 3]
    return r[r.judge.isin(keep)]


def ols_slopes(g, values):
    X = np.c_[np.ones(len(g)), g.p, g.actual_accuracy]
    b = np.linalg.lstsq(X, values, rcond=None)[0]
    return b[1], b[2]


def region_slopes(pts, beta, tau=1.0, shape="normal"):
    """Sim analogue of the real slope: same points, same OLS, per judge; median."""
    per = []
    for j, g in pts.groupby("judge"):
        sim = pd.DataFrame([expected_measures(p, a, 0.0, beta, tau, shape) for p, a in zip(g.p, g.actual_accuracy)])
        per.append({m + k: v for m in MEASURES for k, v in zip(("_p", "_a"), ols_slopes(g, sim[m].to_numpy()))})
    return pd.DataFrame(per).median().to_dict()


def real_slopes(pts):
    per = []
    for j, g in pts.groupby("judge"):
        per.append({m + k: v for m in MEASURES for k, v in zip(("_p", "_a"), ols_slopes(g, g[m].to_numpy()))})
    return pd.DataFrame(per).median().to_dict()


if __name__ == "__main__":
    out = {"ref": REF, "betas": BETAS, "taus": TAUS, "shapes": SHAPES, "measures": MEASURES,
           "p_curve": P_CURVE, "a_curve": A_CURVE, "s_curve": S_CURVE, "configs": []}
    for shape in SHAPES:
        for tau in TAUS:
            for beta in BETAS:
                cfg = {"shape": shape, "tau": tau, "beta": beta,
                       "slopes": slopes_at(REF["p"], REF["a"], beta, tau, shape),
                       "at_s0": {m: expected_measures(REF["p"], REF["a"], 0.0, beta, tau, shape)[m] for m in MEASURES},
                       "vs_p": {m: [] for m in MEASURES}, "vs_a": {m: [] for m in MEASURES}, "vs_s": {m: [] for m in MEASURES}}
                for key, grid, f in (("vs_p", P_CURVE, lambda x: (x, REF["a"], 0.0)),
                                     ("vs_a", A_CURVE, lambda x: (REF["p"], x, 0.0)),
                                     ("vs_s", S_CURVE, lambda x: (REF["p"], REF["a"], x))):
                    for x in grid:
                        e = expected_measures(*f(x), beta, tau, shape)
                        for m in MEASURES:
                            cfg[key][m].append(round(float(e[m]), 5))
                out["configs"].append(cfg)
    print("swept", len(out["configs"]), "configs")

    pts = real_points()
    out["real_region"] = {"judges": sorted(pts.judge.unique()), "n_points": int(len(pts)),
                          "real": real_slopes(pts),
                          "sim": {str(b): region_slopes(pts, b) for b in BETAS}}
    print("real (median over", pts.judge.nunique(), "judges):", {k: round(v, 2) for k, v in out["real_region"]["real"].items()})
    for b in BETAS:
        print(f"sim region beta={b}:", {k: round(v, 2) for k, v in out["real_region"]["sim"][str(b)].items()})

    print("\nslopes at reference, by tau x beta (normal / bimodal): EO_p, EO_a, HSPP_p, HSPP_a, EO_s")
    for c in out["configs"]:
        if c["beta"] in (0, 2, 4, 8):
            s = c["slopes"]
            print(f"{c['shape']:8s} tau={c['tau']:<4} beta={c['beta']:<4} "
                  f"{s['p']['equal_opportunity']:6.2f} {s['a']['equal_opportunity']:6.2f} "
                  f"{s['p']['hspp']:6.2f} {s['a']['hspp']:6.2f} {s['s']['equal_opportunity']:6.2f}")
    with open("difficulty_gap_results.json", "w") as f:
        json.dump(out, f)
    print("wrote difficulty_gap_results.json")
