"""
Meta-evaluation sweep with the difficulty coupling beta (synth/difficulty.py), for several
assumed betas: real-data calibration of beta is deferred.

For each beta: each measure's change with p, accuracy and s, others held fixed, as exact
expectations. Writes difficulty_results.json.

Run with: python difficulty_sweep.py
"""

import json

import numpy as np

from synth.difficulty import expected_measures

BETAS = [0.0, 0.5, 1.0, 2.0]
MEASURES = ["panickssery", "equal_opportunity", "hspp", "roytburg"]
P_GRID = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
A_GRID = [round(x, 2) for x in np.arange(0.55, 0.96, 0.05)]
S_GRID = [round(x, 2) for x in np.arange(-2, 2.01, 0.25)]
REF = {"p": 0.86, "a": 0.64}   # median natural p and accuracy of the real null judges
EPS = 1e-3


def slope(knob, beta, p, a, s=0.0):
    hi = expected_measures(p + EPS * (knob == "p"), a + EPS * (knob == "a"), s + EPS * (knob == "s"), beta)
    lo = expected_measures(p - EPS * (knob == "p"), a - EPS * (knob == "a"), s - EPS * (knob == "s"), beta)
    return {m: (hi[m] - lo[m]) / (2 * EPS) for m in MEASURES}


def ranges(beta):
    """Mean range of each measure over one knob's grid at s = 0 (confounds) and over s."""
    out = {}
    for knob, grid, others in (("p", P_GRID, [(a, 0.0) for a in A_GRID]),
                               ("a", A_GRID, [(p, 0.0) for p in P_GRID])):
        r = []
        for o, s in others:
            vals = [expected_measures(x, o, s, beta) if knob == "p" else expected_measures(o, x, s, beta) for x in grid]
            r.append({m: max(v[m] for v in vals) - min(v[m] for v in vals) for m in MEASURES})
        out[knob] = {m: float(np.mean([x[m] for x in r])) for m in MEASURES}
    return out


if __name__ == "__main__":
    rows = []
    for beta in BETAS:
        for p in P_GRID:
            for a in A_GRID:
                for s in S_GRID:
                    rows.append({"beta": beta, **expected_measures(p, a, s, beta), "self_lean": s})
    summary = {}
    print(f"Slopes at real-data reference p={REF['p']}, a={REF['a']}, s=0 (per unit of each knob)")
    print(f"{'beta':>5} {'knob':>4} " + " ".join(f"{m:>18}" for m in MEASURES))
    for beta in BETAS:
        summary[beta] = {k: slope(k, beta, REF["p"], REF["a"]) for k in ("p", "a", "s")}
        summary[beta]["at_s0"] = {m: v for m, v in expected_measures(REF["p"], REF["a"], 0.0, beta).items() if m in MEASURES}
        for k in ("p", "a", "s"):
            print(f"{beta:>5} {k:>4} " + " ".join(f"{summary[beta][k][m]:>18.3f}" for m in MEASURES))
        print(f"{beta:>5} {'M@0':>4} " + " ".join(f"{summary[beta]['at_s0'][m]:>18.3f}" for m in MEASURES))
    print("\nMean range over the whole grid at s=0 (confound sensitivity)")
    for beta in BETAS:
        r = ranges(beta)
        for k in ("p", "a"):
            print(f"{beta:>5} {k:>4} " + " ".join(f"{r[k][m]:>18.3f}" for m in MEASURES))
        summary[beta]["grid_range"] = r
    with open("difficulty_results.json", "w") as f:
        json.dump({"rows": rows, "summary": {str(b): v for b, v in summary.items()}}, f)
    print("\nwrote difficulty_results.json")
