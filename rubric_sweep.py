"""
Meta-evaluation of the rubric self-preference measures (measures/rubric.py) in the rubric
simulator (synth/rubric.py): requirements R1-R4 for several assumed difficulty couplings
(beta) and generator halos (h).

    R1  insensitive to confounds   d M / d p, d M / d a, d M / d leniency at s = 0
    R2  zero without bias          M at s = 0
    R3  responds to bias           d M / d s at s = 0
    R4  same response everywhere   max / min of d M / d s over the (p, a, leniency) grid

Reference point (IFEval-like): own quality p 0.80, others q 0.75, judge accuracy 0.85.
Writes rubric_results.json. Run with: python rubric_sweep.py
"""

import json
import itertools

import numpy as np

from measures.rubric import MEASURES
from synth.rubric import expected_measures

REF = {"p": 0.80, "a": 0.85, "lenient": 0.0}
P_GRID = [0.6, 0.7, 0.8, 0.9]
A_GRID = [0.7, 0.8, 0.9]
LEN_GRID = [-1.0, 0.0, 1.0]
BETAS = [0.0, 1.0, 2.0]
HALOS = [0.0, 0.5, 1.0]
H = 1e-3


def _m(p, a, s, lenient, beta, h):
    return expected_measures(p, a, s, lenient=lenient, beta=beta, h=h)


def d(knob, p, a, lenient, beta, h, s=0.0):
    x = dict(p=p, a=a, s=s, lenient=lenient)
    hi, lo = dict(x), dict(x)
    hi[knob] += H
    lo[knob] -= H
    e1, e0 = _m(beta=beta, h=h, **hi), _m(beta=beta, h=h, **lo)
    return {m: (e1[m] - e0[m]) / (2 * H) for m in MEASURES}


def requirements(beta, h):
    r = {"R1_p": d("p", **REF, beta=beta, h=h), "R1_a": d("a", **REF, beta=beta, h=h),
         "R1_len": d("lenient", **REF, beta=beta, h=h),
         "R2": {m: v for m, v in _m(REF["p"], REF["a"], 0.0, REF["lenient"], beta, h).items() if m in MEASURES},
         "R3": d("s", **REF, beta=beta, h=h)}
    ds = [d("s", p, a, l, beta, h) for p, a, l in itertools.product(P_GRID, A_GRID, LEN_GRID)]
    r["R4"] = {m: max(x[m] for x in ds) / min(x[m] for x in ds) for m in MEASURES}
    return r


if __name__ == "__main__":
    out = {"ref": REF, "measures": MEASURES, "configs": []}
    for beta, h in itertools.product(BETAS, HALOS):
        r = requirements(beta, h)
        out["configs"].append({"beta": beta, "h": h, **r})
        print(f"\nbeta={beta} h={h}")
        print(f"{'':11s}" + "".join(f"{m:>15s}" for m in MEASURES))
        for k, lab in (("R1_p", "dM/dp"), ("R1_a", "dM/da"), ("R1_len", "dM/dlenient"), ("R2", "M at s=0"),
                       ("R3", "dM/ds"), ("R4", "R4 max/min")):
            print(f"{lab:11s}" + "".join(f"{r[k][m]:15.3f}" for m in MEASURES))
    with open("rubric_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote rubric_results.json")
