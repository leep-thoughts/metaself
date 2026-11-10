"""
Meta-evaluation of Yang et al.'s SPB (measures/yang.py) next to the four earlier measures,
in the continuous-quality world (synth/continuous.py).

Requirements (as on the explainer page):
  R1 insensitive to confounds   d M / d p, d M / d a at s = 0 (also d M / d pos)
  R2 zero without bias          M at s = 0
  R3 responds to bias           d M / d s at s = 0
  R4 same response everywhere   spread of d M / d s over the (p, a) grid (max / min)

Swept nuisances, Yang-specific: sigma_e (noise in the reference judges' quality scores),
eps (equal-quality tolerance), pos (position bias). Writes yang_results.json.
Run with: python yang_sweep.py
"""

import json

import numpy as np

from synth.continuous import EPS, SIGMA_E, expected_measures

MEASURES = ["panickssery", "equal_opportunity", "hspp", "roytburg", "yang"]
REF = {"p": 0.86, "a": 0.64}
P_GRID = [round(x, 2) for x in np.arange(0.50, 0.951, 0.05)]
A_GRID = [round(x, 2) for x in np.arange(0.55, 0.951, 0.05)]
SIGMAS = [0.0, 0.25, 0.5, 1.0]
EPSS = [0.05, 0.17, 0.5]
POSS = [0.0, 1.0, 2.0]
H = 1e-3


def d(knob, p, a, s=0.0, pos=0.0, sigma_e=SIGMA_E, eps=EPS):
    args = dict(p=p, a=a, s=s, pos=pos, sigma_e=sigma_e, eps=eps)
    hi, lo = dict(args), dict(args)
    hi[knob] += H
    lo[knob] -= H
    e_hi, e_lo = expected_measures(**hi), expected_measures(**lo)
    return {m: (e_hi[m] - e_lo[m]) / (2 * H) for m in MEASURES}


def requirements(pos=0.0, sigma_e=SIGMA_E, eps=EPS):
    kw = dict(pos=pos, sigma_e=sigma_e, eps=eps)
    at0 = expected_measures(REF["p"], REF["a"], 0.0, **kw)
    out = {"R1_p": d("p", REF["p"], REF["a"], **kw), "R1_a": d("a", REF["p"], REF["a"], **kw),
           "R2": {m: at0[m] for m in MEASURES}, "R3": d("s", REF["p"], REF["a"], **kw),
           "residual_gap": at0["residual_gap"]}
    ds = [d("s", p, a, **kw) for p in P_GRID for a in A_GRID]
    out["R4_ratio"] = {m: max(x[m] for x in ds) / min(x[m] for x in ds) for m in MEASURES}
    grid = [expected_measures(p, a, 0.0, **kw) for p in P_GRID for a in A_GRID]
    out["range_s0"] = {m: max(x[m] for x in grid) - min(x[m] for x in grid) for m in MEASURES}
    return out


def show(title, r):
    print(f"\n{title}   (residual quality gap in 'equal' own pairs: {r['residual_gap']:.2f} SD)")
    print(f"{'':10s}" + "".join(f"{m:>18s}" for m in MEASURES))
    for k, label in (("R1_p", "dM/dp"), ("R1_a", "dM/da"), ("R2", "M at s=0"), ("R3", "dM/ds"),
                     ("R4_ratio", "R4 max/min"), ("range_s0", "grid range")):
        print(f"{label:10s}" + "".join(f"{r[k][m]:18.3f}" for m in MEASURES))


if __name__ == "__main__":
    out = {"ref": REF, "defaults": {"sigma_e": SIGMA_E, "eps": EPS}, "measures": MEASURES}
    out["default"] = requirements()
    show(f"defaults: sigma_e={SIGMA_E}, eps={EPS}, pos=0", out["default"])
    out["by_sigma"] = {str(x): requirements(sigma_e=x) for x in SIGMAS}
    out["by_eps"] = {str(x): requirements(eps=x) for x in EPSS}
    out["by_pos"] = {str(x): requirements(pos=x) for x in POSS}
    print("\nYang only, by nuisance:  dM/dp   dM/da   M(s=0)   dM/ds   R4   residual gap")
    for name, group in (("sigma_e", "by_sigma"), ("eps", "by_eps"), ("pos", "by_pos")):
        for x, r in out[group].items():
            print(f"  {name}={x:<5} {r['R1_p']['yang']:7.3f} {r['R1_a']['yang']:7.3f} {r['R2']['yang']:8.3f} "
                  f"{r['R3']['yang']:7.3f} {r['R4_ratio']['yang']:5.2f} {r['residual_gap']:8.2f}")
    out["d_pos"] = d("pos", REF["p"], REF["a"], pos=1.0)
    print("\nd M / d pos at pos = 1:", {m: round(v, 3) for m, v in out["d_pos"].items()})
    with open("yang_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote yang_results.json")
