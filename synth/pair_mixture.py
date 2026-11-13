"""
Simulator slopes computed the way the real grid computes them (real/slope_gap.py).

The real grid moves a judge's p by reweighting its rows, and those rows come from many
generator pairs (units) with very different win rates p_u (5-95%: 0.14-0.97). So the p it
reaches is a MIXTURE of pairs, and each pair carries its own lean b_u = gamma * logit(p_u).
A point derivative of synth/difficulty.py at one p (all items sharing one lean) overstates
how fast the pooled lean moves with pooled p.

Here, per real judge: units with the real p_u and sizes n_u (Roytburg null pairs), each unit's
TPR/FPR exact from synth/difficulty.expected_measures(p_u, a + delta, s=0, beta, gamma).
Random reweightings of the units (log-normal weights) and accuracy shifts delta give pooled
(p, actual accuracy, measures); OLS of each measure on (p, accuracy) over the draws = the
simulator's grid slopes. Median over judges, as the real grid.

Run with: python -m synth.pair_mixture     (prints; compare with real/pair_check_results.json)
"""

import numpy as np
import pandas as pd

from synth.difficulty import expected_measures

A_REF = 0.64                      # median real null-judge accuracy (difficulty_sweep.REF)
DELTAS = np.linspace(-0.07, 0.07, 8)
N_DRAWS = 400
WEIGHT_SD = 1.0
STRATA = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])   # difficulty strata within a pair, logit shifts
                                                  # (gives p spread ~0.18 within pair, as real)
RHO = (0.16, 0.41)   # consensus-error shares (real/item_check.py, confident items)
MEASURES = ["panickssery", "equal_opportunity", "hspp"]


def unit_table(p_u, beta, gamma, strata, rho=(0.0, 0.0)):
    """Rows = (pair, difficulty stratum) cells; the lean follows the PAIR's win rate."""
    rows, cell_p = [], []
    lg = lambda x: np.log(x / (1 - x))
    for i, pp in enumerate(p_u):
        pp = min(max(pp, 0.02), 0.98)
        for k, sh in enumerate(strata):
            p = 1 / (1 + np.exp(-(lg(pp) + sh)))
            p = min(max(p, 0.01), 0.99)
            cell_p.append(p)
            for j, dlt in enumerate(DELTAS):
                m = expected_measures(p, A_REF + dlt, 0.0, beta, gamma=gamma, pair_p=pp, rho=rho)
                tpr = m["equal_opportunity"] + 1 - m["hspp"]
                rows.append((len(cell_p) - 1, j, p, tpr, m["hspp"]))
    t = pd.DataFrame(rows, columns=["u", "d", "p", "tpr", "fpr"])
    return (np.array(cell_p), t.pivot(index="u", columns="d", values="tpr").to_numpy(),
            t.pivot(index="u", columns="d", values="fpr").to_numpy())


def judge_slopes(p_u, n_u, beta, gamma, rng, strata=STRATA, rho=(0.0, 0.0)):
    p_u, TPR, FPR = unit_table(np.asarray(p_u), beta, gamma, strata, rho)
    n_u = np.repeat(np.asarray(n_u, float) / len(strata), len(strata))
    X, Y = [], {m: [] for m in MEASURES}
    for _ in range(N_DRAWS):
        w = n_u * np.exp(rng.normal(0, WEIGHT_SD, len(p_u)))
        j = rng.integers(len(DELTAS))
        tpr_u, fpr_u = TPR[:, j], FPR[:, j]
        pa, pb = w @ p_u, w @ (1 - p_u)
        tpr, fpr = (w * p_u) @ tpr_u / pa, (w * (1 - p_u)) @ fpr_u / pb
        p = pa / w.sum()
        acc = p * tpr + (1 - p) * (1 - fpr)
        X.append((p, acc))
        Y["panickssery"].append(p * tpr + (1 - p) * fpr - 0.5)
        Y["equal_opportunity"].append(tpr + fpr - 1)
        Y["hspp"].append(fpr)
    X = np.column_stack([np.array(X), np.ones(len(X))])
    out = {}
    for m in MEASURES:
        b = np.linalg.lstsq(X, np.array(Y[m]), rcond=None)[0]
        out[m + "_p"], out[m + "_a"] = b[0], b[1]
    return out


def main():
    from real.null_injection import load
    from real.pair_check import units
    u, gamma_fit = units(load())
    rng = np.random.default_rng(0)
    print(f"real units per judge, gamma fitted {gamma_fit:.2f}")
    print("real grid: EO p 0.68 a 1.15 | HSPP p 0.79 a 0.06 | Pan p 0.18 a 1.01;  pair_model: EO p 0.77 a 0.80")
    configs = [("base (beta, no pair, no rho)", 0.79, 0.0, (0.0, 0.0)),
               ("+ pair lean", 0.79, round(gamma_fit, 2), (0.0, 0.0)),
               ("+ consensus errors", 0.79, 0.0, RHO),
               ("+ pair lean + consensus errors", 0.79, round(gamma_fit, 2), RHO)]
    for label, beta, gamma, rho in configs:
        res = pd.DataFrame([judge_slopes(d.p, d.n, beta, gamma, rng, STRATA, rho) for _, d in u.groupby("judge")])
        med = res.median()
        print(f"{label:32s}: " + ", ".join(f"{k} {v:.2f}" for k, v in med.items()))

if __name__ == "__main__":
    main()
