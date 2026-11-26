"""
Is removing the lean (real/pair_check.py, copy 1 = "no_pair") a valid operation? Three checks on
Roytburg's null pairs, each built from the real rows so item structure and noise stay realistic.

Per unit u = (judge, task, opponent, proxy): l = z*L_u + b_u + e (z = +1 if A is right, else -1),
with L_u, b_u from real/pair_check.units and e the real residual.

1. recovery    Simulated votes with a KNOWN lean, b_true = b_bar + gamma*(logit p_u - mean), and a
               known constant self-preference s (0 or 0.5) added to every vote:
                   truth     z*L_u + e + b_bar + s            (no differences between opponents)
                   leaned    truth + gamma*(logit p_u - mean)
                   removed   leaned, then copy-1 removal estimated from the leaned votes
                   zeroed    leaned, then the b = 0 removal
               Want: removed matches truth (slopes and levels, s kept); zeroed loses s.
2. wrong       The removal assumes the lean is the same for right and wrong answers. Simulate a
               lean that differs only in where it lands (same average per unit):
                   wrong_only   the lean differences fall only on votes where A is wrong (x2)
                   right_only   only on votes where A is right (x2)
               then apply the same copy-1 removal. Want to know: how far from truth?
3. mirror      Real votes: if the lean is one shift for right and wrong answers, z*(l - b_u) has
               the same distribution for right and wrong A. Quantiles of each, pooled over units.

Every version goes through the same pipeline as pair_check (grid slopes on p and accuracy).
Also the natural (unweighted) level of each measure per judge, to see whether s survives.

Run with: python -m real.lean_checks       writes real/lean_checks_results.json
"""

import json
import os

import numpy as np
import pandas as pd

from real import null_injection
from real.pair_check import FE, UNIT, pipeline, units, with_l
from real.tune_accuracy import weighted_stats

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lean_checks_results.json")
S = 0.5


def attach(d, u):
    cols = ["b", "L", "lp", "lp_bar", "b_bar"]
    d = d.merge(u[UNIT + cols], on=UNIT, how="inner").reset_index(drop=True)
    d["z"] = np.where(d.a_correct, 1.0, -1.0)
    d["e"] = d.l - d.z * d.L - d.b
    return d


def remove(d, l, to_zero=False):
    """Copy-1 removal estimated from votes l (additive assumption): l - (b_hat - b_hat_bar)."""
    t = d[UNIT + ["a_correct", "n_w"]].assign(l=l)
    g = t.groupby(UNIT + ["a_correct"]).l.mean().unstack("a_correct")
    b_hat = ((g[True] + g[False]) / 2).rename("b_hat").reset_index()
    n = t.groupby(UNIT).size().rename("n").reset_index()
    b_hat = b_hat.merge(n, on=UNIT)
    b_hat["wb"] = b_hat.b_hat * b_hat.n
    b_hat["b_hat_bar"] = b_hat.groupby(FE).wb.transform("sum") / b_hat.groupby(FE).n.transform("sum")
    m = d[UNIT].merge(b_hat, on=UNIT, how="left")
    shift = m.b_hat - (0 if to_zero else m.b_hat_bar)
    return l - np.nan_to_num(shift.to_numpy())


def levels(d, l):
    """Natural measures per judge (unit weights), median over judges."""
    x = with_l(d, l)
    x["q"] = null_injection.sigmoid(x.l)
    x["acc"] = np.where(x.a_correct, x.q, 1 - x.q)
    rows = [weighted_stats(g, np.ones(len(g))) for _, g in x.groupby("judge")]
    return pd.DataFrame(rows).median().round(4).to_dict()


def mirror(d):
    r = d.z * (d.l - d.b)
    qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    right, wrong = r[d.a_correct], r[~d.a_correct]
    sd = d.assign(r=r).groupby(UNIT + ["a_correct"]).r.std().unstack("a_correct").dropna()
    return {"quantiles": qs, "right": right.quantile(qs).round(3).tolist(), "wrong": wrong.quantile(qs).round(3).tolist(),
            "median_sd_ratio_right_over_wrong": float((sd[True] / sd[False]).median()),
            "share_units_sd_right_gt_wrong": float((sd[True] > sd[False]).mean())}


def main():
    d0 = null_injection.load()
    u, gamma = units(d0)
    d = attach(d0, u)
    d["n_w"] = 1
    dev = gamma * (d.lp - d.lp_bar)
    out = {"gamma": gamma, "rows": len(d), "mirror": mirror(d)}
    print("mirror:", out["mirror"], flush=True)

    def run(name, l):
        r = pipeline(with_l(d, l))
        out[name] = {"grid": r["grid"]["median"], "levels": levels(d, l)}
        g = r["grid"]["median"]
        print(f"{name:22s} " + ", ".join(f"{k} {v:+.2f}" for k, v in g.items()), flush=True)
        json.dump(out, open(OUT, "w"), indent=1, default=float)

    base = (d.z * d.L + d.e + d.b_bar).to_numpy()
    for s in (0.0, S):
        truth = base + s
        leaned = truth + dev.to_numpy()
        run(f"recovery_s{s}_truth", truth)
        run(f"recovery_s{s}_leaned", leaned)
        run(f"recovery_s{s}_removed", remove(d, leaned))
        run(f"recovery_s{s}_zeroed", remove(d, leaned, to_zero=True))
    wrong = ~d.a_correct.to_numpy()
    run("wrong_only_leaned", base + 2 * dev.to_numpy() * wrong)
    run("wrong_only_removed", remove(d, base + 2 * dev.to_numpy() * wrong))
    run("right_only_leaned", base + 2 * dev.to_numpy() * ~wrong)
    run("right_only_removed", remove(d, base + 2 * dev.to_numpy() * ~wrong))


if __name__ == "__main__":
    main()
