"""
Does the generator-pair lean explain the real slopes? Counterfactuals on Roytburg's null pairs.

Unit u = (judge, task, opponent, proxy). Per unit, from J's log-odds l for proxy K over R:
    mean l | K correct = L_u + b_u        mean l | K wrong = -L_u + b_u
so b_u = lean toward K (the A slot of the measures) whatever the correctness, L_u = strength.
gamma = slope of b_u on logit(p_u) within judge x task (p_u = K's win rate in the unit).

Versions of the votes, each run through the same pipeline as real/slope_gap.py
(what_moves_p regressions and the LP grid slopes):
    real        as released
    no_pair     every unit's lean set to its judge x task mean: l - (b_u - mean b)
    pair_model  lean replaced by the simulator's form: mean b + gamma * (logit p_u - mean)
If no_pair kills the slopes and pair_model restores them, b_pair = gamma * logit(p) is the cause
and the simulator form is enough.

Run with: python -m real.pair_check          writes real/pair_check_results.json
"""

import json
import os

import numpy as np
import pandas as pd

from real import null_injection
from real import slope_gap as sg

UNIT = ["judge", "task", "opponent", "proxy"]
FE = ["judge", "task"]
N_REPS = 4
sg.WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", sg.WORKERS))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pair_check_results.json")


def units(d):
    g = d.groupby(UNIT + ["a_correct"]).l.agg(["mean", "size"]).unstack("a_correct").dropna()
    u = pd.DataFrame({"b": (g["mean"][True] + g["mean"][False]) / 2, "L": (g["mean"][True] - g["mean"][False]) / 2,
                      "n": g["size"][True] + g["size"][False]})
    u["p"] = d.groupby(UNIT).a_correct.mean().loc[u.index]
    u = u.reset_index()
    u = u[(u.p > 0.02) & (u.p < 0.98)].copy()
    u["lp"] = np.log(u.p / (1 - u.p))
    wmean = lambda c: u.groupby(FE)[c].transform(lambda x: np.average(x, weights=u.loc[x.index, "n"]))
    u["b_bar"], u["lp_bar"] = wmean("b"), wmean("lp")
    x, y = u.lp - u.lp_bar, u.b - u.b_bar
    gamma = float(np.sum(u.n * x * y) / np.sum(u.n * x * x))
    u["b_model"] = u.b_bar + gamma * x
    return u, gamma


def with_l(d, l):
    """Same rows, new log-odds; q_proxy recomputed as in real/null_injection.load."""
    d = d.copy()
    d["l"] = l
    group = ["task", "judge", "opponent", "problem_idx"]
    q0 = null_injection.sigmoid(d.l)
    total = q0.groupby([d[c] for c in group]).transform("sum")
    count = q0.groupby([d[c] for c in group]).transform("count")
    d["q_proxy"] = np.where(count > 1, (total - q0) / (count - 1).clip(lower=1), np.nan)
    return d


def pipeline(d):
    sg.load = lambda: d                      # slope_gap.prepare() reads through this name
    df = sg.prepare()
    wm = sg.what_moves_p(df)
    grid = sg.grid_summary(sg.run_grid(df, n_reps=N_REPS))
    return {"what_moves_p": {k: wm[k] for k in ("opponent", "proxy", "difficulty")}, "grid": grid}


def main():
    d = null_injection.load()
    u, gamma = units(d)
    print(f"{len(u)} units, gamma (lean per unit logit p, within judge x task) = {gamma:.3f}")
    key = pd.MultiIndex.from_frame(d[UNIT])
    ui = u.set_index(UNIT)
    b = ui.b.reindex(key).to_numpy()
    shift_none = np.nan_to_num(b - ui.b_bar.reindex(key).to_numpy())          # units without both sides: 0
    shift_model = np.nan_to_num(b - ui.b_model.reindex(key).to_numpy())
    versions = {"real": d.l.to_numpy(), "no_pair": d.l.to_numpy() - shift_none,
                "pair_model": d.l.to_numpy() - shift_model}
    out = {"gamma": gamma, "n_units": int(len(u))}
    for name, l in versions.items():
        out[name] = pipeline(with_l(d, l))
        g = out[name]["grid"]["median"]
        print(f"{name:10s} grid median: " + ", ".join(f"{k} {v:.2f}" for k, v in g.items()), flush=True)
        json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
