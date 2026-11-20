"""
One boost per generator instead of one per pair (Bradley-Terry style), on Roytburg's null pairs.

Vote model: J's log-odds for proxy K over reference R on an item
    l = z * L + beta_J(K) - beta_J(R) + e          z = +1 if K's answer is correct, -1 if not
beta_J(g): how much judge J favours generator g's answers, right or wrong. Only differences are
identified; convention: mean beta_J over the generators J saw = 0.

Two steps:
  1. per pair (judge, task, opponent R, proxy K): b_u = (mean l | K correct + mean l | K wrong) / 2,
     free of the strength L (as real/pair_check.units), with its sampling variance
  2. per judge: weighted least squares b_u = beta(K) - beta(R) over all its pairs (tasks pooled)

Reported per judge:
  - additivity: chi2/df of the fit against b_u's own sampling noise (~1 = per-generator boosts
    explain the pair boosts; >> 1 = genuine pair effects), and R^2
  - corr(beta_J(g), theta_g), theta_g = logit of g's accuracy over Chen's problems (all tasks):
    high = the boost is favour for strength; low = favour for something else (style, family)
Then the counterfactual of real/pair_check.py with the pair lean replaced by beta(K) - beta(R):
if the grid slopes come back like the real ones, one boost per generator is enough.

Run with: python -m real.generator_boost     writes real/generator_boost_{judges,betas}.csv,
                                              real/generator_boost_results.json
"""

import json
import os

import numpy as np
import pandas as pd

from real import final_answers, null_injection
from real.pair_check import UNIT, pipeline, with_l

HERE = os.path.dirname(os.path.abspath(__file__))


def pair_boosts(d):
    g = d.groupby(UNIT + ["a_correct"]).l.agg(["mean", "var", "size"]).unstack("a_correct").dropna()
    var_t, var_f = g["var"][True], g["var"][False]
    n_t, n_f = g["size"][True], g["size"][False]
    u = pd.DataFrame({"b": (g["mean"][True] + g["mean"][False]) / 2,
                      "se2": (var_t / n_t + var_f / n_f) / 4, "n": n_t + n_f}).reset_index()
    return u[np.isfinite(u.se2) & (u.se2 > 0)]


def fit_judge(uj):
    gens = sorted(set(uj.proxy) | set(uj.opponent))
    ix = {g: i for i, g in enumerate(gens)}
    X = np.zeros((len(uj), len(gens)))
    X[np.arange(len(uj)), uj.proxy.map(ix)] = 1
    X[np.arange(len(uj)), uj.opponent.map(ix)] = -1
    w = 1 / uj.se2.to_numpy()
    # constraint sum(beta) = 0, as an extra heavily weighted row
    Xc = np.vstack([X * np.sqrt(w)[:, None], 1e3 * np.ones(len(gens))])
    yc = np.r_[uj.b.to_numpy() * np.sqrt(w), 0.0]
    beta = np.linalg.lstsq(Xc, yc, rcond=None)[0]
    fit = X @ beta
    res = uj.b.to_numpy() - fit
    df = len(uj) - (len(gens) - 1)
    chi2 = float(np.sum(w * res ** 2))
    ybar = np.average(uj.b, weights=w)
    r2 = 1 - np.sum(w * res ** 2) / np.sum(w * (uj.b - ybar) ** 2)
    return pd.Series(beta, index=gens), fit, {"pairs": len(uj), "generators": len(gens), "chi2_per_df": chi2 / df,
                                               "r2_weighted": float(r2)}


def main():
    d = null_injection.load()
    fa = final_answers.load()
    acc = fa.groupby("model").correct.mean().clip(0.01, 0.99)
    theta = np.log(acc / (1 - acc))
    u = pair_boosts(d)
    rows, betas, fitted = [], [], []
    for J, uj in u.groupby("judge"):
        beta, fit, stats = fit_judge(uj.reset_index(drop=True))
        th = theta.reindex(beta.index)
        stats.update({"judge": J, "corr_beta_theta": float(np.corrcoef(beta, th)[0, 1]),
                      "slope_beta_on_theta": float(np.polyfit(th, beta, 1)[0])})
        rows.append(stats)
        betas.append(pd.DataFrame({"judge": J, "generator": beta.index, "beta": beta.values, "theta": th.values}))
        fitted.append(uj.assign(b_gen=fit))
    js = pd.DataFrame(rows).set_index("judge")
    bt = pd.concat(betas)
    js.to_csv(os.path.join(HERE, "generator_boost_judges.csv"))
    bt.to_csv(os.path.join(HERE, "generator_boost_betas.csv"), index=False)
    print(js.round(3).to_string())
    print("\nmean beta by generator (over judges), with theta:")
    print(bt.groupby("generator").agg(beta=("beta", "mean"), sd=("beta", "std"), judges=("beta", "size"),
                                      theta=("theta", "first")).sort_values("theta").round(2).to_string())

    # counterfactual: pair boost replaced by beta(K) - beta(R), everything else kept
    f = pd.concat(fitted).set_index(UNIT)
    key = pd.MultiIndex.from_frame(d[UNIT])
    shift = np.nan_to_num(f.b.reindex(key).to_numpy() - f.b_gen.reindex(key).to_numpy())
    out = {"judges": js.reset_index().to_dict(orient="records")}
    for name, l in (("real", d.l.to_numpy()), ("generator_boost", d.l.to_numpy() - shift)):
        out[name] = pipeline(with_l(d, l))
        g = out[name]["grid"]["median"]
        print(f"{name:16s} grid median: " + ", ".join(f"{k} {v:.2f}" for k, v in g.items()), flush=True)
    json.dump(out, open(os.path.join(HERE, "generator_boost_results.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
