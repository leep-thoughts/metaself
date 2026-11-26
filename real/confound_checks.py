"""
Follow-ups on the confounds page (pairwise null test, Roytburg's null pairs, true s = 0).

quick (seconds to minutes):
  1. split_half     How much of the lean b_u (unit = judge, task, opponent, proxy) is estimation
                    noise? Problems split in random halves; b_u per half, minus the judge x task
                    mean; reliability = Spearman-Brown of the half-half correlation. Same for the
                    part the win-rate rule leaves (b_u - b_model).
  2. mechanical     Slopes with evaluation accuracy that the measures' definitions alone imply when
                    there is no lean at all (TPR = TNR = accuracy): panickssery = p*a + (1-p)(1-a) - .5,
                    equal_opportunity = 0, HSPP = 1 - a, roytburg = 0. Evaluated on the reachable grid
                    points (real/plot_grid_points.csv, version real) with the same per-judge OLS on (p, a).
  3. roytburg_acc   roytburg one target model K at a time (as real/pair_control), in thirds of each
                    judge's items by a covariate that moves evaluation accuracy; OLS within
                    (judge, K) of the value on [evaluation accuracy, K's win rate in the third].
                    Covariates: confidence (the grid's) and problem_acc (other judges' accuracy on
                    the problem).
  4. oracle_L       L measured separately with the oracle: the judge's log-odds for the right answer
                    over a wrong one by two OTHER models on the same problem, both orders
                    (infer/data/beta_verdicts_<judge>.jsonl). Then the lean two ways per unit:
                    b_right = mean l | A right - L, b_wrong = mean l | A wrong + L (equal if the lean
                    is the same for right and wrong answers), and the win-rate slope (gamma) of each.
grid (hours): the b = 0 version of the votes (l - b_u) through the pair_check pipeline.

Run with: python -m real.confound_checks quick | grid     writes real/confound_checks_<part>.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from real import null_injection
from real.pair_check import FE, UNIT, pipeline, units, with_l
from real.tune_accuracy import _leave_judge_out

HERE = os.path.dirname(os.path.abspath(__file__))
N_SPLITS = 20
N_BINS = 3


def split_half(d, u, gamma):
    rng = np.random.default_rng(0)
    probs = d[["task", "problem_idx"]].drop_duplicates()
    out = []
    for s in range(N_SPLITS):
        half = probs.assign(h=rng.integers(2, size=len(probs)))
        dh = d.merge(half, on=["task", "problem_idx"])
        g = dh.groupby(UNIT + ["h", "a_correct"]).l.mean().unstack("a_correct").dropna()
        b = ((g[True] + g[False]) / 2).unstack("h").dropna()
        b = b.join(u.set_index(UNIT)[["lp", "lp_bar"]], how="inner")
        for h in (0, 1):
            b[f"d{h}"] = b[h] - b.groupby(level=FE)[h].transform("mean")       # within judge x task
            b[f"r{h}"] = b[f"d{h}"] - gamma * (b.lp - b.lp_bar)                 # what the rule leaves
        sb = lambda r: 2 * r / (1 + r)
        out.append({"lean": sb(np.corrcoef(b.d0, b.d1)[0, 1]), "left_by_rule": sb(np.corrcoef(b.r0, b.r1)[0, 1]),
                    "units": len(b)})
    o = pd.DataFrame(out).median().to_dict()
    print(f"split-half reliability: lean {o['lean']:.2f}, part the rule leaves {o['left_by_rule']:.2f} "
          f"({int(o['units'])} units, median of {N_SPLITS} splits)")
    return o


def mechanical():
    g = pd.read_csv(os.path.join(HERE, "plot_grid_points.csv"))
    g = g[g.version == "real"] if "version" in g else g
    p, a = g.p, g.actual_accuracy
    g = g.assign(m_panickssery=p * a + (1 - p) * (1 - a) - 0.5, m_equal_opportunity=0.0, m_hspp=1 - a, m_roytburg=0.0)
    rows = []
    for j, gj in g.groupby("judge"):
        if gj.target_p.nunique() < 3 or len(gj) < 6:
            continue
        X = np.c_[np.ones(len(gj)), gj.p, gj.actual_accuracy]
        r = {"judge": j}
        for m in ("panickssery", "equal_opportunity", "hspp", "roytburg"):
            for pre, col in (("mech_", "m_" + m), ("real_", m)):
                ok = gj[col].notna().to_numpy()
                b = np.linalg.lstsq(X[ok], gj[col].to_numpy()[ok], rcond=None)[0]
                r[pre + m + "_a"] = b[2]
        rows.append(r)
    o = pd.DataFrame(rows).drop(columns="judge").median().round(3).to_dict()
    print("accuracy slopes, definition alone vs real:", o)
    return o


def roytburg_acc(d):
    d = d.copy()
    d["q"] = null_injection.sigmoid(d.l)
    d["acc"] = np.where(d.a_correct, d.q, 1 - d.q)
    d["abs_l"] = d.l.abs()
    d["confidence"] = _leave_judge_out(d, "abs_l", ["task", "opponent", "proxy", "problem_idx"])
    d["problem_acc"] = _leave_judge_out(d, "acc", ["task", "problem_idx"])
    out = {}
    rows_all = []
    for cov in ("confidence", "problem_acc"):
        x = d.dropna(subset=[cov, "q_proxy"]).copy()
        # thirds within judge x task, so the task mix is the same in every third
        x["bin"] = x.groupby(["judge", "task"])[cov].transform(
            lambda s: pd.qcut(s.rank(method="first"), N_BINS, labels=False))
        acc = x.groupby(["judge", "bin"]).acc.mean().rename("eval_acc")
        w = x[~x.a_correct].assign(diff=lambda t: t.q - t.q_proxy)
        jk = w.groupby(["judge", "proxy", "bin"]).agg(value=("diff", "mean"), n=("diff", "size"))
        p = x.groupby(["judge", "proxy", "bin"]).a_correct.mean().rename("p")
        jk = jk.join(p).join(acc, on=["judge", "bin"]).reset_index()
        jk = jk[jk.n >= 10]          # ~80 wrong-A rows per (judge, target): ~26 per third
        jk = jk[jk.groupby(["judge", "proxy"]).bin.transform("size") == N_BINS]
        g = jk.groupby(["judge", "proxy"])
        lp = np.log(jk.p.clip(0.01, 0.99) / (1 - jk.p.clip(0.01, 0.99)))
        X = np.c_[jk.eval_acc - g.eval_acc.transform("mean"), lp - lp.groupby([jk.judge, jk.proxy]).transform("mean")]
        y = jk.value - g.value.transform("mean")
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        keys = list(g.groups)
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(500):                                   # resample (judge, target) pairings
            pick = [keys[i] for i in rng.integers(len(keys), size=len(keys))]
            idx = np.concatenate([g.indices[k] for k in pick])
            boot.append(np.linalg.lstsq(X[idx], y.to_numpy()[idx], rcond=None)[0][0])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        per = [np.polyfit(t.eval_acc, t.value, 1)[0] for _, t in g if t.eval_acc.nunique() > 1]
        rng_acc = g.eval_acc.agg(lambda z: z.max() - z.min()).mean()
        out[cov] = {"slope_acc_pooled": b[0], "ci": [lo, hi], "slope_acc_median": float(np.median(per)),
                    "slope_winrate_logit": b[1], "acc_range": rng_acc, "n_judge_target": int(g.ngroups)}
        print(f"roytburg per target, accuracy moved by {cov}: slope {b[0]:+.3f} [{lo:+.2f}, {hi:+.2f}] (median per pairing "
              f"{np.median(per):+.3f}), accuracy range {rng_acc:.3f}, {g.ngroups} pairings")
        rows_all.append(jk.assign(covariate=cov))
    pd.concat(rows_all).to_csv(os.path.join(HERE, "confound_checks_roytburg_acc.csv"), index=False)
    return out


def oracle_L(d, u):
    from real.yang_null import DATA
    ext = []
    for f in sorted(os.listdir(DATA)):
        if not f.startswith("beta_verdicts_"):
            continue
        b = pd.DataFrame([json.loads(l) for l in open(os.path.join(DATA, f))])
        b = b[b.verdict_mass >= 0.5]
        pc, pw = np.where(b.game == 1, b.prob_A, b.prob_B), np.where(b.game == 1, b.prob_B, b.prob_A)
        b = b.assign(lc=np.log(pc / pw))
        b = b[np.isfinite(b.lc)]
        ext.append(b.groupby(["task", "problem_idx"]).lc.mean().rename("L_ext").reset_index()
                   .assign(judge=f[len("beta_verdicts_"):-len(".jsonl")]))
    m = d.merge(pd.concat(ext), on=["judge", "task", "problem_idx"])
    g = m.groupby(UNIT + ["a_correct"]).agg(l=("l", "mean"), Le=("L_ext", "mean"), n=("l", "size")).unstack("a_correct").dropna()
    g = g[(g["n"][True] >= 5) & (g["n"][False] >= 5)]
    x = pd.DataFrame({"b_right": g["l"][True] - g["Le"][True], "b_wrong": g["l"][False] + g["Le"][False],
                      "L_votes": (g["l"][True] - g["l"][False]) / 2, "L_oracle": (g["Le"][True] + g["Le"][False]) / 2,
                      "n": g["n"][True] + g["n"][False]}).reset_index().merge(u[UNIT + ["lp"]], on=UNIT)
    dm = lambda c: x[c] - x.groupby(FE)[c].transform("mean")
    xl = dm("lp")
    gam = lambda c: float(np.sum(x.n * xl * dm(c)) / np.sum(x.n * xl * xl))
    o = {"units": len(x), "rows_matched": len(m), "L_votes_mean": x.L_votes.mean(), "L_oracle_mean": x.L_oracle.mean(),
         "corr_L": np.corrcoef(x.L_votes, x.L_oracle)[0, 1], "b_right_mean": x.b_right.mean(), "b_wrong_mean": x.b_wrong.mean(),
         "corr_b_right_b_wrong_within": np.corrcoef(dm("b_right"), dm("b_wrong"))[0, 1],
         "gamma_right": gam("b_right"), "gamma_wrong": gam("b_wrong")}
    print("oracle L:", {k: round(float(v), 3) for k, v in o.items()})
    return o


def main(part):
    d = null_injection.load()
    u, gamma = units(d)
    path = os.path.join(HERE, f"confound_checks_{part}.json")
    if part == "quick":
        out = {"split_half": split_half(d, u, gamma), "mechanical": mechanical(), "roytburg_acc": roytburg_acc(d),
               "oracle_L": oracle_L(d, u)}
    else:
        key = pd.MultiIndex.from_frame(d[UNIT])
        b = np.nan_to_num(u.set_index(UNIT).b.reindex(key).to_numpy())
        out = {"b_zero": pipeline(with_l(d, d.l.to_numpy() - b))}
        print("b_zero grid median:", out["b_zero"]["grid"]["median"], flush=True)
    json.dump(out, open(path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "quick")
