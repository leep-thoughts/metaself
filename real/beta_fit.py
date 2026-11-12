"""
Fit beta (synth/difficulty.py) from third-party judgments: infer/beta_data.py pairs, judged on
GPU by infer/beta_judges_submit.sh (Chen's direct prompt, both orders).

Per pair, l = mean over the two orders of log P(correct answer) / P(wrong answer), tie mass
dropped. Averaging the orders cancels position bias, and there is no self-lean (neither answer
is J's), so in the simulator's terms l estimates L(x) = L0 + beta * c(x).

c is latent; J's own correctness y ~ Bernoulli(sigmoid(c)), c ~ N(mu, 1), with mu set per
(judge, task) from J's solve rate over ALL of Chen's problems. Then
    E[l | y] = L0_task + beta * E[c | y]      -> beta = within-task slope of mean l on E[c | y]
Also reported: the same with a difficulty control (share of the 18 models solving x), to
separate "J solved it" from "it is easy"; bootstrap CIs over problems.

Keeps pairs with verdict_mass >= MIN_MASS in both orders (the first token was A/T/B).

Run with: python -m real.beta_fit      -> real/beta_fit_results.csv
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from infer.yang_data import DATA, TASKS, load_answers
from synth.difficulty import _SHAPES, expected_measures, sigmoid, solve_mu

MIN_MASS = 0.5
N_BOOT = 500
CLIP = 1e-4
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "beta_fit_results.csv")


def solve_rates():
    """{(judge, task): J's solve rate over all of Chen's problems}."""
    out = {}
    for task in TASKS:
        ans = load_answers(task)
        for m in {m for m, _ in ans}:
            ys = [v[2] for (mm, _), v in ans.items() if mm == m]
            out[(m, task)] = float(np.mean(ys))
    return out


def e_c_given_y(p):
    """E[c | y=1], E[c | y=0] for c ~ N(mu, 1), mu solved from p = E[sigmoid(c)]."""
    nodes, w = _SHAPES["normal"]
    c = solve_mu(p) + nodes
    s = sigmoid(c)
    return (w * s) @ c / (w @ s), (w * (1 - s)) @ c / (w @ (1 - s))


def load(path):
    games = {}
    for r in map(json.loads, open(path)):
        if r.get("skipped") or r.get("verdict_mass") is None or r["verdict_mass"] < MIN_MASS:
            continue
        pa, pb = r["prob_A"], r["prob_B"]
        if pa + pb <= 0:
            continue
        # game 1: correct answer in A; game 2: swapped, correct answer in B
        pc = (pa if r["game"] == 1 else pb) / (pa + pb)
        pc = min(max(pc, CLIP), 1 - CLIP)
        games.setdefault(r["id"], {**{k: r[k] for k in ("task", "problem_idx", "judge", "j_correct", "solve_rate")}})
        games[r["id"]][f"l{r['game']}"] = np.log(pc / (1 - pc))
    df = pd.DataFrame(games.values())
    df = df.dropna(subset=["l1", "l2"])
    df["l"] = (df.l1 + df.l2) / 2
    return df


def fit(d):
    """beta = within-task slope of mean l on E[c | y]; and the y-gap with a difficulty control."""
    x = d.ec - d.groupby("task").ec.transform("mean")
    y = d.l - d.groupby("task").l.transform("mean")
    beta = (x @ y) / (x @ x)
    # with difficulty: l ~ task + ec + logit(solve_rate)
    X = pd.get_dummies(d.task, dtype=float)
    X["ec"] = d.ec
    sr = d.solve_rate.clip(0.02, 0.98)
    X["diff"] = np.log(sr / (1 - sr))
    coef = np.linalg.lstsq(X.values, d.l.values, rcond=None)[0]
    return {"beta": beta, "beta_ctrl_difficulty": coef[-2], "difficulty_coef": coef[-1]}


def main():
    rates = solve_rates()
    rng = np.random.default_rng(0)
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA, "beta_verdicts_*.jsonl"))):
        d = load(path)
        if d.empty:
            continue
        judge = d.judge.iloc[0]
        ec = {t: e_c_given_y(rates[(judge, t)]) for t in TASKS if (judge, t) in rates}
        d["ec"] = [ec[t][0] if y else ec[t][1] for t, y in zip(d.task, d.j_correct)]
        est = fit(d)
        probs = d.problem_idx.astype(str) + d.task
        keys = probs.unique()
        boots = []
        for _ in range(N_BOOT):
            pick = pd.Series(rng.choice(keys, len(keys))).value_counts()
            b = d.assign(k=probs).merge(pick.rename("w"), left_on="k", right_index=True)
            boots.append(fit(b.loc[b.index.repeat(b.w)]))
        bd = pd.DataFrame(boots)
        rows.append({"judge": judge, "n": len(d), "n_j_wrong": int((~d.j_correct).sum()),
                     "acc_j_right": (sigmoid(d.l[d.j_correct]) ).mean(),
                     "acc_j_wrong": (sigmoid(d.l[~d.j_correct])).mean(),
                     "l_gap": d.l[d.j_correct].mean() - d.l[~d.j_correct].mean(), **est,
                     **{k + "_lo": bd[k].quantile(.025) for k in est},
                     **{k + "_hi": bd[k].quantile(.975) for k in est}})
    if not rows:
        raise SystemExit("no infer/data/beta_verdicts_*.jsonl yet: run bash infer/beta_judges_submit.sh")
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS, index=False)
    print(res[["judge", "n", "n_j_wrong", "acc_j_right", "acc_j_wrong", "beta", "beta_lo", "beta_hi",
               "beta_ctrl_difficulty"]].round(3).to_string(index=False))

    # check: simulator slopes at the median fitted beta (compare with the real slopes)
    from difficulty_sweep import REF, slope
    b = float(res.beta.median())
    print(f"\nmedian beta {b:.2f}; simulator at p={REF['p']}, a={REF['a']}, s=0:")
    for knob in ("p", "a"):
        print(f"  slope with {knob}:", {m: round(v, 2) for m, v in slope(knob, b, REF["p"], REF["a"]).items()})


if __name__ == "__main__":
    main()
