"""
Three CPU checks on the pairwise null test (Roytburg's pairs), framed by the decomposition
    measure = lean + gap in accuracy between D_A (A right) and D_B (A wrong).
With soft votes q = P(pick A): acc on D_A = mean q | A right, acc on D_B = mean (1 - q) | A wrong,
equal_opportunity = acc_DA - acc_DB, HSPP = 1 - acc_DB.

1. difficulty   Reweight so D_A and D_B have the same mix of cells, then recompute. Cells:
                    pairs        (task, opponent, target model)
                    + problem    also the problem's difficulty: other judges' accuracy on it (quintiles)
                    + solve      also the share of the other generators that solve it (quintiles)
                How much of the gap is because D_A and D_B hold different pairs / harder problems?
2. own_vs_null  Same problems and opponent: J's own answer (Roytburg's J_vs_R runs) vs other models
                outside J's family whose generation quality on the task is within DELTA_Q of J's
                (their pairs exist only where they share J's correctness). Per judge x task:
                own minus null in log-odds, separately on D_A and D_B. Equal shifts on both = a
                constant lean; a shift only on D_B = J misses its own errors specifically.
3. length       Is the lean toward the model that usually wins still there among pairs of similar
                length? Units' leans (pair_check.units) on rows with |log length ratio| < SIMILAR,
                and gamma with and without controlling for the units' length ratio.

Run with: python -m real.gap_checks     writes real/gap_checks_results.json
"""

import glob
import json
import math
import os

import numpy as np
import pandas as pd

from infer.yang_data import TASKS, load_answers
from real import null_injection
from real import slope_gap as sg
from real.lean_rest import lengths
from real.pair_check import FE, UNIT, units
from real.roytburg import CACHE_GLOB, EXCLUDED_JUDGES, REPO_DIR, _is_verdict, _p_candidate, _read, family

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gap_checks_results.json")
N_Q = 5
DELTA_Q = 0.05
SIMILAR = 0.2
N_BOOT = 300


def stats(d, w=None):
    w = np.ones(len(d)) if w is None else w
    r, q = d.a_correct.to_numpy(), d.q.to_numpy()
    acc_a = np.sum(w[r] * q[r]) / np.sum(w[r])
    acc_b = np.sum(w[~r] * (1 - q[~r])) / np.sum(w[~r])
    return {"acc_DA": acc_a, "acc_DB": acc_b, "equal_opportunity": acc_a - acc_b, "hspp": 1 - acc_b}


def difficulty():
    df = sg.prepare()                                  # q, acc, problem_acc, solve_rate, ...
    for c in ("problem_acc", "solve_rate"):
        df[c + "_bin"] = df.groupby(["judge", "task"])[c].transform(
            lambda s: pd.qcut(s.rank(method="first"), N_Q, labels=False))
    cells = {"none": [], "pairs": ["task", "opponent", "proxy"],
             "+ problem": ["task", "opponent", "proxy", "problem_acc_bin"],
             "+ solve": ["task", "opponent", "proxy", "solve_rate_bin"]}
    out = {}
    for name, keys in cells.items():
        per = []
        for j, g in df.groupby("judge"):
            g = g.reset_index(drop=True)
            if not keys:
                per.append(stats(g))
                continue
            c = g.groupby(keys).a_correct.agg(["sum", "size"])
            c = c[(c["sum"] > 0) & (c["sum"] < c["size"])]
            g = g.join(c, on=keys, how="inner").reset_index(drop=True)
            share = g["size"] / g.drop_duplicates(keys)["size"].sum()
            n_side = np.where(g.a_correct, g["sum"], g["size"] - g["sum"])
            per.append(stats(g, (share / n_side).to_numpy()))
        out[name] = pd.DataFrame(per).median().round(4).to_dict()
        print(f"difficulty matching [{name:9s}] " + ", ".join(f"{k} {v:.3f}" for k, v in out[name].items()), flush=True)
    return out


def own_pairs(eps=1e-6):
    rows = []
    for task in TASKS:
        for d in sorted(glob.glob(os.path.join(REPO_DIR, CACHE_GLOB.format(task=task)))):
            judge, opponent = d.split(os.sep)[-2:]
            if judge in EXCLUDED_JUDGES or family(judge) == family(opponent):
                continue
            by = {}
            for g in (1, 2):
                for r in _read(os.path.join(d, f"J_vs_R_game{g}.jsonl")):
                    p_a = max(_p_candidate(r), eps)
                    p_b = max(r["prob_response2"] if r["response1_key"] == "judge" else r["prob_response1"], eps)
                    row = by.setdefault(r["example_id"], {"task": task, "judge": judge, "opponent": opponent,
                                                          "problem_idx": r["example_id"], "a_correct": r["category"] == "lsp", "ok": True})
                    row[f"l_{g}"] = math.log(p_a / p_b)
                    row["ok"] &= _is_verdict(r)
            rows.extend(by.values())
    o = pd.DataFrame(rows).dropna(subset=["l_1", "l_2"])
    o = o[o.ok]
    o["l"] = (o.l_1 + o.l_2) / 2
    return o


def own_vs_null():
    quality = {}
    for t in TASKS:
        a = pd.Series({k: v[2] for k, v in load_answers(t).items()})
        for m, s in a.groupby(level=0).mean().items():
            quality[(t, m)] = s
    own = own_pairs()
    null = null_injection.load()
    null = null[[abs(quality[(t, k)] - quality[(t, j)]) <= DELTA_Q for t, k, j in zip(null.task, null.proxy, null.judge)]]
    key = ["task", "judge", "opponent", "problem_idx"]
    nm = null.groupby(key).l.mean().rename("l_null")
    m = own.join(nm, on=key, how="inner")
    m["diff"] = m.l - m.l_null

    def summarise(x):
        g = x.groupby(["judge", "task", "a_correct"])["diff"].mean().unstack("a_correct")
        return {"D_A (own right)": float(g[True].median()), "D_B (own wrong)": float(g[False].median())}

    est = summarise(m)
    probs = m[["task", "problem_idx"]].drop_duplicates().to_numpy()
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(N_BOOT):
        pick = pd.DataFrame(probs[rng.integers(len(probs), size=len(probs))], columns=["task", "problem_idx"])
        boots.append(summarise(m.merge(pick, on=["task", "problem_idx"])))
    b = pd.DataFrame(boots)
    out = {k: {"own_minus_null": v, "lo": float(b[k].quantile(.025)), "hi": float(b[k].quantile(.975))} for k, v in est.items()}
    out["rows"] = len(m)
    out["judge_tasks"] = int(m.groupby(["judge", "task"]).ngroups)
    print("own minus quality-matched null (log-odds, median over judge x task):",
          {k: f"{v['own_minus_null']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}]" for k, v in out.items() if isinstance(v, dict)}, flush=True)
    return out


def length():
    d0 = null_injection.load()
    d, _ = lengths(d0)
    d["lr"] = d.len_a - d.len_b
    out = {}
    for name, sub in (("all", d), ("similar length", d[d.lr.abs() < SIMILAR])):
        u, gamma = units(sub.drop(columns=["len_a", "len_b", "own_correct", "lr"]))
        lr = sub.groupby(UNIT).lr.mean().rename("lr").reset_index()
        u = u.merge(lr, on=UNIT)
        dm = lambda c: u[c] - u.groupby(FE)[c].transform("mean")
        X = np.c_[dm("lp"), dm("lr")]
        beta = np.linalg.lstsq(X * np.sqrt(u.n.to_numpy())[:, None], dm("b") * np.sqrt(u.n), rcond=None)[0]
        out[name] = {"rows": len(sub), "units": len(u), "gamma": gamma, "gamma_holding_length": float(beta[0]),
                     "lean_per_log_length_ratio": float(beta[1]), "mean_abs_log_length_ratio": float(sub.lr.abs().mean())}
        print(f"length [{name}]: {out[name]}", flush=True)
    return out


def main():
    out = {"length": length(), "own_vs_null": own_vs_null()}
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    out["difficulty"] = difficulty()
    json.dump(out, open(OUT, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
