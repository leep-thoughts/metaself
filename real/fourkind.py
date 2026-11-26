"""
Analysis of the four-kinds pairs (infer/fourkind_data.py): gap in accuracy vs lean.

Per row, l = J's log-odds for A over B, mean of both orders (tie mass dropped). Per judge, task and
condition (null: A = another model; own: A = J's own answer), the mean l of each kind:
    RW (A right, B wrong)   WR (A wrong, B right)   RR (both right)   WW (both wrong)
The lean is measured where correctness can't play a role: b_RR, b_WW. Then, in log-odds,
    accuracy on D_A  (A right)  L_A = l_RW - b         accuracy on D_B  (A wrong)  L_B = b - l_WR
    gap = L_A - L_B = l_RW + l_WR - 2b
with b = b_RR, b_WW or their mean (they bracket the answer if they differ). The usual estimate
(real/pair_check) takes b = (l_RW + l_WR) / 2, i.e. assumes gap = 0.
Also: does the lean toward the model that usually wins show up in RR and WW pairs? Slope of l on
logit(K's solve rate) - logit(R's solve rate) within judge x task, per kind.
CIs: bootstrap over problems.

Run with: python -m real.fourkind      writes real/fourkind_results.json
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from infer.yang_data import DATA, TASKS, load_answers

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fourkind_results.json")
MIN_MASS = 0.5
N_BOOT = 500


def load():
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA, "fourkind_verdicts_*.jsonl"))):
        for r in map(json.loads, open(path)):
            if r.get("verdict_mass") is None or r["verdict_mass"] < MIN_MASS:
                continue
            pa, pb = (r["prob_A"], r["prob_B"]) if r["game"] == 1 else (r["prob_B"], r["prob_A"])
            if pa <= 0 or pb <= 0:
                continue
            rows.append({k: r[k] for k in ("id", "judge", "task", "problem_idx", "kind", "target", "opponent", "own")}
                        | {"game": r["game"], "lo": np.log(pa / pb)})
    d = pd.DataFrame(rows)
    d = d.groupby(["id", "judge", "task", "problem_idx", "kind", "target", "opponent", "own"]).lo.agg(["mean", "size"])
    d = d[d["size"] == 2].rename(columns={"mean": "l"}).reset_index()
    q = {}
    for t in TASKS:
        a = pd.Series({k: v[2] for k, v in load_answers(t).items()})
        for m, s in a.groupby(level=0).mean().items():
            q[(t, m)] = float(np.clip(s, 0.02, 0.98))
    lg = lambda p: np.log(p / (1 - p))
    d["strength"] = [lg(q[(t, k)]) - lg(q[(t, r)]) for t, k, r in zip(d.task, d.target, d.opponent)]
    return d


def summary(d):
    m = d.groupby(["judge", "task", "own", "kind"]).l.mean().unstack("kind")
    m = m.dropna(subset=["RW", "WR", "RR", "WW"])
    out = pd.DataFrame({"b_RR": m.RR, "b_WW": m.WW, "b_usual": (m.RW + m.WR) / 2})
    for name, b in (("RR", m.RR), ("WW", m.WW), ("mid", (m.RR + m.WW) / 2)):
        out[f"L_A_{name}"], out[f"L_B_{name}"] = m.RW - b, b - m.WR
        out[f"gap_{name}"] = m.RW + m.WR - 2 * b
    return out.reset_index()


def main():
    d = load()
    print(f"{len(d)} pairs with both orders; judges {d.judge.nunique()}")
    s = summary(d)
    probs = d[["task", "problem_idx"]].drop_duplicates().to_numpy()
    rng = np.random.default_rng(0)
    cols = [c for c in s.columns if c.startswith(("b_", "L_", "gap_"))]
    boots = []
    for _ in range(N_BOOT):
        pick = pd.DataFrame(probs[rng.integers(len(probs), size=len(probs))], columns=["task", "problem_idx"])
        pick["problem_idx"] = pick.problem_idx.astype(int)
        bd = d.merge(pick, on=["task", "problem_idx"])
        boots.append(summary(bd).groupby("own")[cols].median())
    b = pd.concat(boots)
    out = {}
    for own, g in s.groupby("own"):
        key = "own" if own else "null"
        bo = b.loc[own]
        out[key] = {c: {"median": float(g[c].median()), "lo": float(bo[c].quantile(.025)), "hi": float(bo[c].quantile(.975))}
                    for c in cols}
        out[key]["judge_tasks"] = len(g)
        print(f"\n{key} ({len(g)} judge x task), median [95% CI]")
        for c in cols:
            v = out[key][c]
            print(f"   {c:10s} {v['median']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}]")
    # lean toward the model that usually wins, per kind (null pairs), within judge x task
    nl = d[~d.own].copy()
    for c in ("l", "strength"):
        nl[c + "_dm"] = nl[c] - nl.groupby(["judge", "task", "kind"])[c].transform("mean")
    out["slope_on_strength"] = {k: float(np.sum(g.l_dm * g.strength_dm) / np.sum(g.strength_dm ** 2))
                                for k, g in nl.groupby("kind")}
    print("\nslope of l on strength (logit K - logit R), null pairs:", {k: round(v, 3) for k, v in out["slope_on_strength"].items()})
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
