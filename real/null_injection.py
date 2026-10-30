"""
Two meta-evaluation tests on real judgments, with a known true bias.

Base data: Roytburg et al.'s proxy judgments (real/roytburg.py load_proxy_pairs): judge J
rates model K's answer (A) against reference R's answer (B). J wrote neither, and we keep
only pairs where J shares no family with K or R, so true self_lean ~ 0. p, actual
accuracy, difficulty and every other confound stay as they are in real data.

1. Null test: every measure should read ~0 at any p and actual accuracy.
2. Injection test: add a known log-odds shift s to J's preference for A,
       q(x) = sigmoid(l(x) + s),   l(x) = J's log-odds for A over B (mean of both orders)
   and check each measure rises with s, by the same amount whatever p and accuracy are.

Votes are soft: q(x) = P(vote A), so each measure returns its expected value over the
judge's own vote randomness (no sampling noise). Roytburg's measure takes, for each A, the
other cross-family proxies on the same (judge, reference, problem) as its matched proxies
-- they share A's correctness by construction; s is injected into A only.

Run with: python -m real.null_injection
"""

import os

import numpy as np
import pandas as pd

from measures import panickssery, equal_opportunity, hspp, roytburg
from real.roytburg import load_proxy_pairs

N_BOOT = 300
SEED = 0
S_GRID = [round(s, 2) for s in np.arange(-2, 2.01, 0.25)]
SPLITS = ("all", "task", "opponent", "proxy")
HERE = os.path.dirname(os.path.abspath(__file__))
NULL_PATH = os.path.join(HERE, "null_results.csv")
INJECTION_PATH = os.path.join(HERE, "injection_results.csv")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load() -> pd.DataFrame:
    """Cross-family proxy pairs, plus each row's matched-proxy log-odds for roytburg."""
    df = load_proxy_pairs()
    df = df[df.cross_family].reset_index(drop=True)
    group = ["task", "judge", "opponent", "problem_idx"]
    # mean of sigmoid(l) over the OTHER proxies on the same item (leave-one-out)
    q0 = sigmoid(df.l)
    total = q0.groupby([df[c] for c in group]).transform("sum")
    count = q0.groupby([df[c] for c in group]).transform("count")
    df["q_proxy"] = np.where(count > 1, (total - q0) / (count - 1).clip(lower=1), np.nan)
    return df


def stats(truth, q, q_proxy) -> dict:
    """Knob readings + the four measures for one sample, soft votes q = P(vote A)."""
    truth = np.asarray(truth, dtype=bool)
    q, q_proxy = np.asarray(q, dtype=float), np.asarray(q_proxy, dtype=float)
    t, v = list(truth), list(q)
    out = {"p": truth.mean(),
           "actual_accuracy": float(np.mean(np.where(truth, q, 1 - q))),
           "panickssery": panickssery.estimate_bias(v)}
    has_both = 0 < truth.sum() < len(truth)
    out["equal_opportunity"] = equal_opportunity.estimate_bias(t, v) if has_both else np.nan
    out["hspp"] = hspp.estimate_bias(t, v) if not truth.all() else np.nan
    proxy = [None if np.isnan(x) else x for x in q_proxy]
    matched = (~truth) & ~np.isnan(q_proxy)
    out["roytburg"] = roytburg.estimate_bias(t, v, proxy) if matched.any() else np.nan
    out["roytburg_n"] = int(matched.sum())
    return out


def groups(df):
    for split in SPLITS:
        keys = ["judge"] if split == "all" else ["judge", split]
        for key, g in df.groupby(keys):
            key = key if isinstance(key, tuple) else (key,)
            yield split, key[0], (key[1] if len(key) > 1 else "all"), g.reset_index(drop=True)


def null_test(df, rng) -> pd.DataFrame:
    rows = []
    for split, judge, sl, g in groups(df):
        truth, q, qp = g.a_correct.to_numpy(), sigmoid(g.l.to_numpy()), g.q_proxy.to_numpy()
        point = stats(truth, q, qp)
        idx_by_problem = list(g.groupby(["task", "problem_idx"]).indices.values())
        boots = []
        for _ in range(N_BOOT):
            pick = rng.integers(len(idx_by_problem), size=len(idx_by_problem))
            idx = np.concatenate([idx_by_problem[i] for i in pick])
            boots.append(stats(truth[idx], q[idx], qp[idx]))
        boots = pd.DataFrame(boots)
        row = {"split": split, "judge": judge, "slice": sl, "n_pairs": len(g),
               "n_problems": len(idx_by_problem), "roytburg_n": point.pop("roytburg_n")}
        for k, v in point.items():
            row[k] = v
            row[k + "_lo"], row[k + "_hi"] = np.nanpercentile(boots[k], [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows)


def injection_test(df) -> pd.DataFrame:
    rows = []
    for split, judge, sl, g in groups(df):
        truth, l, qp = g.a_correct.to_numpy(), g.l.to_numpy(), g.q_proxy.to_numpy()
        for s in S_GRID:
            st = stats(truth, sigmoid(l + s), qp)
            rows.append({"split": split, "judge": judge, "slice": sl, "s": s, "n_pairs": len(g), **st})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load()
    print(f"{len(df)} cross-family pairs, {df.judge.nunique()} judges")
    null = null_test(df, np.random.default_rng(SEED))
    null.to_csv(NULL_PATH, index=False)
    inj = injection_test(df)
    inj.to_csv(INJECTION_PATH, index=False)

    measures = ["panickssery", "equal_opportunity", "hspp", "roytburg"]
    a = null[null.split == "all"].set_index("judge")
    print("\n== Null test (true s ~ 0), per judge")
    print(a[["n_pairs", "p", "actual_accuracy"] + measures].sort_values("p").round(3).to_string())
    for split in ("opponent", "task", "proxy"):
        s = null[(null.split == split) & (null.n_pairs >= 30)]
        print(f"   within-judge corr with p across {split}: " + ", ".join(
            f"{m}={s.groupby('judge').apply(lambda g: g[['p', m]].corr().iloc[0, 1]).mean():.2f}" for m in measures))

    print("\n== Injection test: rise from s=0 to s=1, per judge")
    j = inj[inj.split == "all"].pivot_table(index="judge", columns="s", values=measures)
    rise = pd.DataFrame({m: j[(m, 1.0)] - j[(m, 0.0)] for m in measures})
    print(rise.round(3).to_string())
    print("   spread of the rise across judges (max - min): " +
          ", ".join(f"{m}={rise[m].max() - rise[m].min():.3f}" for m in measures))
    print(f"\nwrote {NULL_PATH}\nwrote {INJECTION_PATH}")
