"""
Why does HSPP-rubric rise with generation quality in the null test? (Rubric-based setting.)

Null test as real/rubric_placebo.py: judge J, target G outside J's family, comparison models C =
outside J's and G's families. HSPP-rubric_J(G) = mean verdict on G's truly failed criteria minus
the same on C's.

Per judge J, on its verdicts for truly failed criteria of every generator outside J's family:
    v = alpha_item + lambda_G + delta * length + e
alpha_item   how easily this criterion is passed by J when it is failed (criterion difficulty)
lambda_G     J going easy on generator G's failures, whatever the criterion
length       log length of the answer minus the mean over generators on the same prompt
(two-way fixed effects, alternating means). Then HSPP-rubric_J(G) splits exactly into
    criteria part    mean alpha over G's failed criteria - over C's   (which criteria each fails)
    generator part   lambda_G - mean lambda over C's failed criteria
    length part      delta * (mean length dev on G's failures - on C's)
    rest
and the slope of each part on G's quality gap (pass rate minus C's) adds up to HSPP-rubric's slope.
Also: is lambda_G shared across judges (mean over judges outside J's and G's families)?
Is the per-generator going-easy explained by answer length?

Run with: [RUBRIC_DATASET=healthbench] python -m real.rubric_decomp
          writes real/rubric_decomp_results[_healthbench].{csv,json}
"""

import json
import os

import numpy as np
import pandas as pd

from infer.ifeval_rubric import DATASET, family
from real.rubric_real import units

HERE = os.path.dirname(os.path.abspath(__file__))
ANS = os.path.join(HERE, "..", "infer", "data", "ifeval" if DATASET == "ifeval" else "healthbench")
OUT = os.path.join(HERE, "rubric_decomp_results" + ("" if DATASET == "ifeval" else f"_{DATASET}"))
N_ITER = 30


def answer_lengths():
    rows = []
    for f in sorted(os.listdir(ANS)):
        if f.startswith("answers_") and f.endswith(".jsonl"):
            for line in open(os.path.join(ANS, f)):
                r = json.loads(line)
                rows.append({"key": str(r["key"]), "generator": r["model"], "len": np.log1p(len(r.get("response") or ""))})
    a = pd.DataFrame(rows).drop_duplicates(["key", "generator"])
    a["len_dev"] = a.len - a.groupby("key").len.transform("mean")
    return a


def fit(d):
    """v = alpha_item + lambda_gen + delta * len_dev, alternating means (balanced enough)."""
    v = d.v.to_numpy()
    item, gen = d.item.to_numpy(), d.generator.to_numpy()
    x = d.len_dev.to_numpy()
    lam = pd.Series(0.0, index=np.unique(gen))
    delta = 0.0
    for _ in range(N_ITER):
        alpha = pd.Series(v - lam.reindex(gen).to_numpy() - delta * x).groupby(item).mean()
        r = v - alpha.reindex(item).to_numpy() - delta * x
        lam = pd.Series(r).groupby(gen).mean()
        lam -= lam.mean()
        r = v - alpha.reindex(item).to_numpy() - lam.reindex(gen).to_numpy()
        delta = float(np.sum(x * r) / np.sum(x * x)) if np.sum(x * x) > 0 else 0.0
    return alpha, lam, delta


def main():
    df = units()
    df["key"] = df.key.astype(str)
    df = df.merge(answer_lengths(), on=["key", "generator"], how="left")
    df["len_dev"] = df.len_dev.fillna(0.0)
    pq = df.groupby("generator").y.mean()
    glen = df.drop_duplicates(["generator", "key"]).groupby("generator").len.mean()
    rows, lams, deltas = [], {}, {}
    for J, dj in df.groupby("judge"):
        fj = family(J)
        f = dj[(~dj.y) & (dj.generator.map(family) != fj)].copy()
        if len(f) < 200:
            continue
        alpha, lam, delta = fit(f)
        lams[J], deltas[J] = lam, delta
        f["alpha"] = alpha.reindex(f.item).to_numpy()
        f["lam"] = lam.reindex(f.generator).to_numpy()
        f["lenp"] = delta * f.len_dev
        for G in f.generator.unique():
            fg = family(G)
            o, c = f[f.generator == G], f[f.generator.map(family) != fg]
            if len(o) < 20 or len(c) < 20:
                continue
            others = [g for g in pq.index if family(g) not in (fj, fg)]
            part = {k: o[k].mean() - c[k].mean() for k in ("v", "alpha", "lam", "lenp")}
            rows.append({"judge": J, "generator": G, "gap": pq[G] - pq[others].mean(),
                         "hspp_rubric": part["v"], "criteria_part": part["alpha"], "generator_part": part["lam"],
                         "length_part": part["lenp"], "rest": part["v"] - part["alpha"] - part["lam"] - part["lenp"]})
    r = pd.DataFrame(rows)
    r.to_csv(OUT + ".csv", index=False)
    parts = ["hspp_rubric", "criteria_part", "generator_part", "length_part", "rest"]
    x = r.gap - r.groupby("judge").gap.transform("mean")
    slope = {p: float(np.sum(x * (r[p] - r.groupby("judge")[p].transform("mean"))) / np.sum(x * x)) for p in parts}
    pooled = {p: float(np.polyfit(r.gap, r[p], 1)[0]) for p in parts}
    # is going easy on G shared across judges? J's lambda_G vs mean lambda_G of judges outside J's and G's families
    sh = []
    for J, lam in lams.items():
        for G, val in lam.items():
            ks = [lams[K][G] for K in lams if K != J and family(K) not in (family(J), family(G)) and G in lams[K].index]
            if ks:
                sh.append((val, np.mean(ks), pq[G], glen.get(G, np.nan)))
    sh = np.array(sh, dtype=float)
    ok = np.isfinite(sh).all(axis=1)
    out = {"dataset": DATASET, "pairs": len(r), "slope_within_judge": slope, "slope_pooled": pooled,
           "length_coef_median": float(np.median(list(deltas.values()))),
           "going_easy_corr_with_other_judges": float(np.corrcoef(sh[ok, 0], sh[ok, 1])[0, 1]),
           "going_easy_corr_with_pass_rate": float(np.corrcoef(sh[ok, 0], sh[ok, 2])[0, 1]),
           "going_easy_corr_with_answer_length": float(np.corrcoef(sh[ok, 0], sh[ok, 3])[0, 1])}
    json.dump(out, open(OUT + ".json", "w"), indent=1)
    print(f"{DATASET}: {len(r)} null pairs")
    for p in parts:
        print(f"   {p:15s} slope on quality gap: within judge {slope[p]:+.3f}, pooled {pooled[p]:+.3f}")
    print({k: round(v, 3) for k, v in out.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
