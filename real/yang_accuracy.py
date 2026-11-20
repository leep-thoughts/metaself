"""
Evaluation-accuracy confound for Yang et al.'s measure (null test and form-B injection).

Within each judge, Chen's problems are split into thirds by difficulty: the share of the 18 models
that solve the problem (a covariate fixed before judging). The judge's evaluation accuracy in each third
comes from the difficulty-calibration runs on the same problems (infer/data/beta_verdicts_<judge>.jsonl:
the judge compares a right and a wrong answer by two other models; accuracy = mean P(pick the right one)
over both orders). For every judge, target G outside its family and third: Yang's measure on the null
pairs (real/yang_null.py) and its response to form-B injection; then OLS within (judge, target) of the
measure on [evaluation accuracy, target's generation quality in that third] (want 0 on accuracy).

Run with: python -m real.yang_accuracy      writes real/yang_accuracy_results.csv
"""

import json
import os

import numpy as np
import pandas as pd

from real import final_answers
from real.roytburg import family
from real.yang_null import DATA, null_pairs

N_BINS = 3
PI = 0.1
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yang_accuracy_results.csv")


def judge_accuracy(J):
    path = os.path.join(DATA, f"beta_verdicts_{J}.jsonl")
    if not os.path.exists(path):
        return None
    d = pd.DataFrame([json.loads(l) for l in open(path)])
    d = d[(d.verdict_mass >= 0.5)]
    pc = np.where(d.game == 1, d.prob_A, d.prob_B) / (d.prob_A + d.prob_B)
    d = d.assign(pc=pc)
    return d.groupby(["task", "problem_idx"]).pc.mean()


def main():
    fa = final_answers.load()
    solve = fa.groupby(["task", "problem_idx"]).correct.mean()
    rows = []
    for path in sorted(p for p in os.listdir(DATA) if p.startswith("verdicts_") and p.endswith(".jsonl")):
        J = path[len("verdicts_"):-len(".jsonl")]
        acc = judge_accuracy(J)
        if acc is None:
            continue
        v = null_pairs(J)
        v["solve"] = [solve.get((t, i), np.nan) for t, i in zip(v.task, v.problem_idx)]
        v = v.dropna(subset=["solve"])
        # thirds of difficulty within each task, so tasks are mixed evenly
        v["bin"] = v.groupby("task").solve.transform(lambda s: pd.qcut(s.rank(method="first"), N_BINS, labels=False))
        accb = pd.DataFrame({"pc": acc}).reset_index()
        accb["solve"] = [solve.get((t, i), np.nan) for t, i in zip(accb.task, accb.problem_idx)]
        edges = {t: v[v.task == t].groupby("bin").solve.max() for t in v.task.unique()}
        accb["bin"] = [int(np.searchsorted(edges[t].values, s)) if t in edges else -1 for t, s in zip(accb.task, accb.solve)]
        acc_bin = accb[accb.bin.between(0, N_BINS - 1)].groupby("bin").pc.mean()
        for b in range(N_BINS):
            vb = v[v.bin == b]
            prob_b = fa.merge(vb[["task", "problem_idx"]].drop_duplicates(), on=["task", "problem_idx"])
            gq = prob_b.groupby("model").correct.mean()
            for G in sorted(set(vb.target)):
                if family(G) == family(J) or G not in gq.index:
                    continue
                w, wo = vb[vb.target == G], vb[(vb.target != G) & (vb.other != G)]
                if len(w) < 10 or len(wo) < 10:
                    continue
                base = (wo.p1 * wo.p2).mean()
                s0 = (w.p1 * w.p2).mean() - base
                s1 = ((w.p1 + PI * (1 - w.p1)) * (w.p2 + PI * (1 - w.p2))).mean() - base
                rows.append({"judge": J, "target": G, "bin": b, "eval_acc": acc_bin.get(b, np.nan),
                             "gen_quality": gq[G], "yang": s0, "yang_resp": (s1 - s0) / PI, "n": len(w)})
    r = pd.DataFrame(rows).dropna(subset=["eval_acc"])
    r.to_csv(OUT, index=False)
    print(r.groupby("bin").agg(eval_acc=("eval_acc", "mean"), yang=("yang", "mean"), resp=("yang_resp", "mean")).round(3))
    for m in ("yang", "yang_resp"):
        d = r.copy()
        g = d.groupby(["judge", "target"])
        d = d[g[m].transform("size") >= 2]
        g = d.groupby(["judge", "target"])
        X = np.c_[d.eval_acc - g.eval_acc.transform("mean"), d.gen_quality - g.gen_quality.transform("mean")]
        y = d[m] - g[m].transform("mean")
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        rng_acc = g.eval_acc.agg(lambda z: z.max() - z.min()).mean()
        print(f"{m:10s}: slope with evaluation accuracy {b[0]:+.3f} (holding generation quality; slope with it {b[1]:+.3f}); "
              f"mean accuracy range within pairing {rng_acc:.3f} -> implied change {b[0] * rng_acc:+.3f}; n {len(d)}")


if __name__ == "__main__":
    main()
