"""
Pairwise same-item judgments (method principles 4, 5 and 7).

Roytburg's release can't give same-item comparisons of targets: on a given problem its targets
always share the judge's own correctness. Here every judge rates EVERY target against the SAME
opponents on the SAME problems, whatever the answers' correctness:

    problems   N_PROBLEMS per task, drawn at random once (same for every judge), without looking at
               labels or votes; only problems all 14 models answered
    opponents  fixed by model property: mistral-7b-v0.3 (weak), gpt-3.5-turbo (mid), gpt-4o (strong);
               none shares a family with any judge
    targets    Chen's 11 judge models outside the judge's family (null test), plus the judge itself
               (own answers, for the detection check)

This supports: null values per judge x target x opponent on identical problems (existence), the
same-item comparison of stronger vs weaker targets and fictional mixtures (trend), quality-matched
comparison targets and a second judge on the same rows (the fix), and real self-judging next to it.
Chen's direct prompt, both orders.

    python -m infer.samepair_data   -> infer/data/samepair_judge_<judge>.jsonl
    GPU: bash infer/samepair_judges_submit.sh   -> infer/data/samepair_verdicts_<judge>.jsonl
"""

import json
import os

import numpy as np

from infer.yang_data import DATA, JUDGES, TASKS, load_answers
from real.roytburg import family

N_PROBLEMS = 100
SEED = 11
OPPONENTS = ["mistral-7b-v0.3", "gpt-3.5-turbo", "gpt-4o"]


def build():
    rng = np.random.default_rng(SEED)
    os.makedirs(DATA, exist_ok=True)
    models = list(JUDGES) + OPPONENTS
    rows = {J: [] for J in JUDGES}
    for task in TASKS:
        ans = load_answers(task)
        idx = sorted({i for (_, i) in ans})
        full = [i for i in idx if all((m, i) in ans for m in models)]
        pick = sorted(rng.choice(full, size=min(N_PROBLEMS, len(full)), replace=False).tolist())
        print(f"{task}: {len(full)} problems answered by all {len(models)} models, using {len(pick)}")
        for J in JUDGES:
            targets = [m for m in JUDGES if family(m) != family(J)] + [J]
            for i in pick:
                for a in targets:
                    for b in OPPONENTS:
                        qa, aa, ca = ans[(a, i)]
                        _, ab, cb = ans[(b, i)]
                        rows[J].append({"id": f"{task}|{i}|{a}|{b}", "task": task, "problem_idx": int(i), "judge": J,
                                        "template": f"chen_direct_{task}", "question": qa, "answer_a": aa, "answer_b": ab,
                                        "target": a, "opponent": b, "own": a == J,
                                        "a_correct": bool(ca), "b_correct": bool(cb)})
    for J, rs in rows.items():
        path = os.path.join(DATA, f"samepair_judge_{J}.jsonl")
        with open(path, "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        print(f"{J:15s} {len(rs):6d} pairs ({sum(r['own'] for r in rs)} own) -> {path}")


if __name__ == "__main__":
    build()
