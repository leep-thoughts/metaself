"""
Calibrating beta (synth/difficulty.py): does a judge evaluate better on problems it solves itself?

    L(x) = L0 + beta * c(x)        evaluation strength rises with the judge's competence on x

Roytburg's release can't answer this (proxies are matched to J's correctness, so "J solved x"
always coincides with "A is correct"). Here J judges pairs written by OTHER models only:
one correct answer W and one wrong answer L on the same problem, both orders, Chen's direct
prompt (the prompt behind the real slopes). Problems are stratified by whether J's own answer
(Chen's release) is correct, so the accuracy gap between the strata identifies beta.

    python -m infer.beta_data      -> infer/data/beta_judge_<judge>.jsonl (input for infer.judge,
                                      --template row --both-orders); analysis: real/beta_fit.py

Per judge, task and stratum (J right / J wrong): up to N_PER_STRATUM problems, one pair each.
W and L are drawn from the 17 other models, excluding J's family (no self-recognition), and
must differ in correctness. Rows carry j_correct (J's own grade) and solve_rate (share of all
18 models that solve x: a difficulty covariate, to separate "J solved it" from "it is easy").
"""

import json
import os
from collections import defaultdict

import numpy as np

from infer.yang_data import DATA, JUDGES, TASKS, load_answers
from real.roytburg import family

N_PER_STRATUM = 250
SEED = 1


def build():
    rng = np.random.default_rng(SEED)
    os.makedirs(DATA, exist_ok=True)
    rows = defaultdict(list)
    for task in TASKS:
        ans = load_answers(task)
        by_problem = defaultdict(dict)
        for (m, i), v in ans.items():
            by_problem[i][m] = v
        for judge in JUDGES:
            strata = {True: [], False: []}
            for i, models in sorted(by_problem.items()):
                if judge not in models:
                    continue
                others = [m for m in models if family(m) != family(judge)]
                right = [m for m in others if models[m][2]]
                wrong = [m for m in others if not models[m][2]]
                if right and wrong:
                    strata[bool(models[judge][2])].append((i, right, wrong))
            for j_ok, items in strata.items():
                pick = rng.permutation(len(items))[:N_PER_STRATUM]
                for k in sorted(pick):
                    i, right, wrong = items[k]
                    w, l = rng.choice(right), rng.choice(wrong)
                    models = by_problem[i]
                    rows[judge].append({
                        "id": f"{task}|{i}|{w}|{l}", "task": task, "problem_idx": int(i), "judge": judge,
                        "template": f"chen_direct_{task}", "question": models[w][0],
                        "answer_a": models[w][1], "answer_b": models[l][1], "correct": "A",
                        "winner": w, "loser": l, "j_correct": j_ok,
                        "solve_rate": float(np.mean([v[2] for v in models.values()]))})
    for judge, rs in rows.items():
        path = os.path.join(DATA, f"beta_judge_{judge}.jsonl")
        with open(path, "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        n_ok = sum(r["j_correct"] for r in rs)
        print(f"{judge:15s} {len(rs):5d} pairs (J right {n_ok}, J wrong {len(rs) - n_ok}) -> {path}")


if __name__ == "__main__":
    build()
