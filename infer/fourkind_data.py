"""
Four kinds of answer pairs: is the null-test signal a gap in accuracy or a lean?

From the votes on right-vs-wrong pairs alone, "accuracy differs between D_A and D_B" and "the judge
leans toward A" can't be told apart (two averages, two unknowns). Pairs where BOTH answers are
right, or BOTH are wrong, have no correctness term, so the vote on them measures the lean directly.

Per judge J and task: problems where the candidate pool has right and wrong answers on both sides,
so all four kinds exist on the SAME problems (difficulty held fixed across kinds):
    RW   A right, B wrong        WR   A wrong, B right
    RR   both right              WW   both wrong
A = target model K (one of Chen's 11 judge models), B = opponent R (one of Chen's 7 opponents),
both outside J's family (null test: J wrote neither). Per problem and kind, N_PER_KIND random (K, R).
Own condition: also K = J (J's own answer) against one right and one wrong R, where available, so
the same split can be made for the judge's own answers.
Chen's direct prompt, both orders (infer/judge.py --template row --both-orders).

    python -m infer.fourkind_data   -> infer/data/fourkind_judge_<judge>.jsonl
    GPU: bash infer/fourkind_judges_submit.sh   -> infer/data/fourkind_verdicts_<judge>.jsonl
    analysis: python -m real.fourkind
"""

import json
import os
from collections import defaultdict

import numpy as np

from infer.yang_data import DATA, JUDGES, TASKS, load_answers
from real.roytburg import family

N_PROBLEMS = 120          # per judge and task
N_PER_KIND = 2            # (K, R) pairs per problem and kind, null condition
SEED = 7
OPPONENTS = ["gpt-3.5-turbo", "gpt-4o", "llama-3.2-1b", "mistral-7b-v0.3", "mistral-small", "phi-3.5-mini", "gemma-2-2b"]
KINDS = {"RW": (True, False), "WR": (False, True), "RR": (True, True), "WW": (False, False)}


def build():
    rng = np.random.default_rng(SEED)
    os.makedirs(DATA, exist_ok=True)
    rows = defaultdict(list)
    for task in TASKS:
        ans = load_answers(task)
        by_problem = defaultdict(dict)
        for (m, i), v in ans.items():
            by_problem[i][m] = v
        for J in JUDGES:
            ks = [m for m in JUDGES if family(m) != family(J)]
            rs = [m for m in OPPONENTS if family(m) != family(J)]
            eligible = []
            for i, models in sorted(by_problem.items()):
                kr = [m for m in ks if m in models and models[m][2]]
                kw = [m for m in ks if m in models and not models[m][2]]
                rr = [m for m in rs if m in models and models[m][2]]
                rw = [m for m in rs if m in models and not models[m][2]]
                if kr and kw and rr and rw:
                    eligible.append((i, {True: kr, False: kw}, {True: rr, False: rw}))
            for k in sorted(rng.permutation(len(eligible))[:N_PROBLEMS]):
                i, kpool, rpool = eligible[k]
                models = by_problem[i]
                solve = float(np.mean([v[2] for v in models.values()]))
                pairs = []
                for kind, (ca, cb) in KINDS.items():
                    combos = [(a, b) for a in kpool[ca] for b in rpool[cb]]
                    for c in rng.permutation(len(combos))[:N_PER_KIND]:
                        pairs.append((kind, *combos[c], False))
                if J in models:                                   # own answer vs one right and one wrong R
                    cj = bool(models[J][2])
                    for cb in (True, False):
                        b = rng.choice(rpool[cb])
                        kind = {(True, False): "RW", (False, True): "WR", (True, True): "RR", (False, False): "WW"}[(cj, cb)]
                        pairs.append((kind, J, b, True))
                for kind, a, b, own in pairs:
                    rows[J].append({
                        "id": f"{task}|{i}|{a}|{b}", "task": task, "problem_idx": int(i), "judge": J,
                        "template": f"chen_direct_{task}", "question": models[a][0],
                        "answer_a": models[a][1], "answer_b": models[b][1],
                        "kind": kind, "target": a, "opponent": b, "own": own,
                        "a_correct": bool(models[a][2]), "b_correct": bool(models[b][2]), "solve_rate": solve})
    for J, rs_ in rows.items():
        path = os.path.join(DATA, f"fourkind_judge_{J}.jsonl")
        with open(path, "w") as f:
            for r in rs_:
                f.write(json.dumps(r) + "\n")
        n_own = sum(r["own"] for r in rs_)
        print(f"{J:15s} {len(rs_):5d} pairs ({n_own} own) -> {path}")


if __name__ == "__main__":
    build()
