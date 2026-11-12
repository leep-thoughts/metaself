"""
Data for Yang et al.'s measure on Chen et al.'s release (measures/yang.py).

    python -m infer.yang_data answers      # rating inputs: every model's answer on sampled problems
    python -m infer.yang_data pairs        # after scoring: per-judge inputs for the GPU jobs

answers -> infer/data/yang_answers.jsonl
    N_PROBLEMS sampled problems per task; all 18 models' answers (11 judges + 7 opponents),
    one row each: id = task|problem_idx|model, question, answer.

pairs (reads infer/data/scores_*.jsonl, one file per reference judge)
    score = mean over reference judges (Yang: mean of two). Equal quality: |gap| <= EPS.
    -> infer/data/yang_judge_<judge>.jsonl, input for infer.judge (--template row --both-orders),
       all with Yang's A/B preference prompt (PROMPT; no tie option):
        kind=self  judge J's own answer (target, slot A in game 1) vs opponent R, equal quality.
                   Only the 7 opponents: the pairs Chen also released verdicts for.
        kind=null  Yang's B_q(i): two other models' answers j, k, each within EPS of J's answer
                   and of each other. One row per unordered pair; both orders give both ordered
                   pairs. At most MAX_NULL_PAIRS per judge, sampled uniformly.
    -> infer/data/yang_self_chen.jsonl: the same self pairs with Chen's released verdicts
       (A/T/B prompt), a no-GPU cross-check.
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

from real.chen import OPPONENTS, REPO_DIR, TASKS, _judge_and_opponent

N_PROBLEMS = 150
EPS = 0.25
MAX_NULL_PAIRS = 2000
PROMPT = "yang_preference"
SEED = 0
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# Chen's judge names -> Hugging Face ids, and GPUs (L40S 48 GB) for bf16 inference
JUDGES = {
    "qwen-2.5-3b": ("Qwen/Qwen2.5-3B-Instruct", 1), "qwen-2.5-7b": ("Qwen/Qwen2.5-7B-Instruct", 1),
    "qwen-2.5-14b": ("Qwen/Qwen2.5-14B-Instruct", 1), "qwen-2.5-32b": ("Qwen/Qwen2.5-32B-Instruct", 2),
    "qwen-2.5-72b": ("Qwen/Qwen2.5-72B-Instruct", 4), "llama-3.2-3b": ("meta-llama/Llama-3.2-3B-Instruct", 1),
    "llama-3.1-8b": ("meta-llama/Llama-3.1-8B-Instruct", 1), "llama-3.1-70b": ("meta-llama/Llama-3.1-70B-Instruct", 4),
    "llama-3.3-70b": ("meta-llama/Llama-3.3-70B-Instruct", 4), "gemma-2-9b": ("google/gemma-2-9b-it", 1),
    "gemma-2-27b": ("google/gemma-2-27b-it", 2),
}


def _files(task):
    return sorted(glob.glob(os.path.join(REPO_DIR, f"sp/{task}/*/*_eval.jsonl")))


def load_answers(task):
    """{(model, problem_idx): (question, answer, correct)} for every model in Chen's files."""
    out = {}
    for path in _files(task):
        judge, opp = _judge_and_opponent(path)
        with open(path) as f:
            for idx, line in enumerate(f):
                r = json.loads(line)
                for model, key in ((judge, "assistent_1"), (opp, "assistent_2")):
                    out.setdefault((model, idx), (r["problem"], r[f"{key}_answer"], r[f"{key}_is_correct"]))
    return out


def problems(task, n_total):
    rng = np.random.default_rng(SEED + TASKS.index(task))
    return sorted(rng.choice(n_total, size=min(N_PROBLEMS, n_total), replace=False).tolist())


def build_answers():
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "yang_answers.jsonl")
    n = 0
    with open(path, "w") as f:
        for task in TASKS:
            ans = load_answers(task)
            n_total = 1 + max(i for _, i in ans)
            for idx in problems(task, n_total):
                for (model, i), (q, a, c) in sorted(ans.items()):
                    if i == idx:
                        f.write(json.dumps({"id": f"{task}|{idx}|{model}", "task": task, "problem_idx": idx,
                                            "model": model, "correct": c, "question": q, "answer": a}) + "\n")
                        n += 1
    print(f"wrote {n} answers -> {path}")


def load_scores():
    per = defaultdict(dict)
    raters = []
    for path in sorted(glob.glob(os.path.join(DATA, "scores_*.jsonl"))):
        rater = os.path.basename(path)[len("scores_"):-len(".jsonl")]
        raters.append(rater)
        for line in open(path):
            r = json.loads(line)
            if r.get("score") is not None:
                per[r["id"]][rater] = r["score"]
    full = {k: float(np.mean(list(v.values()))) for k, v in per.items() if len(v) == len(raters)}
    print(f"raters: {raters}; answers scored by all: {len(full)}")
    return full, raters


def _ab(p):
    """[A, T, B] -> P(A) with the tie dropped."""
    a, _, b = p
    return a / (a + b) if a + b > 0 else 0.5


def build_pairs():
    scores, raters = load_scores()
    scores_of = lambda task, idx, m: scores[f"{task}|{idx}|{m}"]
    rng = np.random.default_rng(SEED)
    self_rows, null_rows, judge_rows = [], defaultdict(list), defaultdict(list)
    for task in TASKS:
        ans = load_answers(task)
        s = {(m, i): scores[f"{task}|{i}|{m}"] for (m, i) in ans if f"{task}|{i}|{m}" in scores}
        by_problem = defaultdict(dict)
        for (m, i), v in s.items():
            by_problem[i][m] = v
        # self pairs: Chen's verdicts for J vs R where the two answers are equal quality
        for path in _files(task):
            judge, opp = _judge_and_opponent(path)
            with open(path) as f:
                for idx, line in enumerate(f):
                    sc = by_problem.get(idx, {})
                    if judge in sc and opp in sc and abs(sc[judge] - sc[opp]) <= EPS:
                        r = json.loads(line)
                        judge_rows[judge].append(("self", task, idx, judge, opp))
                        self_rows.append({"task": task, "judge": judge, "opponent": opp, "problem_idx": idx,
                                          "gap": sc[judge] - sc[opp],
                                          "p_own_1": _ab(r["game_1_spb_score"]),            # own shown first (A)
                                          "p_own_2": 1 - _ab(r["game_2_spb_score"]),        # own shown second (B)
                                          "tie_1": r["game_1_spb_score"][1], "tie_2": r["game_2_spb_score"][1]})
        # null pairs: B_q(i) for each judge
        for idx, sc in by_problem.items():
            for judge in JUDGES:
                if judge not in sc:
                    continue
                near = sorted(m for m in sc if m != judge and abs(sc[m] - sc[judge]) <= EPS)
                for a in range(len(near)):
                    for b in range(a + 1, len(near)):
                        j, k = near[a], near[b]
                        if abs(sc[j] - sc[k]) <= EPS:
                            null_rows[judge].append(("null", task, idx, j, k))
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "yang_self_chen.jsonl"), "w") as f:
        for r in self_rows:
            f.write(json.dumps(r) + "\n")
    texts = {t: load_answers(t) for t in TASKS}
    for judge in JUDGES:
        nulls = null_rows.get(judge, [])
        if len(nulls) > MAX_NULL_PAIRS:
            nulls = [nulls[i] for i in sorted(rng.choice(len(nulls), MAX_NULL_PAIRS, replace=False))]
        rows = judge_rows.get(judge, []) + nulls
        if not rows:
            continue
        with open(os.path.join(DATA, f"yang_judge_{judge}.jsonl"), "w") as f:
            for kind, task, idx, j, k in rows:
                q, aj, _ = texts[task][(j, idx)]
                _, ak, _ = texts[task][(k, idx)]
                f.write(json.dumps({"id": f"{kind}|{task}|{idx}|{j}|{k}", "kind": kind, "template": PROMPT,
                                    "task": task, "problem_idx": idx, "judge": judge, "target": j, "other": k,
                                    "gap": scores_of(task, idx, j) - scores_of(task, idx, k),
                                    "question": q, "answer_a": aj, "answer_b": ak}) + "\n")
        print(f"{judge:14s} self pairs {len(judge_rows.get(judge, [])):5d}   null pairs {len(nulls):5d} (x2 orders each)")
