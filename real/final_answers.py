"""
Each model's final answer to each problem, from Chen et al.'s release (data/llm-sp), so
two wrong answers can be compared: is a proxy's wrong answer the same wrong answer the
judge itself gave?

MATH500: last \\boxed{...}, lightly normalized. MMLU: the letter in "final answer is $$X$$"
(the format Chen's generators use). MBPP+: not extracted (code has
no single final answer) -> None.

Row index = problem_idx in the Roytburg release (same order in every Chen file).
Run with: python -m real.final_answers   (prints extraction coverage and a validity check)
"""

import glob
import json
import os
import re

import pandas as pd

from real.chen import REPO_DIR, _judge_and_opponent

TASKS = ["math500", "mmlu", "mbpp-plus"]   # mbpp-plus: correctness only, answer None


def _last_boxed(text: str):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j + 1:k]
    return None


def _norm_math(s):
    if s is None:
        return None
    for a, b in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("dfrac", "frac"),
                 ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""), ("$", ""), (" ", "")):
        s = s.replace(a, b)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.rstrip(".")
    return s or None


# Chen's generators end with "The final answer is $$X$$". Looser patterns ("answer is (X)")
# disagree with Chen's grades on ~10% of answers, so unmatched answers stay None.
_MMLU = [re.compile(r"\$\$\s*\(?([A-J])\)?\s*\$\$")]


def _mmlu_letter(text):
    for pat in _MMLU:
        m = pat.findall(text)
        if m:
            return m[-1].upper()
    return None


def extract(task, text):
    if task == "math500":
        return _norm_math(_last_boxed(text))
    return _mmlu_letter(text) if task == "mmlu" else None


def load() -> pd.DataFrame:
    """One row per (task, model, problem_idx): answer (None if not found), correct."""
    rows = {}
    for task in TASKS:
        for path in sorted(glob.glob(os.path.join(REPO_DIR, f"sp/{task}/*/*_eval.jsonl"))):
            judge, opponent = _judge_and_opponent(path)
            with open(path) as f:
                for idx, line in enumerate(f):
                    r = json.loads(line)
                    for model, key in ((judge, "assistent_1"), (opponent, "assistent_2")):
                        if (task, model, idx) not in rows:
                            rows[(task, model, idx)] = {"task": task, "model": model, "problem_idx": idx,
                                                        "answer": extract(task, r[f"{key}_answer"]),
                                                        "correct": r[f"{key}_is_correct"]}
    return pd.DataFrame(rows.values())


if __name__ == "__main__":
    df = load()
    print(df.groupby("task").answer.apply(lambda a: a.notna().mean()).round(3).to_string())
    # validity: two models with the same extracted answer should get the same grade
    for task, g in df[df.answer.notna()].groupby("task"):
        m = g.merge(g, on=["task", "problem_idx", "answer"])
        m = m[m.model_x < m.model_y]
        print(task, "same answer -> same grade:", round((m.correct_x == m.correct_y).mean(), 3), "n =", len(m))
