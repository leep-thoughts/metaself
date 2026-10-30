"""
Chen et al. 2025 ("Do LLM Evaluators Prefer Themselves for a Reason?",
arXiv 2504.03846) released judge verdicts: github.com/wlchen0206/llm-sp, cloned into
data/llm-sp/. This is the real-data source with votes already attached.

Each file is one (judge, opponent) on one task. Each row is one problem with the
judge's OWN answer (assistent_1) and the opponent's answer (assistent_2), each graded
by the task's checker -- that grade is our oracle. The judge saw the pair twice:

    game 1: own answer in position A, opponent in B
    game 2: opponent in position A, own answer in B

and *_spb_score is its probability over the verdict tokens [A, Tie, B]. In the
direct setting these are the judge's token probabilities. In the cot / long_cot
settings the judge reasons first, and the score is one-hot on its parsed verdict.

Settings:
    direct    sp/                           11 judges (Qwen-2.5, Llama-3.x, Gemma-2)
    cot       harmful_sp_mitigation/*/cot/      same models, reasoning before verdict
    long_cot  harmful_sp_mitigation/*/long cot/ R1-distill judges, judging their own
                                                (new) answers
Opponents (all settings): GPT-4o, GPT-3.5-turbo, Mistral-small, Mistral-7B-v0.3,
Phi-3.5-mini, Gemma-2-2B, Llama-3.2-1B.
"""

import glob
import json
import os

import pandas as pd

REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "llm-sp")
REPO_URL = "https://github.com/wlchen0206/llm-sp.git"

TASKS = ["math500", "mmlu", "mbpp-plus"]
OPPONENTS = ["gemma-2-2b", "gpt-3.5-turbo", "gpt-4o", "llama-3.2-1b",
             "mistral-7b-v0.3", "mistral-small", "phi-3.5-mini"]
SETTING_GLOB = {
    "direct": "sp/{task}/*/*_eval.jsonl",
    "cot": "harmful_sp_mitigation/{task}/cot/*/*_eval.jsonl",
    "long_cot": "harmful_sp_mitigation/{task}/long cot/*/*_eval.jsonl",
}


def download():
    """Shallow-clone the release (~820 MB) into data/llm-sp/. Skips it if already there."""
    if not os.path.isdir(REPO_DIR):
        os.system(f"git clone --depth 1 -q {REPO_URL} '{REPO_DIR}'")


def _judge_and_opponent(path: str):
    """'qwen-2.5-7b_gpt-4o_eval.jsonl' -> ('qwen-2.5-7b', 'gpt-4o'); drops the '_reasoning' tag."""
    base = os.path.basename(path)[:-len("_eval.jsonl")]
    opponent = next(o for o in OPPONENTS if base.endswith("_" + o))
    judge = base[:-len(opponent) - 1]
    return judge.removesuffix("_reasoning"), opponent


def load(setting: str = "direct", tasks=TASKS, text: bool = False) -> pd.DataFrame:
    """
    One row per (task, judge, opponent, problem).

    Columns: setting, task, family, judge, opponent, problem_id, own_correct,
    other_correct, and for each order g in {1, 2} the judge's p_own_g, p_tie_g,
    p_other_g (g=1: own answer shown first; g=2: own answer shown second).
    text=True also keeps problem, own_answer, other_answer, output_1, output_2.
    """
    rows = []
    for task in tasks:
        for path in sorted(glob.glob(os.path.join(REPO_DIR, SETTING_GLOB[setting].format(task=task)))):
            judge, opponent = _judge_and_opponent(path)
            family = os.path.basename(os.path.dirname(path))
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    a1, t1, b1 = r["game_1_spb_score"]  # own = A
                    a2, t2, b2 = r["game_2_spb_score"]  # own = B
                    row = {
                        "setting": setting, "task": task, "family": family,
                        "judge": judge, "opponent": opponent,
                        "problem_id": r["unique_id"],
                        "own_correct": r["assistent_1_is_correct"],
                        "other_correct": r["assistent_2_is_correct"],
                        "p_own_1": a1, "p_tie_1": t1, "p_other_1": b1,
                        "p_own_2": b2, "p_tie_2": t2, "p_other_2": a2,
                    }
                    if text:
                        row.update(problem=r["problem"],
                                   own_answer=r["assistent_1_answer"],
                                   other_answer=r["assistent_2_answer"],
                                   output_1=r.get("game_1_output"),
                                   output_2=r.get("game_2_output"))
                    rows.append(row)
    return pd.DataFrame(rows)


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """Pairs with exactly one correct answer: the oracle names a winner. own_wins = z(x) > 0."""
    out = df[df.own_correct != df.other_correct].copy()
    out["own_wins"] = out.own_correct
    return out


if __name__ == "__main__":
    download()
    for setting in SETTING_GLOB:
        df = load(setting)
        u = usable(df)
        print(f"{setting:8s} judges={df.judge.nunique():2d}  rows={len(df):6d}  usable={len(u):6d}  "
              f"p_own_wins={u.own_wins.mean():.2f}")
        # Sanity check on the column mapping: in both orders the judge should
        # back its own answer more when that answer is the correct one.
        for g in (1, 2):
            m = u.groupby("own_wins")[f"p_own_{g}"].mean()
            print(f"    order {g}: mean p_own | own wins={m[True]:.2f}, | other wins={m[False]:.2f}")
