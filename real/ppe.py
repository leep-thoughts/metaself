"""
PPE correctness sets (Frick et al. 2024, "How to Evaluate Reward Models for RLHF"):
the real-data source we drive the three knobs from.

Five verifiable benchmarks (MATH, MMLU-Pro, GPQA, MBPP-Plus, IFEval). Each row is one
(prompt, generator) with 32 sampled responses, each graded correct/incorrect by the
benchmark's own checker -- that grade is our oracle. Generators: Llama-3-8B-Instruct,
Gemma-2-9B-it, Claude-3-Haiku, GPT-4o-mini.

Every row has 0 < mean_score < 1 (the generator got the prompt right on some samples
and wrong on others). A prompt is usually answered by one generator only; the
cross-generator pairs we need for self-preference come from the prompts that more
than one generator answered (see shared_prompts()).
"""

import os
import urllib.request

import pandas as pd

BENCHMARKS = ["MATH", "MMLU-Pro", "GPQA", "MBPP-Plus", "IFEval"]
N_SAMPLES = 32
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ppe")

# Column that identifies the underlying task. question_id is unique per
# (prompt, generator) row, so it can't be used to match generators on the same task.
PROMPT_KEY = {
    "MATH": "problem",
    "MMLU-Pro": "question",
    "GPQA": "Question",
    "MBPP-Plus": "task_id",
    "IFEval": "key",
}

_URL = "https://huggingface.co/datasets/lmarena-ai/PPE-{b}-Best-of-K/resolve/main/data/train-00000-of-00001.parquet"


def path(benchmark: str) -> str:
    return os.path.join(DATA_DIR, f"{benchmark}.parquet")


def download(force: bool = False):
    """Fetch the five parquet files (~61 MB) into data/ppe/. Skips files already there."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for b in BENCHMARKS:
        if force or not os.path.exists(path(b)):
            urllib.request.urlretrieve(_URL.format(b=b), path(b))


def load_raw(benchmark: str) -> pd.DataFrame:
    """The parquet as published: one row per (prompt, generator), response_1..response_32 wide."""
    return pd.read_parquet(path(benchmark))


def load_responses(benchmark: str) -> pd.DataFrame:
    """
    Long format: one row per sampled response.

    Columns: benchmark, prompt_key, model, sample (0-31), correct (bool), response,
    prompt (the exact text the generator saw), solve_rate (this generator's mean
    correctness on this prompt -- a per-item difficulty covariate).
    """
    raw = load_raw(benchmark)
    rows = []
    for _, r in raw.iterrows():
        for i in range(N_SAMPLES):
            rows.append({
                "benchmark": benchmark,
                "prompt_key": str(r[PROMPT_KEY[benchmark]]),
                "model": r["model_name"],
                "sample": i,
                "correct": bool(r["scores"][i]),
                "response": r[f"response_{i + 1}"],
                "prompt": r["prompt"],
                "solve_rate": r["mean_score"],
            })
    return pd.DataFrame(rows)


def shared_prompts(responses: pd.DataFrame) -> pd.DataFrame:
    """Keep only prompts answered by two or more generators: the ones we can pair across models."""
    n_models = responses.groupby("prompt_key")["model"].transform("nunique")
    return responses[n_models >= 2]


if __name__ == "__main__":
    download()
    for b in BENCHMARKS:
        df = load_responses(b)
        shared = shared_prompts(df)
        print(f"{b:10s} prompts={df.prompt_key.nunique():4d}  "
              f"shared={shared.prompt_key.nunique():4d}  "
              f"responses={len(df):6d}  acc={df.correct.mean():.3f}")
