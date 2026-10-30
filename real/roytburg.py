"""
Roytburg et al. 2026 judge-swap release (github.com/djroytburg/sanity_checks_for_self_preference),
verifiable tasks only: judge_swap_null_verif_smoke2/<task>/cache/<judge>/<reference>/.
Cloned (sparse, cache folders only, ~53 MB) into data/roytburg/.

Same judges, opponents ("references") and problems as the Chen et al. release
(real/chen.py); example_id is the row index in Chen's files. Roytburg re-ran the
self-judgments, so self and proxy probabilities below come from one consistent run --
they differ from Chen's spb scores for the same pair.

Per (judge J, reference R, example):
    J_vs_R_game{1,2}  J judges its own answer vs R's (game 1: own first, game 2: own second)
    K_vs_R_game{1,2}  J judges proxy K's answer vs R's, for every other judge model K whose
                      answer has the same correctness as J's on that example
Each record gives prob_response1 / prob_response2 (tie is the remaining mass).

Only a subset of Chen's usable pairs was run (~14k self-judgments; ~2.9k where J's own
answer is wrong).
"""

import glob
import json
import math
import os
import subprocess
from collections import defaultdict

import pandas as pd

REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "roytburg")
REPO_URL = "https://github.com/djroytburg/sanity_checks_for_self_preference.git"
CACHE_GLOB = "judge_swap_null_verif_smoke2/{task}/cache/*/*"
TASKS = ["math500", "mmlu", "mbpp-plus"]


def download():
    """Blobless clone + checkout of the three cache folders only. Rerun if the proxy drops files."""
    if not os.path.isdir(REPO_DIR):
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--no-checkout",
                        REPO_URL, REPO_DIR], check=True)
    paths = [f"judge_swap_null_verif_smoke2/{t}/cache" for t in TASKS]
    subprocess.run(["git", "-c", "gc.auto=0", "checkout", "-q", "HEAD", "--", *paths], cwd=REPO_DIR, check=True)


def _p_candidate(r) -> float:
    """Probability J favours the judge's or proxy's answer, whichever slot it was shown in."""
    return r["prob_response1"] if r["response1_key"] in ("judge", "proxy") else r["prob_response2"]


def _read(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load(tasks=TASKS) -> pd.DataFrame:
    """
    One row per (task, judge, reference, example) that has a self-judgment.

    Columns: task, judge, opponent, problem_idx (row in Chen's file), own_correct,
    s_self_1, s_self_2 (J's probability for its own answer, own shown first / second),
    s_self (mean of the two), s_proxy (mean over matched proxies and both orders; NaN if
    no proxy), n_proxies.
    """
    rows = []
    for task in tasks:
        for d in sorted(glob.glob(os.path.join(REPO_DIR, CACHE_GLOB.format(task=task)))):
            judge, opponent = d.split(os.sep)[-2:]
            self_p = defaultdict(dict)
            category = {}
            for g in (1, 2):
                for r in _read(os.path.join(d, f"J_vs_R_game{g}.jsonl")):
                    self_p[r["example_id"]][g] = _p_candidate(r)
                    category[r["example_id"]] = r["category"]
            proxy_p, proxies = defaultdict(list), defaultdict(set)
            for g in (1, 2):
                for r in _read(os.path.join(d, f"K_vs_R_game{g}.jsonl")):
                    proxy_p[r["example_id"]].append(_p_candidate(r))
                    proxies[r["example_id"]].add(r["proxy"])
            for ex, p in self_p.items():
                pp = proxy_p.get(ex, [])
                rows.append({
                    "task": task, "judge": judge, "opponent": opponent, "problem_idx": ex,
                    "own_correct": category[ex] == "lsp",
                    "s_self_1": p.get(1), "s_self_2": p.get(2),
                    "s_self": sum(p.values()) / len(p),
                    "s_proxy": sum(pp) / len(pp) if pp else float("nan"),
                    "n_proxies": len(proxies.get(ex, ())),
                })
    return pd.DataFrame(rows)


def family(model: str) -> str:
    """Model family, for keeping null judgments free of same-family affinity."""
    m = model.lower()
    for key in ("qwen", "llama", "gemma", "gpt", "mistral", "phi"):
        if key in m:
            return key
    raise ValueError(model)


def load_proxy_pairs(tasks=TASKS, eps: float = 1e-6) -> pd.DataFrame:
    """
    The K_vs_R judgments as pairs in their own right: judge J rates proxy K's answer (A)
    against reference R's answer (B). J wrote neither, so its true self_lean toward A is
    ~0 -- a real-data null, as long as J shares no family with K or R (cross_family).

    One row per (task, judge, opponent, proxy, problem). Columns: a_correct (K's answer is
    the correct one), l_1, l_2 (J's log-odds for A over B in each order, tie mass
    dropped; game 1 shows A first), l (their mean), cross_family.
    """
    rows = []
    for task in tasks:
        for d in sorted(glob.glob(os.path.join(REPO_DIR, CACHE_GLOB.format(task=task)))):
            judge, opponent = d.split(os.sep)[-2:]
            by_key = {}
            for g in (1, 2):
                for r in _read(os.path.join(d, f"K_vs_R_game{g}.jsonl")):
                    p_a = max(_p_candidate(r), eps)
                    p_b = max(r["prob_response2"] if r["response1_key"] == "proxy" else r["prob_response1"], eps)
                    row = by_key.setdefault((r["example_id"], r["proxy"]), {
                        "task": task, "judge": judge, "opponent": opponent, "proxy": r["proxy"],
                        "problem_idx": r["example_id"], "a_correct": r["category"] == "lsp"})
                    row[f"l_{g}"] = math.log(p_a / p_b)
            rows.extend(by_key.values())
    df = pd.DataFrame(rows).dropna(subset=["l_1", "l_2"])
    df["l"] = (df.l_1 + df.l_2) / 2
    fj = df.judge.map(family)
    df["cross_family"] = (fj != df.proxy.map(family)) & (fj != df.opponent.map(family))
    return df.reset_index(drop=True)


if __name__ == "__main__":
    # Reproduces the verifiable-task block of Roytburg et al. Table 1 (ILSP_orig, ILSP_upd, N).
    from measures import roytburg

    df = load()
    for (task, judge), g in df.groupby(["task", "judge"]):
        ilsp = g[~g.own_correct]
        matched = ilsp.dropna(subset=["s_proxy"])
        eqb = roytburg.estimate_bias(matched.own_correct, matched.s_self, matched.s_proxy)
        print(f"{task:10s} {judge:14s} ILSP_orig={100 * ilsp.s_self.mean():5.1f}  "
              f"ILSP_upd={100 * eqb:6.1f}  N={len(matched)}")
