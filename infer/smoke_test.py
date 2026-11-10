"""
GPU smoke test for the inference stack. Loads one judge once and runs every mode on
infer/fixtures/pairs.jsonl (4 pairs with a known correct answer, both orders).

    python -m infer.smoke_test [--model meta-llama/Meta-Llama-3-8B-Instruct]

Pass criteria (printed at the end):
  - engine loads and every mode returns one row per input
  - direct templates: median verdict_mass >= 0.5 (labels are what the model wants to say first)
  - cot: verdict parsed on >= 75% of rows
  - rating: score parsed on >= 75% of rows
  - generation: non-empty text
Accuracy is printed but not asserted -- a weak judge can still be a working pipeline.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from infer import judge
from infer.prompts import TEMPLATES

FIXTURE = Path(__file__).parent / "fixtures" / "pairs.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    pairs = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    both = judge.with_both_orders(pairs)
    t0 = time.time()
    llm = judge.load(args.model, max_model_len=args.max_model_len)
    print(f"engine loaded in {time.time() - t0:.0f}s")
    failures = []

    for name, t in TEMPLATES.items():
        if t["mode"] != "direct":
            continue
        try:
            out = judge.judge_direct(llm, name, [dict(r) for r in both])
        except Exception as e:  # noqa: BLE001 -- e.g. no system role in the chat template
            failures.append(f"{name}: {e}")
            continue
        mass = statistics.median(r["verdict_mass"] for r in out)
        first = t["choices"][0]
        # correct label is the first choice when the correct answer is in slot A
        correct = [(r["a_correct"] == (r["game"] == 1)) for r in out]
        acc = statistics.mean((r[f"prob_{first}"] > 0.5) == c for r, c in zip(out, correct))
        print(f"{name:24s} median verdict_mass {mass:.2f}  acc {acc:.2f}  top tokens {[r['top_token'] for r in out]}")
        if mass < 0.5:
            failures.append(f"{name}: verdict_mass {mass:.2f}")

    out = judge.judge_cot(llm, "chen_cot_math500", [dict(r) for r in both], max_tokens=1024)
    parsed = statistics.mean(r["verdict"] is not None for r in out)
    print(f"{'chen_cot_math500':24s} parsed {parsed:.2f}  verdicts {[r['verdict'] for r in out]}")
    if parsed < 0.75:
        failures.append(f"chen_cot_math500: parsed {parsed:.2f}")

    rows = [{"id": p["id"], "question": p["question"], "answer": p["answer_a"]} for p in pairs]
    out = judge.rate(llm, "yang_quality", rows, max_tokens=1024)
    parsed = statistics.mean(r["score"] is not None for r in out)
    print(f"{'yang_quality':24s} parsed {parsed:.2f}  scores {[r['score'] for r in out]}")
    if parsed < 0.75:
        failures.append(f"yang_quality: parsed {parsed:.2f}")

    out = judge.generate(llm, [{"id": p["id"], "question": p["question"]} for p in pairs], max_tokens=256)
    print(f"{'generation':24s} {out[0]['text'][:80]!r}")
    if not all(r["text"].strip() for r in out):
        failures.append("generation: empty text")

    print("PASS" if not failures else "FAIL\n  " + "\n  ".join(failures))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
