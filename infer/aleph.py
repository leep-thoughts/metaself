"""
Aleph (Vulcan's free in-house inference gateway) as a second backend for infer/, for
models it hosts (e.g. gpt-oss-120b). OpenAI-compatible; no GPU or Slurm job needed.

    set -a; source ~/.aleph_tyk.env; set +a          # puts TYK_KEY in the environment
    python -m infer.aleph --model gpt-oss-120b --template yang_quality \
        --input infer/data/yang_answers.jsonl --output infer/data/scores_gpt-oss-120b.jsonl

Same input / output rows as infer/judge.py (rating templates only for now). Resumable:
rows whose id is already in --output are skipped, so rerun after an interruption.
Aleph is a shared proof of concept: keep --workers small, and on 429/503 it waits as the
response asks (models that are scaled to zero take a few minutes to start).
Docs: https://inference.vulcan.alliancecan.ca/  (key: ~/.aleph_tyk.env, never commit it)
"""

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import APIStatusError, APITimeoutError, APIConnectionError, OpenAI

from infer.prompts import TEMPLATES, messages, parse_rating

BASE_URL = "https://inference.vulcan.alliancecan.ca/v1"


def client():
    key = os.environ.get("TYK_KEY")
    if not key:
        raise SystemExit("TYK_KEY not set: run  set -a; source ~/.aleph_tyk.env; set +a")
    return OpenAI(base_url=BASE_URL, api_key=key, timeout=300.0, max_retries=0)


def _wait_hint(err) -> float:
    """Seconds to wait from a 429/503 response: Retry-After header, else a number in the body."""
    try:
        h = err.response.headers.get("retry-after")
        if h:
            return float(h)
        m = re.search(r"(\d+)\s*(second|sec|s\b|minute|min)", err.response.text, re.I)
        if m:
            return float(m.group(1)) * (60 if m.group(2).lower().startswith("min") else 1)
    except Exception:  # noqa: BLE001
        pass
    return 60.0


def call(cl, model, msgs, max_tokens, reasoning_effort=None, attempts=8):
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    for i in range(attempts):
        try:
            r = cl.chat.completions.create(model=model, messages=msgs, max_tokens=max_tokens,
                                           temperature=0.0, **extra)
            c = r.choices[0]
            return {"text": c.message.content or "", "finish_reason": c.finish_reason,
                    "usage": r.usage.model_dump() if r.usage else None}
        except APIStatusError as e:
            if e.status_code in (429, 503):
                time.sleep(min(_wait_hint(e), 600))
                continue
            if e.status_code in (502, 504):
                time.sleep(30 * (i + 1))
                continue
            raise
        except (APITimeoutError, APIConnectionError):
            time.sleep(30 * (i + 1))
    raise RuntimeError(f"gave up after {attempts} attempts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--template", default="yang_quality")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--reasoning-effort", default=None, help="for reasoning models, e.g. low / medium")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if TEMPLATES[args.template]["mode"] != "rating":
        raise SystemExit("aleph.py handles rating templates; pairwise verdicts need first-token probabilities")

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    done = set()
    if os.path.exists(args.output):
        done = {json.loads(l)["id"] for l in open(args.output) if l.strip()}
    todo = [r for r in rows if r["id"] not in done][: args.limit]
    print(f"{len(rows)} rows, {len(done)} done, {len(todo)} to run on {args.model}", flush=True)

    cl, lock, t0, n = client(), threading.Lock(), time.time(), 0
    out = open(args.output, "a")

    def work(r):
        res = call(cl, args.model, messages(args.template, question=r["question"], answer=r["answer"]),
                   args.max_tokens, args.reasoning_effort)
        score, dims = parse_rating(res["text"], args.template)
        keep = {k: v for k, v in r.items() if k not in ("question", "answer")}
        return {**keep, "rater": args.model, "score": score, "dims": dims, "text": res["text"],
                "finish_reason": res["finish_reason"], "usage": res["usage"]}

    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(work, r) for r in todo]
        for f in as_completed(futs):
            rec = f.result()
            with lock:
                out.write(json.dumps(rec) + "\n")
                out.flush()
                n += 1
                if n % 50 == 0 or n == len(todo):
                    rate = n / (time.time() - t0)
                    print(f"{n}/{len(todo)}  {rate * 3600:.0f}/h  eta {(len(todo) - n) / rate / 3600:.1f}h", flush=True)
    out.close()


if __name__ == "__main__":
    main()
