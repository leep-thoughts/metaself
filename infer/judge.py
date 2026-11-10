"""
vLLM inference for judges, raters and generators.

    python -m infer.judge --model meta-llama/Meta-Llama-3-8B-Instruct --template chen_direct_math500 \
        --input pairs.jsonl --output verdicts.jsonl [--both-orders]

Input JSONL, one object per line:
    pairwise ("direct"/"cot" templates): id, question, answer_a, answer_b   (other keys are copied through)
    rating ("yang_quality"):             id, question, answer
    generation (--template none):        id, question  [, system]

Output adds, per row:
    direct  p_<choice> (first-token probability, unnormalized), verdict_mass (their sum),
            prob_<choice> (normalized over choices; NaN if verdict_mass == 0), top_token
    cot     text, verdict (parsed; None if missing), prob_<choice> one-hot -- as in Chen et al.
    rating  text, score (parsed overall score; None if missing)
    gen     text
With --both-orders each pair is judged twice: game 1 as given, game 2 with A and B swapped
(swapped rows keep the original id and get game=2; answer_a/answer_b stay as given).

Direct verdicts read the top-K logprobs at the first generated position (K=20, as in Chen
et al. and Roytburg et al.). Token variants of a label ("A", " A", "▁A") are summed.
verdict_mass shows how much probability the labels got: a low value means the model wanted
to start with something else (e.g. "The"), which is the failure in the Roytburg release
(real/roytburg.py, Gemma-2-27B). Filter on it instead of renormalizing silently.
"""

import argparse
import json
import math
import re

from infer.prompts import TEMPLATES, messages

DEFAULT_ENGINE = dict(
    dtype="auto",                  # bf16 for most current checkpoints (Roytburg used float16)
    gpu_memory_utilization=0.90,
    enable_prefix_caching=True,    # the same question/answers reappear across games and judges
    seed=0,
)
TOP_LOGPROBS = 20


def load(model: str, **engine_kwargs):
    from vllm import LLM
    return LLM(model=model, **{**DEFAULT_ENGINE, **engine_kwargs})


def _sampling(**kw):
    from vllm import SamplingParams
    return SamplingParams(**kw)


def _label_probs(top: dict, choices: list[str]) -> dict:
    """Sum the probability of every token in the top-K whose text is a label (after stripping)."""
    p = {c: 0.0 for c in choices}
    for lp in top.values():
        tok = (lp.decoded_token or "").replace("▁", " ").strip()
        if tok in p:
            p[tok] += math.exp(lp.logprob)
    return p


def judge_direct(llm, template: str, rows: list[dict], top_logprobs: int = TOP_LOGPROBS) -> list[dict]:
    choices = TEMPLATES[template]["choices"]
    convs = [messages(template, question=r["question"], answer_a=r["answer_a"], answer_b=r["answer_b"]) for r in rows]
    outs = llm.chat(convs, _sampling(temperature=0.0, max_tokens=1, logprobs=top_logprobs), use_tqdm=True)
    results = []
    for r, o in zip(rows, outs):
        comp = o.outputs[0]
        p = _label_probs(comp.logprobs[0], choices) if comp.logprobs else {c: 0.0 for c in choices}
        mass = sum(p.values())
        results.append({**r, **{f"p_{c}": p[c] for c in choices}, "verdict_mass": mass,
                        **{f"prob_{c}": (p[c] / mass if mass > 0 else float("nan")) for c in choices},
                        "top_token": comp.text})
    return results


def _strip_think(text: str) -> str:
    return text.split("</think>")[-1] if "</think>" in text else text


def judge_cot(llm, template: str, rows: list[dict], max_tokens: int = 4096, temperature: float = 0.0,
              long_cot: bool = False) -> list[dict]:
    """Free-form reasoning, then a parsed verdict. long_cot pre-fills "<think>" (Chen/Roytburg)."""
    t = TEMPLATES[template]
    convs = []
    for r in rows:
        m = messages(template, question=r["question"], answer_a=r["answer_a"], answer_b=r["answer_b"])
        if long_cot:
            m[-1]["content"] += "\n"
            m.append({"role": "assistant", "content": "<think>"})
        convs.append(m)
    outs = llm.chat(convs, _sampling(temperature=temperature, max_tokens=max_tokens),
                    add_generation_prompt=not long_cot, continue_final_message=long_cot, use_tqdm=True)
    results = []
    for r, o in zip(rows, outs):
        text = o.outputs[0].text
        found = re.findall(t["verdict_regex"], _strip_think(text))
        verdict = found[-1] if found else None
        results.append({**r, "text": text, "verdict": verdict,
                        **{f"prob_{c}": float(verdict == c) if verdict else float("nan") for c in t["choices"]}})
    return results


def rate(llm, template: str, rows: list[dict], max_tokens: int = 2048) -> list[dict]:
    t = TEMPLATES[template]
    convs = [messages(template, question=r["question"], answer=r["answer"]) for r in rows]
    outs = llm.chat(convs, _sampling(temperature=0.0, max_tokens=max_tokens), use_tqdm=True)
    results = []
    for r, o in zip(rows, outs):
        text = o.outputs[0].text
        m = re.search(t["score_regex"], _strip_think(text))
        results.append({**r, "text": text, "score": float(m.group(1)) if m else None})
    return results


def generate(llm, rows: list[dict], max_tokens: int = 2048, temperature: float = 0.0) -> list[dict]:
    convs = [([{"role": "system", "content": r["system"]}] if r.get("system") else [])
             + [{"role": "user", "content": r["question"]}] for r in rows]
    outs = llm.chat(convs, _sampling(temperature=temperature, max_tokens=max_tokens), use_tqdm=True)
    return [{**r, "text": o.outputs[0].text} for r, o in zip(rows, outs)]


def with_both_orders(rows: list[dict]) -> list[dict]:
    """Game 1 as given, game 2 with the answers swapped in the prompt."""
    out = []
    for r in rows:
        out.append({**r, "game": 1})
        out.append({**r, "game": 2, "answer_a": r["answer_b"], "answer_b": r["answer_a"],
                    "_orig_a": r["answer_a"], "_orig_b": r["answer_b"]})
    return out


def _restore(rows: list[dict]) -> list[dict]:
    for r in rows:
        if r.get("game") == 2:
            r["answer_a"], r["answer_b"] = r.pop("_orig_a"), r.pop("_orig_b")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--template", required=True, help=f"one of {sorted(TEMPLATES)} or 'none' (plain generation)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--both-orders", action="store_true")
    ap.add_argument("--long-cot", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--tensor-parallel", type=int, default=1)
    args = ap.parse_args()

    with open(args.input) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    mode = "gen" if args.template == "none" else TEMPLATES[args.template]["mode"]
    if args.both_orders:
        if mode not in ("direct", "cot"):
            raise SystemExit("--both-orders only applies to pairwise templates")
        rows = with_both_orders(rows)

    engine = {"tensor_parallel_size": args.tensor_parallel}
    if args.max_model_len:
        engine["max_model_len"] = args.max_model_len
    llm = load(args.model, **engine)
    extra = {"max_tokens": args.max_tokens} if args.max_tokens else {}
    if mode == "direct":
        out = judge_direct(llm, args.template, rows)
    elif mode == "cot":
        out = judge_cot(llm, args.template, rows, temperature=args.temperature, long_cot=args.long_cot, **extra)
    elif mode == "rating":
        out = rate(llm, args.template, rows, **extra)
    else:
        out = generate(llm, rows, temperature=args.temperature, **extra)
    out = _restore(out)

    with open(args.output, "w") as f:
        for r in out:
            f.write(json.dumps({**r, "judge_model": args.model, "template": args.template}) + "\n")
    print(f"wrote {len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
