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
Resumable: rerun with the same --output to continue (rows saved every --chunk).
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
import os
import math
import re

from infer.prompts import TEMPLATES, messages, parse_rating

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


MIN_OUTPUT = 256      # free-form outputs: skip a row if fewer tokens than this fit after its prompt


def _chat(llm, convs, max_tokens, min_out, chat_kw=None, **sampling):
    """llm.chat with each request's output capped to what fits in the context window.
    Rows whose prompt leaves fewer than min_out tokens are not run: their output is None."""
    tok, limit = llm.get_tokenizer(), llm.model_config.max_model_len
    chat_kw = chat_kw or {}
    add_gen = chat_kw.get("add_generation_prompt", True)
    params, keep = [], []
    for i, c in enumerate(convs):
        text = tok.apply_chat_template(c, tokenize=False, add_generation_prompt=add_gen,
                                       continue_final_message=chat_kw.get("continue_final_message", False))
        room = limit - len(tok(text, add_special_tokens=False).input_ids) - 1
        if room >= min_out:
            keep.append(i)
            params.append(_sampling(max_tokens=min(max_tokens, room), **sampling))
    if len(keep) < len(convs):
        print(f"skipping {len(convs) - len(keep)} of {len(convs)} rows: prompt too long for max_model_len={limit}", flush=True)
    outs = llm.chat([convs[i] for i in keep], params, use_tqdm=True, **chat_kw) if keep else []
    full = [None] * len(convs)
    for i, o in zip(keep, outs):
        full[i] = o
    return full


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
    outs = _chat(llm, convs, 1, 1, temperature=0.0, logprobs=top_logprobs)
    results = []
    for r, o in zip(rows, outs):
        if o is None:
            results.append({**r, "skipped": "prompt_too_long"})
            continue
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
    outs = _chat(llm, convs, max_tokens, MIN_OUTPUT, temperature=temperature,
                 chat_kw={"add_generation_prompt": not long_cot, "continue_final_message": long_cot})
    results = []
    for r, o in zip(rows, outs):
        if o is None:
            results.append({**r, "skipped": "prompt_too_long", "verdict": None})
            continue
        text = o.outputs[0].text
        found = re.findall(t["verdict_regex"], _strip_think(text))
        verdict = found[-1] if found else None
        results.append({**r, "text": text, "verdict": verdict,
                        **{f"prob_{c}": float(verdict == c) if verdict else float("nan") for c in t["choices"]}})
    return results


def rate(llm, template: str, rows: list[dict], max_tokens: int = 2048) -> list[dict]:
    convs = [messages(template, question=r["question"], answer=r["answer"]) for r in rows]
    outs = _chat(llm, convs, max_tokens, MIN_OUTPUT, temperature=0.0)
    results = []
    for r, o in zip(rows, outs):
        if o is None:
            results.append({**r, "skipped": "prompt_too_long", "score": None})
            continue
        text = o.outputs[0].text
        score, dims = parse_rating(text, template)
        results.append({**r, "text": text, "score": score, "dims": dims,
                        "finish_reason": o.outputs[0].finish_reason})
    return results


def generate(llm, rows: list[dict], max_tokens: int = 2048, temperature: float = 0.0) -> list[dict]:
    convs = [([{"role": "system", "content": r["system"]}] if r.get("system") else [])
             + [{"role": "user", "content": r["question"]}] for r in rows]
    outs = _chat(llm, convs, max_tokens, MIN_OUTPUT, temperature=temperature)
    return [{**r, "text": o.outputs[0].text} if o is not None else {**r, "skipped": "prompt_too_long"}
            for r, o in zip(rows, outs)]


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
    ap.add_argument("--template", required=True, help=f"one of {sorted(TEMPLATES)}, 'none' (plain generation) or 'row' (per-row 'template' field)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--both-orders", action="store_true")
    ap.add_argument("--long-cot", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=500, help="rows per save")
    ap.add_argument("--pipeline-parallel", type=int, default=1,
                    help="split layers across GPUs; needed for pre-quantized bitsandbytes models")
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="cap the batch; large-vocab models (Command-R+, 256k) OOM in the sampler otherwise")
    args = ap.parse_args()

    with open(args.input) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    # --template row: each row names its own template (e.g. one Chen prompt per task)
    for r in rows:
        r["template"] = r.get("template") if args.template == "row" else args.template
    if any(t is None for t in (r["template"] for r in rows)):
        raise SystemExit("--template row needs a 'template' field on every row")
    groups = {}
    for r in rows:
        groups.setdefault(r["template"], []).append(r)
    modes = {t: ("gen" if t == "none" else TEMPLATES[t]["mode"]) for t in groups}
    if args.both_orders and any(m not in ("direct", "cot") for m in modes.values()):
        raise SystemExit("--both-orders only applies to pairwise templates")

    engine = {"tensor_parallel_size": args.tensor_parallel, "pipeline_parallel_size": args.pipeline_parallel}
    if args.max_model_len:
        engine["max_model_len"] = args.max_model_len
    if args.max_num_seqs:
        engine["max_num_seqs"] = args.max_num_seqs
    # resumable: rows already in --output (same id, game, template) are skipped; results are
    # appended every --chunk rows, so a timed-out job keeps its work. Resubmit to continue.
    key = lambda r: (r["id"], r.get("game"), r["template"])
    done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            done = {key(json.loads(l)) for l in f if l.strip()}
    work = []
    for template, group in groups.items():
        if args.both_orders:
            group = with_both_orders(group)
        todo = [r for r in group if key(r) not in done]
        if todo:
            work.append((template, todo))
    n_todo = sum(len(g) for _, g in work)
    print(f"{sum(len(g) for g in groups.values()) * (2 if args.both_orders else 1)} requests, "
          f"{len(done)} already in {args.output}, {n_todo} to run", flush=True)
    if not n_todo:
        return

    llm = load(args.model, **engine)
    extra = {"max_tokens": args.max_tokens} if args.max_tokens else {}
    n = 0
    for template, group in work:
        mode = modes[template]
        for i in range(0, len(group), args.chunk):
            chunk = group[i:i + args.chunk]
            if mode == "direct":
                res = judge_direct(llm, template, chunk)
            elif mode == "cot":
                res = judge_cot(llm, template, chunk, temperature=args.temperature, long_cot=args.long_cot, **extra)
            elif mode == "rating":
                res = rate(llm, template, chunk, **extra)
            else:
                res = generate(llm, chunk, temperature=args.temperature, **extra)
            with open(args.output, "a") as f:
                for r in _restore(res):
                    f.write(json.dumps({**r, "judge_model": args.model}) + "\n")
            n += len(chunk)
            print(f"saved {n}/{n_todo} -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
