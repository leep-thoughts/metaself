"""
IFEval as a rubric self-preference benchmark (setup of Pombal et al. 2026, arXiv 2604.06996).

Data: 541 IFEval prompts with 834 checkable instructions. Rubric text per instruction from
Pombal's conversion (data/pombal-simple-evals/ifeval/converted.jsonl); instruction ids and
arguments from data/pombal-simple-evals/ifeval/raw.jsonl (same order).
True label: Google's IFEval checker, strict mode, as packaged in lm_eval 0.4.11
(lm_eval.tasks.ifeval) from the Alliance wheelhouse.
Judge prompt: Pombal's single-rubric (SR) grader prompt, verbatim; verdict = P(true) from
the logprobs of the true/false token in the JSON reply (falls back to the parsed value).

    python -m infer.ifeval_rubric generate --model gemma-4-31b        # Aleph
    python -m infer.ifeval_rubric label                                # true labels for all answer files
    python -m infer.ifeval_rubric judge --model gemma-4-31b            # Aleph; all answers x criteria
    python -m infer.ifeval_rubric gpu-generate --model llama-3.1-8b    # vLLM, in a GPU job
    python -m infer.ifeval_rubric gpu-judge --model llama-3.1-8b       # vLLM, in a GPU job

Files (infer/data/ifeval/): answers_<model>.jsonl, labels.jsonl, verdicts_<judge>.jsonl.
All steps are resumable (rows already in the output are skipped).
"""

import argparse
import glob
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "data", "pombal-simple-evals", "ifeval")
OUT = os.path.join(HERE, "data", "ifeval")

# Aleph models used as generators AND judges. Settings follow Pombal et al.: every model at its
# own defaults (sampling, reasoning), only the output limit set (MAX_TOKENS = 16,384, their
# max_completion_tokens). Their sampler's temperature is not forwarded to the API, so model
# defaults apply. A thinking-off run (reasoning_effort "none") is kept in infer/data/ifeval_nothink/.
ALEPH = {
    "gemma-4-31b":     {"family": "gemma"},
    "gemma-4-26b-a4b": {"family": "gemma"},
    "qwen38-27b":      {"family": "qwen"},
    "qwen35-122b":     {"family": "qwen"},
    "gpt-oss-120b":    {"family": "gptoss"},
    "gpt-oss-20b":     {"family": "gptoss"},
    "command-r-7b":    {"family": "cohere"},
    "aya-expanse-8b":  {"family": "cohere"},
}
MAX_TOKENS = 16384

GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on whether it follows the rubric item. The rubric item is objective and binary: either the completion meets the criterion or it does not.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with "criteria_met" field.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.

# Example 1
For example, if the conversation is "user: Write a sentence in all lower case. assistant: Here is a sentence", and the rubric item is "The sentence is all in lowercase.", you should return a json like this you should return a json like this:

```json
{
  "criteria_met": false
}
```

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()   # Pombal et al., simple-evals fork, ifeval.py (typos kept)


def prompts():
    raw = [json.loads(l) for l in open(os.path.join(SRC, "raw.jsonl"))]
    conv = [json.loads(l) for l in open(os.path.join(SRC, "converted.jsonl"))]
    out = []
    for r, c in zip(raw, conv):
        assert r["prompt"] == c["prompt"][0]["content"] and len(r["instruction_id_list"]) == len(c["rubrics"])
        out.append({"key": r["key"], "prompt": r["prompt"], "instruction_ids": r["instruction_id_list"],
                    "kwargs": r["kwargs"], "rubrics": [x["criterion"] for x in c["rubrics"]]})
    return out


def _read(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


# ------------------------------------------------------------------ Aleph calls
def _client():
    from openai import OpenAI
    key = os.environ.get("TYK_KEY")
    if not key:
        raise SystemExit("TYK_KEY not set: run  set -a; source ~/.aleph_tyk.env; set +a")
    return OpenAI(base_url="https://inference.vulcan.alliancecan.ca/v1", api_key=key, timeout=600.0, max_retries=0)


def _call(cl, model, messages, max_tokens, logprobs=False, attempts=10):
    from openai import APIConnectionError, APIStatusError, APITimeoutError
    from infer.aleph import _wait_hint
    kw = dict(model=model, messages=messages, max_tokens=max_tokens)   # model defaults otherwise
    if logprobs:
        kw.update(logprobs=True, top_logprobs=5)
    for i in range(attempts):
        try:
            return cl.chat.completions.create(**kw)
        except APIStatusError as e:
            if e.status_code in (429, 503):
                time.sleep(min(_wait_hint(e), 600))
            elif e.status_code in (500, 502, 504) and i < 2:
                time.sleep(30 * (i + 1))
            elif e.status_code in (502, 504):
                time.sleep(30 * (i + 1))
            else:
                raise   # 400 (prompt too long) or repeated 500: deterministic, caller records it
        except (APITimeoutError, APIConnectionError):
            time.sleep(30 * (i + 1))
    raise RuntimeError("gave up")


def _run(rows, fn, out_path, workers):
    lock, t0, n = threading.Lock(), time.time(), 0
    with open(out_path, "a") as f, ThreadPoolExecutor(workers) as pool:
        futs = [pool.submit(fn, r) for r in rows]
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n += 1
                if n % 100 == 0 or n == len(rows):
                    rate = n / (time.time() - t0)
                    print(f"{n}/{len(rows)}  {rate * 3600:.0f}/h  eta {(len(rows) - n) / rate / 3600:.1f}h", flush=True)


# ------------------------------------------------------------------ steps
def generate(model, workers, limit=None):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"answers_{model}.jsonl")
    done = {r["key"] for r in _read(path)}
    todo = [p for p in prompts() if p["key"] not in done][:limit]
    print(f"{model}: {len(done)} done, {len(todo)} to generate", flush=True)
    cl = _client()

    def work(p):
        r = _call(cl, model, [{"role": "user", "content": p["prompt"]}], max_tokens=MAX_TOKENS)
        c = r.choices[0]
        return {"key": p["key"], "model": model, "response": c.message.content or "",
                "finish_reason": c.finish_reason, "usage": r.usage.model_dump() if r.usage else None}
    _run(todo, work, path, workers)


def _checker():
    """lm_eval's IFEval checker without importing lm_eval.tasks (which needs datasets, etc.):
    register the package paths as bare namespaces, then import only the three ifeval modules."""
    import importlib
    import importlib.util
    import sys
    import types
    root = os.path.dirname(importlib.util.find_spec("lm_eval").origin)
    for name, path in (("lm_eval", root), ("lm_eval.tasks", os.path.join(root, "tasks")),
                       ("lm_eval.tasks.ifeval", os.path.join(root, "tasks", "ifeval"))):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [path]
            sys.modules[name] = mod
    return importlib.import_module("lm_eval.tasks.ifeval.instructions_registry")


def label():
    """True labels with the IFEval checker (strict), for every answer file."""
    reg = _checker()
    ps = {p["key"]: p for p in prompts()}
    path = os.path.join(OUT, "labels.jsonl")
    done = {(r["key"], r["model"]) for r in _read(path)}
    n = 0
    with open(path, "a") as f:
        for ap in sorted(glob.glob(os.path.join(OUT, "answers_*.jsonl"))):
            for a in _read(ap):
                if (a["key"], a["model"]) in done or a["finish_reason"] != "stop":
                    continue
                p, resp, follow = ps[a["key"]], a["response"], []
                for iid, kw in zip(p["instruction_ids"], p["kwargs"]):
                    ins = reg.INSTRUCTION_DICT[iid](iid)
                    ins.build_description(**{k: int(v) if isinstance(v, float) and v.is_integer() else v
                                                 for k, v in kw.items() if v is not None})
                    args = ins.get_instruction_args()
                    if args and "prompt" in args:
                        ins.build_description(prompt=p["prompt"])
                    follow.append(bool(resp.strip()) and bool(ins.check_following(resp)))
                f.write(json.dumps({"key": a["key"], "model": a["model"], "follow": follow}) + "\n")
                n += 1
    print(f"labelled {n} new answers -> {path}")


def _p_true(choice):
    """P(criteria_met = true) from the logprobs at the last true/false token; else parsed value."""
    lp = choice.logprobs.content if choice.logprobs and choice.logprobs.content else []
    for tok in reversed(lp):
        if tok.token.strip().lower() in ("true", "false"):
            pt = sum(math.exp(t.logprob) for t in tok.top_logprobs if t.token.strip().lower() == "true")
            pf = sum(math.exp(t.logprob) for t in tok.top_logprobs if t.token.strip().lower() == "false")
            if pt + pf > 0:
                return pt / (pt + pf), pt + pf
    m = re.search(r'"criteria_met"\s*:\s*(true|false)', choice.message.content or "", re.I)
    return (float(m.group(1).lower() == "true"), None) if m else (None, None)


def judge(model, workers, limit=None):
    from openai import APIStatusError
    ps = {p["key"]: p for p in prompts()}
    answers = [a for ap in sorted(glob.glob(os.path.join(OUT, "answers_*.jsonl"))) for a in _read(ap)
               if a["finish_reason"] == "stop" and a["response"].strip()]
    path = os.path.join(OUT, f"verdicts_{model}.jsonl")
    done = {(r["key"], r["generator"], r["criterion"]) for r in _read(path)}
    todo = [(a, i) for a in answers for i in range(len(ps[a["key"]]["rubrics"]))
            if (a["key"], a["model"], i) not in done][:limit]
    print(f"judge {model}: {len(answers)} answers, {len(done)} verdicts done, {len(todo)} to run", flush=True)
    cl = _client()

    def work(t):
        a, i = t
        p = ps[a["key"]]
        convo = f"user: {p['prompt']}\n\nassistant: {a['response']}"
        msg = GRADER_TEMPLATE.replace("<<conversation>>", convo).replace("<<rubric_item>>", p["rubrics"][i])
        try:
            r = _call(cl, model, [{"role": "user", "content": msg}], max_tokens=MAX_TOKENS, logprobs=True)
        except APIStatusError as e:   # recorded with v=None so a rerun skips it; analysis drops it
            return {"key": a["key"], "generator": a["model"], "criterion": i, "judge": model, "v": None,
                    "verdict_mass": None, "error": f"{e.status_code}: {str(e)[:200]}"}
        c = r.choices[0]
        v, mass = _p_true(c)
        return {"key": a["key"], "generator": a["model"], "criterion": i, "judge": model, "v": v,
                "verdict_mass": mass, "text": (c.message.content or "")[:200], "finish_reason": c.finish_reason,
                "usage": r.usage.model_dump() if r.usage else None}
    _run(todo, work, path, workers)


# ------------------------------------------------------------------ self-hosted models (vLLM, GPU)
# Chen et al.'s judges, as generators AND judges. name -> (HF id, GPUs); family = name prefix.
GPU_MODELS = {
    "qwen-2.5-7b": ("Qwen/Qwen2.5-7B-Instruct", 1), "qwen-2.5-32b": ("Qwen/Qwen2.5-32B-Instruct", 2),
    "llama-3.1-8b": ("meta-llama/Llama-3.1-8B-Instruct", 1), "llama-3.1-70b": ("meta-llama/Llama-3.1-70B-Instruct", 4),
    "gemma-2-9b": ("google/gemma-2-9b-it", 1), "gemma-2-27b": ("google/gemma-2-27b-it", 2),
}


def family(model):
    if model in ALEPH:
        return ALEPH[model]["family"]
    return model.split("-")[0]


def _vllm_p_true(comp):
    """P(true) at the last generated true/false token, from vLLM top-k logprobs."""
    for pos in reversed(comp.logprobs or []):
        pt = sum(math.exp(lp.logprob) for lp in pos.values() if (lp.decoded_token or "").strip().lower() == "true")
        pf = sum(math.exp(lp.logprob) for lp in pos.values() if (lp.decoded_token or "").strip().lower() == "false")
        best = max(pos.values(), key=lambda lp: lp.logprob)
        if (best.decoded_token or "").strip().lower() in ("true", "false") and pt + pf > 0:
            return pt / (pt + pf), pt + pf
    m = re.search(r'"criteria_met"\s*:\s*(true|false)', comp.text or "", re.I)
    return (float(m.group(1).lower() == "true"), None) if m else (None, None)


MAX_MODEL_LEN = {"gemma-2-9b": 8192, "gemma-2-27b": 8192}   # Gemma-2's window; others 24,576


def _defaults(llm):
    """The model's own default sampling (its generation_config), as a vLLM server would apply."""
    d = llm.get_default_sampling_params()
    return {k: getattr(d, k) for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")}


def gpu_generate(name, llm):
    from infer.judge import _chat
    path = os.path.join(OUT, f"answers_{name}.jsonl")
    done = {r["key"] for r in _read(path)}
    todo = [p for p in prompts() if p["key"] not in done]
    outs = _chat(llm, [[{"role": "user", "content": p["prompt"]}] for p in todo], MAX_TOKENS, 256, **_defaults(llm))
    with open(path, "a") as f:
        for p, o in zip(todo, outs):
            c = o.outputs[0] if o is not None else None
            f.write(json.dumps({"key": p["key"], "model": name, "response": c.text if c else "",
                                "finish_reason": c.finish_reason if c else "prompt_too_long"}) + "\n")
    print(f"{name}: wrote {len(todo)} answers -> {path}", flush=True)


def gpu_judge(name, llm, chunk=2000):
    from infer.judge import _chat
    ps = {p["key"]: p for p in prompts()}
    answers = [a for ap in sorted(glob.glob(os.path.join(OUT, "answers_*.jsonl"))) for a in _read(ap)
               if a["finish_reason"] == "stop" and a["response"].strip()]
    path = os.path.join(OUT, f"verdicts_{name}.jsonl")
    done = {(r["key"], r["generator"], r["criterion"]) for r in _read(path)}
    todo = [(a, i) for a in answers for i in range(len(ps[a["key"]]["rubrics"])) if (a["key"], a["model"], i) not in done]
    print(f"judge {name}: {len(answers)} answers, {len(done)} done, {len(todo)} to run", flush=True)
    for s in range(0, len(todo), chunk):
        part = todo[s:s + chunk]
        convs = []
        for a, i in part:
            p = ps[a["key"]]
            convo = f"user: {p['prompt']}\n\nassistant: {a['response']}"
            convs.append([{"role": "user", "content": GRADER_TEMPLATE.replace("<<conversation>>", convo).replace("<<rubric_item>>", p["rubrics"][i])}])
        outs = _chat(llm, convs, MAX_TOKENS, 32, logprobs=5, **_defaults(llm))
        with open(path, "a") as f:
            for (a, i), o in zip(part, outs):
                c = o.outputs[0] if o is not None else None
                v, mass = _vllm_p_true(c) if c is not None else (None, None)
                f.write(json.dumps({"key": a["key"], "generator": a["model"], "criterion": i, "judge": name, "v": v,
                                    "verdict_mass": mass, "text": (c.text if c else "")[:200],
                                    "finish_reason": c.finish_reason if c else "prompt_too_long"}) + "\n")
        print(f"saved {min(s + chunk, len(todo))}/{len(todo)}", flush=True)


def gpu_main(step, name):
    from infer.judge import load
    hf, gpus = GPU_MODELS[name]
    llm = load(hf, tensor_parallel_size=gpus, max_model_len=MAX_MODEL_LEN.get(name, 24576))
    os.makedirs(OUT, exist_ok=True)
    if step == "gpu-generate":
        gpu_generate(name, llm)
    else:
        gpu_judge(name, llm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["generate", "label", "judge", "gpu-generate", "gpu-judge"])
    ap.add_argument("--model")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.step.startswith("gpu-"):
        gpu_main(a.step, a.model)
    elif a.step == "generate":
        generate(a.model, a.workers, a.limit)
    elif a.step == "label":
        label()
    else:
        judge(a.model, a.workers, a.limit)
