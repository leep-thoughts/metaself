"""
CPU-only check (no GPU, no vLLM engine): render every template through a model's chat
template and show how the verdict labels tokenize right after the generation prompt.

    python -m infer.check_prompts meta-llama/Meta-Llama-3-8B-Instruct

What to look for
  - each label is ONE token in at least one spelling ("A" or " A"); judge_direct needs that
  - the rendered prompt ends with the assistant header (generation prompt added)
  - models without a system role (e.g. Gemma-2) raise on panickssery_compare -> fold the
    system text into the user turn for those judges
"""

import sys

from transformers import AutoTokenizer

from infer.prompts import TEMPLATES, messages

FIELDS = dict(question="What is 2+2?", answer_a="4", answer_b="5", answer="4")


def main(model: str):
    print(f"== {model}")
    try:
        tok = AutoTokenizer.from_pretrained(model)
    except Exception as e:  # noqa: BLE001
        print(f"TOKENIZER LOAD ERROR: {str(e).splitlines()[0]}")
        return
    for name, t in TEMPLATES.items():
        try:
            text = tok.apply_chat_template(messages(name, **FIELDS), tokenize=False, add_generation_prompt=True)
        except Exception as e:  # noqa: BLE001 -- report and keep going
            print(f"{name:24s} CHAT TEMPLATE ERROR: {e}")
            continue
        n = len(tok(text, add_special_tokens=False).input_ids)
        line = f"{name:24s} {n:5d} tokens  ends with {text[-40:]!r}"
        if "choices" in t:
            spell = {}
            for c in t["choices"]:
                ids = {s: tok(text + s, add_special_tokens=False).input_ids[n:] for s in (c, " " + c)}
                spell[c] = {s: [tok.convert_ids_to_tokens(i) for i in v] for s, v in ids.items()}
            single = all(any(len(v) == 1 for v in s.values()) for s in spell.values())
            line += f"  labels single-token: {single}  {spell}"
        print(line)


if __name__ == "__main__":
    for m in sys.argv[1:] or ["meta-llama/Meta-Llama-3-8B-Instruct"]:
        main(m)
