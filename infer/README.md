# infer/ — local LLM inference (vLLM)

Not used by any result yet. For when we need our own judgments, e.g. Yang's null set,
rubric scores, or new judges.

## Setup (once, CPU node is fine)
```bash
bash infer/setup_vllm.sh        # builds /scratch/haobin4/venvs/vllm, writes infer/requirements.lock.txt
source infer/env.sh             # every session / job
python -m infer.check_prompts meta-llama/Meta-Llama-3-8B-Instruct   # CPU: chat templates + label tokens
```

## Dependencies
| Kind | What | Note |
|---|---|---|
| Modules | `StdEnv/2023 gcc/12.3 cuda/12.9 opencv/4.13.0 python/3.12.4 arrow/23.0.1` | opencv: the wheelhouse ships a placeholder that needs the module |
| Python | `vllm==0.25.0` (pulls torch, transformers), `pandas` | all from the Alliance wheelhouse (`--no-index`); 0.32.0 fails: needs `nccl-extensions`, not in wheelhouse |
| Slurm | `--account=aip-lilimou --gres=gpu:l40s:1` | L40S = 48 GB: 8–9B judges fit on 1 GPU; 27B needs 2 (`--tensor-parallel 2`); 70B needs 4 |
| HF | token at `~/.cache/huggingface/token` | gated: Llama-3.x, Gemma-2. Cached in `/scratch/haobin4/hf_home` already: Llama-3-8B/70B-Instruct, Mistral-7B v0.1/v0.3, Mistral-Large, Phi-3-medium, Aya-23-8B, Command-R+ |

## Use
```bash
sbatch infer/smoke_test.sbatch                 # GPU test, see infer/smoke_test.py for pass criteria
python -m infer.judge --model <hf id> --template chen_direct_math500 \
    --input pairs.jsonl --output out.jsonl --both-orders
```

## Prompt templates (`infer/prompts.py`)
| Key | Paper | Output read | Source of exact text |
|---|---|---|---|
| `chen_direct_*`, `chen_cot_*` | Chen et al. 2025 | first-token P over A/T/B; CoT parses `$$X$$` | Roytburg's reproduction of Chen App. B (Chen's repo has no code) |
| `yang_preference` | Yang et al. 2026 | first-token P over A/B | Table A.15 |
| `yang_structured` | Yang et al. 2026 (mitigation) | first-token P over A/B | Table A.16 |
| `yang_quality` | Yang et al. 2026 (benchmark judges) | parsed overall score 0–10 | Tables A.7–A.14, joined in order |
| `panickssery_compare` | Panickssery et al. 2024 | first-token P over 1/2 | Roytburg's CNN/XSUM replication (official bit.ly link dead) |

Known gaps for exact replication
- Chen: answer-generation prompts and max tokens not released (their answers are, in `data/llm-sp`).
- Yang: sampling settings, system/user split, and how Tables A.7–A.14 join aren't stated; no code.
- Panickssery: self-recognition prompt not copied yet (only the preference prompt).
- Label probabilities: we sum token variants ("A", " A"); Roytburg takes the max. Differences are tiny.

Checked on CPU (`check_prompts`), still unchecked on GPU
- Llama-3-8B: all templates render; every label is one token.
- Phi-3 (sentencepiece): letters are one token; digits "1"/"2" split into "▁" + "1", so
  `panickssery_compare` may lose its label mass. The smoke test's `verdict_mass` will show it.
- Mistral-7B-v0.3 cache is Mistral-native (`tokenizer.model.v3`, no HF tokenizer): load
  with `tokenizer_mode="mistral"` in vLLM.
