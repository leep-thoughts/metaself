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

## Yang et al. on Chen's data: run order
| Step | Command | Where | Output |
|---|---|---|---|
| 0 | `python -m infer.yang_data answers` | CPU (done) | `data/yang_answers.jsonl`: 150 problems/task × 18 models = 8,100 answers |
| 1 | `sbatch infer/aleph_score.sbatch gpt-oss-120b medium` | CPU job, Aleph (free) | `data/scores_gpt-oss-120b.jsonl` |
| 1b | `sbatch infer/download_models.sbatch` | CPU job | weights in `$HF_HOME` (rerun after accepting gated licenses) |
| 2 | `bash infer/yang_raters_submit.sh` | **GPU, you submit** | `data/scores_command-r-plus.jsonl`, `data/scores_olmo-2-32b.jsonl` |
| 3 | `python -m infer.yang_data pairs` | CPU, after ALL score files are complete | `data/yang_judge_<judge>.jsonl`, `data/yang_self_chen.jsonl` |
| 4 | `bash infer/yang_judges_submit.sh` | **GPU, you submit** | `data/verdicts_<judge>.jsonl` |
| 5 | `python -m real.yang_real` | CPU | `real/yang_real_results.csv` |

Aleph key: `~/.aleph_tyk.env` (never commit). Aleph is a shared proof of concept; the client keeps 4 requests in flight.

## IFEval rubric run: order
| Step | Command | Where |
|---|---|---|
| 1 | `sbatch infer/ifeval_aleph.sbatch` | CPU job (Aleph): 8 Aleph models answer the 541 prompts, checker labels, 8 judges grade |
| 2 | `bash infer/ifeval_gpu_submit.sh` | **GPU, you submit** (after step 1's generation is done): 6 Chen models answer, then judge everything; Aleph judges re-run on the new answers |
| 3 | `python -m real.rubric_real` | CPU: per-judge measures with bootstrap CIs |

Simulator side: `python rubric_sweep.py` (synth/rubric.py, measures/rubric.py).
Checker: Google's IFEval code as packaged in `lm_eval==0.4.11` (wheelhouse), loaded without lm_eval's task framework.

## β calibration (synth/difficulty.py): run order
| Step | Command | Where | Output |
|---|---|---|---|
| 1 | `python -m infer.beta_data` | CPU (done) | `data/beta_judge_<judge>.jsonl`: ~1k pairs/judge, other models' answers (one right, one wrong), stratified by whether J solves the problem |
| 2 | `bash infer/beta_judges_submit.sh` | **GPU, you submit** | `data/beta_verdicts_<judge>.jsonl` (Chen direct prompt, both orders) |
| 3 | `python -m real.beta_fit` | CPU | `real/beta_fit_results.csv`, plus simulator slopes at the fitted β |
