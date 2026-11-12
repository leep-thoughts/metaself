# infer/ — local LLM inference (vLLM)

Not used by any result yet. For when we need our own judgments, e.g. Yang's null set,
rubric scores, or new judges.

## Clusters
Everything cluster-specific lives in `infer/clusters/<name>.sh`, picked by `$CC_CLUSTER`
(GPU type and memory, CPU/RAM per GPU, Slurm account, compute-node internet). Paths come from
`$SCRATCH`. Per-machine overrides (e.g. your Narval account): `infer/clusters/local.sh`
(gitignored; copy `local.sh.example`). `python -m infer.cluster <hf id>` gives a model's GPU count
on this cluster; submit scripts skip models that don't fit on one node.

| | Vulcan | Narval |
|---|---|---|
| GPUs per node | 4× L40S 48 GB | 4× A100 40 GB |
| 70B models (Llama-3.x-70B, Qwen-72B) | 4 GPUs | don't fit → run on Vulcan |
| 14B / 32B | 1 / 2 GPUs | 2 / 4 GPUs |
| Compute-node internet | yes | no: download on the login node |
| Aleph (`*_vulcan.sbatch`) | yes | no (scripts refuse) |

Moving to a new cluster (Narval):
```bash
git clone <repo> && cd metaself
cp infer/clusters/local.sh.example infer/clusters/local.sh   # set ACCOUNT (sshare -U -u $USER)
bash infer/setup_vllm.sh                 # login node: venv in $SCRATCH/venvs/vllm + NLTK data
# HF token: ~/.cache/huggingface/token (same account, licenses already accepted)
bash infer/download_models.sbatch        # login node; skips models that don't fit here
# data not in git: from Vulcan, rsync data/ and the infer/data/ inputs the jobs read, e.g.
#   rsync -a data/ infer/data/ narval.alliancecan.ca:<repo>/...   (and outputs back afterwards)
```
Submit from the repo root (job logs go to `logs/`, relative to it). Analysis, labelling and the
Aleph jobs stay on Vulcan: sync GPU outputs (`infer/data/*verdicts*`, `scores_*`, `ifeval/answers_*`) back.

## Setup (once per cluster, login node)
```bash
bash infer/setup_vllm.sh        # builds $SCRATCH/venvs/vllm, writes infer/requirements.lock.txt
source infer/env.sh             # every session / job (loads infer/cluster.sh too)
python -m infer.check_prompts meta-llama/Meta-Llama-3-8B-Instruct   # CPU: chat templates + label tokens
```

## Dependencies
| Kind | What | Note |
|---|---|---|
| Modules | `StdEnv/2023 gcc/12.3 cuda/12.9 opencv/4.13.0 python/3.12.4 arrow/23.0.1` | same on every Alliance cluster (CVMFS); opencv: the wheelhouse ships a placeholder that needs the module |
| Python | `vllm==0.25.0` (pulls torch, transformers), `pandas`, `bitsandbytes`, IFEval checker deps (`lm_eval==0.4.11 --no-deps`, `langdetect`, `nltk`, `immutabledict`, `absl-py`) | all from the Alliance wheelhouse (`--no-index`); 0.32.0 fails: needs `nccl-extensions`, not in wheelhouse |
| Slurm | account + GPU flags from `infer/cluster.sh` (`gpu_flags N`) | |
| HF | token at `~/.cache/huggingface/token` | gated: Llama-3.x, Gemma-2. Weights in `$SCRATCH/hf_home` |

## Use
```bash
sbatch $(gpu_flags 1) infer/smoke_test.sbatch   # GPU test, see infer/smoke_test.py for pass criteria
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
| 1 | `sbatch infer/aleph_score_vulcan.sbatch gpt-oss-120b medium` | CPU job, Aleph (Vulcan) | `data/scores_gpt-oss-120b.jsonl` |
| 1b | `sbatch infer/download_models.sbatch` (Narval: `bash`, login node) | CPU | weights in `$HF_HOME` (rerun after accepting gated licenses) |
| 2 | `bash infer/yang_raters_submit.sh` | **GPU, you submit** | `data/scores_command-r-plus.jsonl`, `data/scores_olmo-2-32b.jsonl` |
| 3 | `python -m infer.yang_data pairs` | CPU, after ALL score files are complete | `data/yang_judge_<judge>.jsonl`, `data/yang_self_chen.jsonl` |
| 4 | `bash infer/yang_judges_submit.sh` | **GPU, you submit** | `data/verdicts_<judge>.jsonl` |
| 5 | `python -m real.yang_real` | CPU | `real/yang_real_results.csv` |

Aleph key: `~/.aleph_tyk.env` (never commit). Aleph is a shared proof of concept; the client keeps 4 requests in flight.

## IFEval rubric run: order
| Step | Command | Where |
|---|---|---|
| 1 | `sbatch infer/ifeval_aleph_vulcan.sbatch` | CPU job (Aleph, Vulcan): 8 Aleph models answer the 541 prompts, checker labels, 8 judges grade |
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
