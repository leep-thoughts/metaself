#!/bin/bash
# STEP 2 of Yang on Chen's data: self-hosted quality scores (GPU). Run from the repo root:
#   bash infer/yang_raters_submit.sh                  # both
#   bash infer/yang_raters_submit.sh command-r-plus   # just one (resumes from its output file)
# Needs: infer/data/yang_answers.jsonl (python -m infer.yang_data answers) and both models
# downloaded (infer/download_models.sbatch). Writes infer/data/scores_<name>.jsonl.
# Step 1 (gpt-oss-120b on Aleph) runs separately, on Vulcan: infer/aleph_score_vulcan.sbatch.
# Rows whose prompt does not fit the context window are written with score null (2 of 8,100).
set -uo pipefail
cd "$(dirname "$0")/.."
source infer/env.sh > /dev/null
COMMON="--template yang_quality --input infer/data/yang_answers.jsonl"
WHICH=${1:-all}

# Command-R+ (Cohere, 104B, pre-quantized bitsandbytes 4-bit, ~63 GB). vLLM cannot split
# pre-quantized bnb weights by tensor parallelism, so the GPUs split the layers (pipeline).
# Slow (~6 s per answer, ~14 h for 8,100); output is saved every 500 rows and resumes.
# --max-num-seqs 64: the default batch of 512 x 256k-token vocab OOMed in the sampler (job 2269458).
G=$(python -m infer.cluster CohereLabs/c4ai-command-r-plus-4bit)
[ "$WHICH" = all ] || [ "$WHICH" = command-r-plus ] && \
sbatch --job-name=rate-cmdrplus $(gpu_flags $G) --time=24:00:00 --export=ALL,HF_HUB_OFFLINE=1 \
    infer/judge.sbatch --model CohereLabs/c4ai-command-r-plus-4bit $COMMON \
    --pipeline-parallel $G --max-model-len 8192 --max-tokens 1536 --max-num-seqs 64 \
    --output infer/data/scores_command-r-plus.jsonl

# OLMo-2-32B-Instruct (AI2, bf16, ~64 GB). Its context window is 4,096 tokens.
G=$(python -m infer.cluster allenai/OLMo-2-0325-32B-Instruct)
[ "$WHICH" = all ] || [ "$WHICH" = olmo-2-32b ] && \
sbatch --job-name=rate-olmo32b $(gpu_flags $G) --time=12:00:00 --export=ALL,HF_HUB_OFFLINE=1 \
    infer/judge.sbatch --model allenai/OLMo-2-0325-32B-Instruct $COMMON \
    --tensor-parallel $G --max-model-len 4096 --max-tokens 1024 \
    --output infer/data/scores_olmo-2-32b.jsonl
