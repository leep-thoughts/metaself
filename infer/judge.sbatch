#!/bin/bash
# Generic judge / rater / generator run. Arguments pass straight to infer.judge:
#   sbatch $(gpu_flags 1) infer/judge.sbatch --model Qwen/Qwen2.5-7B-Instruct --template chen_direct_math500 \
#          --input data/x.jsonl --output data/y.jsonl --both-orders
# From the repo root, after `source infer/env.sh` (GPU flags: gpu_flags N; GPU count: python -m infer.cluster <hf id>):
#   sbatch $(gpu_flags 2) infer/judge.sbatch ... --tensor-parallel 2
#SBATCH --job-name=vllm-judge
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/infer/env.sh"   # submit from the repo root
export HF_HUB_OFFLINE=1   # weights must be cached (no internet on Narval compute nodes)
nvidia-smi -L
python -m infer.judge "$@"
