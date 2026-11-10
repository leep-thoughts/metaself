#!/bin/bash
# Generic judge / rater / generator run. Arguments pass straight to infer.judge:
#   sbatch infer/judge.sbatch --model Qwen/Qwen2.5-7B-Instruct --template chen_direct_math500 \
#          --input data/x.jsonl --output data/y.jsonl --both-orders
# For 27B-class models: sbatch --gres=gpu:l40s:2 infer/judge.sbatch ... --tensor-parallel 2
# For 70B-class models: sbatch --gres=gpu:l40s:4 --mem=128G infer/judge.sbatch ... --tensor-parallel 4
#SBATCH --job-name=vllm-judge
#SBATCH --account=aip-lilimou
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/haobin4/metaself/logs/%x_%j.out

set -euo pipefail
source /scratch/haobin4/metaself/infer/env.sh
nvidia-smi -L
python -m infer.judge "$@"
