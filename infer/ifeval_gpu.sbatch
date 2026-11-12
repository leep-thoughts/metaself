#!/bin/bash
# One self-hosted model on IFEval (GPU). Submitted by infer/ifeval_gpu_submit.sh, which sets
# --gres, --mem and --time per model.   Args: gpu-generate|gpu-judge <model name>
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/infer/env.sh"   # submit from the repo root
export HF_HUB_OFFLINE=1
nvidia-smi -L
python -m infer.ifeval_rubric "$1" --model "$2"
