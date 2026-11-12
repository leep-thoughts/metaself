#!/bin/bash
# One self-hosted model on IFEval (GPU). Submitted by infer/ifeval_gpu_submit.sh, which sets
# --gres, --mem and --time per model.   Args: gpu-generate|gpu-judge <model name>
#SBATCH --account=aip-lilimou
#SBATCH --output=/scratch/haobin4/metaself/logs/%x_%j.out

set -euo pipefail
source /scratch/haobin4/metaself/infer/env.sh
export HF_HUB_OFFLINE=1
nvidia-smi -L
python -m infer.ifeval_rubric "$1" --model "$2"
