#!/bin/bash
# GPU smoke test for the vLLM stack -- run once GPUs are available:
#   cd /scratch/haobin4/metaself && sbatch infer/smoke_test.sbatch [model]
# Default model is already cached in $HF_HOME, so the job needs no download.
#SBATCH --job-name=vllm-smoke
#SBATCH --account=aip-lilimou
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/haobin4/metaself/logs/%x_%j.out

set -euo pipefail
source /scratch/haobin4/metaself/infer/env.sh
export HF_HUB_OFFLINE=1
nvidia-smi -L
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda ok', torch.cuda.is_available())"
python -m infer.smoke_test --model "${1:-meta-llama/Meta-Llama-3-8B-Instruct}"
