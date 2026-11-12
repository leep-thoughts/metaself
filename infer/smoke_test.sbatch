#!/bin/bash
# GPU smoke test for the vLLM stack -- run once GPUs are available:
#   source infer/env.sh && sbatch $(gpu_flags 1) infer/smoke_test.sbatch [model]
# Default model is already cached in $HF_HOME, so the job needs no download.
#SBATCH --job-name=vllm-smoke
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/infer/env.sh"   # submit from the repo root
export HF_HUB_OFFLINE=1
nvidia-smi -L
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda ok', torch.cuda.is_available())"
python -m infer.smoke_test --model "${1:-meta-llama/Meta-Llama-3-8B-Instruct}"
