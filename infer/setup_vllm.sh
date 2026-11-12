#!/bin/bash
# One-time build of the vLLM venv from the Alliance wheelhouse. No GPU needed.
#   bash infer/setup_vllm.sh
# Notes:
#   - vllm 0.32.0 in the wheelhouse needs nccl-extensions, which isn't there -> use 0.25.0.
#   - The wheelhouse's opencv is a placeholder ("opencv-noinstall"); the opencv module provides it.
set -euo pipefail
VLLM_VERSION=${VLLM_VERSION:-0.25.0}

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.9 opencv/4.13.0 python/3.12.4 arrow/23.0.1
export PIP_CACHE_DIR=/scratch/haobin4/cache/pip
VENV=/scratch/haobin4/venvs/vllm

rm -rf "$VENV"
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index "vllm==$VLLM_VERSION" pandas bitsandbytes   # bitsandbytes: 4-bit Command-R+
pip freeze > /scratch/haobin4/metaself/infer/requirements.lock.txt
python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__)"
