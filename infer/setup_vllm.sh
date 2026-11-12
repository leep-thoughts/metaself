#!/bin/bash
# One-time build of the vLLM venv from the Alliance wheelhouse, per cluster. No GPU needed;
# run on a login node (needs internet for the NLTK tokenizer data the IFEval checker uses).
#   bash infer/setup_vllm.sh
# Notes:
#   - vllm 0.32.0 in the wheelhouse needs nccl-extensions, which isn't there -> use 0.25.0.
#   - The wheelhouse's opencv is a placeholder ("opencv-noinstall"); the opencv module provides it.
set -euo pipefail
cd "$(dirname "$0")/.."
source infer/cluster.sh
VLLM_VERSION=${VLLM_VERSION:-0.25.0}

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.9 opencv/4.13.0 python/3.12.4 arrow/23.0.1
export PIP_CACHE_DIR=$WORK/cache/pip
VENV=$WORK/venvs/vllm

rm -rf "$VENV"
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index "vllm==$VLLM_VERSION" pandas bitsandbytes langdetect nltk immutabledict absl-py
pip install --no-index --no-deps lm_eval==0.4.11      # IFEval checker only (infer/ifeval_rubric.py)
pip freeze > infer/requirements.lock.txt
python -c "import nltk; nltk.download('punkt', '$WORK/cache/nltk_data'); nltk.download('punkt_tab', '$WORK/cache/nltk_data')"
python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__)"
