# Source this before any inference work:  source infer/env.sh
# (sourced, not executed -- module loads only stick in the current shell)
# Stack for the vLLM venv built by infer/setup_vllm.sh. Change both together.
# Cluster-specific values (paths, account, GPU type) come from infer/cluster.sh.
source "$(dirname "${BASH_SOURCE[0]}")/cluster.sh"

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.9 opencv/4.13.0 python/3.12.4 arrow/23.0.1

export VLLM_VENV=$WORK/venvs/vllm
source "$VLLM_VENV/bin/activate"

# Keep model weights and caches off /home (50 GB quota) and off $SLURM_TMPDIR (wiped).
export HF_HOME=$WORK/hf_home
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token   # gated models (Llama, Gemma); HF_HOME has no token
export XDG_CACHE_HOME=$WORK/xdg_cache
export PIP_CACHE_DIR=$WORK/cache/pip
export VLLM_CACHE_ROOT=$WORK/cache/vllm
export NLTK_DATA=$WORK/cache/nltk_data
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}   # set to 1 in jobs once weights are cached

cd "$METASELF_REPO"
