#!/bin/bash
# Download the self-hosted reference judges and Chen's 11 judges into $HF_HOME (no GPU).
# Resumable and idempotent: gated repos you haven't accepted are skipped with a message;
# resubmit after accepting them on huggingface.co.
#   sbatch infer/download_models.sbatch      Vulcan (compute nodes have internet)
#   bash infer/download_models.sbatch        Narval: on the LOGIN node (no internet on compute nodes)
#SBATCH --job-name=hf-download
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/infer/env.sh"   # submit from the repo root
if [ -n "${SLURM_JOB_ID:-}" ] && [ "${COMPUTE_INTERNET:-1}" = 0 ]; then
    echo "no internet on $CC_CLUSTER compute nodes: run  bash infer/download_models.sbatch  on the login node" >&2; exit 1
fi
# only models that fit on one GPU node of this cluster (70B skipped on Narval)
MODELS=$(python -c "from infer.yang_data import JUDGES; print(' '.join(r for r, g in JUDGES.values() if g))")
for repo in allenai/OLMo-2-0325-32B-Instruct $MODELS; do
    echo "== $repo $(date +%T)"
    hf download "$repo" --exclude "original/*" --exclude "*.pth" > /dev/null 2> >(tail -2 >&2) \
        && echo "   done" || echo "   FAILED (gated? accept the license on huggingface.co, then resubmit)"
done
du -sh "$HF_HOME/hub"
