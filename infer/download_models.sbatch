#!/bin/bash
# Download the self-hosted reference judges and Chen's 11 judges into $HF_HOME (no GPU).
# Resumable and idempotent: gated repos you haven't accepted are skipped with a message;
# resubmit after accepting them on huggingface.co.
#   sbatch infer/download_models.sbatch
#SBATCH --job-name=hf-download
#SBATCH --account=aip-lilimou
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/haobin4/metaself/logs/%x_%j.out

set -uo pipefail
source /scratch/haobin4/metaself/infer/env.sh
MODELS=$(python -c "from infer.yang_data import JUDGES; print(' '.join(r for r, _ in JUDGES.values()))")
for repo in allenai/OLMo-2-0325-32B-Instruct $MODELS; do
    echo "== $repo $(date +%T)"
    hf download "$repo" --exclude "original/*" --exclude "*.pth" > /dev/null 2> >(tail -2 >&2) \
        && echo "   done" || echo "   FAILED (gated? accept the license on huggingface.co, then resubmit)"
done
du -sh "$HF_HOME/hub"
