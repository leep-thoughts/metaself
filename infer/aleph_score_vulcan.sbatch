#!/bin/bash
# Quality scores from a model hosted on Aleph (no GPU: HTTP calls only). Resumable: resubmit to continue.
#   sbatch infer/aleph_score_vulcan.sbatch [model] [reasoning_effort]
#SBATCH --job-name=aleph-score
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
MODEL=${1:-gpt-oss-120b}
EFFORT=${2:-medium}
source "${SLURM_SUBMIT_DIR:-.}/infer/env.sh"   # submit from the repo root
require_vulcan
set -a; source ~/.aleph_tyk.env; set +a
python -m infer.aleph --model "$MODEL" --template yang_quality --reasoning-effort "$EFFORT" --workers 4 \
    --input infer/data/yang_answers.jsonl --output "infer/data/scores_${MODEL}.jsonl"
