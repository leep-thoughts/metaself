#!/bin/bash
# IFEval rubric run on Aleph (CPU job: HTTP calls only). Steps: every Aleph model answers the
# 541 prompts -> IFEval checker labels -> every Aleph model judges every answer x criterion.
# Resumable: resubmit to continue. Answers from GPU models (infer/ifeval_gpu_submit.sh) that
# exist when the judge step runs are judged too; rerun later to judge any added afterwards.
#   sbatch infer/ifeval_aleph_vulcan.sbatch [generate|judge|all]
#   RUBRIC_DATASET=healthbench sbatch infer/ifeval_aleph_vulcan.sbatch   # HealthBench (Pombal subset)
#SBATCH --job-name=ifeval-aleph
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
STEP=${1:-all}
source "${SLURM_SUBMIT_DIR:-.}/infer/env.sh"   # submit from the repo root
require_vulcan
set -a; source ~/.aleph_tyk.env; set +a
MODELS=$(python -c "from infer.ifeval_rubric import ALEPH; print(' '.join(ALEPH))")
DS=${RUBRIC_DATASET:-ifeval}   # ifeval | healthbench (RUBRIC_DATASET=healthbench sbatch ...)
mkdir -p logs/$DS

if [ "$STEP" = all ] || [ "$STEP" = generate ]; then
    for m in $MODELS; do
        python -m infer.ifeval_rubric generate --model "$m" --workers 8 > "logs/$DS/generate_$m.log" 2>&1 &
    done
    wait
    echo "generation done $(date +%T)"; tail -qn1 logs/$DS/generate_*.log
fi
python -m infer.ifeval_rubric label
if [ "$STEP" = all ] || [ "$STEP" = judge ]; then
    for m in $MODELS; do
        python -m infer.ifeval_rubric judge --model "$m" --workers 8 > "logs/$DS/judge_$m.log" 2>&1 &
    done
    wait
    echo "judging done $(date +%T)"; tail -qn1 logs/$DS/judge_*.log
fi
