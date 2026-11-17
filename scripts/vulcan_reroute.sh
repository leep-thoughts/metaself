#!/bin/bash
# Vulcan side of moving the 70B jobs to Fir. Run ON VULCAN, repo root.
#   bash scripts/vulcan_reroute.sh cancel   cancel the stuck 70B jobs and everything waiting on them
#                                           (asks first; shows the list)
#   bash scripts/vulcan_reroute.sh hb       after Llama-70B's HealthBench answers are pulled from Fir:
#                                           5 smaller GPU judges + Aleph judges on the new answers + analysis
set -euo pipefail
cd "$(dirname "$0")/.."
source infer/env.sh > /dev/null
SKIP=llama-3.1-70b

case "${1:-}" in
cancel)
    JOBS=$(squeue -u "$USER" -h -o "%i %j %T" | awk '$2 ~ /^(yang-(qwen-2.5-72b|llama-3.1-70b|llama-3.3-70b)|ifgen-|ifjudge-|ifeval-aleph|analyses)$/ && $3 == "PENDING"')
    [ -n "$JOBS" ] || { echo "nothing to cancel"; exit 0; }
    echo "$JOBS"; read -rp "cancel these? [y/N] " a
    [ "$a" = y ] && scancel $(echo "$JOBS" | awk '{print $1}') && echo cancelled
    ;;
hb)
    [ -f infer/data/healthbench/answers_$SKIP.jsonl ] || { echo "first: bash scripts/sync_fir.sh pull (Llama-70B answers)"; exit 1; }
    IDS=()
    for m in $(python -c "from infer.ifeval_rubric import GPU_MODELS; print(' '.join(GPU_MODELS))"); do
        [ "$m" = "$SKIP" ] && continue
        g=$(python -c "from infer.ifeval_rubric import GPU_MODELS; print(GPU_MODELS['$m'][1])")
        id=$(RUBRIC_DATASET=healthbench sbatch --parsable --job-name=ifjudge-$m $(gpu_flags $g) --time=08:00:00 \
            infer/ifeval_gpu.sbatch gpu-judge "$m")
        echo "judge $m: job $id"; IDS+=("$id")
    done
    id=$(RUBRIC_DATASET=healthbench sbatch --parsable --job-name=hb-aleph infer/ifeval_aleph_vulcan.sbatch judge)
    echo "Aleph judges on the new answers: job $id"; IDS+=("$id")
    id=$(sbatch --parsable --dependency=afterany:$(IFS=:; echo "${IDS[*]}") scripts/analyses.sbatch healthbench)
    echo "analysis after all: job $id (rerun after Fir's Llama-70B verdicts are pulled)"
    ;;
*) sed -n 2,7p "$0"; exit 1 ;;
esac
