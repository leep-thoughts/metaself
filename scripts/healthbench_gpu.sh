#!/bin/bash
# HealthBench on the 6 self-hosted models (GPU: you run this). From the repo root:
#   bash scripts/healthbench_gpu.sh
# Needs the Aleph HealthBench run finished (job hb-aleph: answers + verdicts from 8 Aleph models).
# Submits (infer/ifeval_gpu_submit.sh with RUBRIC_DATASET=healthbench):
#   1. GPU generation per model (100 prompts)
#   2. GPU judging per model, after all generation (every answer x criterion, Aleph's included)
#   3. CPU: Aleph judges grade the new answers
#   4. CPU, after all of the above: committee labels + measures (scripts/analyses.sbatch healthbench)
set -euo pipefail
cd "$(dirname "$0")/.."
source infer/env.sh > /dev/null
[ "$(ls infer/data/healthbench/answers_*.jsonl 2>/dev/null | wc -l)" -ge 8 ] || \
    { echo "Aleph HealthBench answers missing: RUBRIC_DATASET=healthbench sbatch infer/ifeval_aleph_vulcan.sbatch"; exit 1; }

IDS=$(RUBRIC_DATASET=healthbench bash infer/ifeval_gpu_submit.sh | tee /dev/stderr | sed -n 's/.*job \([0-9]*\).*/\1/p' | paste -sd: -)
[ -n "$IDS" ] || { echo "nothing submitted"; exit 1; }
id=$(sbatch --parsable --dependency=afterany:$IDS --export=ALL,RUBRIC_DATASET=healthbench scripts/analyses.sbatch healthbench)
echo "analysis (CPU, after everything): job $id -> real/rubric_real_healthbench.csv"
