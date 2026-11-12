#!/bin/bash
# IFEval rubric run, self-hosted models (GPU). Run from the repo root:
#   bash infer/ifeval_gpu_submit.sh
# Submits, with Slurm dependencies so you run it once:
#   1. one generation job per model (answers to the 541 prompts)
#   2. after ALL generations: one judge job per model (every answer x criterion, Aleph's included)
#   3. after ALL generations: CPU job that labels the new answers and has the Aleph models judge them
# Needs the Aleph generation (infer/ifeval_aleph_vulcan.sbatch) finished before step 2 starts, so the
# GPU judges see those answers too; judge jobs are resumable, so resubmitting fills any gap.
set -uo pipefail
cd "$(dirname "$0")/.."
source infer/env.sh > /dev/null
MODELS=$(python -c "from infer.ifeval_rubric import GPU_MODELS; print(' '.join(GPU_MODELS))")
gpus() { python -c "from infer.ifeval_rubric import GPU_MODELS; print(GPU_MODELS['$1'][1])"; }

GEN_IDS=()
for m in $MODELS; do
    g=$(gpus "$m")
    [ "$g" -gt 0 ] || { echo "skip $m: does not fit on one $GPU_TYPE node (run it on Vulcan)"; continue; }
    id=$(sbatch --parsable --job-name=ifgen-$m $(gpu_flags $g) --time=02:00:00 infer/ifeval_gpu.sbatch gpu-generate "$m")
    echo "generate $m: job $id"; GEN_IDS+=("$id")
done
DEP="afterok:$(IFS=:; echo "${GEN_IDS[*]}")"

for m in $MODELS; do
    g=$(gpus "$m")
    [ "$g" -gt 0 ] || { echo "skip $m: does not fit on one $GPU_TYPE node (run it on Vulcan)"; continue; }
    id=$(sbatch --parsable --dependency=$DEP --job-name=ifjudge-$m $(gpu_flags $g) --time=08:00:00 infer/ifeval_gpu.sbatch gpu-judge "$m")
    echo "judge $m: job $id (waits for all generation)"
done

# Aleph part: Vulcan only. Elsewhere, sync the new answers back and run it on Vulcan.
if [ "${CC_CLUSTER:-}" != vulcan ]; then
    echo "not on Vulcan: after these jobs, copy infer/data/ifeval/answers_* to Vulcan and run sbatch infer/ifeval_aleph_vulcan.sbatch judge"
    exit 0
fi
# also wait for any Aleph IFEval job still running, so two jobs never append to the same file
RUNNING=$(squeue -u "$USER" -n ifeval-aleph -h -o %i | paste -sd: -)
ADEP=$DEP${RUNNING:+,afterany:$RUNNING}
id=$(sbatch --parsable --dependency=$ADEP infer/ifeval_aleph_vulcan.sbatch judge)
echo "Aleph judges on the new answers (CPU): job $id"
