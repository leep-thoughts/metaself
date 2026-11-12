#!/bin/bash
# IFEval rubric run, self-hosted models (GPU). Run from the repo root:
#   bash infer/ifeval_gpu_submit.sh
# Submits, with Slurm dependencies so you run it once:
#   1. one generation job per model (answers to the 541 prompts)
#   2. after ALL generations: one judge job per model (every answer x criterion, Aleph's included)
#   3. after ALL generations: CPU job that labels the new answers and has the Aleph models judge them
# Needs the Aleph generation (infer/ifeval_aleph.sbatch) finished before step 2 starts, so the
# GPU judges see those answers too; judge jobs are resumable, so resubmitting fills any gap.
set -uo pipefail
cd /scratch/haobin4/metaself
source infer/env.sh > /dev/null
MODELS=$(python -c "from infer.ifeval_rubric import GPU_MODELS; print(' '.join(GPU_MODELS))")
gpus() { python -c "from infer.ifeval_rubric import GPU_MODELS; print(GPU_MODELS['$1'][1])"; }

GEN_IDS=()
for m in $MODELS; do
    g=$(gpus "$m")
    id=$(sbatch --parsable --job-name=ifgen-$m --gres=gpu:l40s:$g --cpus-per-task=$((g * 4)) --mem=$((g * 32 + 32))G \
        --time=02:00:00 infer/ifeval_gpu.sbatch gpu-generate "$m")
    echo "generate $m: job $id"; GEN_IDS+=("$id")
done
DEP="afterok:$(IFS=:; echo "${GEN_IDS[*]}")"

for m in $MODELS; do
    g=$(gpus "$m")
    id=$(sbatch --parsable --dependency=$DEP --job-name=ifjudge-$m --gres=gpu:l40s:$g --cpus-per-task=$((g * 4)) \
        --mem=$((g * 32 + 32))G --time=08:00:00 infer/ifeval_gpu.sbatch gpu-judge "$m")
    echo "judge $m: job $id (waits for all generation)"
done

# also wait for any Aleph IFEval job still running, so two jobs never append to the same file
RUNNING=$(squeue -u "$USER" -n ifeval-aleph -h -o %i | paste -sd: -)
ADEP=$DEP${RUNNING:+,afterany:$RUNNING}
id=$(sbatch --parsable --dependency=$ADEP infer/ifeval_aleph.sbatch judge)
echo "Aleph judges on the new answers (CPU): job $id"
