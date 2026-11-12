#!/bin/bash
# beta calibration (GPU): each Chen judge, Chen's direct prompt, both orders, judges pairs
# written by OTHER models (one correct, one wrong), on problems it solves and problems it fails.
# Run from the repo root:
#   bash infer/beta_judges_submit.sh                           # all 11 judges
#   bash infer/beta_judges_submit.sh qwen-2.5-7b llama-3.1-8b  # only these
# Needs: python -m infer.beta_data  (CPU, done) -> infer/data/beta_judge_<judge>.jsonl
# Writes infer/data/beta_verdicts_<judge>.jsonl (resumable). Then: python -m real.beta_fit
# ~1k pairs x 2 orders per judge, one verdict token each: minutes per judge once loaded.
set -euo pipefail
cd /scratch/haobin4/metaself
source infer/env.sh > /dev/null
JUDGES=${*:-$(python -c "from infer.yang_data import JUDGES; print(' '.join(JUDGES))")}

for j in $JUDGES; do
    in=infer/data/beta_judge_$j.jsonl
    [ -f "$in" ] || { echo "skip $j: $in missing (run python -m infer.beta_data)"; continue; }
    read -r repo gpus < <(python -c "from infer.yang_data import JUDGES; r, g = JUDGES['$j']; print(r, g)")
    id=$(sbatch --parsable --job-name=beta-$j --gres=gpu:l40s:$gpus --cpus-per-task=$((gpus * 4)) \
        --mem=$((gpus * 32 + 32))G --time=02:00:00 --export=ALL,HF_HUB_OFFLINE=1 \
        infer/judge.sbatch --model "$repo" --template row --both-orders --tensor-parallel "$gpus" \
        --max-model-len 8192 --input "$in" --output infer/data/beta_verdicts_$j.jsonl)
    echo "beta $j: job $id"
done
