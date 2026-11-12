#!/bin/bash
# STEP 4 of Yang on Chen's data (GPU): each Chen judge, with Yang's A/B prompt and both orders,
# judges (a) its own answer vs an opponent's and (b) two OTHER models' answers (Yang's null set),
# all equal quality. Run from the repo root:
#   bash infer/yang_judges_submit.sh                           # every judge whose input exists
#   bash infer/yang_judges_submit.sh qwen-2.5-7b llama-3.1-8b  # only these
# Needs step 3 first: python -m infer.yang_data pairs  (after ALL score files are complete),
# which writes infer/data/yang_judge_<judge>.jsonl. Writes infer/data/verdicts_<judge>.jsonl.
# Gated models (Llama-3.3-70B, Gemma-2) need their licenses accepted and the download rerun.
set -euo pipefail
cd /scratch/haobin4/metaself
source infer/env.sh > /dev/null
JUDGES=${*:-$(python -c "from infer.yang_data import JUDGES; print(' '.join(JUDGES))")}

for j in $JUDGES; do
    in=infer/data/yang_judge_$j.jsonl
    [ -f "$in" ] || { echo "skip $j: $in missing (run python -m infer.yang_data pairs)"; continue; }
    read -r repo gpus < <(python -c "from infer.yang_data import JUDGES; r, g = JUDGES['$j']; print(r, g)")
    mem=$((gpus * 32 + 32))G
    sbatch --job-name=yang-$j --gres=gpu:l40s:$gpus --cpus-per-task=$((gpus * 4)) --mem=$mem --time=06:00:00 \
        infer/judge.sbatch --model "$repo" --template row --both-orders --tensor-parallel "$gpus" \
        --max-model-len 8192 --input "$in" --output infer/data/verdicts_$j.jsonl
done
