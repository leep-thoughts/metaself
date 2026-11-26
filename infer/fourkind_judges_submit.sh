#!/bin/bash
# Four kinds of answer pairs (GPU): each Chen judge, Chen's direct prompt, both orders, on pairs
# where A is right/wrong and B is right/wrong (infer/fourkind_data.py), plus its own answers.
# Run from the repo root:
#   bash infer/fourkind_judges_submit.sh                           # all 11 judges
#   bash infer/fourkind_judges_submit.sh qwen-2.5-7b llama-3.1-8b  # only these
# Needs: python -m infer.fourkind_data  (CPU, done) -> infer/data/fourkind_judge_<judge>.jsonl
# Writes infer/data/fourkind_verdicts_<judge>.jsonl (resumable). Then: python -m real.fourkind
# ~3.6k pairs x 2 orders per judge, one verdict token each (about 3.5x the beta runs).
set -euo pipefail
cd "$(dirname "$0")/.."
source infer/env.sh > /dev/null
JUDGES=${*:-$(python -c "from infer.yang_data import JUDGES; print(' '.join(JUDGES))")}

for j in $JUDGES; do
    in=infer/data/fourkind_judge_$j.jsonl
    [ -f "$in" ] || { echo "skip $j: $in missing (run python -m infer.fourkind_data)"; continue; }
    read -r repo gpus < <(python -c "from infer.yang_data import JUDGES; r, g = JUDGES['$j']; print(r, g)")
    [ "$gpus" -gt 0 ] || { echo "skip $j: does not fit on one $GPU_TYPE node (run it on Vulcan)"; continue; }
    id=$(sbatch --parsable --job-name=fourkind-$j $(gpu_flags $gpus) --time=05:00:00 --export=ALL,HF_HUB_OFFLINE=1 \
        infer/judge.sbatch --model "$repo" --template row --both-orders --tensor-parallel "$gpus" \
        --max-model-len 8192 --input "$in" --output infer/data/fourkind_verdicts_$j.jsonl)
    echo "fourkind $j: job $id"
done
