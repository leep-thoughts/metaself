#!/bin/bash
# Same-item pairs (GPU): each Chen judge rates every target against the same 3 opponents
# on the same 300 problems (infer/samepair_data.py), plus its own answers.
# Run from the repo root:
#   bash infer/samepair_judges_submit.sh                           # all 11 judges
#   bash infer/samepair_judges_submit.sh qwen-2.5-7b llama-3.1-8b  # only these
# Needs: python -m infer.samepair_data  (CPU, done) -> infer/data/samepair_judge_<judge>.jsonl
# Writes infer/data/samepair_verdicts_<judge>.jsonl (resumable). Analysis script follows once verdicts exist.
# 6.3k-9k pairs x 2 orders per judge (about 2-2.5x the four-kinds runs, which took 3-21 min per judge).
set -euo pipefail
cd "$(dirname "$0")/.."
source infer/env.sh > /dev/null
JUDGES=${*:-$(python -c "from infer.yang_data import JUDGES; print(' '.join(JUDGES))")}

for j in $JUDGES; do
    in=infer/data/samepair_judge_$j.jsonl
    [ -f "$in" ] || { echo "skip $j: $in missing (run python -m infer.samepair_data)"; continue; }
    read -r repo gpus < <(python -c "from infer.yang_data import JUDGES; r, g = JUDGES['$j']; print(r, g)")
    [ "$gpus" -gt 0 ] || { echo "skip $j: does not fit on one $GPU_TYPE node (run it on Vulcan)"; continue; }
    id=$(sbatch --parsable --job-name=samepair-$j $(gpu_flags $gpus) --time=02:00:00 --export=ALL,HF_HUB_OFFLINE=1 \
        infer/judge.sbatch --model "$repo" --template row --both-orders --tensor-parallel "$gpus" \
        --max-model-len 8192 --input "$in" --output infer/data/samepair_verdicts_$j.jsonl)
    echo "samepair $j: job $id"
done
