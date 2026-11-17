#!/bin/bash
# Yang et al.'s measure on Chen's data, end to end (GPU judging: you run this). From the repo root:
#   bash scripts/yang.sh                          # all judges that fit on this cluster
#   bash scripts/yang.sh qwen-2.5-7b gemma-2-9b   # only these
# 1. CPU, here: equal-quality pairs from the three raters' scores (gpt-oss-120b, OLMo-2-32B,
#    Command-R+; score = mean of the three) -> infer/data/yang_judge_<judge>.jsonl
#    (skipped if those files exist; delete them to rebuild. Drop a rater, e.g. the weak OLMo:
#     rm infer/data/yang_judge_*.jsonl; YANG_RATERS=gpt-oss-120b,command-r-plus bash scripts/yang.sh)
# 2. GPU: one judge job per Chen judge (infer/yang_judges_submit.sh), ~1-6 h each
# 3. CPU, after all judge jobs: real/yang_real.py (scripts/analyses.sbatch yang)
set -euo pipefail
cd "$(dirname "$0")/.."
source infer/env.sh > /dev/null

if compgen -G "infer/data/yang_judge_*.jsonl" > /dev/null; then
    echo "pairs exist: $(ls infer/data/yang_judge_*.jsonl | wc -l) files (delete to rebuild)"
else
    python -m infer.yang_data pairs
fi

IDS=$(bash infer/yang_judges_submit.sh "$@" | tee /dev/stderr | sed -n 's/.*job \([0-9]*\).*/\1/p' | paste -sd: -)
[ -n "$IDS" ] || { echo "no judge jobs submitted"; exit 1; }
id=$(sbatch --parsable --dependency=afterany:$IDS scripts/analyses.sbatch yang)
echo "analysis (CPU, after all judge jobs): job $id -> real/yang_real_results.csv"
