#!/bin/bash
# Resubmit the 70B work on Vulcan (after the Fir detour), with the small jobs no longer waiting on
# the 70B ones. Run from the repo root.
#   Yang:        3 x 70B judges -> analysis (8 judges already done)
#   HealthBench: now      5 smaller GPU judges + Aleph judges on the 13 answer sets already there
#                after    Llama-70B answers -> Llama-70B judge; second pass of the 5 + Aleph for its
#                         answers (each waits for its own first pass: no two jobs append to one file)
#                finally  analysis
set -euo pipefail
cd "$(dirname "$0")/.."
source infer/env.sh > /dev/null
BIG=llama-3.1-70b
gpus() { python -c "from infer.ifeval_rubric import GPU_MODELS; print(GPU_MODELS['$1'][1])"; }
SMALL=$(python -c "from infer.ifeval_rubric import GPU_MODELS; print(' '.join(m for m in GPU_MODELS if m != '$BIG'))")
hb() { RUBRIC_DATASET=healthbench sbatch --parsable "$@"; }

# ---- Yang
Y=$(bash infer/yang_judges_submit.sh qwen-2.5-72b llama-3.1-70b llama-3.3-70b | tee /dev/stderr | sed -n 's/.*job \([0-9]*\).*/\1/p' | paste -sd: -)
id=$(sbatch --parsable --dependency=afterany:$Y scripts/analyses.sbatch yang)
echo "Yang analysis: job $id (after $Y)"

# ---- HealthBench
GEN=$(hb --job-name=ifgen-$BIG $(gpu_flags $(gpus $BIG)) --time=02:00:00 infer/ifeval_gpu.sbatch gpu-generate $BIG)
echo "HB answers $BIG: job $GEN"
ALL=()
for m in $SMALL; do
    a=$(hb --job-name=ifjudge-$m $(gpu_flags $(gpus $m)) --time=08:00:00 infer/ifeval_gpu.sbatch gpu-judge "$m")
    b=$(hb --dependency=afterok:$GEN,afterany:$a --job-name=ifjudge2-$m $(gpu_flags $(gpus $m)) --time=02:00:00 \
        infer/ifeval_gpu.sbatch gpu-judge "$m")
    echo "HB judge $m: job $a now, job $b for $BIG's answers"; ALL+=("$a" "$b")
done
c=$(hb --dependency=afterok:$GEN --job-name=ifjudge-$BIG $(gpu_flags $(gpus $BIG)) --time=08:00:00 infer/ifeval_gpu.sbatch gpu-judge $BIG)
echo "HB judge $BIG: job $c (after its answers)"; ALL+=("$c")
a=$(hb --job-name=hb-aleph infer/ifeval_aleph_vulcan.sbatch judge)
b=$(hb --dependency=afterok:$GEN,afterany:$a --job-name=hb-aleph2 infer/ifeval_aleph_vulcan.sbatch judge)
echo "HB Aleph judges: job $a now, job $b for $BIG's answers"; ALL+=("$a" "$b")
id=$(sbatch --parsable --dependency=afterany:$(IFS=:; echo "${ALL[*]}") scripts/analyses.sbatch healthbench)
echo "HB analysis: job $id (after everything)"
