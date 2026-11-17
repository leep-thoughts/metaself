#!/bin/bash
# Run the 70B jobs that are stuck in Vulcan's queue, on Fir (4x H100 80 GB). Run ON FIR, repo root.
#   bash scripts/fir_70b.sh setup      once: venv (~20 min) + 70B weights (~430 GB, hours), login node
#   bash scripts/fir_70b.sh start      Yang judges (Qwen-72B, Llama-3.1-70B, Llama-3.3-70B) +
#                                      HealthBench answers from Llama-3.1-70B
#   bash scripts/fir_70b.sh hb-judge   HealthBench judging by Llama-3.1-70B (after ALL 14 models'
#                                      HealthBench answers are copied here: scripts/sync_fir.sh push-hb)
# Inputs come from Vulcan (scripts/sync_fir.sh push); outputs go back (scripts/sync_fir.sh pull).
set -euo pipefail
cd "$(dirname "$0")/.."
YANG=(qwen-2.5-72b llama-3.1-70b llama-3.3-70b)
HB=llama-3.1-70b

case "${1:-}" in
setup)
    [ -f infer/clusters/local.sh ] || { echo "first: cp infer/clusters/local.sh.example infer/clusters/local.sh and set ACCOUNT"; exit 1; }
    [ -f ~/.cache/huggingface/token ] || { echo "first: copy your HF token to ~/.cache/huggingface/token (gated Llama)"; exit 1; }
    [ -d "${SCRATCH:-/scratch/$USER}/venvs/vllm" ] || bash infer/setup_vllm.sh
    bash infer/download_models.sbatch Qwen/Qwen2.5-72B-Instruct meta-llama/Llama-3.1-70B-Instruct meta-llama/Llama-3.3-70B-Instruct
    ;;
start)
    source infer/env.sh > /dev/null
    for j in "${YANG[@]}"; do
        [ -f infer/data/yang_judge_$j.jsonl ] || { echo "missing infer/data/yang_judge_$j.jsonl: run scripts/sync_fir.sh push on Vulcan"; exit 1; }
    done
    bash infer/yang_judges_submit.sh "${YANG[@]}"
    g=$(python -c "from infer.ifeval_rubric import GPU_MODELS; print(GPU_MODELS['$HB'][1])")
    id=$(RUBRIC_DATASET=healthbench sbatch --parsable --job-name=hbgen-$HB $(gpu_flags $g) --time=02:00:00 \
        infer/ifeval_gpu.sbatch gpu-generate $HB)
    echo "HealthBench answers $HB: job $id -> infer/data/healthbench/answers_$HB.jsonl"
    ;;
hb-judge)
    source infer/env.sh > /dev/null
    n=$(ls infer/data/healthbench/answers_*.jsonl 2>/dev/null | wc -l)
    [ "$n" -ge 14 ] || { echo "only $n/14 HealthBench answer files here: run scripts/sync_fir.sh push-hb on Vulcan"; exit 1; }
    g=$(python -c "from infer.ifeval_rubric import GPU_MODELS; print(GPU_MODELS['$HB'][1])")
    id=$(RUBRIC_DATASET=healthbench sbatch --parsable --job-name=hbjudge-$HB $(gpu_flags $g) --time=08:00:00 \
        infer/ifeval_gpu.sbatch gpu-judge $HB)
    echo "HealthBench judging by $HB: job $id -> infer/data/healthbench/verdicts_$HB.jsonl"
    ;;
*) sed -n 2,9p "$0"; exit 1 ;;
esac
