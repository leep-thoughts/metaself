#!/bin/bash
# Copy inputs/outputs between Vulcan and Fir for scripts/fir_70b.sh. Run ON VULCAN, repo root.
# Each rsync logs in to Fir (expect a password/MFA prompt). Never deletes anything.
#   bash scripts/sync_fir.sh push      code-independent inputs: Pombal data, Yang pairs (3 judges)
#   bash scripts/sync_fir.sh push-hb   all HealthBench answers (after Llama-70B's answers are pulled
#                                      back and the Vulcan GPU models' answers exist)
#   bash scripts/sync_fir.sh pull      outputs: Yang verdicts (3 judges), HealthBench Llama-70B answers/verdicts
# Code goes via git (git push here, git pull on Fir). Fir repo path: $FIR_REPO (default below).
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${FIR_HOST:-fir.alliancecan.ca}
REMOTE=${FIR_REPO:-/scratch/$USER/metaself}
YANG=(qwen-2.5-72b llama-3.1-70b llama-3.3-70b)
HB=llama-3.1-70b

case "${1:-}" in
push)
    rsync -av --mkpath data/pombal-simple-evals/ifeval data/pombal-simple-evals/hb_data "$HOST:$REMOTE/data/pombal-simple-evals/"
    rsync -av --mkpath $(printf 'infer/data/yang_judge_%s.jsonl ' "${YANG[@]}") "$HOST:$REMOTE/infer/data/"
    ;;
push-hb)
    rsync -av --mkpath infer/data/healthbench/answers_*.jsonl "$HOST:$REMOTE/infer/data/healthbench/"
    ;;
pull)
    for j in "${YANG[@]}"; do rsync -av --mkpath "$HOST:$REMOTE/infer/data/verdicts_$j.jsonl" infer/data/ || true; done
    rsync -av --mkpath "$HOST:$REMOTE/infer/data/healthbench/answers_$HB.jsonl" infer/data/healthbench/ || true
    rsync -av --mkpath "$HOST:$REMOTE/infer/data/healthbench/verdicts_$HB.jsonl" infer/data/healthbench/ || true
    ;;
*) sed -n 2,9p "$0"; exit 1 ;;
esac
