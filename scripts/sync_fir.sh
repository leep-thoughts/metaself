#!/bin/bash
# Copy inputs/outputs between Vulcan and Fir for scripts/fir_70b.sh. Run ON VULCAN, repo root.
#   bash scripts/sync_fir.sh push      inputs: Pombal data (IFEval + HealthBench), Yang pairs (3 judges)
#   bash scripts/sync_fir.sh push-hb   all HealthBench answers (after Llama-70B's are pulled back)
#   bash scripts/sync_fir.sh pull      outputs: Yang verdicts (3 judges), HealthBench Llama-70B answers/verdicts
# Transport (never deletes anything):
#   SYNC=globus (default)  Globus transfer between the two collections; returns a task id, runs in
#                          the background (globus task wait <id>). Needs once: bash scripts/sync_fir.sh globus-setup
#   SYNC=rsync             rsync over ssh to the Fir login node (password/MFA prompt each time)
# Collections (override with env): FIR_EP (Fir, from the Alliance docs), VULCAN_EP (find with
#   globus endpoint search vulcan). Paths: same absolute path on both, $REPO below.
# Code goes via git (git push here, git pull on Fir).
set -euo pipefail
cd "$(dirname "$0")/.."
SYNC=${SYNC:-globus}
HERE=$(pwd)
REMOTE=${FIR_REPO:-/scratch/$USER/metaself}          # repo path on Fir
HOST=${FIR_HOST:-fir.alliancecan.ca}
FIR_EP=${FIR_EP:-8dec4129-9ab4-451d-a45f-5b4b8471f7a3}  # alliancecan#fir-globus
VULCAN_EP=${VULCAN_EP:-}
GLOBUS_VENV=${WORK:-${SCRATCH:-/scratch/$USER}}/venvs/globus
YANG=(qwen-2.5-72b llama-3.1-70b llama-3.3-70b)
HB=llama-3.1-70b

# file lists, relative to the repo root
push_files() {
    find data/pombal-simple-evals/ifeval data/pombal-simple-evals/hb_data -type f
    printf 'infer/data/yang_judge_%s.jsonl\n' "${YANG[@]}"
}
push_hb_files() { ls infer/data/healthbench/answers_*.jsonl; }
pull_files() {
    printf 'infer/data/verdicts_%s.jsonl\n' "${YANG[@]}"
    echo "infer/data/healthbench/answers_$HB.jsonl"
    echo "infer/data/healthbench/verdicts_$HB.jsonl"
}

globus_cmd() {
    [ -x "$GLOBUS_VENV/bin/globus" ] || { echo "run first: bash scripts/sync_fir.sh globus-setup"; exit 1; }
    "$GLOBUS_VENV/bin/globus" "$@"
}

globus_xfer() {   # $1 src endpoint, $2 src root, $3 dst endpoint, $4 dst root, stdin: relative paths
    [ -n "$VULCAN_EP" ] || { echo "set VULCAN_EP (find it: $GLOBUS_VENV/bin/globus endpoint search vulcan)"; exit 1; }
    local batch; batch=$(mktemp)
    while read -r f; do echo "$2/$f $4/$f"; done > "$batch"
    echo "$(wc -l < "$batch") files"
    globus_cmd transfer "$1" "$3" --batch "$batch" --sync-level mtime --preserve-mtime \
        --label "metaself $ACTION $(date +%m%d-%H%M)"
    rm -f "$batch"
}

ACTION=${1:-}
case "$ACTION" in
globus-setup)
    module load python/3.12.4 > /dev/null 2>&1 || true
    [ -d "$GLOBUS_VENV" ] || python -m venv "$GLOBUS_VENV"
    "$GLOBUS_VENV/bin/pip" install -q --upgrade globus-cli
    "$GLOBUS_VENV/bin/globus" login
    echo "Vulcan collection candidates:"; "$GLOBUS_VENV/bin/globus" endpoint search vulcan --filter-scope all | head -20
    echo "then: export VULCAN_EP=<id>   (also activate both collections once in the Globus web app)"
    ;;
push|push-hb|pull)
    list=$( { [ "$ACTION" = push ] && push_files; [ "$ACTION" = push-hb ] && push_hb_files; \
              [ "$ACTION" = pull ] && pull_files; } || true)
    if [ "$SYNC" = rsync ]; then
        if [ "$ACTION" = pull ]; then
            echo "$list" | rsync -av --files-from=- --ignore-missing-args "$HOST:$REMOTE/" ./
        else
            echo "$list" | rsync -av --files-from=- ./ "$HOST:$REMOTE/"
        fi
    elif [ "$ACTION" = pull ]; then
        # only files that exist on Fir (a missing source fails the whole Globus task)
        have=$(globus_cmd ls -r --format unix "$FIR_EP:$REMOTE/infer/data" 2>/dev/null | awk '{print $NF}' | sed 's|^|infer/data/|')
        echo "$list" | grep -xF -f <(echo "$have") | globus_xfer "$FIR_EP" "$REMOTE" "$VULCAN_EP" "$HERE"
    else
        echo "$list" | globus_xfer "$VULCAN_EP" "$HERE" "$FIR_EP" "$REMOTE"
    fi
    ;;
*) sed -n 2,13p "$0"; exit 1 ;;
esac
