#!/bin/bash
# All CPU analyses, as one Slurm job (I may submit this; no GPU). From the repo root:
#   sbatch scripts/analyses.sbatch                 # everything below whose inputs exist
#   sbatch scripts/analyses.sbatch rubric yang     # only these groups
# Groups:
#   rubric       IFEval: real/rubric_real.py, real/rubric_did.py (halo control)
#   healthbench  HealthBench: real/rubric_real.py with committee labels
#   beta         real/beta_fit.py (difficulty coupling)
#   pairs        real/pair_check.py, real/item_check.py, synth/pair_mixture.py (slow: grid reweighting)
#   yang         real/yang_real.py (needs infer/data/verdicts_<judge>.jsonl from scripts/yang.sh)
# Output: real/*_results.* and this job's log, logs/analyses_<jobid>.out
#SBATCH --job-name=analyses
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-.}/infer/env.sh" > /dev/null
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-16}
GROUPS_=${*:-rubric healthbench beta pairs yang}
run() { echo; echo "===== $* ($(date +%T))"; "$@" || echo "!!!!! FAILED: $*"; }
has() { compgen -G "$1" > /dev/null || { echo; echo "----- skip: no $1"; return 1; }; }

for g in $GROUPS_; do case $g in
    rubric)
        has "infer/data/ifeval/verdicts_*.jsonl" && { run python -m real.rubric_real; run python -m real.rubric_did; } ;;
    healthbench)
        has "infer/data/healthbench/verdicts_*.jsonl" && RUBRIC_DATASET=healthbench run python -m real.rubric_real ;;
    beta)
        has "infer/data/beta_verdicts_*.jsonl" && run python -m real.beta_fit ;;
    pairs)
        run python -m real.pair_check; run python -m real.item_check; run python -m synth.pair_mixture ;;
    yang)
        has "infer/data/verdicts_*.jsonl" && run python -m real.yang_real ;;
    *) echo "unknown group $g" ;;
esac; done
echo; echo "done $(date +%T)"
