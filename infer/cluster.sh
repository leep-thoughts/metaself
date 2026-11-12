# Per-cluster settings loader. Sourced by infer/env.sh (so by every job and submit script).
#   infer/clusters/$CC_CLUSTER.sh   committed: GPU type/memory, CPU/RAM per GPU, account, internet
#   infer/clusters/local.sh         gitignored: per-machine overrides (see local.sh.example)
# New cluster = add infer/clusters/<name>.sh ($CC_CLUSTER is set on Alliance systems).
METASELF_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=${SCRATCH:-/scratch/$USER}
_c=$METASELF_REPO/infer/clusters
if [ -f "$_c/${CC_CLUSTER:-none}.sh" ]; then
    source "$_c/$CC_CLUSTER.sh"
else
    echo "infer/cluster.sh: no infer/clusters/${CC_CLUSTER:-?}.sh: add one for this cluster" >&2
fi
[ -f "$_c/local.sh" ] && source "$_c/local.sh"
ACCOUNT=${METASELF_ACCOUNT:-${ACCOUNT:-}}     # env var wins (one-off override)
unset _c
export METASELF_REPO WORK GPU_TYPE GPU_MEM_GB CPUS_PER_GPU MEM_PER_GPU_GB MEM_BASE_GB COMPUTE_INTERNET
if [ -n "${ACCOUNT:-}" ]; then
    export SBATCH_ACCOUNT=$ACCOUNT SALLOC_ACCOUNT=$ACCOUNT    # sbatch/salloc read these
else
    echo "infer/cluster.sh: no Slurm account: set ACCOUNT in infer/clusters/local.sh" >&2
fi

# sbatch resource flags for N GPUs:   sbatch $(gpu_flags 2) ...
gpu_flags() {
    echo "--gres=gpu:$GPU_TYPE:$1 --cpus-per-task=$(($1 * CPUS_PER_GPU)) --mem=$(($1 * MEM_PER_GPU_GB + MEM_BASE_GB))G"
}

# Vulcan-only scripts (*_vulcan.*) call this first.
require_vulcan() {
    [ "${CC_CLUSTER:-}" = vulcan ] || { echo "Vulcan only (Aleph): this is ${CC_CLUSTER:-?}" >&2; exit 1; }
}
