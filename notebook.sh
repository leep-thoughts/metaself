# Interactive allocation on whichever cluster this is (account from infer/cluster.sh)
source "$(dirname "${BASH_SOURCE[0]:-$0}")/infer/cluster.sh"
salloc --time=8:00:00 --mem=4G --cpus-per-task=2
