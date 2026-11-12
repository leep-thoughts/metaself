# Narval (Alliance): 4x A100 40 GB per node; NO internet on compute nodes, so download weights
# on the login node (bash infer/download_models.sbatch). 70B models don't fit on one node.
GPU_TYPE=a100; GPU_MEM_GB=40; CPUS_PER_GPU=12; MEM_PER_GPU_GB=100; MEM_BASE_GB=0
ACCOUNT=                      # set in infer/clusters/local.sh (check: sshare -U -u $USER)
COMPUTE_INTERNET=0
