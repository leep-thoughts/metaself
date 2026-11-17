# Fir (Alliance, SFU): 4x H100 80 GB per node (bundle per GPU: 12 cores, ~280 GB). 70B models fit
# in bf16 on 4 GPUs. Compute-node internet not confirmed in the docs: treated as none, so
# download weights on the login node (bash infer/download_models.sbatch <repo>...).
GPU_TYPE=h100; GPU_MEM_GB=80; CPUS_PER_GPU=12; MEM_PER_GPU_GB=120; MEM_BASE_GB=0
ACCOUNT=                      # set in infer/clusters/local.sh (check: sshare -U -u $USER)
COMPUTE_INTERNET=0
