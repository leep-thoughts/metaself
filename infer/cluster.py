"""
GPUs needed per model on this cluster (GPU_MEM_GB from infer/cluster.sh; default 48 = L40S).
Weights + 10% + 6 GB (KV cache, activations) must fit in 90% of the GPUs' memory; 1, 2 or 4
GPUs (one node). 0 = does not fit on one node here (e.g. 70B bf16 on 4x A100-40GB).

    python -m infer.cluster <hf id>     prints the GPU count
"""

import os
import sys

WEIGHTS_GB = {   # bf16 unless noted
    "Qwen/Qwen2.5-3B-Instruct": 6.2, "Qwen/Qwen2.5-7B-Instruct": 15.2, "Qwen/Qwen2.5-14B-Instruct": 29.5,
    "Qwen/Qwen2.5-32B-Instruct": 65.5, "Qwen/Qwen2.5-72B-Instruct": 145.4,
    "meta-llama/Llama-3.2-3B-Instruct": 6.4, "meta-llama/Llama-3.1-8B-Instruct": 16.1,
    "meta-llama/Meta-Llama-3-8B-Instruct": 16.1,
    "meta-llama/Llama-3.1-70B-Instruct": 141.1, "meta-llama/Llama-3.3-70B-Instruct": 141.1,
    "google/gemma-2-9b-it": 18.5, "google/gemma-2-27b-it": 54.5,
    "allenai/OLMo-2-0325-32B-Instruct": 64.5,
    "CohereLabs/c4ai-command-r-plus-4bit": 63.0,   # bitsandbytes 4-bit; 2 GPUs at 48 GB already tight (OOM at batch 512)
}


def gpus(hf_id: str) -> int:
    mem = float(os.environ.get("GPU_MEM_GB", 48))
    need = WEIGHTS_GB[hf_id] * 1.10 + 6
    return next((n for n in (1, 2, 4) if n * mem * 0.9 >= need), 0)


if __name__ == "__main__":
    print(gpus(sys.argv[1]))
