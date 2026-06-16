#!/bin/bash
# Phase 2 vLLM server — per-request precision switching.
#
# Each request can specify "precision": 3/4/5/6/7/8.
# The custom AnyPrecisionWorker groups decode tokens by precision and
# runs a separate forward pass per precision group within each step.
#
# Usage:
#   bash run_vllm_p2.sh               # default port 8001
#   bash run_vllm_p2.sh --port 8002
#
# Test:
#   curl http://localhost:8001/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"messages":[{"role":"user","content":"What is AI?"}],
#           "precision": 4, "max_tokens": 64}'

set -e

MODEL_PATH="/mnt/scratch/users/psuwannapichat/any-precision-llm/cache/packed/anyprec-(Qwen3-4B)-w8_orig3-gc1-c4_s100_blk512"
VENV="vllm_venv"

module load compiler/GCC/13.2.0 system/CUDA/12.6.0
TORCH_LIB=$VENV/lib/python3.12/site-packages/torch/lib
CUDNN_LIB=$VENV/lib/python3.12/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH=$CUDNN_LIB:$TORCH_LIB:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export VLLM_TORCH_COMPILE_LEVEL=0   # Triton requires Ampere+; disable on V100

cd /mnt/aiongpfs/users/psuwannapichat/work_space/any-precision-llm

$VENV/bin/python3.12 -m any_precision.vllm_integration.p2_server \
    --model "$MODEL_PATH" \
    --max_model_len 2048 \
    --gpu_memory_utilization 0.85 \
    --max_num_seqs 32 \
    --enforce_eager \
    "$@"
