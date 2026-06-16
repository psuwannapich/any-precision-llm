#!/bin/bash
# Launch the any-precision model under vLLM (Phase 1 integration).
#
# Uses vllm_venv (Python 3.12, vllm==0.6.6, torch==2.5.1+cu124).
#
# To choose a precision, add "active_precision": N to config.json's "anyprec" dict.
# Default is 8-bit. Supported: 3,4,5,6,7,8.
#
# Usage:
#   bash run_vllm.sh                  # port 8000, 8-bit
#   bash run_vllm.sh --port 8001
#
# Endpoint is OpenAI-compatible:
#   curl http://localhost:8000/v1/chat/completions ...

set -e

MODEL_PATH="/mnt/scratch/users/psuwannapichat/any-precision-llm/cache/packed/anyprec-(Qwen3-4B)-w8_orig3-gc1-c4_s100_blk512"
VENV="vllm_venv"

module load compiler/GCC/13.2.0 system/CUDA/12.6.0
TORCH_LIB=$VENV/lib/python3.12/site-packages/torch/lib
CUDNN_LIB=$VENV/lib/python3.12/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH=$CUDNN_LIB:$TORCH_LIB:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Disable torch.compile — vLLM's Triton kernels require Ampere (SM 8.0+);
# on Volta (V100) the SDPA fallback in our model handles attention instead.
export VLLM_TORCH_COMPILE_LEVEL=0

cd /mnt/aiongpfs/users/psuwannapichat/work_space/any-precision-llm

$VENV/bin/python3.12 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --dtype float16 \
    --trust-remote-code \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 32 \
    "$@"
