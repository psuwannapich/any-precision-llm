#!/bin/bash
# run_trtllm_serve.sh — launch the OpenAI-compatible Any-Precision TensorRT-LLM
# server (per-request bit-width). Sets up the same native env as
# run_trtllm_test.sh (Blackwell/sm_120: .venv-gpu, torch+tensorrt libs on the
# path, MPI singleton workaround, plugin .so).
#
#   ./run_trtllm_serve.sh                       # defaults: trt_engine, port 8000
#   PORT=8080 ./run_trtllm_serve.sh
#   ENGINE_DIR=... MODEL_PATH=... ./run_trtllm_serve.sh
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PYTHON="${PYTHON:-$REPO_DIR/.venv-gpu/bin/python}"
ENGINE_DIR="${ENGINE_DIR:-$REPO_DIR/trt_engine}"
MODEL_PATH="${MODEL_PATH:-$REPO_DIR/cache/packed/anyprec-(Qwen3-4B)-w8_orig3-gc1-c4_s100_blk512}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL_ID="${MODEL_ID:-anyprec}"

# tensorrt_llm import needs the MPI singleton workaround on this box.
export OMPI_MCA_ess_singleton_isolated="${OMPI_MCA_ess_singleton_isolated:-1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# torch libs + pip tensorrt_libs (libnvinfer.so.10) must be discoverable for the
# plugin .so that build_engine/runtime dlopen.
LIBS="$("$PYTHON" - <<'PY'
import os
p=[]
try:
    import torch; p.append(os.path.join(os.path.dirname(torch.__file__),"lib"))
except Exception: pass
try:
    import tensorrt as trt
    cand=os.path.join(os.path.dirname(os.path.dirname(trt.__file__)),"tensorrt_libs")
    if os.path.isdir(cand): p.append(cand)
except Exception: pass
print(":".join(p))
PY
)"
export LD_LIBRARY_PATH="$LIBS:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export ANYPREC_TRT_PLUGIN="${ANYPREC_TRT_PLUGIN:-$REPO_DIR/any_precision/trtllm_integration/plugin/build/libanyprec_trt_plugin.so}"

echo "engine: $ENGINE_DIR"
echo "model:  $MODEL_PATH"
echo "serve:  http://$HOST:$PORT   (model id '$MODEL_ID')"
exec "$PYTHON" -m any_precision.trtllm_integration.python.server \
  --engine_dir "$ENGINE_DIR" --tokenizer_dir "$MODEL_PATH" \
  --host "$HOST" --port "$PORT" --model_id "$MODEL_ID"
