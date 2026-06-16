#!/bin/bash
# run_trtllm_test.sh — end-to-end Any-Precision x TensorRT-LLM test for the
# LOCAL workstation (RTX 5090 / sm_120 / Blackwell), as opposed to the HPC
# run_*.sh scripts which use `module load` and a V100 (sm_70, which TRT-LLM
# cannot run).
#
# Stages (run a subset by passing them as args; default = all):
#   preflight  - check GPU, CUDA, cmake, venv, tensorrt(_llm), checkpoint
#   cpu        - tests/cpu_validate.py        (weight-layout sanity, no GPU)
#   plugin     - cmake build of libanyprec_trt_plugin.so  (sm_120)
#   engine     - python/build_engine.py        (build the TRT-LLM engine)
#   smoke      - tests/gpu_smoke.py            (generate at every precision)
#
# Examples:
#   ./run_trtllm_test.sh preflight
#   MODEL_PATH=~/models/anyprec-Qwen3-4B ./run_trtllm_test.sh
#   ./run_trtllm_test.sh plugin engine smoke
#
# Override via env: PYTHON, MODEL_PATH, ENGINE_DIR, CUDA_HOME, TRT_ROOT,
#                   CUDA_ARCH, DEFAULT_PRECISION, MAX_BATCH_SIZE, MAX_SEQ_LEN.
set -euo pipefail

# --- repo / paths ----------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# Inside the TRT-LLM Docker container python3 already has tensorrt_llm and there
# is no .venv; natively we use the base venv. PYTHON env overrides either way.
if [ -n "${PYTHON:-}" ]; then
  :
elif [ -f /.dockerenv ]; then
  PYTHON="$(command -v python3)"
elif [ -x "$REPO_DIR/.venv-gpu/bin/python" ]; then
  # native Blackwell/sm_120 env: torch 2.9 cu128 + tensorrt_llm 1.2.1 live here
  PYTHON="$REPO_DIR/.venv-gpu/bin/python"
else
  PYTHON="$REPO_DIR/.venv/bin/python"
fi

# tensorrt_llm import runs MPI_Init; OpenMPI's singleton startup needs orted
# (openmpi-bin), which isn't installed on this WSL box. Tell OpenMPI to init in
# isolated singleton mode so `import tensorrt_llm` works without mpirun.
export OMPI_MCA_ess_singleton_isolated="${OMPI_MCA_ess_singleton_isolated:-1}"
# AP-quantized Qwen3 checkpoint (config.json with an "anyprec" block +
# pytorch_model.bin). NOT the base Qwen3-4B. This is the path quantize.py writes
# for: Qwen3-4B, seed=3 parent=8, c4, 100 examples, seq 512 (the defaults used
# by run_quantize_cpu.sh). Override MODEL_PATH if you quantized differently.
MODEL_PATH="${MODEL_PATH:-$REPO_DIR/cache/packed/anyprec-(Qwen3-4B)-w8_orig3-gc1-c4_s100_blk512}"
ENGINE_DIR="${ENGINE_DIR:-$REPO_DIR/trt_engine}"
PLUGIN_DIR="$REPO_DIR/any_precision/trtllm_integration/plugin"

# RTX 5090 is sm_120; the CMakeLists default (80;86;89;90) does NOT cover it.
CUDA_ARCH="${CUDA_ARCH:-120}"
DEFAULT_PRECISION="${DEFAULT_PRECISION:-8}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"

# --- CUDA toolkit (nvcc not on PATH by default on this box) -----------------
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# TRT-LLM pip wheel bundles TensorRT + libnvinfer inside site-packages; the
# plugin links against those. If TRT_ROOT is unset, try to locate the pip
# `tensorrt_libs` dir so cmake can find NvInfer.h / libnvinfer.
TRT_ROOT="${TRT_ROOT:-}"

c_green() { printf '\033[92m%s\033[0m\n' "$1"; }
c_red()   { printf '\033[91m%s\033[0m\n' "$1"; }
c_blue()  { printf '\033[94m\n=== %s ===\033[0m\n' "$1"; }

# --- ensure torch + tensorrt libs are on the path for the plugin .so ---------
# The plugin links libnvinfer.so.10 (pip tensorrt_libs) and uses cublas/cudart.
# build_engine.py dlopens the plugin BEFORE importing tensorrt_llm, so these
# must be discoverable up front.
add_torch_libs() {
  local libs
  libs="$("$PYTHON" - <<'PY' 2>/dev/null || true
import os
paths = []
try:
    import torch
    paths.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
except Exception:
    pass
try:
    import tensorrt as trt
    base = os.path.dirname(os.path.dirname(trt.__file__))
    cand = os.path.join(base, "tensorrt_libs")
    if os.path.isdir(cand):
        paths.append(cand)
except Exception:
    pass
print(":".join(paths))
PY
)"
  [ -n "$libs" ] && export LD_LIBRARY_PATH="$libs:$LD_LIBRARY_PATH"
  return 0
}

locate_trt_root() {
  [ -n "$TRT_ROOT" ] && return 0
  TRT_ROOT="$("$PYTHON" - <<'PY' 2>/dev/null || true
import os
try:
    import tensorrt as trt
    # headers live in the `tensorrt` pkg; libs in the sibling `tensorrt_libs`.
    base = os.path.dirname(os.path.dirname(trt.__file__))
    cand = os.path.join(base, "tensorrt_libs")
    print(cand if os.path.isdir(cand) else os.path.dirname(trt.__file__))
except Exception:
    pass
PY
)"
  [ -n "$TRT_ROOT" ] && export TRT_ROOT
  return 0
}

# --- stages ----------------------------------------------------------------
stage_preflight() {
  c_blue "preflight"
  local ok=1

  echo "-- GPU --"
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.used --format=csv
    local cc
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')"
    if [ "$cc" -ge 80 ] 2>/dev/null; then
      c_green "compute capability ${cc} >= 80 (TRT-LLM requires sm_80+)"
    else
      c_red "compute capability ${cc} < 80 — TRT-LLM cannot run here"; ok=0
    fi
    echo "NOTE: sm_120/Blackwell + CUDA 13 needs a recent TRT-LLM (>=0.17/1.0);"
    echo "      older wheels target up to sm_90 and will fail to build/run."
  else
    c_red "nvidia-smi not found"; ok=0
  fi

  echo "-- CUDA toolkit --"
  if command -v nvcc >/dev/null; then
    c_green "nvcc: $(nvcc --version | grep release)"
  else
    c_red "nvcc not on PATH (CUDA_HOME=$CUDA_HOME)"; ok=0
  fi
  command -v cmake >/dev/null && c_green "cmake: $(cmake --version | head -1)" \
    || { c_red "cmake not found"; ok=0; }

  echo "-- Python / venv --"
  if [ -x "$PYTHON" ]; then
    c_green "python: $("$PYTHON" --version 2>&1)  ($PYTHON)"
  else
    c_red "venv python not found at $PYTHON  (create with: uv venv && uv pip install -e .)"; ok=0
  fi
  if [ -x "$PYTHON" ]; then
    "$PYTHON" - <<'PY' || ok=0
import importlib, sys
miss = []
for m in ("torch", "numpy", "transformers"):
    try: importlib.import_module(m)
    except Exception: miss.append(m)
for m in ("tensorrt", "tensorrt_llm"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m}: {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m}: MISSING ({type(e).__name__})")
        miss.append(m)
if "torch" not in miss:
    import torch
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
sys.exit(1 if miss else 0)
PY
    [ "$ok" = 1 ] && true || true
  fi

  echo "-- Checkpoint --"
  if [ -f "$MODEL_PATH/config.json" ] && [ -f "$MODEL_PATH/pytorch_model.bin" ]; then
    if "$PYTHON" - "$MODEL_PATH" <<'PY'
import json, sys
c = json.load(open(sys.argv[1] + "/config.json"))
sys.exit(0 if "anyprec" in c else 1)
PY
    then c_green "AP checkpoint OK: $MODEL_PATH"
    else c_red "config.json has no 'anyprec' block — not an AP checkpoint: $MODEL_PATH"; ok=0
    fi
  else
    c_red "no AP checkpoint at $MODEL_PATH (set MODEL_PATH=...; quantize.py to create one)"; ok=0
  fi

  echo "-- TensorRT headers/libs for plugin build --"
  locate_trt_root
  if [ -n "$TRT_ROOT" ]; then
    c_green "TRT_ROOT candidate: $TRT_ROOT"
    ls "$TRT_ROOT"/*nvinfer* 2>/dev/null || \
      find "$TRT_ROOT" -maxdepth 2 -name 'NvInfer.h' 2>/dev/null | head -1 || \
      c_red "  (could not see NvInfer.h/libnvinfer there — pass -DTRT_ROOT explicitly)"
  else
    c_red "TRT_ROOT not located; set TRT_ROOT=/path/to/TensorRT before 'plugin' stage"
  fi

  echo
  [ "$ok" = 1 ] && c_green "preflight: ready" || c_red "preflight: missing prerequisites above"
}

stage_cpu() {
  c_blue "cpu_validate (weight layout, no GPU)"
  "$PYTHON" -m any_precision.trtllm_integration.tests.cpu_validate \
    --model_path "$MODEL_PATH"
}

stage_plugin() {
  c_blue "build plugin (sm_${CUDA_ARCH})"
  locate_trt_root
  local cmake_args=(-B build -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
  [ -n "$TRT_ROOT" ] && cmake_args+=(-DTRT_ROOT="$TRT_ROOT")
  ( cd "$PLUGIN_DIR" \
    && cmake "${cmake_args[@]}" \
    && cmake --build build -j )
  local so="$PLUGIN_DIR/build/libanyprec_trt_plugin.so"
  [ -f "$so" ] && c_green "built: $so" || { c_red "plugin .so not produced"; exit 1; }
  export ANYPREC_TRT_PLUGIN="$so"
}

stage_engine() {
  c_blue "build engine"
  add_torch_libs
  local so="$PLUGIN_DIR/build/libanyprec_trt_plugin.so"
  [ -f "$so" ] && export ANYPREC_TRT_PLUGIN="$so"
  "$PYTHON" -m any_precision.trtllm_integration.python.build_engine \
    --model_path "$MODEL_PATH" \
    --output_dir "$ENGINE_DIR" \
    --default_precision "$DEFAULT_PRECISION" \
    --max_batch_size "$MAX_BATCH_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN"
  c_green "engine -> $ENGINE_DIR"
}

stage_smoke() {
  c_blue "gpu_smoke (generate at every precision)"
  add_torch_libs
  local so="$PLUGIN_DIR/build/libanyprec_trt_plugin.so"
  [ -f "$so" ] && export ANYPREC_TRT_PLUGIN="$so"
  "$PYTHON" -m any_precision.trtllm_integration.tests.gpu_smoke \
    --engine_dir "$ENGINE_DIR" \
    --tokenizer_dir "$MODEL_PATH"
}

# --- dispatch --------------------------------------------------------------
STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(preflight cpu plugin engine smoke)

echo "repo:      $REPO_DIR"
echo "python:    $PYTHON"
echo "model:     $MODEL_PATH"
echo "engine:    $ENGINE_DIR"
echo "cuda_arch: sm_${CUDA_ARCH}   cuda_home: $CUDA_HOME"
echo "stages:    ${STAGES[*]}"

for s in "${STAGES[@]}"; do
  case "$s" in
    preflight) stage_preflight ;;
    cpu)       stage_cpu ;;
    plugin)    stage_plugin ;;
    engine)    stage_engine ;;
    smoke)     stage_smoke ;;
    *) c_red "unknown stage: $s (preflight|cpu|plugin|engine|smoke)"; exit 2 ;;
  esac
done

echo
c_green "done: ${STAGES[*]}"
