#!/bin/bash
# run_trtllm_docker.sh — run the Any-Precision x TensorRT-LLM test stages inside
# NVIDIA's TensorRT-LLM container (the reliable path for Blackwell / sm_120,
# where the repo's pinned torch 2.2.2 cannot run).
#
# It bind-mounts this repo at the SAME absolute path inside the container (so the
# checkpoint under cache/ and all paths resolve unchanged) plus the HF cache, and
# invokes run_trtllm_test.sh with PYTHON=python3 (tensorrt_llm lives on the
# container's system python; there is no .venv inside).
#
# Usage:
#   ./run_trtllm_docker.sh build-image          # build anyprec-trtllm:local (optional)
#   ./run_trtllm_docker.sh preflight            # checks, inside the container
#   ./run_trtllm_docker.sh plugin               # build the plugin .so (compile only, no GPU exec)
#   ./run_trtllm_docker.sh engine smoke         # GPU stages (run when the GPU is free)
#   ./run_trtllm_docker.sh shell                # interactive bash in the container
#
# Image selection (in priority): $TRTLLM_IMAGE, else anyprec-trtllm:local if it
# exists, else the NVIDIA base below. Set a tag that supports sm_120:
#   https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.0.0}"
LOCAL_IMAGE="anyprec-trtllm:local"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

c_red() { printf '\033[91m%s\033[0m\n' "$1"; }

command -v docker >/dev/null || { c_red "docker not found"; exit 1; }

# --- build-image: bake the thin layer (transformers/cmake) over the base ----
if [ "${1:-}" = "build-image" ]; then
  exec docker build -f "$REPO_DIR/docker/Dockerfile.trtllm" \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" -t "$LOCAL_IMAGE" "$REPO_DIR"
fi

# --- choose image -----------------------------------------------------------
if [ -n "${TRTLLM_IMAGE:-}" ]; then
  IMAGE="$TRTLLM_IMAGE"
elif docker image inspect "$LOCAL_IMAGE" >/dev/null 2>&1; then
  IMAGE="$LOCAL_IMAGE"
else
  IMAGE="$BASE_IMAGE"
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  c_red "image '$IMAGE' not present locally."
  echo "  pull it:   docker pull $IMAGE"
  echo "  or build:  ./run_trtllm_docker.sh build-image   (uses BASE_IMAGE=$BASE_IMAGE)"
  echo "  or set:    TRTLLM_IMAGE=<tag> ./run_trtllm_docker.sh ..."
  exit 1
fi

# --- common docker args -----------------------------------------------------
# --gpus is needed for engine/smoke; harmless for plugin (compile-only) and
# preflight. Pass-through env lets the inner script honor your overrides.
DOCKER_ARGS=(
  --rm --gpus all --ipc=host
  --ulimit memlock=-1 --ulimit stack=67108864
  -v "$REPO_DIR":"$REPO_DIR" -w "$REPO_DIR"
  -v "$HF_CACHE":/root/.cache/huggingface
  -e PYTHON=python3
  -e HF_HOME=/root/.cache/huggingface
  -e "CUDA_ARCH=${CUDA_ARCH:-120}"
  -e "MODEL_PATH=${MODEL_PATH:-}"
  -e "ENGINE_DIR=${ENGINE_DIR:-}"
  -e "DEFAULT_PRECISION=${DEFAULT_PRECISION:-8}"
  -e "MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-4}"
  -e "MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}"
)
mkdir -p "$HF_CACHE"

# --- shell: drop into the container -----------------------------------------
if [ "${1:-}" = "shell" ]; then
  exec docker run -it "${DOCKER_ARGS[@]}" "$IMAGE" /bin/bash
fi

# --- default: run the requested stages inside the container -----------------
STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(preflight)
echo "image:  $IMAGE"
echo "mount:  $REPO_DIR -> $REPO_DIR   hf: $HF_CACHE"
echo "stages: ${STAGES[*]}"
exec docker run "${DOCKER_ARGS[@]}" "$IMAGE" \
  bash -lc './run_trtllm_test.sh "$@"' _ "${STAGES[@]}"
