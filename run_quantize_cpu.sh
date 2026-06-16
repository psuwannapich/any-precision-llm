#!/bin/bash
# run_quantize_cpu.sh — produce an Any-Precision Qwen3-4B checkpoint on the LOCAL
# box using CPU only (no GPU contention). This is the checkpoint the TRT-LLM
# test (run_trtllm_test.sh) consumes. Hours-long; run in the background.
#
#   ./run_quantize_cpu.sh                 # defaults: seed 3, parent 8, c4
#   CPU_COUNT=16 ./run_quantize_cpu.sh    # limit worker processes
#
# Output:
#   cache/packed/anyprec-(Qwen3-4B)-w8_orig3-gc1-c4_s100_blk512/
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
CACHE_DIR="${CACHE_DIR:-$REPO_DIR/cache}"

args=(--cpu_only --cache_dir "$CACHE_DIR")
[ -n "${SEED_PRECISION:-}" ]   && args+=(--seed_precision "$SEED_PRECISION")
[ -n "${PARENT_PRECISION:-}" ] && args+=(--parent_precision "$PARENT_PRECISION")
[ -n "${DATASET:-}" ]          && args+=(--dataset "$DATASET")
[ -n "${CPU_COUNT:-}" ]        && args+=(--cpu_count "$CPU_COUNT")

echo "python:  $PYTHON"
echo "model:   $MODEL"
echo "cache:   $CACHE_DIR"
echo "args:    ${args[*]}"
exec "$PYTHON" quantize.py "$MODEL" "${args[@]}"
