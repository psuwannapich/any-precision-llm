"""
batch_decode_benchmark.py

Direct model benchmark: measures decode step latency across bit-widths and
batch sizes, bypassing vLLM scheduling overhead to show the raw speedup.

Key finding: AnyPrecisionLinear has two paths:
  batch ≤ 8  →  matmul_kbit:       reads quantized weights directly
                                    → lower bits = fewer reads → FASTER
  batch > 8  →  dequant_kbit + GEMM: dequantises to fp16 first
                                    → lower bits reduce dequant cost only

This script demonstrates both regimes clearly.

Usage:
  python batch_decode_benchmark.py [--model_path PATH]
"""

import argparse
import sys
import time

import torch
from transformers import AutoTokenizer

DEFAULT_MODEL = (
    "/mnt/scratch/users/psuwannapichat/any-precision-llm/cache/packed/"
    "anyprec-(Qwen3-4B)-w8_orig3-gc1-c4_s100_blk512"
)
PRECISIONS = [3, 4, 8]
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
DECODE_STEPS = 60    # timed steps per run
WARMUP_STEPS = 10


def sep(title="", width=72):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*(width - pad - len(title) - 2)}")
    else:
        print("─" * width)


def bar(speedup, ref=1.0, width=28):
    frac = min(speedup / ref, 2.0) / 2.0
    n = max(0, min(width, int(frac * width)))
    return "█" * n + "░" * (width - n)


def time_decode_steps(model, input_ids, past_key_values, precision, steps, warmup):
    """Time `steps` decode steps using CUDA events (accurate GPU timing)."""
    model.set_precision(precision)

    # Clone KV cache so we can reuse it across runs
    # (we don't update the cache to keep tensor sizes constant per step)
    pkv = past_key_values

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            out = model(input_ids, past_key_values=pkv, use_cache=True)
            pkv_w = out.past_key_values
            torch.cuda.synchronize()

    # Timed run using CUDA events
    events_start = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    events_end   = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

    with torch.no_grad():
        for i in range(steps):
            events_start[i].record()
            out = model(input_ids, past_key_values=pkv, use_cache=True)
            events_end[i].record()

    torch.cuda.synchronize()
    step_ms = [events_start[i].elapsed_time(events_end[i]) for i in range(steps)]
    return sorted(step_ms)  # sorted so p50/p90 are easy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--decode_steps", type=int, default=DECODE_STEPS)
    parser.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    parser.add_argument("--prompt", default="Explain how neural network quantization works.")
    args = parser.parse_args()

    print("\n" + "="*72)
    print("  Any-Precision Batch Decode Benchmark")
    print("  GPU:     ", torch.cuda.get_device_name(0))
    print("  Model:   ", args.model_path.split("/")[-1])
    print("  Steps:   ", args.decode_steps, " timed + ", args.warmup, " warmup per config")
    print("="*72)

    # ── Load model ────────────────────────────────────────────────────────────
    print("\n  Loading tokenizer... ", end="", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print("done")

    print("  Loading model (all precisions)... ", end="", flush=True)
    from any_precision.modules.AnyPrecisionForCausalLM import AnyPrecisionForCausalLM
    model = AnyPrecisionForCausalLM.from_quantized(args.model_path, precisions=PRECISIONS)
    model = model.eval().cuda()
    print(f"done  ({torch.cuda.memory_allocated()/1e9:.1f} GB)")

    # ── Tokenize prompt ───────────────────────────────────────────────────────
    base_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.cuda()
    prompt_len = base_ids.shape[1]
    print(f"\n  Prompt: {repr(args.prompt[:60])}")
    print(f"  Prompt tokens: {prompt_len}")

    # ── Results table -  results[batch][prec] = list of step times (ms) ──────
    results = {b: {} for b in BATCH_SIZES}

    sep("Benchmarking decode steps")
    print(f"  {'Batch':>6}  {'Prec':>5}  {'p50 ms':>8}  {'p90 ms':>8}  "
          f"{'vs 8-bit':>9}  {'Path':>12}  Visual (full=2×)")
    print("  " + "─" * 72)

    prev_batch = None
    for batch in BATCH_SIZES:
        # Build batched input: repeat prompt batch times
        input_ids_batch = base_ids.expand(batch, -1)

        # Prefill: run forward to get KV cache
        print(f"\n  [batch={batch}] prefill... ", end="", flush=True)
        with torch.no_grad():
            # Use 8-bit for prefill (it doesn't matter for timing)
            model.set_precision(8)
            out = model(input_ids_batch, use_cache=True)
            pkv = out.past_key_values

        # Single-token decode input: last token of each sequence
        decode_input = input_ids_batch[:, -1:]
        print("done")

        ref_times = None
        for prec in PRECISIONS:
            print(f"  [batch={batch:2d}, {prec}-bit] ", end="", flush=True)
            step_times = time_decode_steps(
                model, decode_input, pkv, prec,
                steps=args.decode_steps, warmup=args.warmup
            )
            results[batch][prec] = step_times
            p50 = step_times[len(step_times) // 2]
            p90 = step_times[int(len(step_times) * 0.9)]

            if ref_times is None:
                ref_times = step_times

            ref_p50 = ref_times[len(ref_times) // 2] if 8 in [prec] or 8 not in PRECISIONS else None
            speedup_vs_8 = None

            path = "matmul_kbit" if batch <= 8 else "dequant+GEMM"
            print(f"p50={p50:.1f}ms  p90={p90:.1f}ms  [{path}]")

        # Print comparison table for this batch size
        ref_p50 = results[batch][8][len(results[batch][8]) // 2]
        if prev_batch != batch:
            path_label = "matmul_kbit" if batch <= 8 else "dequant+GEMM"
            print(f"\n  ─── batch={batch} ({path_label}) ───")
            prev_batch = batch
        for prec in PRECISIONS:
            times = results[batch][prec]
            p50 = times[len(times) // 2]
            p90 = times[int(len(times) * 0.9)]
            speedup = ref_p50 / p50
            b = bar(speedup)
            flag = "  ◄ baseline" if prec == 8 else f"  ×{speedup:.3f}"
            print(f"  {batch:>6}  {prec:>4}-bit  {p50:>8.2f}  {p90:>8.2f}  "
                  f"×{speedup:>6.3f}  {path_label:>12}  {b}{flag}")

    # ── Summary: speedup heatmap across batch × precision ────────────────────
    sep("Speedup Heatmap  (vs 8-bit, p50 decode step time)")
    header = f"  {'Batch':>6}  {'Path':>12}  " + "  ".join(
        f"{p}-bit speedup" for p in PRECISIONS)
    print(header)
    print("  " + "─" * 65)

    for batch in BATCH_SIZES:
        ref_p50 = results[batch][8][len(results[batch][8]) // 2]
        path = "matmul_kbit" if batch <= 8 else "dequant+GEMM"
        parts = [f"  {batch:>6}  {path:>12}  "]
        for prec in PRECISIONS:
            times = results[batch][prec]
            p50 = times[len(times) // 2]
            speedup = ref_p50 / p50
            trend = "▲" if speedup > 1.03 else ("▼" if speedup < 0.97 else "─")
            parts.append(f"  ×{speedup:.3f} {trend}  ")
        print("".join(parts))

    # ── Throughput: tokens/s per bit-width ────────────────────────────────────
    sep("Throughput  (tok/s total = batch / step_time)")
    print(f"  {'Batch':>6}  {'Path':>12}  " + "  ".join(
        f"{p}-bit tok/s" for p in PRECISIONS))
    print("  " + "─" * 65)

    for batch in BATCH_SIZES:
        path = "matmul_kbit" if batch <= 8 else "dequant+GEMM"
        parts = [f"  {batch:>6}  {path:>12}  "]
        for prec in PRECISIONS:
            times = results[batch][prec]
            p50_ms = times[len(times) // 2]
            tps = batch / (p50_ms / 1000)
            parts.append(f"  {tps:>8.1f}  ")
        print("".join(parts))

    # ── Key insight ───────────────────────────────────────────────────────────
    sep("Key Insight")
    print("""
  AnyPrecisionLinear uses two kernels based on batch size:

    batch ≤ 8  →  matmul_kbit   : reads quantized bits + LUT directly.
                                   3-bit reads 3/8 = 37.5% of 8-bit weight data.
                                   Result: lower bits = fewer memory reads = FASTER.

    batch > 8  →  dequant + GEMM: dequantises weights to fp16 first.
                                   Dequant saves I/O but GEMM runs on fp16 anyway.
                                   Result: speedup is smaller (only dequant stage).

  Practical implication:
    • Serve ≤8 concurrent requests per precision  →  full bit-width speedup
    • Higher concurrency uses GEMM path           →  partial speedup
    • vLLM max_num_seqs=8 maximises the advantage of lower bits
""")
    sep()


if __name__ == "__main__":
    main()
