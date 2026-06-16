"""
bit_speedup_demo.py

Isolates and reproduces the bit-width throughput speedup by testing only
the quantized linear layer kernel — no attention, no server, no batching noise.

Two kernel paths are tested separately:
  matmul_kbit   — batch 1-8   (decode path: token-by-token generation)
  dequant_kbit  — batch > 8   (prefill path: processing a prompt)

Each (batch_size, precision) combination is pre-warmed to eliminate JIT spikes,
then timed with per-iteration CUDA synchronisation.
Speedup is reported vs 8-bit baseline at each batch size.
"""

import statistics
import sys
import time

import torch
import torch.nn as nn

try:
    from any_precision_ext import matmul_kbit, dequant_kbit
except ImportError:
    print("ERROR: any_precision_ext not found. Install the CUDA extension first.")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────

PRECISIONS   = [3, 4, 5, 6, 7, 8]
IN_FEATURES  = 2560   # Qwen3-4B hidden size
OUT_FEATURES = 2560
DEVICE       = "cuda"
DTYPE        = torch.float16
WARMUP       = 20
REPEATS      = 100

# batch sizes that exercise matmul_kbit (≤8) and dequant_kbit (>8)
DECODE_BATCHES  = [1, 2, 4, 8]
PREFILL_BATCHES = [16, 64, 256, 512]


# ── Build fake quantized weights ──────────────────────────────────────────────

def make_weights(precision: int):
    """Allocate one AnyPrecisionLinear layer's buffers on GPU."""
    K32 = IN_FEATURES // 32
    qweight = torch.randint(0, 2**31, (precision, OUT_FEATURES, K32),
                            dtype=torch.int32, device=DEVICE)
    lut = torch.randn(OUT_FEATURES, 2 ** precision,
                      dtype=DTYPE, device=DEVICE)
    return qweight, lut


def make_input(batch: int):
    return torch.randn(batch, IN_FEATURES, dtype=DTYPE, device=DEVICE)


# ── Timing helper ─────────────────────────────────────────────────────────────

def time_kernel(fn, warmup: int, repeats: int) -> float:
    """Return median latency in ms over `repeats` timed calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    lo, hi = len(times) // 4, 3 * len(times) // 4
    return statistics.median(times[lo:hi])


# ── Report helper ─────────────────────────────────────────────────────────────

def sep(title="", width=72):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*(width-pad-len(title)-2)}")
    else:
        print("─" * width)

def bar(frac, width=20):
    n = max(0, min(width, int(frac * width)))
    return "█" * n + "░" * (width - n)


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_decode_bench():
    """matmul_kbit path: batch 1-8, used during token-by-token decode."""
    sep("matmul_kbit  (decode path, batch ≤ 8)")
    print("  Speedup shown vs 8-bit baseline at the same batch size.\n")

    # Pre-build all weights so allocation doesn't pollute timing
    weights = {p: make_weights(p) for p in PRECISIONS}

    for bs in DECODE_BATCHES:
        x = make_input(bs)
        timings = {}
        for prec in PRECISIONS:
            qw, lut = weights[prec]
            fn = lambda qw=qw, lut=lut: matmul_kbit(x, qw, lut, prec)
            timings[prec] = time_kernel(fn, WARMUP, REPEATS)

        ref = timings[8]
        print(f"  batch={bs}")
        print(f"  {'Bits':>5}  {'ms':>7}  {'Speedup':>9}  Visual (full=×2)")
        print("  " + "─" * 50)
        for prec in PRECISIONS:
            ms = timings[prec]
            speedup = ref / ms
            b = bar(speedup / 2.0)
            flag = " ◄ baseline" if prec == 8 else ""
            print(f"  {prec:>4}-bit  {ms:>7.3f}  ×{speedup:>6.3f}  {b}{flag}")
        print()


def run_prefill_bench():
    """dequant_kbit path: batch > 8, used during prompt prefill."""
    sep("dequant_kbit + torch.matmul  (prefill path, batch > 8)")
    print("  Only the dequantisation step scales with bits;")
    print("  torch.matmul cost is constant. Speedup is smaller.\n")

    weights = {p: make_weights(p) for p in PRECISIONS}

    for bs in PREFILL_BATCHES:
        x = make_input(bs)
        timings = {}
        for prec in PRECISIONS:
            qw, lut = weights[prec]
            def fn(qw=qw, lut=lut, x=x, prec=prec):
                w = dequant_kbit(qw, lut, prec)
                return torch.matmul(x, w.T)
            timings[prec] = time_kernel(fn, WARMUP, REPEATS)

        ref = timings[8]
        print(f"  batch={bs}")
        print(f"  {'Bits':>5}  {'ms':>7}  {'Speedup':>9}  Visual (full=×2)")
        print("  " + "─" * 50)
        for prec in PRECISIONS:
            ms = timings[prec]
            speedup = ref / ms
            b = bar(speedup / 2.0)
            flag = " ◄ baseline" if prec == 8 else ""
            print(f"  {prec:>4}-bit  {ms:>7.3f}  ×{speedup:>6.3f}  {b}{flag}")
        print()


def run_dequant_only_bench():
    """Isolate just dequant_kbit to show the pure weight-decode speedup."""
    sep("dequant_kbit only  (weight decode cost, independent of batch)")
    print("  Pure cost of converting packed integers → fp16 weight matrix.\n")

    weights = {p: make_weights(p) for p in PRECISIONS}

    timings = {}
    for prec in PRECISIONS:
        qw, lut = weights[prec]
        fn = lambda qw=qw, lut=lut, prec=prec: dequant_kbit(qw, lut, prec)
        timings[prec] = time_kernel(fn, WARMUP, REPEATS)

    ref = timings[8]
    print(f"  layer: {IN_FEATURES}→{OUT_FEATURES}")
    print(f"  {'Bits':>5}  {'ms':>7}  {'Speedup':>9}  Visual (full=×3)")
    print("  " + "─" * 50)
    for prec in PRECISIONS:
        ms = timings[prec]
        speedup = ref / ms
        b = bar(speedup / 3.0)
        flag = " ◄ baseline" if prec == 8 else ""
        print(f"  {prec:>4}-bit  {ms:>7.3f}  ×{speedup:>6.3f}  {b}{flag}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    gpu = torch.cuda.get_device_name(0)
    sep("Bit-Width Kernel Speedup Demo")
    print(f"  GPU        : {gpu}")
    print(f"  Layer size : {IN_FEATURES} → {OUT_FEATURES}")
    print(f"  Warmup     : {WARMUP}  |  Repeats: {REPEATS}")
    print(f"  Timing     : median of middle 50% (outliers removed)")
    print(f"\n  This benchmark isolates ONLY the quantized linear kernel.")
    print(f"  No attention, no server, no batching variance.\n")

    # Pre-warm GPU
    _ = torch.randn(1024, 1024, device=DEVICE, dtype=DTYPE) @ \
        torch.randn(1024, 1024, device=DEVICE, dtype=DTYPE)
    torch.cuda.synchronize()

    run_decode_bench()
    run_dequant_only_bench()
    run_prefill_bench()

    sep("Summary")
    print("""
  Lower bits win CLEARLY when:
    1. matmul_kbit, batch=4-8  →  ×1.2–1.5× faster  (decode bottleneck)
    2. dequant_kbit alone       →  ×2–3× faster       (weight decode)

  Lower bits win WEAKLY when:
    3. dequant + matmul, large batch  →  ×1.0–1.1×  (matmul dominates)

  Lower bits show NO gain when:
    4. batch=1                  →  attention & memory latency dominate
    5. Full server benchmark    →  irregular batching + response variance
                                   mask the kernel-level speedup
""")
    sep()


if __name__ == "__main__":
    main()
