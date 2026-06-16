"""
Any-Precision LLM — Inference & Benchmark Script

Three benchmark modes that expose bit-width speedup at different levels:

  1. Micro-benchmark  — one AnyPrecisionLinear layer, no attention overhead.
                        Tests matmul_kbit (bs≤8) and dequant_kbit+matmul (bs>8).
  2. Prefill          — full-model forward pass on a long prompt (N tokens).
                        Linear layers dominate; speedup from dequant_kbit.
  3. Decode           — autoregressive token generation (batch=1).
                        Attention-bound; bit-width effect is smallest here.

Usage:
  python inference_benchmark.py [options]

  --model_path PATH       packed model dir (default: Qwen3-4B anyprec)
  --precisions 3 5 8      which precisions to test (default: all 3-8)
  --prompt TEXT           base prompt text
  --prefill_len N         token count for prefill bench (default 512)
  --max_new_tokens N      tokens to generate in decode bench (default 64)
  --warmup_runs N         warmup iterations (default 2)
  --bench_runs N          timed iterations (default 5)
  --skip_micro            skip linear layer micro-benchmark
  --skip_prefill          skip prefill benchmark
  --skip_decode           skip decode benchmark
  --skip_demo             skip inference demo
  --skip_fp16             skip FP16 baseline
"""

import argparse
import gc
import time
import os
import warnings
import logging

import torch
from transformers import AutoTokenizer

DEFAULT_MODEL = (
    "/mnt/scratch/users/psuwannapichat/any-precision-llm/cache/packed/"
    "anyprec-(Qwen3-4B)-w8_orig3-gc1-c4_s100_blk512"
)
DEFAULT_PROMPT = "Explain the concept of neural network quantization in simple terms."
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_PREFILL_LEN = 512
DEFAULT_WARMUP = 2
DEFAULT_BENCH_RUNS = 5
# Batch sizes for micro-benchmark: covers matmul_kbit (≤8) and dequant_kbit (>8) paths
MICRO_BATCH_SIZES = [1, 4, 8, 16, 64, 256, 512]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def gpu_mem_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def gpu_mem_reserved_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1e9
    return 0.0


def reset_cuda_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_mem_gb():
    cuda_sync()
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def sep(title="", width=76):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*(width - pad - len(title) - 2)}")
    else:
        print("─" * width)


def make_long_input(tokenizer, prompt, target_len, device="cpu"):
    """Repeat/tile prompt tokens to reach target_len, then truncate."""
    base = tokenizer(prompt, return_tensors="pt").input_ids[0]
    if len(base) == 0:
        base = torch.tensor([tokenizer.eos_token_id])
    repeats = (target_len // len(base)) + 2
    tiled = base.repeat(repeats)[:target_len]
    return tiled.unsqueeze(0).to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_anyprec_model(model_path, precisions=None):
    from any_precision import AnyPrecisionForCausalLM

    sep("Loading Any-Precision Model")
    print(f"  Path      : {model_path}")
    print(f"  Precisions: {precisions or 'all supported'}")

    t0 = time.perf_counter()
    model = AnyPrecisionForCausalLM.from_quantized(
        model_path,
        precisions=precisions,
        trust_remote_code=True,
    )
    load_time = time.perf_counter() - t0

    model.eval()
    print(f"  Load time : {load_time:.2f}s")
    print(f"  VRAM used : {gpu_mem_gb():.2f} GB  (reserved {gpu_mem_reserved_gb():.2f} GB)")
    print(f"  Supported : {model.supported_bits}-bit")
    print(f"  Active    : {model.precisions}-bit")
    return model


def load_fp16_model(hf_model_id="Qwen/Qwen3-4B"):
    from transformers import AutoModelForCausalLM

    sep("Loading FP16 Baseline")
    print(f"  Model: {hf_model_id}")
    cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    t0 = time.perf_counter()
    try:
        m = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache,
            trust_remote_code=True,
        )
        m.eval()
        print(f"  Load time : {time.perf_counter()-t0:.2f}s   VRAM: {gpu_mem_gb():.2f} GB")
        return m
    except Exception as e:
        print(f"  Skipped — {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark 1: Linear layer micro-benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_micro_benchmark(model, active_precisions, warmup, bench_runs, device="cuda"):
    """
    Directly benchmark a single AnyPrecisionLinear layer at varying batch sizes.
    Removes all attention/KV-cache overhead — pure quantized matmul cost.

    Path selection (from AnyPrecisionLinear.forward):
        bs ≤ 8  →  matmul_kbit   (reads w_bits qweight rows per call)
        bs > 8  →  dequant_kbit + torch.matmul  (dequant reads w_bits rows)
    """
    sep("Micro-Benchmark: Single AnyPrecisionLinear Layer")

    layer = model.ap_linears[0]
    in_f, out_f = layer.in_features, layer.out_features
    print(f"  Layer shape : ({out_f}, {in_f})  [out × in]")
    print(f"  Batch sizes : {MICRO_BATCH_SIZES}")
    print(f"  matmul_kbit path: bs ≤ 8   |   dequant_kbit path: bs > 8")

    # Pre-warm ALL (batch_size, precision) combos before timing any of them.
    # Each unique combo triggers a new CUDA kernel variant compilation; if we
    # only warm up within the inner loop a previous combo's first-time compile
    # can still poison the first timed run of a later combo.
    print(f"  Pre-warming all (batch × precision) kernel variants...", end="", flush=True)
    warmup_iters = max(warmup, 5)
    for bs in MICRO_BATCH_SIZES:
        x_w = torch.randn(bs, in_f, device=device, dtype=torch.float16)
        for prec in active_precisions:
            layer.set_precision(prec)
            for _ in range(warmup_iters):
                with torch.no_grad():
                    layer(x_w)
    cuda_sync()
    print(" done\n")

    col_w = 16
    header_precs = "  ".join(f"{p}-bit".center(col_w) for p in active_precisions)
    print(f"  {'batch':>6}   {header_precs}")
    print("  " + "─" * (8 + (col_w + 2) * len(active_precisions)))

    results = {}

    for bs in MICRO_BATCH_SIZES:
        x = torch.randn(bs, in_f, device=device, dtype=torch.float16)
        path = "dequant" if bs > 8 else "matmul "
        row_vals = {}

        for prec in active_precisions:
            layer.set_precision(prec)
            run_ms = []
            for _ in range(bench_runs):
                cuda_sync()
                t0 = time.perf_counter()
                with torch.no_grad():
                    layer(x)
                cuda_sync()
                run_ms.append((time.perf_counter() - t0) * 1000)
            run_ms.sort()
            # median of middle half to reject outliers
            lo, hi = len(run_ms) // 4, 3 * len(run_ms) // 4
            row_vals[prec] = sum(run_ms[lo:hi]) / max(len(run_ms[lo:hi]), 1)

        ref_ms = row_vals[max(active_precisions)]
        cells = []
        for prec in active_precisions:
            ms = row_vals[prec]
            speedup = ref_ms / ms
            cells.append(f"{ms:.3f}ms (×{speedup:.2f})".center(col_w))
        print(f"  {bs:>4} {path}   {'  '.join(cells)}")
        results[bs] = row_vals

    print(f"\n  Values: avg latency (ms) and speedup vs {max(active_precisions)}-bit.")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark 2: Prefill benchmark
# ──────────────────────────────────────────────────────────────────────────────

def _prefill_once(model, input_ids, precision, device):
    ids = input_ids.to(device)
    mask = torch.ones_like(ids)
    reset_cuda_stats()
    cuda_sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        model(ids, attention_mask=mask, precision=precision)
    cuda_sync()
    return time.perf_counter() - t0, peak_mem_gb()


def run_prefill_benchmark(model, input_ids_long, active_precisions, warmup, bench_runs, device="cuda"):
    """
    Benchmark full-model forward pass (no token generation) on a long sequence.
    Linear layers are dominant at long sequence lengths.
    """
    seq_len = input_ids_long.shape[1]
    sep(f"Prefill Benchmark  ({seq_len} input tokens)")
    print(f"  Measures time for one forward pass over {seq_len} tokens.\n")

    results = []
    for prec in active_precisions:
        print(f"  {prec}-bit  warmup...", end="", flush=True)
        for _ in range(warmup):
            _prefill_once(model, input_ids_long, prec, device)
        print(" timing...", end="", flush=True)

        times, peaks = [], []
        for _ in range(bench_runs):
            t, pk = _prefill_once(model, input_ids_long, prec, device)
            times.append(t)
            peaks.append(pk)

        avg_t = sum(times) / len(times)
        throughput = seq_len / avg_t
        print(f" {avg_t*1000:.1f}ms  ({throughput:.0f} tok/s)  peak {sum(peaks)/len(peaks):.2f}GB")
        results.append({
            "precision": prec,
            "avg_latency_ms": avg_t * 1000,
            "throughput_tok_s": throughput,
            "peak_vram_gb": sum(peaks) / len(peaks),
        })

    # Print summary table
    print()
    ref = results[-1]
    print(f"  {'Prec':>6}  {'Latency (ms)':>13}  {'Tok/s':>8}  {'Speedup':>8}  {'VRAM (GB)':>10}")
    print("  " + "─" * 56)
    for r in results:
        speedup = ref["avg_latency_ms"] / r["avg_latency_ms"]
        print(f"  {r['precision']:>4}-bit  {r['avg_latency_ms']:>13.1f}  "
              f"{r['throughput_tok_s']:>8.0f}  {speedup:>7.2f}×  {r['peak_vram_gb']:>10.2f}")
    print(f"  (Speedup relative to {ref['precision']}-bit)")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark 3: Decode benchmark (autoregressive generation)
# ──────────────────────────────────────────────────────────────────────────────

def _decode_once(model, tokenizer, input_ids, precision, max_new_tokens, device):
    ids = input_ids.to(device)
    mask = torch.ones_like(ids)
    reset_cuda_stats()
    cuda_sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            ids,
            attention_mask=mask,
            pad_token_id=tokenizer.eos_token_id,
            precision=precision,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    cuda_sync()
    elapsed = time.perf_counter() - t0
    n_new = out.shape[1] - ids.shape[1]
    return elapsed, n_new, peak_mem_gb()


def _fp16_decode_once(model, tokenizer, input_ids, max_new_tokens, device):
    ids = input_ids.to(device)
    mask = torch.ones_like(ids)
    reset_cuda_stats()
    cuda_sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            ids,
            attention_mask=mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    cuda_sync()
    elapsed = time.perf_counter() - t0
    n_new = out.shape[1] - ids.shape[1]
    return elapsed, n_new, peak_mem_gb()


def run_decode_benchmark(model, tokenizer, input_ids, active_precisions,
                         max_new_tokens, warmup, bench_runs, device="cuda"):
    sep(f"Decode Benchmark  (batch=1, generate {max_new_tokens} tokens)")
    print("  Note: attention/KV-cache dominates at batch=1; bit-width speedup is small.\n")

    results = []
    for prec in active_precisions:
        print(f"  {prec}-bit  warmup...", end="", flush=True)
        for _ in range(warmup):
            _decode_once(model, tokenizer, input_ids, prec, max_new_tokens, device)
        print(" timing...", end="", flush=True)

        times, peaks = [], []
        for _ in range(bench_runs):
            t, n_new, pk = _decode_once(model, tokenizer, input_ids, prec, max_new_tokens, device)
            times.append(t)
            peaks.append(pk)

        avg_t = sum(times) / len(times)
        tp = n_new / avg_t
        print(f" {avg_t:.3f}s  ({tp:.1f} tok/s)  peak {sum(peaks)/len(peaks):.2f}GB")
        results.append({
            "precision": prec,
            "avg_latency_s": avg_t,
            "throughput_tok_s": tp,
            "tokens_generated": n_new,
            "peak_vram_gb": sum(peaks) / len(peaks),
        })

    print()
    ref = results[-1]
    print(f"  {'Prec':>6}  {'Latency (s)':>12}  {'Tok/s':>8}  {'Speedup':>8}  {'VRAM (GB)':>10}")
    print("  " + "─" * 52)
    for r in results:
        speedup = ref["avg_latency_s"] / r["avg_latency_s"]
        print(f"  {r['precision']:>4}-bit  {r['avg_latency_s']:>12.3f}  "
              f"{r['throughput_tok_s']:>8.1f}  {speedup:>7.2f}×  {r['peak_vram_gb']:>10.2f}")
    print(f"  (Speedup relative to {ref['precision']}-bit)")
    return results


def run_fp16_decode_benchmark(model, tokenizer, input_ids, max_new_tokens,
                              warmup, bench_runs, device="cuda"):
    print(f"  fp16  warmup...", end="", flush=True)
    for _ in range(warmup):
        _fp16_decode_once(model, tokenizer, input_ids, max_new_tokens, device)
    print(" timing...", end="", flush=True)

    times, peaks = [], []
    for _ in range(bench_runs):
        t, n_new, pk = _fp16_decode_once(model, tokenizer, input_ids, max_new_tokens, device)
        times.append(t)
        peaks.append(pk)

    avg_t = sum(times) / len(times)
    print(f" {avg_t:.3f}s  ({n_new/avg_t:.1f} tok/s)  peak {sum(peaks)/len(peaks):.2f}GB")
    return {"precision": "fp16", "avg_latency_s": avg_t,
            "throughput_tok_s": n_new / avg_t, "tokens_generated": n_new,
            "peak_vram_gb": sum(peaks) / len(peaks)}


# ──────────────────────────────────────────────────────────────────────────────
# Inference demo
# ──────────────────────────────────────────────────────────────────────────────

def run_inference_demo(model, tokenizer, input_ids, precisions, max_new_tokens, device="cuda"):
    sep("Inference Demo")
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"  Prompt: {prompt_text}\n")
    for prec in precisions:
        ids = input_ids.to(device)
        mask = torch.ones_like(ids)
        cuda_sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                ids, attention_mask=mask, pad_token_id=tokenizer.eos_token_id,
                precision=prec, max_new_tokens=max_new_tokens,
                do_sample=False, use_cache=True,
            )
        cuda_sync()
        elapsed = time.perf_counter() - t0
        n_new = out.shape[1] - ids.shape[1]
        generated = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        print(f"  ── {prec}-bit ──")
        print(f"  {generated.strip()}")
        print(f"  [{n_new} tokens in {elapsed:.2f}s | {n_new/elapsed:.1f} tok/s | peak {peak_mem_gb():.2f} GB]\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Any-Precision LLM inference & benchmark")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--precisions", nargs="+", type=int, default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prefill_len", type=int, default=DEFAULT_PREFILL_LEN,
                        help="Input token count for prefill benchmark")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--warmup_runs", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--bench_runs", type=int, default=DEFAULT_BENCH_RUNS)
    parser.add_argument("--skip_demo", action="store_true")
    parser.add_argument("--skip_micro", action="store_true")
    parser.add_argument("--skip_prefill", action="store_true")
    parser.add_argument("--skip_decode", action="store_true")
    parser.add_argument("--skip_fp16", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sep("Environment")
    print(f"  Device : {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU    : {props.name}")
        print(f"  VRAM   : {props.total_memory / 1e9:.1f} GB")
    print(f"  PyTorch: {torch.__version__}")

    sep("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    print(f"  Prompt tokens: {input_ids.shape[1]}")

    model = load_anyprec_model(args.model_path, precisions=args.precisions)
    active_precisions = sorted(model.precisions)

    if device == "cuda" and next(model.model.parameters()).device.type == "cpu":
        print("  Moving model to GPU...")
        model.model.to(device)

    if not args.skip_demo:
        run_inference_demo(model, tokenizer, input_ids, active_precisions, args.max_new_tokens, device)

    if not args.skip_micro:
        run_micro_benchmark(model, active_precisions, args.warmup_runs, args.bench_runs, device)

    if not args.skip_prefill:
        long_ids = make_long_input(tokenizer, args.prompt, args.prefill_len)
        print(f"\n  Prefill input: {long_ids.shape[1]} tokens")
        run_prefill_benchmark(model, long_ids, active_precisions,
                              args.warmup_runs, args.bench_runs, device)

    if not args.skip_decode:
        run_decode_benchmark(model, tokenizer, input_ids, active_precisions,
                             args.max_new_tokens, args.warmup_runs, args.bench_runs, device)

    if not args.skip_fp16 and device == "cuda":
        sep("FP16 Baseline — Decode")
        print("  Freeing any-precision model to free VRAM...")
        del model
        gc.collect()
        torch.cuda.empty_cache()

        fp16_model = load_fp16_model()
        if fp16_model is not None:
            r = run_fp16_decode_benchmark(fp16_model, tokenizer, input_ids,
                                          args.max_new_tokens, args.warmup_runs, args.bench_runs, device)
            sep()
            print(f"  FP16 decode: {r['avg_latency_s']:.3f}s  {r['throughput_tok_s']:.1f} tok/s"
                  f"  {r['peak_vram_gb']:.2f}GB")
            del fp16_model
            gc.collect()
            torch.cuda.empty_cache()

    sep()


if __name__ == "__main__":
    main()
