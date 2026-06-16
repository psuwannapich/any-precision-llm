"""
Server throughput benchmark — measures end-to-end performance at each bit width
through the live HTTP API.

Three scenarios per precision:
  1. Serial      — one request at a time (baseline latency)
  2. Batch-8     — 8 concurrent requests (tests dynamic batching gain)
  3. Batch-max   — max_batch_size concurrent requests (peak throughput)

Usage:
  python server_benchmark.py [--host HOST] [--port PORT]
                              [--precisions 3 4 5 6 7 8]
                              [--warmup N] [--runs N]
                              [--max_tokens N] [--batch_size N]
"""

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import requests

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
DEFAULT_PROMPT = (
    "Explain what neural network quantization is in 2-3 sentences."
)
DEFAULT_MAX_TOKENS = 64
DEFAULT_WARMUP = 2
DEFAULT_RUNS = 5
DEFAULT_BATCH_SIZES = [1, 4, 8]


# ──────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────────────────────────────────────

def _post(base_url: str, precision: int, prompt: str,
          max_tokens: int, timeout: int = 120) -> dict:
    """Send a single non-streaming chat completion request, return timing info."""
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "precision": precision,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    t0 = time.perf_counter()
    r = requests.post(f"{base_url}/v1/chat/completions",
                      json=payload, timeout=timeout)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    n_tokens = data["usage"]["completion_tokens"]
    return {"latency": elapsed, "tokens": n_tokens}


def _run_concurrent(base_url: str, precision: int, prompt: str,
                    max_tokens: int, batch_size: int) -> dict:
    """
    Fire `batch_size` requests simultaneously.
    Returns wall-clock time, total tokens, per-request latencies.
    """
    results = []
    t_wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = [
            pool.submit(_post, base_url, precision, prompt, max_tokens)
            for _ in range(batch_size)
        ]
        for f in as_completed(futures):
            results.append(f.result())

    wall_time = time.perf_counter() - t_wall_start
    total_tokens = sum(r["tokens"] for r in results)
    latencies = [r["latency"] for r in results]
    return {
        "wall_time": wall_time,
        "total_tokens": total_tokens,
        "latencies": latencies,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark core
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_precision(base_url: str, precision: int, prompt: str,
                        max_tokens: int, batch_sizes: List[int],
                        warmup: int, runs: int) -> Dict:
    print(f"\n  [{precision}-bit]  warmup...", end="", flush=True)

    # Warmup — always serial
    for _ in range(warmup):
        _post(base_url, precision, prompt, max_tokens)
    print(" done")

    results_by_batch = {}

    for bs in batch_sizes:
        label = f"batch={bs}"
        run_data = []

        print(f"    {label}  ", end="", flush=True)
        for i in range(runs):
            d = _run_concurrent(base_url, precision, prompt, max_tokens, bs)
            run_data.append(d)
            tok_s = d["total_tokens"] / d["wall_time"]
            print(f".", end="", flush=True)

        wall_times  = [d["wall_time"]   for d in run_data]
        total_toks  = [d["total_tokens"] for d in run_data]
        all_lats    = [lat for d in run_data for lat in d["latencies"]]

        avg_wall   = statistics.mean(wall_times)
        avg_tokens = statistics.mean(total_toks)

        results_by_batch[bs] = {
            "avg_wall_s":        avg_wall,
            "avg_total_tokens":  avg_tokens,
            "throughput_tok_s":  avg_tokens / avg_wall,
            "throughput_req_s":  bs / avg_wall,
            "p50_latency_s":     statistics.median(all_lats),
            "p99_latency_s":     sorted(all_lats)[int(len(all_lats) * 0.99)],
        }
        r = results_by_batch[bs]
        print(f"  wall={avg_wall:.2f}s  {r['throughput_tok_s']:.1f} tok/s  "
              f"p50={r['p50_latency_s']:.2f}s")

    return results_by_batch


# ──────────────────────────────────────────────────────────────────────────────
# Results table
# ──────────────────────────────────────────────────────────────────────────────

def sep(title="", width=78):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*(width - pad - len(title) - 2)}")
    else:
        print("─" * width)


def print_table(all_results: dict, batch_sizes: List[int]):
    precisions = sorted(all_results.keys())

    for bs in batch_sizes:
        sep(f"Batch size = {bs}")
        print(f"  {'Prec':>6}  {'Tok/s':>8}  {'Req/s':>7}  "
              f"{'Wall (s)':>9}  {'p50 lat':>8}  {'p99 lat':>8}  {'Speedup':>8}")
        print("  " + "─" * 66)

        ref = all_results[precisions[-1]][bs]   # 8-bit is baseline
        for prec in precisions:
            r = all_results[prec][bs]
            speedup = r["throughput_tok_s"] / ref["throughput_tok_s"]
            print(f"  {prec:>4}-bit  "
                  f"{r['throughput_tok_s']:>8.1f}  "
                  f"{r['throughput_req_s']:>7.2f}  "
                  f"{r['avg_wall_s']:>9.2f}  "
                  f"{r['p50_latency_s']:>7.2f}s  "
                  f"{r['p99_latency_s']:>7.2f}s  "
                  f"{'×'+str(round(speedup,2)):>8}")
        print(f"  (Speedup vs 8-bit)")

    # Cross-batch summary: how much batching helps for each precision
    sep("Batching gain  (throughput at batch=N vs batch=1)")
    print(f"  {'Prec':>6}  " + "  ".join(f"bs={bs:>2} tok/s  gain" for bs in batch_sizes[1:]))
    print("  " + "─" * (10 + 20 * (len(batch_sizes) - 1)))
    for prec in precisions:
        serial = all_results[prec][batch_sizes[0]]["throughput_tok_s"]
        parts = [f"  {prec:>4}-bit  "]
        for bs in batch_sizes[1:]:
            tp = all_results[prec][bs]["throughput_tok_s"]
            parts.append(f"{tp:>8.1f}  ×{tp/serial:.2f}    ")
        print("".join(parts))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--precisions", nargs="+", type=int, default=None,
                        help="Precisions to test (default: all from /v1/info)")
    parser.add_argument("--batch_sizes", nargs="+", type=int,
                        default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Fetch supported precisions from server
    info = requests.get(f"{base_url}/v1/info", timeout=10).json()
    supported = info["supported_precisions"]
    precisions = sorted(args.precisions or supported)
    for p in precisions:
        if p not in supported:
            print(f"ERROR: precision {p} not supported by server (has {supported})")
            sys.exit(1)

    sep("Server Benchmark Configuration")
    print(f"  Server     : {base_url}")
    print(f"  GPU        : {info.get('gpu', 'N/A')}  ({info.get('vram_total_gb', '?')} GB)")
    print(f"  Model      : {info['model']}")
    print(f"  Precisions : {precisions}")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  Max tokens : {args.max_tokens}")
    print(f"  Runs       : {args.runs}  (warmup={args.warmup})")
    print(f"  Prompt     : {args.prompt[:60]}...")

    sep("Running Benchmarks")
    all_results = {}
    for prec in precisions:
        all_results[prec] = benchmark_precision(
            base_url=base_url,
            precision=prec,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            batch_sizes=args.batch_sizes,
            warmup=args.warmup,
            runs=args.runs,
        )

    print_table(all_results, args.batch_sizes)
    sep()

    # Reset server metrics
    requests.get(f"{base_url}/v1/metrics")


if __name__ == "__main__":
    main()
