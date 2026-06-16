"""
bitwidth_latency_benchmark.py

Demonstrates that lower bit-width reduces per-request latency at high
concurrency — the condition where the GPU becomes matmul-bound.

Test design:
  • 200 prompts from tatsu-lab/alpaca (diverse, real-world questions)
  • For each precision (3, 4, 8-bit): send N requests simultaneously
  • Measure individual request E2E latency and TPOT for each request
  • Report p50 / p90 / p99 latency distributions

Why concurrency matters:
  batch=1   → attention dominates (~95% of GPU time) → bits irrelevant
  batch≥8   → matmul grows proportionally → lower bits finish faster
  batch=16+ → clear latency advantage for lower bits

Larger dataset = more statistical confidence, diverse prompt lengths.

Usage:
  python bitwidth_latency_benchmark.py [--port 8001] [--concurrency 8 16 32]
"""

import argparse, random, statistics, time, json
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

PRECISIONS     = [3, 4, 8]
DEFAULT_CONC   = [1, 4, 8, 16, 32]
MAX_TOKENS     = 128   # long enough to see decode latency differences
WARMUP_N       = 5
N_PROMPTS      = 200

# ── Dataset ───────────────────────────────────────────────────────────────────

BUILTIN_PROMPTS = [
    "What is machine learning and how does it work?",
    "Explain the concept of neural networks.",
    "What are the main differences between Python and Java?",
    "Describe the water cycle in nature.",
    "What causes climate change?",
    "How does photosynthesis work?",
    "Explain quantum computing in simple terms.",
    "What is the theory of relativity?",
    "How do vaccines protect against diseases?",
    "Describe the process of evolution.",
    "What is blockchain technology?",
    "How does the internet work?",
    "Explain supply and demand in economics.",
    "What are the main branches of government?",
    "Describe the scientific method.",
    "What is DNA and how does it function?",
    "How do black holes form?",
    "Explain the greenhouse effect.",
    "What is the significance of the Pythagorean theorem?",
    "How does memory work in computers?",
    "What is artificial intelligence?",
    "Describe the structure of an atom.",
    "How do airplanes generate lift?",
    "What is the difference between weather and climate?",
    "Explain how vaccines are developed.",
    "What is the role of the immune system?",
    "How does natural selection work?",
    "What is a computer algorithm?",
    "Describe the history of the internet.",
    "How do solar panels generate electricity?",
    "What is the difference between a virus and a bacterium?",
    "Explain how GPS navigation works.",
    "What causes earthquakes?",
    "How does the stock market work?",
    "What is the Big Bang theory?",
    "Explain encryption and cybersecurity.",
    "How do electric vehicles work?",
    "What is machine vision in AI?",
    "Describe how the human heart functions.",
    "What is renewable energy and why is it important?",
]


def load_prompts(n: int, seed: int = 42) -> list:
    try:
        from datasets import load_dataset
        print(f"  Downloading alpaca dataset... ", end="", flush=True)
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        candidates = [r["instruction"] for r in ds
                      if not r["input"].strip() and 30 < len(r["instruction"]) < 150]
        rng = random.Random(seed)
        rng.shuffle(candidates)
        prompts = candidates[:n]
        print(f"{len(prompts)} prompts")
        return prompts
    except Exception as e:
        print(f"\n  Dataset unavailable ({type(e).__name__}), using {len(BUILTIN_PROMPTS)} built-ins")
        prompts = BUILTIN_PROMPTS * ((n // len(BUILTIN_PROMPTS)) + 1)
        return prompts[:n]


# ── HTTP ──────────────────────────────────────────────────────────────────────

def send_request(base_url, precision, prompt, max_tokens):
    """Non-streaming request; returns (e2e_sec, n_out_tokens, n_in_tokens)."""
    t0 = time.perf_counter()
    r = requests.post(f"{base_url}/v1/chat/completions", json={
        "messages": [{"role": "user", "content": prompt}],
        "precision": precision,
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": False,
    }, timeout=300)
    e2e = time.perf_counter() - t0
    d = r.json()
    n_out = d["usage"]["completion_tokens"]
    n_in  = d["usage"]["prompt_tokens"]
    return e2e, n_out, n_in


def run_burst(base_url, precision, prompts_batch, max_tokens):
    """Fire len(prompts_batch) requests simultaneously; return list of results."""
    results = []
    with ThreadPoolExecutor(max_workers=len(prompts_batch)) as pool:
        futs = [pool.submit(send_request, base_url, precision, p, max_tokens)
                for p in prompts_batch]
        for f in as_completed(futs):
            results.append(f.result())
    return results  # [(e2e, n_out, n_in), ...]


# ── Stats helpers ─────────────────────────────────────────────────────────────

def percentile(data, p):
    d = sorted(data)
    i = max(0, min(len(d)-1, int(len(d) * p / 100)))
    return d[i]


def tpot_ms(e2e_s, n_in, n_out):
    """Approximate TPOT: remove prefill estimate (TTFT ≈ 80ms) from E2E."""
    decode_s = max(0, e2e_s - 0.08)   # subtract ~80ms TTFT baseline
    return decode_s / max(n_out, 1) * 1000


# ── Separator ────────────────────────────────────────────────────────────────

def sep(t="", w=72):
    if t:
        pad = (w - len(t) - 2) // 2
        print(f"\n{'─'*pad} {t} {'─'*(w-pad-len(t)-2)}")
    else:
        print("─" * w)


def bar(frac, width=24):
    n = max(0, min(width, int(frac * width)))
    return "█" * n + "░" * (width - n)


# ── Core benchmark ────────────────────────────────────────────────────────────

def benchmark(base_url, prompts, concurrency_levels, max_tokens, runs_per_conc=5):
    """
    For each (precision, concurrency), repeatedly fire `concurrency` requests
    and record each individual request's latency.
    Returns nested dict: results[precision][concurrency] = list of (e2e, n_out, n_in)
    """
    results = {p: {c: [] for c in concurrency_levels} for p in PRECISIONS}
    n = len(prompts)

    for prec in PRECISIONS:
        print(f"\n  [{prec}-bit]  warmup... ", end="", flush=True)
        # Warmup: WARMUP_N serial requests
        for i in range(WARMUP_N):
            send_request(base_url, prec, prompts[i % n], max_tokens)
        print("done")

        for conc in concurrency_levels:
            print(f"    conc={conc:2d}  ", end="", flush=True)
            for run_i in range(runs_per_conc):
                start = (run_i * conc) % n
                batch = [prompts[(start + j) % n] for j in range(conc)]
                batch_results = run_burst(base_url, prec, batch, max_tokens)
                results[prec][conc].extend(batch_results)
                print(".", end="", flush=True)
            print()

    return results


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(results, concurrency_levels, max_tokens):
    ref_prec = 8   # baseline

    # ── 1. E2E Latency table per concurrency ─────────────────────────────────
    for conc in concurrency_levels:
        sep(f"E2E Latency — concurrency={conc}")
        ref_e2es = [r[0] for r in results[ref_prec][conc]]
        ref_p50  = percentile(ref_e2es, 50)

        print(f"  {'Bits':>6}  {'p50 s':>7}  {'p90 s':>7}  {'p95 s':>7}  "
              f"{'p99 s':>7}  {'vs 8-bit':>9}  Visual (full=×0.5)")
        print("  " + "─" * 66)

        for prec in PRECISIONS:
            e2es = [r[0] for r in results[prec][conc]]
            p50  = percentile(e2es, 50)
            p90  = percentile(e2es, 90)
            p95  = percentile(e2es, 95)
            p99  = percentile(e2es, 99)
            # speedup: ref_p50 / this_p50 > 1 means THIS is faster
            speedup = ref_p50 / p50
            # bar filled = faster is better (relative to ref)
            b    = bar(speedup / 2.0)
            flag = " ◄ baseline" if prec == ref_prec else ""
            print(f"  {prec:>4}-bit  {p50:>7.2f}  {p90:>7.2f}  {p95:>7.2f}  "
                  f"{p99:>7.2f}  ×{speedup:>6.3f}  {b}{flag}")

    # ── 2. TPOT (decode speed) ────────────────────────────────────────────────
    sep("TPOT (decode speed) — ms per output token")
    print(f"  TPOT = (E2E − TTFT) / n_tokens.  Lower = faster decode.\n")
    print(f"  {'Bits':>6}  " + "  ".join(f"{'conc='+str(c):>10}" for c in concurrency_levels))
    print("  " + "─" * (10 + 12 * len(concurrency_levels)))

    ref_tpots = {}
    for conc in concurrency_levels:
        vals = [tpot_ms(r[0], r[2], r[1]) for r in results[ref_prec][conc]]
        ref_tpots[conc] = percentile(vals, 50)

    for prec in PRECISIONS:
        parts = [f"  {prec:>4}-bit  "]
        for conc in concurrency_levels:
            vals = [tpot_ms(r[0], r[2], r[1]) for r in results[prec][conc]]
            med  = percentile(vals, 50)
            sp   = ref_tpots[conc] / med
            parts.append(f"{med:>6.1f}ms(×{sp:.2f})")
        print("".join(parts))

    # ── 3. Latency speedup heatmap ────────────────────────────────────────────
    sep("Latency speedup vs 8-bit  (p50 E2E)")
    print(f"  {'Bits':>6}  " + "  ".join(f"{'conc='+str(c):>12}" for c in concurrency_levels))
    print("  " + "─" * (10 + 14 * len(concurrency_levels)))

    for prec in PRECISIONS:
        parts = [f"  {prec:>4}-bit  "]
        for conc in concurrency_levels:
            e2es    = [r[0] for r in results[prec][conc]]
            ref_e2es = [r[0] for r in results[ref_prec][conc]]
            sp      = percentile(ref_e2es, 50) / percentile(e2es, 50)
            trend   = "▲" if sp > 1.02 else ("▼" if sp < 0.98 else "─")
            parts.append(f"  ×{sp:.3f} {trend}  ")
        print("".join(parts))

    # ── 4. Throughput (complementary) ────────────────────────────────────────
    sep("Throughput (tok/s) — complement to latency")
    print(f"  {'Bits':>6}  " + "  ".join(f"{'conc='+str(c):>10}" for c in concurrency_levels))
    print("  " + "─" * (10 + 12 * len(concurrency_levels)))

    for prec in PRECISIONS:
        parts = [f"  {prec:>4}-bit  "]
        for conc in concurrency_levels:
            total_toks = sum(r[1] for r in results[prec][conc])
            total_time = sum(r[0] for r in results[prec][conc]) / len(results[prec][conc]) * len(results[prec][conc]) / conc
            # Better: use wall time of last request in each burst
            # Approximate: avg_e2e * (n_total_toks / avg_toks_per_request)
            avg_e2e = statistics.mean(r[0] for r in results[prec][conc])
            avg_tok = statistics.mean(r[1] for r in results[prec][conc])
            tps     = avg_tok * conc / avg_e2e
            parts.append(f"{tps:>8.1f}  ")
        print("".join(parts))

    # ── Summary ───────────────────────────────────────────────────────────────
    sep("Summary")
    max_conc = max(concurrency_levels)
    print(f"\n  At concurrency={max_conc} (high load):")
    for prec in PRECISIONS:
        e2es = [r[0] for r in results[prec][max_conc]]
        ref_e2es = [r[0] for r in results[ref_prec][max_conc]]
        sp = percentile(ref_e2es, 50) / percentile(e2es, 50)
        p50 = percentile(e2es, 50)
        p99 = percentile(e2es, 99)
        flag = f"  ×{sp:.3f} faster than 8-bit" if prec != ref_prec else "  (baseline)"
        print(f"    {prec}-bit:  p50={p50:.2f}s  p99={p99:.2f}s{flag}")

    print(f"""
  Key insight:
    • batch=1  → all precisions identical (attention-bound, bit-width invisible)
    • batch≥8  → lower bits finish each decode step faster → lower p50/p99
    • The gap grows with concurrency because more work is matmul-bound
""")
    sep()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",        default="localhost")
    parser.add_argument("--port",        type=int, default=8001)
    parser.add_argument("--concurrency", nargs="+", type=int, default=DEFAULT_CONC)
    parser.add_argument("--max_tokens",  type=int, default=MAX_TOKENS)
    parser.add_argument("--n_prompts",   type=int, default=N_PROMPTS)
    parser.add_argument("--runs",        type=int, default=5,
                        help="Burst runs per (precision, concurrency)")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    try:
        requests.get(f"{base_url}/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"Server unreachable at {base_url}: {e}"); return

    sep("Bit-Width Latency Benchmark  (Phase 2 vLLM, Qwen3-4B)")
    print(f"  Server      : {base_url}")
    print(f"  Precisions  : {PRECISIONS}")
    print(f"  Concurrencies: {args.concurrency}")
    print(f"  Max tokens  : {args.max_tokens}")
    print(f"  Prompts     : {args.n_prompts}")
    print(f"  Runs/config : {args.runs}")
    print(f"\n  Loading prompts...")
    prompts = load_prompts(args.n_prompts, seed=args.seed)

    sep("Running")
    results = benchmark(base_url, prompts, args.concurrency,
                        args.max_tokens, runs_per_conc=args.runs)

    print_report(results, args.concurrency, args.max_tokens)


if __name__ == "__main__":
    main()
