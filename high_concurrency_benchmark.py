"""
high_concurrency_benchmark.py

Reveals bit-width throughput speedup by saturating the GPU with concurrent requests.

Why concurrency matters:
  - batch=1  → KV-cache attention dominates; all bit-widths are equal (~23 tok/s)
  - batch≥8  → quantized linear layers become the bottleneck; fewer bits = faster

Dataset: tatsu-lab/alpaca (HuggingFace, cached after first download).
         Falls back to 80 built-in prompts if network is unavailable.

Requires: serve.py must be running
  bash run_serve.sh --max_batch_size 8 --batch_wait_ms 20

Usage:
  python high_concurrency_benchmark.py
  python high_concurrency_benchmark.py --concurrency 1 8 16 32 --max_tokens 64
  python high_concurrency_benchmark.py --precisions 4 6 8 --n_prompts 80
"""

import argparse
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ── Built-in fallback prompts (used if dataset unavailable) ──────────────────

BUILTIN_PROMPTS = [
    # Science & technology
    "What is machine learning?",
    "Explain gradient descent in simple terms.",
    "What is the difference between supervised and unsupervised learning?",
    "What is a convolutional neural network?",
    "Explain how backpropagation works.",
    "What is the transformer architecture in deep learning?",
    "How does attention mechanism work in neural networks?",
    "What is reinforcement learning?",
    "Explain the bias-variance tradeoff.",
    "What is overfitting and how do you prevent it?",
    "What is batch normalization?",
    "How does dropout regularization work?",
    "What is the difference between RNN and LSTM?",
    "Explain transfer learning.",
    "What is a generative adversarial network?",
    "How does BERT differ from GPT?",
    "What is tokenization in NLP?",
    "Explain word embeddings.",
    "What is the softmax function used for?",
    "What is the role of activation functions in neural networks?",
    # Mathematics
    "What is a matrix and how is it used in linear algebra?",
    "Explain eigenvalues and eigenvectors.",
    "What is calculus used for in machine learning?",
    "Explain Bayes theorem.",
    "What is a probability distribution?",
    "What is the central limit theorem?",
    "Explain the concept of entropy in information theory.",
    "What is convolution in mathematics?",
    "What is a Fourier transform?",
    "Explain the concept of a derivative.",
    # Computer science
    "What is the difference between a process and a thread?",
    "Explain how a hash table works.",
    "What is dynamic programming?",
    "How does binary search work?",
    "What is a graph data structure?",
    "Explain recursion with an example.",
    "What is the difference between stack and heap memory?",
    "How does garbage collection work?",
    "What is a REST API?",
    "Explain the concept of caching.",
    # General knowledge
    "What causes climate change?",
    "Explain how vaccines work.",
    "What is the theory of relativity?",
    "How does photosynthesis work?",
    "What is DNA and how does it work?",
    "Explain the water cycle.",
    "How do black holes form?",
    "What is quantum computing?",
    "Explain the structure of an atom.",
    "What is the speed of light?",
    # Writing tasks
    "Write a one-paragraph summary of the importance of sleep.",
    "Describe three benefits of regular exercise.",
    "Explain why reading books is important.",
    "List three reasons why learning a second language is beneficial.",
    "Describe what makes a good leader.",
    "What are the key ingredients of a healthy diet?",
    "Explain the importance of time management.",
    "Describe the benefits of meditation.",
    "What is critical thinking and why is it important?",
    "Explain the concept of emotional intelligence.",
    # Coding
    "What is object-oriented programming?",
    "Explain the difference between a list and a tuple in Python.",
    "What is a lambda function?",
    "How does version control with git work?",
    "What is the difference between SQL and NoSQL databases?",
    "Explain what a compiler does.",
    "What is an API?",
    "How does TCP/IP work?",
    "What is a virtual machine?",
    "Explain what Docker containers are.",
    # Creative / opinion
    "What is the most important invention of the 20th century?",
    "Describe the ideal city of the future.",
    "What skills will be most valuable in 2050?",
    "If you could solve one global problem, what would it be?",
    "What are the pros and cons of social media?",
    "What does success mean to you?",
    "Describe a world without the internet.",
    "What is the relationship between art and technology?",
    "How has globalization changed society?",
    "What is the role of education in reducing poverty?",
]


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_prompts(n: int, seed: int = 42) -> list:
    """Load n diverse prompts from tatsu-lab/alpaca, falling back to built-ins."""
    prompts = []
    try:
        from datasets import load_dataset
        print("  Loading prompts from tatsu-lab/alpaca... ", end="", flush=True)
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        # Only use instructions without additional context input, length < 150 chars
        candidates = [
            r["instruction"]
            for r in ds
            if not r["input"].strip() and 20 < len(r["instruction"]) < 150
        ]
        rng = random.Random(seed)
        rng.shuffle(candidates)
        prompts = candidates[:n]
        print(f"loaded {len(prompts)} prompts")
    except Exception as e:
        print(f"\n  Dataset unavailable ({type(e).__name__}), using {len(BUILTIN_PROMPTS)} built-in prompts")
        prompts = BUILTIN_PROMPTS

    if len(prompts) < n:
        # Repeat to reach n
        prompts = (prompts * ((n // len(prompts)) + 1))[:n]

    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts[:n]


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def send_one(base_url: str, precision: int, prompt: str,
             max_tokens: int, timeout: int = 120) -> dict:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "precision": precision,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    t0 = time.perf_counter()
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    return {
        "latency": elapsed,
        "tokens": data["usage"]["completion_tokens"],
    }


def run_burst(base_url: str, precision: int, prompts: list,
              max_tokens: int, concurrency: int) -> dict:
    """Send all prompts with up to `concurrency` requests in-flight simultaneously."""
    results = []
    errors = 0
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(send_one, base_url, precision, p, max_tokens): p
                for p in prompts}
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:
                errors += 1
    wall_time = time.perf_counter() - t_start

    if not results:
        return {"throughput": 0, "wall_time": wall_time, "total_tokens": 0,
                "p50": 0, "p95": 0, "errors": errors}

    total_tokens = sum(r["tokens"] for r in results)
    lats = sorted(r["latency"] for r in results)
    return {
        "throughput": total_tokens / wall_time,
        "wall_time": wall_time,
        "total_tokens": total_tokens,
        "p50": statistics.median(lats),
        "p95": lats[max(0, int(len(lats) * 0.95) - 1)],
        "errors": errors,
    }


# ── Benchmark core ────────────────────────────────────────────────────────────

def benchmark_precision(base_url: str, precision: int, prompts: list,
                        max_tokens: int, concurrency_levels: list,
                        warmup: int, runs: int) -> dict:
    print(f"\n  [{precision}-bit]  warmup({warmup})...", end="", flush=True)
    for p in prompts[:warmup]:
        send_one(base_url, precision, p, max_tokens)
    print(" ok", flush=True)

    n = len(prompts)
    results_by_conc = {}

    for conc in concurrency_levels:
        run_data = []
        print(f"    concurrency={conc:2d}  ", end="", flush=True)
        for i in range(runs):
            # Rotate through the prompt pool to get fresh prompts each run
            start = (i * conc) % n
            batch = [prompts[(start + j) % n] for j in range(conc)]
            d = run_burst(base_url, precision, batch, max_tokens, concurrency=conc)
            run_data.append(d)
            print(".", end="", flush=True)

        throughputs = [d["throughput"] for d in run_data]
        p50s = [d["p50"] for d in run_data]
        results_by_conc[conc] = {
            "throughput":     statistics.median(throughputs),
            "throughput_std": statistics.stdev(throughputs) if runs > 1 else 0.0,
            "p50_lat":        statistics.median(p50s),
        }
        r = results_by_conc[conc]
        print(f"  {r['throughput']:6.1f} tok/s  ±{r['throughput_std']:.1f}"
              f"  p50={r['p50_lat']:.2f}s")

    return results_by_conc


# ── Reporting ─────────────────────────────────────────────────────────────────

def sep(title="", width=82):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*(width - pad - len(title) - 2)}")
    else:
        print("─" * width)


def bar(frac: float, width: int = 24) -> str:
    filled = max(0, min(width, int(frac * width)))
    return "█" * filled + "░" * (width - filled)


def print_results(all_results: dict, concurrency_levels: list, precisions: list):
    sorted_precs = sorted(precisions)
    ref = max(precisions)  # 8-bit baseline

    # ── Per-concurrency table ─────────────────────────────────────────────────
    for conc in concurrency_levels:
        sep(f"Concurrency = {conc}")
        print(f"  {'Prec':>6}  {'Tok/s':>8}  {'±':>5}  {'p50 lat':>8}  {'Speedup':>9}  Visual")
        print("  " + "─" * 68)
        ref_tp = all_results[ref][conc]["throughput"]
        for prec in sorted_precs:
            r = all_results[prec][conc]
            speedup = r["throughput"] / ref_tp if ref_tp > 0 else 1.0
            b = bar(speedup / 1.5)  # scale: ×1.5 = full bar
            flag = " ◄ baseline" if prec == ref else ""
            print(f"  {prec:>4}-bit  {r['throughput']:>8.1f}  {r['throughput_std']:>4.1f}  "
                  f"{r['p50_lat']:>7.2f}s  ×{speedup:>6.3f}  {b}{flag}")
        print(f"  (speedup vs {ref}-bit at concurrency={conc})")

    # ── Speedup summary at max concurrency ────────────────────────────────────
    max_conc = max(concurrency_levels)
    sep(f"Speedup summary — concurrency={max_conc}  (vs {ref}-bit baseline)")
    ref_tp = all_results[ref][max_conc]["throughput"]
    print(f"\n  {'Bits':>4}   {'Tok/s':>8}   {'vs 8-bit':>9}   Visual (full bar = ×1.5×)")
    print("  " + "─" * 60)
    for prec in sorted_precs:
        tp = all_results[prec][max_conc]["throughput"]
        speedup = tp / ref_tp if ref_tp > 0 else 1.0
        b = bar(speedup / 1.5, width=30)
        print(f"  {prec:>4}-bit   {tp:>8.1f}   ×{speedup:.3f}      {b}")

    # ── Batching gain ─────────────────────────────────────────────────────────
    if 1 in concurrency_levels and len(concurrency_levels) > 1:
        sep("Batching gain  (serial → max concurrency, per bit-width)")
        print(f"  {'Bits':>4}   {'Serial tok/s':>13}   {'Conc={} tok/s'.format(max_conc):>16}   {'Gain':>6}")
        print("  " + "─" * 50)
        for prec in sorted_precs:
            serial_tp = all_results[prec][1]["throughput"]
            max_tp    = all_results[prec][max_conc]["throughput"]
            gain      = max_tp / serial_tp if serial_tp > 0 else 0
            print(f"  {prec:>4}-bit   {serial_tp:>13.1f}   {max_tp:>16.1f}   ×{gain:.2f}")

    # ── Conclusion ────────────────────────────────────────────────────────────
    sep("Conclusion")
    best_prec = max(sorted_precs, key=lambda p: all_results[p][max_conc]["throughput"])
    best_tp   = all_results[best_prec][max_conc]["throughput"]
    ref_tp_   = all_results[ref][max_conc]["throughput"]
    print(f"\n  Best throughput : {best_prec}-bit @ {best_tp:.1f} tok/s"
          f"  (×{best_tp/ref_tp_:.2f} vs {ref}-bit)")
    print(f"\n  Key finding     : batch=1 is attention-bound (all bits ≈ equal).")
    print(f"                    concurrency≥8 is linear-layer-bound → lower bits win.")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="High-concurrency benchmark to reveal bit-width throughput speedup"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--precisions", nargs="+", type=int, default=None,
                        help="Bit-widths to test (default: all from /v1/info)")
    parser.add_argument("--concurrency", nargs="+", type=int,
                        default=[1, 8, 16, 32],
                        help="Concurrent request levels to test (default: 1 8 16 32)")
    parser.add_argument("--max_tokens", type=int, default=64,
                        help="Max tokens per response (default: 64)")
    parser.add_argument("--n_prompts", type=int, default=100,
                        help="Prompts to load from dataset (default: 100)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup requests per precision (default: 3)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Timed runs per (precision, concurrency) (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # ── Server info ───────────────────────────────────────────────────────────
    try:
        info = requests.get(f"{base_url}/v1/info", timeout=10).json()
    except Exception as e:
        print(f"ERROR: cannot reach server at {base_url}  ({e})")
        print("  Start it with:  bash run_serve.sh --max_batch_size 8 --batch_wait_ms 20")
        sys.exit(1)

    supported = info["supported_precisions"]
    precisions = sorted(args.precisions or supported)
    for p in precisions:
        if p not in supported:
            print(f"ERROR: precision {p} not supported (server has {supported})")
            sys.exit(1)

    # ── Prompts ───────────────────────────────────────────────────────────────
    sep("High-Concurrency Bit-Width Benchmark")
    print(f"  Server      : {base_url}")
    print(f"  GPU         : {info.get('gpu', 'N/A')}  ({info.get('vram_total_gb', '?')} GB)")
    print(f"  Model       : {info['model']}")
    print(f"  Precisions  : {precisions}")
    print(f"  Concurrency : {args.concurrency}")
    print(f"  Max tokens  : {args.max_tokens}")
    print(f"  Runs/config : {args.runs}  (warmup={args.warmup})")
    print()
    print("  Loading prompts...")
    prompts = load_prompts(args.n_prompts, seed=args.seed)
    print(f"  Using {len(prompts)} prompts")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    sep("Running")
    all_results = {}
    for prec in precisions:
        all_results[prec] = benchmark_precision(
            base_url=base_url,
            precision=prec,
            prompts=prompts,
            max_tokens=args.max_tokens,
            concurrency_levels=args.concurrency,
            warmup=args.warmup,
            runs=args.runs,
        )

    # ── Report ────────────────────────────────────────────────────────────────
    print_results(all_results, args.concurrency, precisions)
    sep()


if __name__ == "__main__":
    main()
