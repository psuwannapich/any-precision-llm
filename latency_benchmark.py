"""
latency_benchmark.py

Detailed latency analysis of the Phase 2 vLLM server.

Metrics measured:
  TTFT  — Time to First Token   (prefill cost; prompt length matters)
  TPOT  — Time per Output Token (decode cost; bit-width matters here)
  E2E   — End-to-end latency    (TTFT + TPOT × n_tokens)

Three test scenarios:
  1. Serial latency vs precision        (batch=1, isolates pure decode speed)
  2. Latency vs prompt length           (shows TTFT scaling with input length)
  3. Latency under load (p50/p95/p99)   (concurrent requests, realistic SLA)

Usage:
  python latency_benchmark.py [--port 8001] [--runs 20]
"""

import argparse, json, statistics, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

PRECISIONS = [3, 4, 8]
BASE_URL   = "http://localhost:8001"

# Prompts of different lengths (approx token counts vary by tokenizer)
SHORT_PROMPT  = "What is artificial intelligence?"                # ~8 tokens
MEDIUM_PROMPT = (
    "Explain the concept of machine learning in detail, covering "
    "supervised learning, unsupervised learning, and reinforcement "
    "learning. Include examples of real-world applications for each "
    "category and discuss the main algorithms used."
)                                                                  # ~50 tokens
LONG_PROMPT   = (
    "You are a computer science professor. Write a comprehensive "
    "explanation of deep learning, covering: (1) the history of "
    "neural networks from the 1950s to present, (2) the mathematical "
    "foundations including gradient descent and backpropagation, "
    "(3) key architectures: CNNs, RNNs, LSTMs, and Transformers, "
    "(4) training techniques like batch normalization, dropout, and "
    "learning rate scheduling, (5) major breakthroughs including "
    "AlexNet, ResNet, BERT, and GPT, and (6) current challenges "
    "such as interpretability, data efficiency, and energy consumption. "
    "Provide concrete examples and equations where appropriate."
)                                                                  # ~120 tokens


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def measure_e2e(base_url, precision, prompt, max_tokens) -> dict:
    """Non-streaming: total wall time = TTFT + generation time."""
    t0 = time.perf_counter()
    r = requests.post(f"{base_url}/v1/chat/completions", json={
        "messages": [{"role": "user", "content": prompt}],
        "precision": precision,
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": False,
        "stream": False,
    }, timeout=120)
    e2e = time.perf_counter() - t0
    data = r.json()
    n_out = data["usage"]["completion_tokens"]
    n_in  = data["usage"]["prompt_tokens"]
    return {"e2e": e2e, "n_out": n_out, "n_in": n_in}


def measure_streaming(base_url, precision, prompt, max_tokens) -> dict:
    """Streaming: capture TTFT and per-token timing."""
    token_times = []
    t_start = time.perf_counter()
    t_first = None

    with requests.post(f"{base_url}/v1/chat/completions", json={
        "messages": [{"role": "user", "content": prompt}],
        "precision": precision,
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": False,
        "stream": True,
    }, timeout=120, stream=True) as resp:
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta", {}).get("content", "")
                if delta:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now - t_start
                    token_times.append(now)
            except Exception:
                pass

    t_end = time.perf_counter()
    n_tok = len(token_times)
    ttft  = t_first if t_first is not None else (t_end - t_start)
    e2e   = t_end - t_start
    # TPOT = time to generate all output tokens / number of tokens
    # Exclude TTFT from decode time to isolate per-decode-step cost
    tpot  = (e2e - ttft) / max(n_tok - 1, 1) if n_tok > 1 else e2e
    return {"ttft": ttft, "tpot": tpot * 1000, "e2e": e2e, "n_tok": n_tok}


def pct(data, p):
    data = sorted(data)
    idx  = max(0, min(len(data)-1, int(len(data) * p / 100)))
    return data[idx]


def sep(t="", w=70):
    if t: p=(w-len(t)-2)//2; print(f"\n{'─'*p} {t} {'─'*(w-p-len(t)-2)}")
    else: print("─"*w)


# ── Scenario 1: Serial latency vs precision ───────────────────────────────────

def bench_serial_precision(base_url, runs, max_tokens, prompt_label, prompt):
    sep(f"Serial latency — {prompt_label} prompt,  max_tokens={max_tokens}")
    print(f"  Prompt  : {prompt[:70]}...")
    print(f"  Runs    : {runs}\n")

    print(f"  {'Bits':>6}  {'TTFT ms':>9}  {'TPOT ms/tok':>12}  {'E2E s':>7}  {'Tok':>5}")
    print("  " + "─"*52)

    results = {}
    for prec in PRECISIONS:
        ttfts, tpots, e2es, n_toks = [], [], [], []
        for _ in range(runs):
            d = measure_streaming(base_url, prec, prompt, max_tokens)
            ttfts.append(d["ttft"] * 1000)
            tpots.append(d["tpot"])
            e2es.append(d["e2e"])
            n_toks.append(d["n_tok"])

        results[prec] = dict(ttft=ttfts, tpot=tpots, e2e=e2es, n_tok=n_toks)
        med_ttft = statistics.median(ttfts)
        med_tpot = statistics.median(tpots)
        med_e2e  = statistics.median(e2es)
        med_ntok = statistics.median(n_toks)
        print(f"  {prec:>4}-bit  {med_ttft:>9.1f}  {med_tpot:>12.2f}  "
              f"{med_e2e:>7.2f}  {med_ntok:>5.0f}")

    # Speedup table
    ref_tpot = statistics.median(results[8]["tpot"])
    ref_ttft = statistics.median(results[8]["ttft"])
    print(f"\n  Speedup vs 8-bit:")
    print(f"  {'Bits':>6}  {'TPOT speedup':>14}  {'TTFT speedup':>13}")
    print("  " + "─"*38)
    for prec in PRECISIONS:
        tpot_sp = ref_tpot / statistics.median(results[prec]["tpot"])
        ttft_sp = ref_ttft / statistics.median(results[prec]["ttft"])
        flag = " ◄ baseline" if prec == 8 else ""
        print(f"  {prec:>4}-bit  {tpot_sp:>12.3f}×  {ttft_sp:>12.3f}×{flag}")

    return results


# ── Scenario 2: TTFT vs prompt length ────────────────────────────────────────

def bench_ttft_vs_length(base_url, runs):
    sep("TTFT vs prompt length (prefill speed)")
    prompts = [
        ("short (~8 tok)",    SHORT_PROMPT,  8),
        ("medium (~50 tok)",  MEDIUM_PROMPT, 16),
        ("long (~120 tok)",   LONG_PROMPT,   16),
    ]
    print(f"  Shows how TTFT scales with input length at each precision.\n")
    print(f"  {'Prompt':>16}  " + "  ".join(f"{p}-bit TTFT" for p in PRECISIONS))
    print("  " + "─"*60)

    for label, prompt, max_tokens in prompts:
        row = []
        for prec in PRECISIONS:
            ttfts = [measure_streaming(base_url, prec, prompt, max_tokens)["ttft"] * 1000
                     for _ in range(runs)]
            row.append(f"{statistics.median(ttfts):>10.1f}ms")
        print(f"  {label:>16}  " + "  ".join(row))


# ── Scenario 3: Latency percentiles under concurrent load ────────────────────

def bench_latency_under_load(base_url, runs_per_conc, max_tokens):
    sep("Latency percentiles under load  (mixed 3+4+8-bit requests)")
    conc_levels = [1, 4, 8, 16]

    def _req(prec, prompt):
        d = measure_e2e(base_url, prec, prompt, max_tokens)
        return d["e2e"]

    prompts_pool = [
        "Explain neural networks briefly.",
        "What is climate change?",
        "How does the internet work?",
        "Describe the water cycle.",
        "What is quantum physics?",
        "Explain supply and demand.",
        "How do vaccines work?",
        "What is machine learning?",
    ]

    print(f"  max_tokens={max_tokens}  runs/config={runs_per_conc}\n")
    print(f"  {'Conc':>6}  {'p50 s':>7}  {'p90 s':>7}  {'p99 s':>7}  {'Mean s':>7}  {'Tok/s':>8}")
    print("  " + "─"*50)

    for conc in conc_levels:
        all_lats = []
        all_toks = []
        t_total_start = time.perf_counter()
        prec_cycle = [PRECISIONS[i % len(PRECISIONS)] for i in range(conc)]

        for _ in range(runs_per_conc):
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=conc) as pool:
                futs = [pool.submit(_req, prec_cycle[i],
                                    prompts_pool[i % len(prompts_pool)])
                        for i in range(conc)]
                for f in as_completed(futs):
                    all_lats.append(f.result())
            all_toks.append(conc * max_tokens)  # approx

        total_wall = time.perf_counter() - t_total_start
        tps = sum(all_toks) / total_wall

        p50 = pct(all_lats, 50)
        p90 = pct(all_lats, 90)
        p99 = pct(all_lats, 99)
        mean = statistics.mean(all_lats)
        print(f"  {conc:>6}  {p50:>7.2f}  {p90:>7.2f}  {p99:>7.2f}  {mean:>7.2f}  {tps:>8.1f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",       default="localhost")
    parser.add_argument("--port",       type=int, default=8001)
    parser.add_argument("--runs",       type=int, default=15,
                        help="Timed runs per config (default 15)")
    parser.add_argument("--max_tokens", type=int, default=64)
    args = parser.parse_args()

    global BASE_URL
    BASE_URL = f"http://{args.host}:{args.port}"

    try:
        requests.get(f"{BASE_URL}/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"Server unreachable: {e}"); return

    sep("Phase 2 Latency Benchmark — Qwen3-4B (V100)")
    print(f"  Server     : {BASE_URL}")
    print(f"  Precisions : {PRECISIONS}")
    print(f"  Runs/config: {args.runs}")
    print(f"  Max tokens : {args.max_tokens}")

    # Warmup
    print("\n  Warmup... ", end="", flush=True)
    for prec in PRECISIONS:
        measure_e2e(BASE_URL, prec, SHORT_PROMPT, 16)
    print("done")

    # 1. Serial latency (decode bottleneck)
    serial_results = bench_serial_precision(
        BASE_URL, args.runs, args.max_tokens, "medium", MEDIUM_PROMPT
    )

    # 2. TTFT vs prompt length (prefill bottleneck)
    bench_ttft_vs_length(BASE_URL, runs=8)

    # 3. Latency under load
    bench_latency_under_load(BASE_URL, runs_per_conc=5, max_tokens=args.max_tokens)

    # ── Summary ────────────────────────────────────────────────────────────
    sep("Summary")
    print("""
  TTFT (Time to First Token):
    Driven by prefill (prompt processing). Longer prompt → higher TTFT.
    Bit-width has minimal effect on TTFT (attention dominates prefill).

  TPOT (Time per Output Token):
    Driven by decode kernel. Lower bits → fewer memory reads → lower TPOT.
    This is where bit-width speedup is measurable at batch=1.

  Latency under load (p99):
    At high concurrency, queuing adds to latency. vLLM's continuous
    batching keeps p99 reasonable by interleaving requests.
""")
    sep()


if __name__ == "__main__":
    main()
