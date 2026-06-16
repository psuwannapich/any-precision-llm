"""
benchmark_api.py — benchmark per-bit-width latency by calling the OpenAI-
compatible Any-Precision server over HTTP (server.py). Proves that lower
bit-width = faster inference at the API level.

Each timed run is ONE streaming chat completion, consumed to completion (so no
server-side generation is stranded), capturing:
  * TTFT  — time to first token: request send -> first content chunk.
  * TPOT  — mean inter-token latency = (t_last - t_first) / (n_tokens - 1),
            i.e. the per-output-token decode latency as seen over the API.
  * total — end-to-end latency for the whole completion.
Reported as the median over `iters` runs (after `warmup`).

Run (server must be up: ./run_trtllm_serve.sh):
  python -m any_precision.trtllm_integration.tests.benchmark_api \
      --base_url http://localhost:8000/v1 --output_len 128 --iters 6 --warmup 1
(uses the `openai` SDK, already in .venv-gpu)
"""
import argparse
import random
import statistics
import time

from openai import OpenAI

PROMPT = "Write a detailed paragraph about the history of the city of Paris."


def _one_stream(client, model, precision, max_tokens):
    """One streaming completion, fully drained. Returns (ttft, total, n_tokens)."""
    t0 = time.perf_counter()
    first = last = None
    n = 0
    stream = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": PROMPT}],
        max_tokens=max_tokens, temperature=0.0, stream=True,
        extra_body={"precision": precision})
    for chunk in stream:
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta.content
        if d:
            now = time.perf_counter()
            if first is None:
                first = now
            last = now
            n += 1
    ttft = (first - t0) if first is not None else float("nan")
    total = (last - t0) if last is not None else float("nan")
    return ttft, total, n


def measure(client, model, p, max_tokens, iters, warmup):
    for _ in range(warmup):
        _one_stream(client, model, p, max_tokens)
    ttfts, totals, tpots, ntok = [], [], [], 0
    for _ in range(iters):
        ttft, total, n = _one_stream(client, model, p, max_tokens)
        ttfts.append(ttft)
        totals.append(total)
        ntok = n
        if n > 1:
            tpots.append((total - ttft) / (n - 1))
    return (statistics.median(ttfts), statistics.median(tpots),
            statistics.median(totals), ntok)


def run_sweep(client, model, precisions, args):
    """Per-bit-width sweep (the original benchmark)."""
    print(f"\n=== per-bit-width sweep ===")
    print(f"{'bits':>4} | {'TTFT ms':>8} | {'TPOT ms/tok':>11} | {'total ms':>9} | "
          f"{'tok/s':>7} | {'~tokens':>7} | {'speedup':>7}")
    print("-" * 78)
    rows = []
    for p in precisions:
        ttft, tpot, total, ntok = measure(client, model, p, args.output_len,
                                          args.iters, args.warmup)
        rows.append((p, ttft, tpot, total, ntok))
    base = max(r[2] for r in rows)
    for p, ttft, tpot, total, ntok in rows:
        toks = 1.0 / tpot if tpot > 0 else float("inf")
        print(f"{p:>4} | {ttft*1e3:>8.1f} | {tpot*1e3:>11.3f} | {total*1e3:>9.1f} | "
              f"{toks:>7.1f} | {ntok:>7d} | {base/tpot:>6.2f}x")
    print("-" * 78)
    asc = all(rows[i][2] <= rows[i + 1][2] + 1e-9 for i in range(len(rows) - 1))
    print(f"decode latency monotonic with bit-width (lower bit = faster): "
          f"{'YES ✓' if asc else 'NO'}")


def run_mix(client, model, percents, precisions, args):
    """Mixed-traffic workload: X% of requests use the highest precision, the
    rest use a randomly-chosen LOWER precision. Shows aggregate serving latency
    / throughput as a function of the high-precision share X.

    X=100 (all-high) is the baseline == the all-8-bit point of the sweep."""
    high = max(precisions)
    lowers = [p for p in precisions if p < high]
    rng = random.Random(args.seed)
    n = args.requests

    # warm the server up once at the highest precision
    for _ in range(args.warmup):
        _one_stream(client, model, high, args.output_len)

    print(f"\n=== mixed-precision workload ===")
    print(f"high={high}-bit  lower(round-robin)={lowers}  requests/point={n}  "
          f"output_len={args.output_len}")
    print(f"{'X% high':>7} | {'high/low':>9} | {'mean lat s':>10} | "
          f"{'p50 lat s':>9} | {'agg tok/s':>9} | {'speedup vs all-high':>19}")
    print("-" * 78)

    baseline_mean = None
    for X in [100] + [x for x in percents if x != 100]:
        n_high = round(n * X / 100.0)
        n_low = n - n_high
        # balance the lower-precision requests across all lowers (round-robin)
        # so each mix point uses the same lower-precision composition; the only
        # variable is the high/low ratio X.
        low_precs = [lowers[i % len(lowers)] for i in range(n_low)]
        precs = [high] * n_high + low_precs
        rng.shuffle(precs)
        lats, toks = [], []
        wall0 = time.perf_counter()
        for p in precs:
            _ttft, total, ntok = _one_stream(client, model, p, args.output_len)
            lats.append(total)
            toks.append(ntok)
        wall = time.perf_counter() - wall0
        mean_lat = statistics.mean(lats)
        p50 = statistics.median(lats)
        agg_toks = sum(toks) / wall                      # serving throughput
        if X == 100:
            baseline_mean = mean_lat
        speedup = baseline_mean / mean_lat if mean_lat else float("nan")
        tag = "  (baseline)" if X == 100 else ""
        print(f"{X:>7} | {n_high:>3}/{n - n_high:<5} | {mean_lat:>10.3f} | "
              f"{p50:>9.3f} | {agg_toks:>9.1f} | {speedup:>17.2f}x{tag}")
    print("-" * 78)
    print("Lower X (more low-precision traffic) => lower mean latency / higher "
          "throughput, vs serving everything at the highest precision.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="anyprec")
    ap.add_argument("--api_key", default="none")
    ap.add_argument("--output_len", type=int, default=128)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--precisions", default=None, help="comma list, e.g. 3,4,5,6,7,8")
    # mixed-traffic mode
    ap.add_argument("--mix", default=None,
                    help="comma list of high-precision percentages, e.g. "
                         "25,33,50,66,75 — runs the mixed-traffic workload")
    ap.add_argument("--requests", type=int, default=12,
                    help="requests per mix point (mixed-traffic mode)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    if args.precisions:
        precisions = [int(x) for x in args.precisions.split(",")]
    else:
        models = client.models.list()
        precisions = next((m.supported_precisions for m in models.data
                           if getattr(m, "supported_precisions", None)),
                          [3, 4, 5, 6, 7, 8])

    print(f"server={args.base_url}  model={args.model}  output_len={args.output_len}")
    print(f"prompt={PROMPT!r}")

    if args.mix:
        percents = [int(x) for x in args.mix.split(",")]
        run_mix(client, args.model, percents, precisions, args)
    else:
        run_sweep(client, args.model, precisions, args)


if __name__ == "__main__":
    main()
