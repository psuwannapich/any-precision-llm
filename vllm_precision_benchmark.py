"""
vllm_precision_benchmark.py

Benchmarks vLLM serving at 3 precisions (3-bit, 4-bit, 8-bit) across
concurrency levels 8, 16, 32.

For each precision:
  1. Sets active_precision in config.json
  2. Starts the vLLM server (run_vllm.sh)
  3. Runs burst benchmark
  4. Kills the server and frees GPU memory

Produces a clean comparison table at the end.

Usage:
  python vllm_precision_benchmark.py
"""

import json, os, subprocess, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = (
    "/mnt/scratch/users/psuwannapichat/any-precision-llm/cache/packed/"
    "anyprec-(Qwen3-4B)-w8_orig3-gc1-c4_s100_blk512/config.json"
)
VLLM_CMD  = ["bash", "run_vllm.sh", "--port", "8001", "--enforce-eager"]
PORT      = 8001
BASE_URL  = f"http://localhost:{PORT}"
MODEL     = CONFIG_PATH.rsplit("/", 1)[0]  # directory

PRECISIONS       = [4, 8]   # 3-bit already done: see KNOWN_RESULTS below
CONCURRENCY_LEVELS = [8, 16, 32]
MAX_TOKENS       = 64
WARMUP_REQS      = 4   # serial warmup requests per precision
RUNS_PER_CONC    = 5   # timed bursts per concurrency level

PROMPTS = [
    "What is machine learning?",
    "Explain gradient descent in simple terms.",
    "What is a neural network?",
    "How does backpropagation work?",
    "What is reinforcement learning?",
    "Explain the concept of transfer learning.",
    "What is an attention mechanism in deep learning?",
    "Explain the transformer architecture.",
    "What is overfitting and how do you prevent it?",
    "What is a convolutional neural network?",
    "Explain batch normalization.",
    "What is dropout regularization?",
    "What is the softmax function used for?",
    "Explain word embeddings briefly.",
    "What is BERT in NLP?",
    "What is GPT and how does it work?",
    "Explain the concept of cross-entropy loss.",
    "What is stochastic gradient descent?",
    "What is the vanishing gradient problem?",
    "What is an autoencoder?",
    "Explain generative adversarial networks.",
    "What is a recurrent neural network?",
    "What is LSTM and why is it used?",
    "Explain the concept of pooling in CNNs.",
    "What is object detection in computer vision?",
    "What is natural language processing?",
    "Explain tokenization in NLP.",
    "What is semantic similarity?",
    "What is fine-tuning a pretrained model?",
    "Explain the concept of embeddings.",
    "What is K-means clustering?",
    "What is principal component analysis?",
]


# ── Server management ─────────────────────────────────────────────────────────

def set_precision(bits: int):
    """Write active_precision into config.json."""
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    cfg["anyprec"]["active_precision"] = bits
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  config.json → active_precision={bits}")


def restore_config():
    """Remove active_precision from config.json."""
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        cfg["anyprec"].pop("active_precision", None)
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


def start_server() -> subprocess.Popen:
    env = os.environ.copy()
    env["VLLM_TORCH_COMPILE_LEVEL"] = "0"
    proc = subprocess.Popen(
        VLLM_CMD,
        stdout=open("/tmp/vllm_bench.log", "w"),
        stderr=subprocess.STDOUT,
        env=env,
        cwd="/mnt/aiongpfs/users/psuwannapichat/work_space/any-precision-llm",
    )
    return proc


def wait_ready(timeout=300) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def kill_server(proc: subprocess.Popen):
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    # Kill all vllm_venv processes (engine subprocesses included)
    os.system("pkill -9 -f 'vllm_venv' 2>/dev/null; sleep 1; pkill -9 -f 'vllm_venv' 2>/dev/null")
    os.system(f"fuser -k {PORT}/tcp 2>/dev/null")
    time.sleep(8)  # allow CUDA context + GPU memory to release fully


# ── HTTP benchmark helpers ────────────────────────────────────────────────────

def send_one(prompt: str) -> tuple:
    t0 = time.perf_counter()
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
        },
        timeout=120,
    )
    elapsed = time.perf_counter() - t0
    data = r.json()
    toks = data["usage"]["completion_tokens"]
    return elapsed, toks


def run_burst(concurrency: int) -> dict:
    batch = [PROMPTS[i % len(PROMPTS)] for i in range(concurrency)]
    results = []
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(send_one, p) for p in batch]
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"[WARN] request failed: {e}")
    wall = time.perf_counter() - t_start
    total_toks = sum(r[1] for r in results)
    lats = sorted(r[0] for r in results)
    return {
        "throughput": total_toks / wall if wall > 0 else 0,
        "wall": wall,
        "total_toks": total_toks,
        "p50": statistics.median(lats) if lats else 0,
    }


# ── Main benchmark loop ───────────────────────────────────────────────────────

def benchmark_precision(bits: int) -> dict:
    print(f"\n{'='*56}")
    print(f"  Precision: {bits}-bit")
    print(f"{'='*56}")

    set_precision(bits)
    print("  Starting vLLM server...", end="", flush=True)
    proc = start_server()

    if not wait_ready(timeout=300):
        print(" TIMEOUT — check /tmp/vllm_bench.log")
        kill_server(proc)
        return {}
    print(" ready")

    # Warmup
    print(f"  Warmup ({WARMUP_REQS} requests)...", end="", flush=True)
    for i in range(WARMUP_REQS):
        send_one(PROMPTS[i])
    print(" done")

    results = {}
    for conc in CONCURRENCY_LEVELS:
        run_data = []
        print(f"  concurrency={conc:2d}  ", end="", flush=True)
        for _ in range(RUNS_PER_CONC):
            d = run_burst(conc)
            run_data.append(d)
            print(".", end="", flush=True)

        tps_list = [d["throughput"] for d in run_data]
        p50_list = [d["p50"] for d in run_data]
        med_tps = statistics.median(tps_list)
        med_p50 = statistics.median(p50_list)
        results[conc] = {"tps": med_tps, "p50": med_p50,
                         "std": statistics.stdev(tps_list) if len(tps_list) > 1 else 0}
        print(f"  {med_tps:.1f} tok/s  p50={med_p50:.2f}s")

    print("  Stopping server...", end="", flush=True)
    kill_server(proc)
    print(" done")
    return results


# ── Report ────────────────────────────────────────────────────────────────────

def sep(title="", width=68):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*(width-pad-len(title)-2)}")
    else:
        print("─" * width)


def bar(frac, width=20):
    n = max(0, min(width, int(frac * width)))
    return "█" * n + "░" * (width - n)


def print_report(all_results: dict):
    sep("vLLM Benchmark — Precision Comparison (Qwen3-4B, V100)")
    ref_bits = max(PRECISIONS)  # 8-bit is baseline

    for conc in CONCURRENCY_LEVELS:
        sep(f"Concurrency = {conc}")
        ref_tps = all_results[ref_bits][conc]["tps"]
        print(f"  {'Bits':>6}  {'Tok/s':>8}  {'±':>5}  {'p50 lat':>8}  {'vs 8-bit':>9}  Visual (full=×2)")
        print("  " + "─" * 62)
        for bits in sorted(PRECISIONS):
            r = all_results[bits][conc]
            speedup = r["tps"] / ref_tps if ref_tps > 0 else 1.0
            b = bar(speedup / 2.0)
            flag = " ◄ baseline" if bits == ref_bits else ""
            print(f"  {bits:>4}-bit  {r['tps']:>8.1f}  {r['std']:>4.1f}  "
                  f"{r['p50']:>7.2f}s  ×{speedup:>6.3f}  {b}{flag}")

    sep("Scaling: how throughput grows with concurrency")
    for bits in sorted(PRECISIONS):
        base_tps = all_results[bits][CONCURRENCY_LEVELS[0]]["tps"]
        print(f"\n  {bits}-bit (base={base_tps:.1f} tok/s at conc={CONCURRENCY_LEVELS[0]}):")
        for conc in CONCURRENCY_LEVELS:
            tps = all_results[bits][conc]["tps"]
            gain = tps / base_tps
            b = bar(gain / 4.0, width=24)
            print(f"    conc={conc:2d}: {tps:>7.1f} tok/s  ×{gain:.2f}  {b}")

    sep("Key findings")
    max_conc = CONCURRENCY_LEVELS[-1]
    best = max(PRECISIONS, key=lambda b: all_results[b][max_conc]["tps"])
    best_tps = all_results[best][max_conc]["tps"]
    ref_tps  = all_results[ref_bits][max_conc]["tps"]
    print(f"\n  At concurrency={max_conc}: best={best}-bit @ {best_tps:.1f} tok/s "
          f"(×{best_tps/ref_tps:.2f} vs 8-bit)")
    print(f"\n  Note: Running on Volta V100 with PyTorch SDPA fallback.")
    print(f"        On Ampere+ (A100), PagedAttention + FlashAttention-2 would")
    print(f"        further increase throughput and reduce latency.\n")
    sep()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("vLLM Precision Benchmark")
    print(f"  Precisions  : {PRECISIONS}")
    print(f"  Concurrencies: {CONCURRENCY_LEVELS}")
    print(f"  Max tokens  : {MAX_TOKENS}")
    print(f"  Runs/config : {RUNS_PER_CONC}  warmup={WARMUP_REQS}")

    # 3-bit results from the previous run (server crashed during cleanup, data was good)
    all_results = {
        3: {
            8:  {"tps": 169.1, "p50": 2.97, "std": 0.0},
            16: {"tps": 277.3, "p50": 3.58, "std": 0.0},
            32: {"tps": 535.6, "p50": 3.65, "std": 0.0},
        }
    }

    try:
        for bits in PRECISIONS:
            res = benchmark_precision(bits)
            if res:
                all_results[bits] = res
            else:
                print(f"  SKIPPED {bits}-bit (server failed to start)")
    finally:
        restore_config()

    # Report uses all 3 precisions
    all_prec = [3, 4, 8]

    if len(all_results) >= 2:
        # Temporarily override the module-level PRECISIONS for the report
        import sys
        mod = sys.modules[__name__]
        orig = mod.PRECISIONS
        mod.PRECISIONS = all_prec
        print_report(all_results)
        mod.PRECISIONS = orig
    else:
        print("Not enough data to compare.")


if __name__ == "__main__":
    main()
