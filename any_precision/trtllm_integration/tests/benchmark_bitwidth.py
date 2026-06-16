"""
benchmark_bitwidth.py — measure TensorRT-LLM inference latency at every supported
Any-Precision bit-width, from a SINGLE engine (precision is a runtime switch).

Goal: show how decode latency scales with weight bit-width. Lower bit-width reads
fewer bit-planes of the packed weight per matmul, so decode (which is weight-
memory-bound at batch=1) should get faster as precision drops.

Methodology:
  * One warmed-up engine; per-precision warmup too (first launch of each
    precision's kernels is slower).
  * end_id=-1 forces exactly `output_len` tokens so every run is the same work.
  * TPOT (time-per-output-token, i.e. pure decode latency) is isolated with the
    two-point method:   TPOT = (lat[output_len] - lat[1]) / (output_len - 1)
    which cancels the one-off prefill + fixed overhead.
  * Each point is the median of `iters` timed runs (after `warmup` runs).

Run (in .venv-gpu, with OMPI_MCA_ess_singleton_isolated=1 and the plugin built):
  python -m any_precision.trtllm_integration.tests.benchmark_bitwidth \
      --engine_dir trt_engine \
      --tokenizer_dir <packed ckpt> \
      --output_len 128 --iters 10 --warmup 3 --batch_size 1
"""
import argparse
import json
import os
import statistics
import time

import torch

from ..python._plugin_loader import load_plugin, set_precision


def _time_generate(runner, input_batch, out_len, end_id, iters, warmup):
    """Return median wall-clock seconds to generate exactly out_len tokens."""
    def once():
        runner.generate(input_batch, max_new_tokens=out_len, end_id=end_id,
                        pad_id=0, temperature=0.0, top_p=1.0)
        torch.cuda.synchronize()

    for _ in range(warmup):
        once()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        once()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine_dir", required=True)
    ap.add_argument("--tokenizer_dir", default=None)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--output_len", type=int, default=128)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--precisions", default=None,
                    help="comma list, e.g. 3,4,5,6,7,8 (default: all supported)")
    args = ap.parse_args()

    load_plugin()
    with open(os.path.join(args.engine_dir, "anyprec_meta.json")) as f:
        meta = json.load(f)
    supported = meta["supported_bits"]
    precisions = ([int(x) for x in args.precisions.split(",")]
                  if args.precisions else supported)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        args.tokenizer_dir or args.engine_dir, trust_remote_code=True)
    ids = tok(args.prompt, return_tensors="pt").input_ids[0].cuda()
    input_batch = [ids] * args.batch_size

    from tensorrt_llm.runtime import ModelRunner
    runner = ModelRunner.from_dir(engine_dir=args.engine_dir)

    print(f"engine={args.engine_dir}  batch={args.batch_size}  "
          f"prompt_tokens={len(ids)}  output_len={args.output_len}  "
          f"iters={args.iters} (warmup {args.warmup})")
    print(f"{'bits':>4} | {'total ms':>9} | {'prefill ms':>10} | "
          f"{'TPOT ms/tok':>11} | {'decode tok/s':>12} | {'speedup':>7}")
    print("-" * 70)

    rows = []
    for p in precisions:
        set_precision(p)
        # two-point: 1 token (≈prefill) and output_len tokens
        lat1, _ = _time_generate(runner, input_batch, 1, -1,
                                 args.iters, args.warmup)
        latN, _ = _time_generate(runner, input_batch, args.output_len, -1,
                                 args.iters, args.warmup)
        tpot = (latN - lat1) / max(args.output_len - 1, 1)   # s/token
        rows.append((p, latN, lat1, tpot))

    # speedup relative to the highest bit-width (the slowest expected)
    base_tpot = max(r[3] for r in rows)
    for p, latN, lat1, tpot in rows:
        tok_s = (args.batch_size / tpot) if tpot > 0 else float("inf")
        print(f"{p:>4} | {latN*1e3:>9.2f} | {lat1*1e3:>10.2f} | "
              f"{tpot*1e3:>11.3f} | {tok_s:>12.1f} | {base_tpot/tpot:>6.2f}x")

    print("-" * 70)
    asc = all(rows[i][3] <= rows[i + 1][3] + 1e-9 for i in range(len(rows) - 1))
    print("decode latency is monotonically NON-decreasing with bit-width: "
          f"{'YES ✓ (lower bit = faster)' if asc else 'NO'}")


if __name__ == "__main__":
    main()
