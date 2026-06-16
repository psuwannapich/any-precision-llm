"""
gpu_smoke.py — end-to-end GPU smoke test for the Any-Precision TensorRT-LLM
backend. Complements tests/cpu_validate.py (which only checks the weight layout
on CPU); this one actually deserializes the engine, loads the compiled plugin,
and runs generation at every supported bit-width from the *single* engine.

Assumes the plugin .so and the engine have already been built (see
run_trtllm_test.sh, stages `plugin` and `engine`). It does not build anything.

What it asserts:
  * the plugin .so loads and registers (engine deserialization needs it),
  * generation runs at each supported precision without error,
  * each precision returns non-empty text,
  * (optional) lower precision != higher precision output, i.e. the runtime
    precision switch actually changes the math from one shared engine.

Run:
  python -m any_precision.trtllm_integration.tests.gpu_smoke \
      --engine_dir ./trt_engine \
      --tokenizer_dir /path/to/anyprec-Qwen3-4B
"""

import argparse
import json
import os
import time

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
_n_fail = 0


def check(name, cond):
    global _n_fail
    print(f"  [{PASS if cond else FAIL}] {name}")
    if not cond:
        _n_fail += 1
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine_dir", required=True)
    ap.add_argument("--tokenizer_dir", default=None,
                    help="defaults to the engine dir if it holds tokenizer files")
    ap.add_argument("--prompt", default="Explain quantization in one sentence.")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--precisions", default=None,
                    help="comma list, e.g. 3,4,8; default = all supported")
    ap.add_argument("--cpp_runtime", action="store_true",
                    help="use ModelRunnerCpp instead of the Python ModelRunner")
    args = ap.parse_args()

    # Import lazily so a missing tensorrt_llm produces a clear message, not an
    # import error at module load.
    import torch
    from ..python.runtime import AnyPrecisionTRTRunner

    check("CUDA visible to torch", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(f"  device: {torch.cuda.get_device_name(0)} "
              f"(sm_{''.join(map(str, torch.cuda.get_device_capability(0)))})")

    meta_path = os.path.join(args.engine_dir, "anyprec_meta.json")
    check("anyprec_meta.json next to engine", os.path.exists(meta_path))
    with open(meta_path) as f:
        meta = json.load(f)
    supported = meta["supported_bits"]
    print(f"  supported_bits = {supported}")

    if args.precisions:
        precisions = [int(x) for x in args.precisions.split(",")]
    else:
        precisions = supported

    print("\n[1] load engine + plugin")
    t0 = time.time()
    runner = AnyPrecisionTRTRunner(
        args.engine_dir, tokenizer_dir=args.tokenizer_dir,
        use_cpp_runtime=args.cpp_runtime)
    check(f"engine deserialized + plugin registered ({time.time()-t0:.1f}s)", True)

    print("\n[2] generate at each precision (one shared engine)")
    outputs = {}
    for p in precisions:
        if p not in supported:
            check(f"precision {p} is supported", False)
            continue
        t0 = time.time()
        text = runner.generate(args.prompt, precision=p,
                               max_new_tokens=args.max_new_tokens)
        dt = time.time() - t0
        outputs[p] = text
        tps = args.max_new_tokens / dt if dt > 0 else 0.0
        print(f"\n  --- precision {p} ({dt:.2f}s, ~{tps:.1f} tok/s) ---")
        print(f"  {text!r}")
        check(f"precision {p}: non-empty output", bool(text and text.strip()))

    print("\n[3] runtime precision switch actually changes output")
    if len(outputs) >= 2:
        lo, hi = min(outputs), max(outputs)
        check(f"precision {lo} output differs from precision {hi}",
              outputs[lo] != outputs[hi])
    else:
        print("  (skipped: need >=2 precisions)")

    print("\n" + "=" * 60)
    if _n_fail == 0:
        print("  ALL GPU SMOKE CHECKS PASSED")
    else:
        print(f"  {_n_fail} CHECK(S) FAILED")
    print("=" * 60)
    return _n_fail


if __name__ == "__main__":
    raise SystemExit(main())
