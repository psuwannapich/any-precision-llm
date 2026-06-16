# Any-Precision × TensorRT-LLM integration

End-to-end path to run an Any-Precision (LUT bit-packed) Qwen3 model inside a
TensorRT-LLM engine, with the bit-width selectable **at runtime** from a single
engine — the same "one model, many precisions" property the PyTorch path has.

```
HF AP checkpoint ──convert──▶ fused numpy weights ──build──▶ TRT-LLM engine ──serve──▶ /generate?precision=3..8
   (qweight+luts)              (q/k/v fused)         (+ AP plugin)        (runtime per-request precision)
```

## ⚠️ Hardware / software requirements (read first)

| Requirement | Why |
|---|---|
| **GPU compute capability ≥ 8.0** (A100/A6000/L40/H100) | TensorRT-LLM dropped Volta/Turing; its prebuilt wheels target sm_80+. **The V100 (sm_70) in this repo's default environment cannot run the resulting engine.** The Any-Precision kernels themselves are sm_70+, but TRT-LLM gates the whole engine. |
| TensorRT-LLM ≥ 0.11, TensorRT ≥ 9 (10 recommended) | Plugin uses `IPluginV2DynamicExt`; build script uses the Qwen2/Qwen3 model definition. |
| CUDA Toolkit (matches TRT-LLM), cuBLAS | Plugin links cudart + cublas. |
| CMake ≥ 3.18 | Builds the plugin `.so`. |

This integration is **scaffolding verified for structure and weight-layout
correctness on CPU**, but it has **not been executed on a TRT-LLM GPU** (the
available hardware is a V100). Expect to make small TRT-LLM-version adjustments
in `python/build_engine.py` (module names / weight-loading) — those parts are
the most version-sensitive. Everything precision-specific lives in the plugin
and is version-stable.

## Why a custom plugin?

TensorRT-LLM only understands its own quant formats (FP8, INT8 SmoothQuant,
INT4 AWQ/GPTQ). Any-Precision uses a **per-output-row LUT over bit-plane-packed
weights** decoded by custom CUDA kernels. To run the *real* AP weights (not a
re-quantized approximation) the projection matmul must be a custom op — the
`AnyPrecisionLinear` TensorRT plugin in `plugin/`.

The plugin **reuses the upstream kernels verbatim** (`../modules/kernels/matmul.cuh`,
`dequant.cuh`) so the math is identical to the PyTorch path:

- `M ≤ 8`  → fused `matmul_kbit` (reads packed bits directly)
- `M > 8`  → `dequant_kbit` → cuBLAS HGEMM (matches `AnyPrecisionLinear.forward`)

Precision is a **process-global** (`ap_set_precision`, read in `enqueue`), not a
build-time constant — mirroring the vLLM `precision_manager`. One engine serves
bits 3–8.

## Layout

```
trtllm_integration/
├── plugin/                       # custom TensorRT plugin (C++/CUDA)
│   ├── anyPrecisionKernels.{h,cu}   launchers reusing matmul.cuh/dequant.cuh + cuBLAS
│   ├── anyPrecisionPlugin.{h,cpp}   IPluginV2DynamicExt + creator + C ABI
│   └── CMakeLists.txt
├── python/
│   ├── _plugin_loader.py          load/register .so, set_precision()
│   ├── ap_linear.py               functional + Module TRT-LLM building block
│   ├── convert_checkpoint.py      AP .bin -> fused numpy weights (CPU, verified)
│   ├── build_engine.py            swap Linears in TRT-LLM Qwen -> AP plugin, build
│   └── runtime.py                 ModelRunner wrapper + per-request precision
├── precision_manager.py          per-request precision registry
└── server.py                     FastAPI /generate with `precision` field
```

## Build & run (on an sm_80+ box)

### 1. Build the plugin
```bash
cd any_precision/trtllm_integration/plugin
cmake -B build -DTRT_ROOT=/path/to/TensorRT -DCMAKE_CUDA_ARCHITECTURES="80;90"
cmake --build build -j
# -> build/libanyprec_trt_plugin.so   (or set $ANYPREC_TRT_PLUGIN)
```

### 2. Build the engine
```bash
python -m any_precision.trtllm_integration.python.build_engine \
    --model_path /path/to/anyprec-Qwen3-4B \
    --output_dir ./trt_engine \
    --default_precision 8 --max_batch_size 8 --max_seq_len 4096
```

### 3. Serve
```bash
python -m any_precision.trtllm_integration.server \
    --engine_dir ./trt_engine \
    --tokenizer_dir /path/to/anyprec-Qwen3-4B --port 8002

curl -s localhost:8002/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"Explain quantization.","precision":4,"max_new_tokens":64}'
```

Or programmatically:
```python
from any_precision.trtllm_integration.python.runtime import AnyPrecisionTRTRunner
r = AnyPrecisionTRTRunner("./trt_engine", tokenizer_dir="/path/to/anyprec-Qwen3-4B")
print(r.generate("Hello", precision=3, max_new_tokens=32))   # 3-bit
print(r.generate("Hello", precision=8, max_new_tokens=32))   # 8-bit, same engine
```

## Verified vs. unverified

| Component | Status |
|---|---|
| Weight layout / q-k-v fusion / LUT flattening | ✅ verified on the real checkpoint (CPU) |
| Plugin source (kernels reused from upstream) | ✅ compiles against TRT headers; ❓ not run on GPU here |
| `build_engine.py` TRT-LLM model swap | ❓ version-sensitive; needs an sm_80+ box to validate |
| Runtime / server | ❓ depends on the engine build |

## Mixed precision & batching

The plugin precision is global, so a single engine call is single-precision.
For a MAS-style mixed-precision batch, group requests by precision and issue one
`generate_batch(..., precision=p)` per group (same approach as the vLLM Phase-2
worker). `precision_manager.PrecisionManager` tracks per-request precision.

## Notes / future work

- **IPluginV3**: TRT 10 deprecates `IPluginV2DynamicExt`. The plugin still works
  but migrating to `IPluginV3` is recommended long-term.
- **No tensor parallelism**: built for `tp_size=1`. TP would require sharding the
  packed `qweight`/`luts` along N and is not implemented.
- **Dense path accuracy**: the `M>8` path dequantizes then HGEMMs with fp32
  accumulation (`CUBLAS_COMPUTE_32F`) to match torch's fp16 matmul.
```
