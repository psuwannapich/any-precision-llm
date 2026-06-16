"""
plugin_trt_check.py — verify the AnyPrecisionLinear plugin produces correct
numbers *inside a real TensorRT engine*, isolated from the TRT-LLM Qwen model.

Builds a 1-op network (input -> AnyPrecisionLinear plugin -> output), runs it on
a random input, and compares against the reference any_precision_ext.matmul_kbit
(the proven-correct PyTorch kernel) using the same converted weights.

Run (in .venv-gpu with any_precision_ext on PYTHONPATH and OMPI_MCA_ess_singleton_isolated=1):
  python -m any_precision.trtllm_integration.tests.plugin_trt_check \
      --model_path <packed ckpt> --module layers.0.mlp.down --precision 8 --M 4
"""
import argparse
import numpy as np
import torch
import tensorrt as trt

from ..python import convert_checkpoint as cc
from ..python._plugin_loader import load_plugin, set_precision
from ..python import ap_linear


def build_engine(qweight, luts, supported_bits, N, K, seed, parent, precision, M):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    x = network.add_input("x", trt.DataType.HALF, (M, K))

    creator = trt.get_plugin_registry().get_plugin_creator(
        "AnyPrecisionLinear", "1", "")

    def i32(name, v):
        return trt.PluginField(name, np.array([v], np.int32),
                               trt.PluginFieldType.INT32)
    fields = [
        i32("N", N), i32("K", K), i32("seed_bits", seed),
        i32("parent_bits", parent), i32("default_precision", precision),
        trt.PluginField("supported_bits", np.asarray(supported_bits, np.int32),
                        trt.PluginFieldType.INT32),
        trt.PluginField("qweight", np.ascontiguousarray(qweight, np.int32).reshape(-1),
                        trt.PluginFieldType.INT32),
        trt.PluginField("luts", np.ascontiguousarray(luts, np.float16).reshape(-1),
                        trt.PluginFieldType.FLOAT16),
        trt.PluginField("bias", np.zeros((0,), np.float16),
                        trt.PluginFieldType.FLOAT16),
    ]
    plugin = creator.create_plugin("ap", trt.PluginFieldCollection(fields))
    layer = network.add_plugin_v2([x], plugin)
    layer.get_output(0).name = "y"
    network.mark_output(layer.get_output(0))

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    return builder.build_serialized_network(network, config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--module", default="layers.0.mlp.down")
    ap.add_argument("--precision", type=int, default=8)
    ap.add_argument("--M", type=int, default=4)
    args = ap.parse_args()

    load_plugin()
    conv = cc.convert(args.model_path)
    meta = conv["meta"]
    d = conv["ap"][args.module]
    qweight, luts = d["qweight"], d["luts"]
    parent = meta["parent_bits"]
    N = qweight.shape[1]
    K = qweight.shape[2] * 32
    p = args.precision

    torch.manual_seed(0)
    x = torch.randn(args.M, K, dtype=torch.float16, device="cuda")

    # reference
    from any_precision_ext import matmul_kbit
    off = sum(N * (2 ** b) for b in meta["supported_bits"] if b < p)
    lut_p = torch.from_numpy(luts[off:off + N * (2 ** p)].reshape(N, 2 ** p)).cuda()
    qw_t = torch.from_numpy(qweight).cuda()
    y_ref = matmul_kbit(x, qw_t, lut_p, p)

    # TRT engine
    set_precision(p)
    serialized = build_engine(qweight, luts, meta["supported_bits"], N, K,
                              meta["seed_bits"], parent, p, args.M)
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = rt.deserialize_cuda_engine(serialized)
    ctx = engine.create_execution_context()
    y_trt = torch.empty(args.M, N, dtype=torch.float16, device="cuda")
    ctx.set_input_shape("x", (args.M, K))
    ctx.set_tensor_address("x", x.data_ptr())
    ctx.set_tensor_address("y", y_trt.data_ptr())
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    diff = (y_trt.float() - y_ref.float()).abs()
    print(f"module={args.module} N={N} K={K} precision={p} M={args.M}")
    print(f"  y_ref[0,:5]  = {y_ref[0,:5].tolist()}")
    print(f"  y_trt[0,:5]  = {y_trt[0,:5].tolist()}")
    print(f"  max abs diff = {diff.max().item():.5f}   mean = {diff.mean().item():.6f}")
    ok = diff.max().item() < 0.5
    print("  RESULT:", "PASS (plugin matches reference in TRT)" if ok
          else "FAIL (plugin output wrong inside TRT engine)")


if __name__ == "__main__":
    main()
