"""
build_engine.py — build a TensorRT-LLM engine for an Any-Precision Qwen3 model.

Strategy: reuse TensorRT-LLM's own Qwen model definition (correct RoPE, KV
cache, gpt_attention plugin, RMSNorm, sampling) and swap only the projection
Linear submodules for the AnyPrecisionLinear plugin. The converter already
fuses q/k/v to match TRT-LLM's fused-QKV layout, so the swap is 1:1:

    transformer.layers.{i}.attention.qkv   <- ap[layers.{i}.attn.qkv]
    transformer.layers.{i}.attention.dense <- ap[layers.{i}.attn.o]
    transformer.layers.{i}.mlp.gate        <- ap[layers.{i}.mlp.gate]
    transformer.layers.{i}.mlp.fc          <- ap[layers.{i}.mlp.up]
    transformer.layers.{i}.mlp.proj        <- ap[layers.{i}.mlp.down]

Dense tensors (embeddings, RMSNorm weights, q/k norm, lm_head) are loaded into
the TRT-LLM model's parameters unchanged.

Version note: written against tensorrt_llm >= 0.11 (Qwen2/Qwen3 family). The
module names above and the weight-loading helpers are the parts most likely to
need small adjustments on other TRT-LLM versions; everything precision-specific
lives in the plugin and is version-stable.

Usage:
  python -m any_precision.trtllm_integration.python.build_engine \
      --model_path /path/to/anyprec-Qwen3-4B \
      --output_dir ./trt_engine \
      --default_precision 8 --max_batch_size 8 --max_seq_len 4096
"""

import argparse
import os

import numpy as np
import torch

from ._plugin_loader import load_plugin
from .ap_linear import AnyPrecisionLinear
from . import convert_checkpoint as cc

# Map TRT-LLM attention/mlp submodule attribute names -> our ap dict suffixes.
ATTN_MAP = {"qkv": "attn.qkv", "dense": "attn.o"}
MLP_MAP = {"gate": "mlp.gate", "fc": "mlp.up", "proj": "mlp.down"}


def _make_ap_linear(meta, in_features, out_features, default_precision):
    return AnyPrecisionLinear(
        in_features=in_features, out_features=out_features,
        supported_bits=meta["supported_bits"],
        seed_bits=meta["seed_bits"], parent_bits=meta["parent_bits"],
        default_precision=default_precision, bias=False)


def _swap_linears(trtllm_model, converted, default_precision):
    """Replace projection Linears in a TRT-LLM Qwen model with AP plugins."""
    meta = converted["meta"]
    ap = converted["ap"]
    hidden = meta["hidden_size"]
    inter = meta["intermediate_size"]
    n_head = meta["num_attention_heads"]
    n_kv = meta["num_key_value_heads"]
    head_dim = meta["head_dim"]
    qkv_out = (n_head + 2 * n_kv) * head_dim

    layers = trtllm_model.transformer.layers
    for i, layer in enumerate(layers):
        # attention
        for attr, suffix in ATTN_MAP.items():
            d = ap[f"layers.{i}.{suffix}"]
            out_f = qkv_out if attr == "qkv" else hidden
            in_f = hidden
            lin = _make_ap_linear(meta, in_f, out_f, default_precision)
            lin.load_weights(d["qweight"], d["luts"])
            setattr(layer.attention, attr, lin)
        # mlp (SwiGLU): gate/fc are hidden->inter, proj is inter->hidden
        for attr, suffix in MLP_MAP.items():
            d = ap[f"layers.{i}.{suffix}"]
            if attr == "proj":
                in_f, out_f = inter, hidden
            else:
                in_f, out_f = hidden, inter
            lin = _make_ap_linear(meta, in_f, out_f, default_precision)
            lin.load_weights(d["qweight"], d["luts"])
            setattr(layer.mlp, attr, lin)
    return trtllm_model


def _load_dense_weights(trtllm_model, converted):
    """Load the non-quantized tensors into TRT-LLM model parameters by name."""
    dense = converted["dense"]
    # Map HF names -> TRT-LLM parameter names (Qwen2/Qwen3).
    name_map = {
        "model.embed_tokens.weight": "transformer.vocab_embedding.weight",
        "model.norm.weight": "transformer.ln_f.weight",
        "lm_head.weight": "lm_head.weight",
    }
    n_layers = converted["meta"]["num_hidden_layers"]
    for i in range(n_layers):
        hf = f"model.layers.{i}"
        tl = f"transformer.layers.{i}"
        name_map[f"{hf}.input_layernorm.weight"] = f"{tl}.input_layernorm.weight"
        name_map[f"{hf}.post_attention_layernorm.weight"] = f"{tl}.post_layernorm.weight"
        # Qwen3 per-head q/k RMSNorm (attribute names vary; try common ones).
        name_map[f"{hf}.self_attn.q_norm.weight"] = f"{tl}.attention.q_layernorm.weight"
        name_map[f"{hf}.self_attn.k_norm.weight"] = f"{tl}.attention.k_layernorm.weight"

    params = dict(trtllm_model.named_parameters())
    loaded, missing = 0, []
    for hf_name, arr in dense.items():
        tl_name = name_map.get(hf_name)
        if tl_name is None or tl_name not in params:
            missing.append(hf_name)
            continue
        params[tl_name].value = np.ascontiguousarray(arr)
        loaded += 1
    if missing:
        print(f"[build_engine] WARNING: {len(missing)} dense tensors had no "
              f"matching TRT-LLM param (check version naming): {missing[:6]}...")
    print(f"[build_engine] loaded {loaded} dense tensors")
    return trtllm_model


def build(model_path, output_dir, default_precision=8, max_batch_size=8,
          max_input_len=2048, max_seq_len=4096, dtype="float16"):
    load_plugin()  # register the AP plugin before the network references it

    import tensorrt_llm
    from tensorrt_llm import Mapping
    from tensorrt_llm.builder import build as trtllm_build
    from tensorrt_llm.models.qwen.model import QWenForCausalLM
    from tensorrt_llm.models.qwen.config import QWenConfig
    from tensorrt_llm.plugin import PluginConfig

    converted = cc.convert(model_path)
    meta = converted["meta"]

    # Build a TRT-LLM Qwen config from the AP metadata.
    cfg = QWenConfig(
        architecture="Qwen3ForCausalLM",
        dtype=dtype,
        num_hidden_layers=meta["num_hidden_layers"],
        num_attention_heads=meta["num_attention_heads"],
        num_key_value_heads=meta["num_key_value_heads"],
        hidden_size=meta["hidden_size"],
        intermediate_size=meta["intermediate_size"],
        vocab_size=meta["vocab_size"],
        head_size=meta["head_dim"],
        max_position_embeddings=meta["max_position_embeddings"],
        norm_epsilon=meta["rms_norm_eps"],
        rotary_base=meta["rope_theta"],
        qwen_type="qwen3",
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
    )

    model = QWenForCausalLM(cfg)
    model = _swap_linears(model, converted, default_precision)
    model = _load_dense_weights(model, converted)

    # Enable the gpt_attention + paged-KV plugins for an efficient runtime.
    plugin_cfg = PluginConfig()
    plugin_cfg.gpt_attention_plugin = dtype
    plugin_cfg.paged_kv_cache = True
    plugin_cfg.remove_input_padding = True

    build_cfg = tensorrt_llm.BuildConfig(
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
        max_seq_len=max_seq_len,
        plugin_config=plugin_cfg,
    )

    os.makedirs(output_dir, exist_ok=True)
    engine = trtllm_build(model, build_cfg)
    engine.save(output_dir)

    # Stash AP metadata next to the engine for the runtime/server.
    import json
    with open(os.path.join(output_dir, "anyprec_meta.json"), "w") as f:
        json.dump({**meta, "default_precision": default_precision}, f, indent=2)
    print(f"[build_engine] engine + meta written to {output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--default_precision", type=int, default=8)
    ap.add_argument("--max_batch_size", type=int, default=8)
    ap.add_argument("--max_input_len", type=int, default=2048)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()
    build(args.model_path, args.output_dir, args.default_precision,
          args.max_batch_size, args.max_input_len, args.max_seq_len, args.dtype)


if __name__ == "__main__":
    main()
