"""
convert_checkpoint.py — turn an Any-Precision HF checkpoint into the flat numpy
weights consumed by the TensorRT-LLM AnyPrecisionLinear plugin.

Per quantized linear the plugin needs:
  qweight : int32   [parent_bits, N, K//32]
  luts    : float16 flat = concat_b( lut_b.reshape(-1) ), b ascending

Attention q/k/v projections are fused into one `qkv` linear by concatenating
along the output (N) axis — valid because AP packing is per-output-row and
q/k/v share K (hidden_size). down/gate/up and o_proj are kept as-is.

Non-quantized tensors (embeddings, RMSNorm weights, q/k norm, lm_head) are
passed through unchanged for the surrounding TRT-LLM model definition.

Usage (optional materialization to disk):
  python -m any_precision.trtllm_integration.python.convert_checkpoint \
      --model_path /path/to/anyprec-Qwen3-4B --out_dir ./trt_ckpt
build_engine.py can also call convert() directly to avoid a 5 GB intermediate.
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

ATTN_PROJS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_PROJS = ["gate_proj", "up_proj", "down_proj"]


def _supported_bits(config: dict) -> List[int]:
    ap = config["anyprec"]
    return list(range(ap["seed_precision"], ap["parent_precision"] + 1))


def _flat_luts(sd: dict, prefix: str, bits: List[int]) -> np.ndarray:
    """concat_b lut_b.reshape(-1) for one linear, ascending bit order."""
    parts = []
    for b in bits:
        lut = sd[f"{prefix}.lut{b}"]
        parts.append(lut.to(torch.float16).contiguous().view(-1).numpy())
    return np.concatenate(parts)


def _fused_qkv_luts(sd: dict, layer_prefix: str, bits: List[int]) -> np.ndarray:
    """Row-concat q,k,v LUTs per bit, then flatten/concat over bits.

    Order must match the qweight fusion (q, then k, then v along N).
    """
    parts = []
    for b in bits:
        per_bit = []
        for proj in ["q_proj", "k_proj", "v_proj"]:
            lut = sd[f"{layer_prefix}.self_attn.{proj}.lut{b}"]
            per_bit.append(lut.to(torch.float16).contiguous().numpy())  # [Nx, 2^b]
        parts.append(np.concatenate(per_bit, axis=0).reshape(-1))       # [Nq+Nk+Nv, 2^b]
    return np.concatenate(parts)


def _qweight(sd: dict, prefix: str) -> np.ndarray:
    return sd[f"{prefix}.qweight"].to(torch.int32).contiguous().numpy()


def convert(model_path: str, config: Optional[dict] = None) -> Dict:
    """Load the AP checkpoint and return a dict of numpy weights.

    Returns:
      {
        "meta": {...},
        "ap":   { "<module path>": {"qweight":..., "luts":...}, ... },
        "dense":{ "<name>": np.ndarray, ... }   # fp16/fp32 passthrough tensors
      }
    """
    if config is None:
        with open(os.path.join(model_path, "config.json")) as f:
            config = json.load(f)

    bits = _supported_bits(config)
    seed, parent = bits[0], bits[-1]
    n_layers = config["num_hidden_layers"]

    bin_path = os.path.join(model_path, "pytorch_model.bin")
    sd = torch.load(bin_path, map_location="cpu", mmap=True, weights_only=False)

    ap: Dict[str, Dict[str, np.ndarray]] = {}
    dense: Dict[str, np.ndarray] = {}

    def put_dense(key):
        dense[key] = sd[key].to(torch.float16).contiguous().numpy()

    # global / tied tensors
    put_dense("model.embed_tokens.weight")
    put_dense("model.norm.weight")
    if "lm_head.weight" in sd:
        put_dense("lm_head.weight")

    for i in range(n_layers):
        lp = f"model.layers.{i}"

        # norms (dense passthrough)
        for nm in ["input_layernorm.weight", "post_attention_layernorm.weight",
                   "self_attn.q_norm.weight", "self_attn.k_norm.weight"]:
            key = f"{lp}.{nm}"
            if key in sd:
                put_dense(key)

        # fused QKV
        qkv_qw = np.concatenate(
            [_qweight(sd, f"{lp}.self_attn.{p}") for p in ["q_proj", "k_proj", "v_proj"]],
            axis=1)
        ap[f"layers.{i}.attn.qkv"] = {
            "qweight": qkv_qw,
            "luts": _fused_qkv_luts(sd, lp, bits),
        }
        # output projection
        ap[f"layers.{i}.attn.o"] = {
            "qweight": _qweight(sd, f"{lp}.self_attn.o_proj"),
            "luts": _flat_luts(sd, f"{lp}.self_attn.o_proj", bits),
        }
        # mlp
        for src, dst in [("gate_proj", "gate"), ("up_proj", "up"),
                         ("down_proj", "down")]:
            ap[f"layers.{i}.mlp.{dst}"] = {
                "qweight": _qweight(sd, f"{lp}.mlp.{src}"),
                "luts": _flat_luts(sd, f"{lp}.mlp.{src}", bits),
            }

    meta = {
        "supported_bits": bits,
        "seed_bits": seed,
        "parent_bits": parent,
        "num_hidden_layers": n_layers,
        "hidden_size": config["hidden_size"],
        "intermediate_size": config["intermediate_size"],
        "num_attention_heads": config["num_attention_heads"],
        "num_key_value_heads": config["num_key_value_heads"],
        "head_dim": config.get("head_dim",
                               config["hidden_size"] // config["num_attention_heads"]),
        "vocab_size": config["vocab_size"],
        "rms_norm_eps": config.get("rms_norm_eps", 1e-6),
        "rope_theta": config.get("rope_theta", 1e6),
        "max_position_embeddings": config.get("max_position_embeddings", 40960),
        "tie_word_embeddings": config.get("tie_word_embeddings", False),
    }
    return {"meta": meta, "ap": ap, "dense": dense}


def save_npz(converted: Dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    flat = {}
    for name, d in converted["ap"].items():
        flat[f"ap::{name}::qweight"] = d["qweight"]
        flat[f"ap::{name}::luts"] = d["luts"]
    for name, arr in converted["dense"].items():
        flat[f"dense::{name}"] = arr
    np.savez(os.path.join(out_dir, "anyprec_weights.npz"), **flat)
    with open(os.path.join(out_dir, "anyprec_meta.json"), "w") as f:
        json.dump(converted["meta"], f, indent=2)
    print(f"Saved {len(flat)} arrays + meta to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    converted = convert(args.model_path)
    save_npz(converted, args.out_dir)


if __name__ == "__main__":
    main()
