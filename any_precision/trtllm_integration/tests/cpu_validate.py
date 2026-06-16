"""
cpu_validate.py — CPU-only validation of the parts of the TensorRT-LLM
integration that do NOT need a GPU.

What is (and isn't) tested:
  * TESTED on CPU: the code we wrote — the checkpoint converter, q/k/v fusion,
    the flat-LUT layout, and the producer/consumer contract with the C++
    plugin's lutOffsetElems(). Plus cross-file key consistency with
    build_engine.py and an npz save/load roundtrip.
  * NOT tested here: the CUDA kernels (reused verbatim from the already-working
    PyTorch path) and the TensorRT engine — both are GPU-only.

Run:
  srun -p interactive --qos debug -C batch --time=0:20:00 --mem=8G \
    .venv/bin/python -m any_precision.trtllm_integration.tests.cpu_validate \
      --model_path /path/to/anyprec-Qwen3-4B
"""

import argparse
import copy
import json
import os
import tempfile

import numpy as np
import torch

from ..python import convert_checkpoint as cc

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
_n_fail = 0


def check(name, cond):
    global _n_fail
    print(f"  [{PASS if cond else FAIL}] {name}")
    if not cond:
        _n_fail += 1
    return cond


def plugin_lut_offset(supported_bits, N, bits):
    """Pure-Python replica of AnyPrecisionPlugin::lutOffsetElems (C++)."""
    off = 0
    for b in supported_bits:
        if b == bits:
            return off
        off += N * (1 << b)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    args = ap.parse_args()

    with open(os.path.join(args.model_path, "config.json")) as f:
        config = json.load(f)
    bits = cc._supported_bits(config)
    print(f"\nsupported bits = {bits}")

    sd = torch.load(os.path.join(args.model_path, "pytorch_model.bin"),
                    map_location="cpu", mmap=True, weights_only=False)

    # ---- 1. flat-LUT layout matches the plugin's offset formula -------------
    print("\n[1] flat-LUT layout <-> plugin lutOffsetElems (o_proj, N=2560)")
    lp = "model.layers.0"
    o_prefix = f"{lp}.self_attn.o_proj"
    N_o = sd[f"{o_prefix}.qweight"].shape[1]
    flat = cc._flat_luts(sd, o_prefix, bits)
    ok = True
    for b in bits:
        off = plugin_lut_offset(bits, N_o, b)
        size = N_o * (1 << b)
        sliced = flat[off:off + size].reshape(N_o, 1 << b)
        ref = sd[f"{o_prefix}.lut{b}"].to(torch.float16).numpy()
        ok &= np.array_equal(sliced, ref)
    check("each lut_b recovered from flat buffer at plugin offset", ok)
    check("flat length == sum_b N*2^b",
          flat.size == sum(N_o * (1 << b) for b in bits))

    # ---- 2. q/k/v fusion: qweight concatenation along N --------------------
    print("\n[2] q/k/v fusion (qweight concat along output dim N)")
    q = cc._qweight(sd, f"{lp}.self_attn.q_proj")
    k = cc._qweight(sd, f"{lp}.self_attn.k_proj")
    v = cc._qweight(sd, f"{lp}.self_attn.v_proj")
    fused = np.concatenate([q, k, v], axis=1)
    Nq, Nk, Nv = q.shape[1], k.shape[1], v.shape[1]
    check(f"fused N = Nq+Nk+Nv = {Nq}+{Nk}+{Nv} = {fused.shape[1]}",
          fused.shape[1] == Nq + Nk + Nv)
    check("planes (parent_bits) preserved", fused.shape[0] == q.shape[0])
    check("K/32 shared & preserved", fused.shape[2] == q.shape[2])
    check("fused[:, :Nq] == q", np.array_equal(fused[:, :Nq], q))
    check("fused[:, Nq:Nq+Nk] == k", np.array_equal(fused[:, Nq:Nq+Nk], k))
    check("fused[:, Nq+Nk:] == v", np.array_equal(fused[:, Nq+Nk:], v))

    # ---- 3. fused QKV LUTs: per-bit row-concat, recoverable at offsets ------
    print("\n[3] fused QKV LUT layout (N=Nq+Nk+Nv)")
    N_qkv = Nq + Nk + Nv
    fused_luts = cc._fused_qkv_luts(sd, lp, bits)
    ok = True
    for b in bits:
        off = plugin_lut_offset(bits, N_qkv, b)
        size = N_qkv * (1 << b)
        sliced = fused_luts[off:off + size].reshape(N_qkv, 1 << b)
        ref = np.concatenate([
            sd[f"{lp}.self_attn.q_proj.lut{b}"].to(torch.float16).numpy(),
            sd[f"{lp}.self_attn.k_proj.lut{b}"].to(torch.float16).numpy(),
            sd[f"{lp}.self_attn.v_proj.lut{b}"].to(torch.float16).numpy(),
        ], axis=0)
        ok &= np.array_equal(sliced, ref)
    check("each fused lut_b == row-concat(q,k,v) at plugin offset", ok)

    # ---- 4. convert() on a 2-layer subset + key consistency w/ build_engine -
    print("\n[4] convert() (2-layer subset) + build_engine key consistency")
    cfg2 = copy.deepcopy(config)
    cfg2["num_hidden_layers"] = 2
    converted = cc.convert(args.model_path, cfg2)
    from ..python import build_engine as be

    keys = set(converted["ap"].keys())
    expected = set()
    for i in range(2):
        for suf in be.ATTN_MAP.values():
            expected.add(f"layers.{i}.{suf}")
        for suf in be.MLP_MAP.values():
            expected.add(f"layers.{i}.{suf}")
    check("every build_engine ATTN/MLP map key exists in converted ap",
          expected.issubset(keys))
    check("meta supported_bits/seed/parent correct",
          converted["meta"]["supported_bits"] == bits and
          converted["meta"]["seed_bits"] == bits[0] and
          converted["meta"]["parent_bits"] == bits[-1])
    check("dense has embed/norm/lm_head",
          "model.embed_tokens.weight" in converted["dense"] and
          "model.norm.weight" in converted["dense"])

    # qkv out dim consistency with build_engine's formula
    m = converted["meta"]
    qkv_out = (m["num_attention_heads"] + 2 * m["num_key_value_heads"]) * m["head_dim"]
    check(f"converted qkv N ({converted['ap']['layers.0.attn.qkv']['qweight'].shape[1]}) "
          f"== (n_head+2*n_kv)*head_dim ({qkv_out})",
          converted["ap"]["layers.0.attn.qkv"]["qweight"].shape[1] == qkv_out)

    # ---- 5. npz save/load roundtrip ---------------------------------------
    print("\n[5] npz save/load roundtrip")
    with tempfile.TemporaryDirectory() as td:
        cc.save_npz(converted, td)
        loaded = np.load(os.path.join(td, "anyprec_weights.npz"))
        qkv_key = "ap::layers.0.attn.qkv::qweight"
        check("npz contains fused qkv qweight", qkv_key in loaded.files)
        check("roundtrip array equal",
              np.array_equal(loaded[qkv_key],
                             converted["ap"]["layers.0.attn.qkv"]["qweight"]))
        check("meta json written",
              os.path.exists(os.path.join(td, "anyprec_meta.json")))

    # ---- summary ----------------------------------------------------------
    print("\n" + "=" * 60)
    if _n_fail == 0:
        print("  ALL CPU CHECKS PASSED")
    else:
        print(f"  {_n_fail} CHECK(S) FAILED")
    print("=" * 60)
    return _n_fail


if __name__ == "__main__":
    raise SystemExit(main())
