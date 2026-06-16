"""
ap_linear.py — TensorRT-LLM building blocks for Any-Precision linear layers.

Two forms:
  * any_precision_linear(...)  : functional — inserts the AnyPrecisionLinear
                                  plugin into the current TRT-LLM network and
                                  returns the output Tensor.
  * AnyPrecisionLinear(...)    : a tensorrt_llm.module.Module that is a drop-in
                                  for tensorrt_llm.layers.Linear, so an existing
                                  model definition can be patched layer-by-layer.

Weight layout (per linear, produced by convert_checkpoint.py):
  qweight : int32   [parent_bits, N, K//32]   bit-plane packed
  luts    : float16 [sum_b N*2^b]             supported LUTs concatenated, in
                                              ascending bit order
  bias    : float16 [N] or None

The plugin selects the active precision at runtime from the process-global set
by set_precision(); it is NOT a build-time constant, so a single engine serves
every supported bit-width.
"""

from typing import List, Optional, Sequence

import numpy as np

try:
    import tensorrt as trt
    from tensorrt_llm._common import default_trtnet
    from tensorrt_llm.functional import Tensor, _create_tensor
    from tensorrt_llm.module import Module
    _HAS_TRTLLM = True
except Exception:  # pragma: no cover - import guarded so the file is importable
    _HAS_TRTLLM = False
    Module = object  # type: ignore

from ._plugin_loader import load_plugin

_PLUGIN_NAME = "AnyPrecisionLinear"
_PLUGIN_VERSION = "1"
_PLUGIN_NAMESPACE = ""


def _get_creator():
    load_plugin()  # ensures REGISTER_TENSORRT_PLUGIN ran
    creator = trt.get_plugin_registry().get_plugin_creator(
        _PLUGIN_NAME, _PLUGIN_VERSION, _PLUGIN_NAMESPACE)
    if creator is None:
        raise RuntimeError(
            f"Plugin creator {_PLUGIN_NAME} v{_PLUGIN_VERSION} not registered. "
            "Did libanyprec_trt_plugin.so load correctly?")
    return creator


def _i32(name, value):
    return trt.PluginField(name, np.array([value], dtype=np.int32),
                           trt.PluginFieldType.INT32)


def any_precision_linear(
    x: "Tensor",
    qweight: np.ndarray,           # int32 [parent_bits, N, K//32]
    luts: np.ndarray,              # float16 flat, concatenated per bit
    supported_bits: Sequence[int],
    N: int,
    K: int,
    seed_bits: int,
    parent_bits: int,
    default_precision: int,
    bias: Optional[np.ndarray] = None,   # float16 [N]
) -> "Tensor":
    """Insert one AnyPrecisionLinear plugin into the active network."""
    if not _HAS_TRTLLM:
        raise RuntimeError("tensorrt_llm is required to build the network.")

    creator = _get_creator()
    fields = [
        _i32("N", N),
        _i32("K", K),
        _i32("seed_bits", seed_bits),
        _i32("parent_bits", parent_bits),
        _i32("default_precision", default_precision),
        trt.PluginField("supported_bits",
                        np.asarray(supported_bits, dtype=np.int32),
                        trt.PluginFieldType.INT32),
        trt.PluginField("qweight",
                        np.ascontiguousarray(qweight, dtype=np.int32).reshape(-1),
                        trt.PluginFieldType.INT32),
        trt.PluginField("luts",
                        np.ascontiguousarray(luts, dtype=np.float16).reshape(-1),
                        trt.PluginFieldType.FLOAT16),
    ]
    if bias is not None:
        fields.append(trt.PluginField(
            "bias", np.ascontiguousarray(bias, dtype=np.float16).reshape(-1),
            trt.PluginFieldType.FLOAT16))
    else:
        fields.append(trt.PluginField(
            "bias", np.zeros((0,), dtype=np.float16),
            trt.PluginFieldType.FLOAT16))

    pfc = trt.PluginFieldCollection(fields)
    plugin = creator.create_plugin("anyprec_linear", pfc)
    layer = default_trtnet().add_plugin_v2([x.trt_tensor], plugin)
    return _create_tensor(layer.get_output(0), layer)


class AnyPrecisionLinear(Module):
    """Drop-in replacement for tensorrt_llm.layers.Linear backed by the plugin.

    Construct it, then call .load_weights(...) with the converted numpy arrays
    before building the engine. forward(x) emits the plugin op.
    """

    def __init__(self, in_features: int, out_features: int,
                 supported_bits: Sequence[int], seed_bits: int,
                 parent_bits: int, default_precision: int = 8,
                 bias: bool = False, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.supported_bits = list(supported_bits)
        self.seed_bits = seed_bits
        self.parent_bits = parent_bits
        self.default_precision = default_precision
        self.use_bias = bias
        self.dtype = dtype

        # populated by load_weights()
        self._qweight: Optional[np.ndarray] = None
        self._luts: Optional[np.ndarray] = None
        self._bias: Optional[np.ndarray] = None

    def load_weights(self, qweight: np.ndarray, luts: np.ndarray,
                     bias: Optional[np.ndarray] = None):
        self._qweight = np.ascontiguousarray(qweight, dtype=np.int32)
        self._luts = np.ascontiguousarray(luts, dtype=np.float16)
        self._bias = (np.ascontiguousarray(bias, dtype=np.float16)
                      if bias is not None else None)
        return self

    def forward(self, x: "Tensor", *args, **kwargs) -> "Tensor":
        # Accept (and ignore) the extra args TRT-LLM's Column/RowLinear may pass
        # (lora_layer_params, all_reduce_params, ...): the plugin handles only
        # the dense projection; those features are out of scope.
        if self._qweight is None:
            raise RuntimeError("AnyPrecisionLinear.load_weights() not called.")
        return any_precision_linear(
            x, self._qweight, self._luts, self.supported_bits,
            N=self.out_features, K=self.in_features,
            seed_bits=self.seed_bits, parent_bits=self.parent_bits,
            default_precision=self.default_precision, bias=self._bias)


def concat_qkv(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Fuse q/k/v qweight by concatenating along the output (N) axis.

    AP packing is per-output-row and q/k/v share K (hidden_size), so a fused
    QKV projection is exactly the row-concatenation of the three packed tensors.
    Shapes: [parent_bits, Nx, K//32] -> [parent_bits, Nq+Nk+Nv, K//32].
    """
    return np.concatenate([q, k, v], axis=1)


def concat_qkv_luts(luts: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate per-bit LUT lists for q,k,v along the row (N) axis, per bit.

    Each element of `luts` is the flat concat-over-bits LUT for one projection;
    here we expect already-per-bit-split arrays. See convert_checkpoint.py which
    builds the fused flat buffer directly, so this helper is provided for
    completeness / testing.
    """
    return np.concatenate(list(luts), axis=0)
