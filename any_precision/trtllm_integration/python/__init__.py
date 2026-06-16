"""
TensorRT-LLM integration for Any-Precision LLMs.

Public surface:
    load_plugin()        -> load & register the libanyprec_trt_plugin.so
    set_precision(bits)  -> set the active bit-width for the next engine run
    AnyPrecisionLinear   -> TRT-LLM drop-in linear backed by the LUT plugin
    any_precision_linear -> functional form used inside model definitions

See ../README.md for the full build/run pipeline and hardware requirements
(TensorRT-LLM requires compute capability >= 8.0; this repo's default GPU is a
V100/sm_70 and cannot run the resulting engine).
"""

from ._plugin_loader import load_plugin, set_precision, get_precision  # noqa: F401
from .ap_linear import AnyPrecisionLinear, any_precision_linear  # noqa: F401
