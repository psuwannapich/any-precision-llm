from . import modules
from .modules import AnyPrecisionForCausalLM


def __getattr__(name):
    if name == "quantization":
        from . import quantization as _q
        return _q
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
