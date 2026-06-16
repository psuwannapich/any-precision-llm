"""
_plugin_loader.py — load the compiled AnyPrecisionLinear TensorRT plugin and
expose the process-global precision control.

The plugin .so (libanyprec_trt_plugin.so) is built from ../plugin via CMake.
Loading it with ctypes triggers REGISTER_TENSORRT_PLUGIN, so the creator becomes
visible to `trt.get_plugin_registry()`. The same .so exports the C ABI
`ap_set_precision` / `ap_get_precision`, which set the precision read by every
plugin instance inside enqueue().
"""

import ctypes
import glob
import os
from typing import Optional

_PLUGIN_LIB: Optional[ctypes.CDLL] = None

# Where to look for the built .so, in priority order.
_SEARCH_GLOBS = [
    os.environ.get("ANYPREC_TRT_PLUGIN", ""),
    os.path.join(os.path.dirname(__file__), "..", "plugin", "build",
                 "libanyprec_trt_plugin.so"),
    os.path.join(os.path.dirname(__file__), "..", "plugin",
                 "libanyprec_trt_plugin.so"),
    "libanyprec_trt_plugin.so",
]


def _find_library() -> str:
    for pattern in _SEARCH_GLOBS:
        if not pattern:
            continue
        for hit in glob.glob(pattern):
            if os.path.isfile(hit):
                return os.path.abspath(hit)
    raise FileNotFoundError(
        "libanyprec_trt_plugin.so not found. Build it with:\n"
        "  cd any_precision/trtllm_integration/plugin\n"
        "  cmake -B build -DTRT_ROOT=/path/to/TensorRT && cmake --build build -j\n"
        "or set $ANYPREC_TRT_PLUGIN to the .so path."
    )


def load_plugin(path: Optional[str] = None) -> ctypes.CDLL:
    """Load and register the plugin .so exactly once. Returns the CDLL handle."""
    global _PLUGIN_LIB
    if _PLUGIN_LIB is not None:
        return _PLUGIN_LIB

    so_path = path or _find_library()
    lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)

    # Declare the C ABI signatures we use.
    lib.ap_set_precision.argtypes = [ctypes.c_int]
    lib.ap_set_precision.restype = None
    lib.ap_get_precision.argtypes = []
    lib.ap_get_precision.restype = ctypes.c_int
    if hasattr(lib, "ap_register_plugin"):
        lib.ap_register_plugin.restype = ctypes.c_bool
        lib.ap_register_plugin()  # idempotent; REGISTER macro usually suffices

    _PLUGIN_LIB = lib
    return lib


def set_precision(bits: int) -> None:
    """Set the active bit-width (3..8) for subsequent engine executions."""
    load_plugin().ap_set_precision(int(bits))


def get_precision() -> int:
    """Return the currently active bit-width (0 if unset -> plugin default)."""
    return int(load_plugin().ap_get_precision())
