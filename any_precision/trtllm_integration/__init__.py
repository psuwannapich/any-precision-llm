"""
Any-Precision TensorRT-LLM integration.

Pipeline:
  1. build the plugin .so      (plugin/CMakeLists.txt)
  2. build the engine          (python/build_engine.py)
  3. serve / generate          (python/runtime.py, server.py)

The precision control and plugin loader are importable without TensorRT-LLM
installed; the model-definition / runtime pieces import tensorrt_llm lazily.
See README.md for hardware requirements (compute capability >= 8.0).
"""

from .python._plugin_loader import load_plugin, set_precision, get_precision  # noqa: F401
from .precision_manager import PrecisionManager  # noqa: F401
