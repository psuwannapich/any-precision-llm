"""
precision_manager.py — per-request precision registry for the TensorRT-LLM
backend, mirroring the vLLM integration's precision_manager.

The TRT plugin reads a single process-global precision in enqueue(), so the
server registers (request_id -> precision) here and the runner applies it via
apply() right before each engine call. Because the precision is global, a single
engine call must be single-precision; batch mixed-precision requests by grouping
them per precision (see python/runtime.py:generate_batch).
"""

import threading
from typing import Dict, Optional

from .python._plugin_loader import set_precision

DEFAULT_PRECISION = 8


class PrecisionManager:
    _instance: Optional["PrecisionManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._map: Dict[str, int] = {}
        self._rw = threading.Lock()

    @classmethod
    def get(cls) -> "PrecisionManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register(self, request_id: str, precision: int) -> None:
        with self._rw:
            self._map[request_id] = precision

    def lookup(self, request_id: str) -> int:
        with self._rw:
            return self._map.get(request_id, DEFAULT_PRECISION)

    def unregister(self, request_id: str) -> None:
        with self._rw:
            self._map.pop(request_id, None)

    @staticmethod
    def apply(precision: int) -> None:
        """Push the precision into the plugin global (read by enqueue())."""
        set_precision(precision)
