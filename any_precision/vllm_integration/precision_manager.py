"""
precision_manager.py — Thread-safe registry of per-request precision.

The server registers (request_id → precision) before calling engine.generate().
The custom ModelRunner reads it to determine which precision to use per sequence.

Design notes:
  - Uses a plain dict + threading.Lock (no queues needed: requests arrive on
    the async event loop, are registered synchronously, then consumed once by
    the model runner worker thread).
  - Precision is looked up by request_id and auto-removed after first use
    to avoid unbounded growth.
  - DEFAULT_PRECISION is returned for any request_id not found in the registry
    (e.g. health-check dummy requests that vLLM fires internally).
"""

import threading
from typing import Dict, Optional

DEFAULT_PRECISION = 8   # used when no precision registered for a request


class PrecisionManager:
    """Global singleton registry: request_id → precision int."""

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
        """Return precision for request_id; returns DEFAULT_PRECISION if unknown."""
        with self._rw:
            return self._map.get(request_id, DEFAULT_PRECISION)

    def unregister(self, request_id: str) -> None:
        with self._rw:
            self._map.pop(request_id, None)

    def lookup_batch(self, request_ids) -> Dict[str, int]:
        """Return {request_id: precision} for all ids in the iterable."""
        with self._rw:
            return {rid: self._map.get(rid, DEFAULT_PRECISION) for rid in request_ids}
