from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from threading import Lock
from typing import Iterator


class RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: defaultdict[str, int] = defaultdict(int)
        self._stage_total_ms: defaultdict[str, float] = defaultdict(float)
        self._stage_count: defaultdict[str, int] = defaultdict(int)

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def observe_ms(self, name: str, elapsed_ms: float) -> None:
        with self._lock:
            self._stage_total_ms[name] += elapsed_ms
            self._stage_count[name] += 1

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe_ms(name, (time.perf_counter() - start) * 1000)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            stages = {}
            for name, total_ms in self._stage_total_ms.items():
                count = self._stage_count[name]
                stages[name] = {
                    "count": count,
                    "total_ms": round(total_ms, 2),
                    "avg_ms": round(total_ms / count, 2) if count else 0.0,
                }
            return {
                "counters": dict(self._counters),
                "stages": stages,
            }

