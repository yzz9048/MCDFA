"""Small shared helpers: seeding, device selection, JSON-safe conversion."""

from __future__ import annotations

import contextlib
import json
import random
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


def resolve_device(requested: str = "auto") -> torch.device:
    requested = (requested or "auto").casefold()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(requested)


@contextlib.contextmanager
def temporary_seed(seed: int) -> Iterator[None]:
    """Seed all RNGs and restore their exact prior state on exit."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def stable_seed(*parts: object) -> int:
    """Deterministic 31-bit seed independent of Python's hash randomisation."""
    import zlib
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(zlib.crc32(payload)) & 0x7FFFFFFF


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_json(path: Path, value: Any, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=indent)


class ResourceMeter:
    """Measure peak additional RSS (CPU) and peak CUDA allocation."""

    def __init__(self, device: torch.device, interval: float = 0.01):
        self.device = device
        self.interval = interval
        self.baseline_rss = 0
        self.peak_rss = 0
        self.baseline_cuda = 0
        self.peak_cuda = 0
        self._stop = threading.Event()
        self._thread = None
        self.available = False

    def _sample(self) -> None:
        try:
            import psutil
            process = psutil.Process()
            self.peak_rss = max(self.peak_rss, process.memory_info().rss)
        except Exception:
            pass

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self) -> "ResourceMeter":
        try:
            import psutil
            self.baseline_rss = psutil.Process().memory_info().rss
            self.peak_rss = self.baseline_rss
            self.available = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        except Exception:
            self.available = False
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.baseline_cuda = torch.cuda.memory_allocated(self.device)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sample()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.peak_cuda = torch.cuda.max_memory_allocated(self.device)

    @property
    def rss_delta_mb(self) -> float:
        if not self.available:
            return 0.0
        return max(0.0, (self.peak_rss - self.baseline_rss) / (1024.0 ** 2))

    @property
    def cuda_delta_mb(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return max(0.0, (self.peak_cuda - self.baseline_cuda) / (1024.0 ** 2))

    @property
    def resource_mb(self) -> float:
        return self.cuda_delta_mb if self.device.type == "cuda" else self.rss_delta_mb


class Stopwatch:
    def __init__(self):
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Stopwatch":
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start
