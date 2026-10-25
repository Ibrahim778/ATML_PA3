"""Timers, peak-memory tracking, sizes and parameter counts."""
import time
from contextlib import contextmanager

import torch


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.elapsed_before = 0.0  # carried over from a resumed run

    def elapsed(self):
        return self.elapsed_before + time.perf_counter() - self.t0


@contextmanager
def timed(store: dict, key: str):
    t0 = time.perf_counter()
    yield
    store[key] = store.get(key, 0.0) + time.perf_counter() - t0


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 ** 3
    if device.type == "mps":
        # MPS has no peak counter; report the driver allocation at call time.
        return torch.mps.driver_allocated_memory() / 1024 ** 3
    return float("nan")


def tensor_bytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)
