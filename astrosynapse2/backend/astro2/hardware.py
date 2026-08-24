"""Low-overhead hardware and runtime telemetry."""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass
from typing import Any

import psutil


@dataclass(slots=True)
class RateMeter:
    started_at: float
    last_at: float
    last_games: int = 0
    last_decisions: int = 0
    last_games_per_second: float = 0.0
    last_decisions_per_second: float = 0.0

    @classmethod
    def start(cls) -> RateMeter:
        now = time.monotonic()
        return cls(started_at=now, last_at=now)

    def sample(self, games: int, decisions: int) -> dict[str, float]:
        now = time.monotonic()
        game_delta = games - self.last_games
        decision_delta = decisions - self.last_decisions
        # Actor batches arrive in bursts. Zero-delta 1 Hz telemetry must not
        # move the measurement boundary forward or the next completed batch
        # appears hundreds of times faster than its actual wall-clock rate.
        # A forced final zero-delta snapshot keeps the last complete interval.
        if game_delta > 0 or decision_delta > 0:
            elapsed = max(1e-6, now - self.last_at)
            if game_delta > 0:
                self.last_games_per_second = game_delta / elapsed
            if decision_delta > 0:
                self.last_decisions_per_second = decision_delta / elapsed
            self.last_at = now
            self.last_games = games
            self.last_decisions = decisions
        result = {
            "games_per_second": self.last_games_per_second,
            "decisions_per_second": self.last_decisions_per_second,
            "elapsed_seconds": now - self.started_at,
        }
        return result


def system_snapshot() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    process = psutil.Process()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": "Apple silicon" if platform.machine() == "arm64" else platform.processor(),
        "cpu_logical": psutil.cpu_count(logical=True) or os.cpu_count() or 1,
        "cpu_physical": psutil.cpu_count(logical=False) or 0,
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_total_bytes": memory.total,
        "memory_available_bytes": memory.available,
        "memory_percent": memory.percent,
        "process_rss_bytes": process.memory_info().rss,
        "swap_total_bytes": swap.total,
        "swap_used_bytes": swap.used,
        "swap_free_bytes": swap.free,
        "swap_percent": swap.percent,
        "swap_in_bytes": swap.sin,
        "swap_out_bytes": swap.sout,
        "python": platform.python_version(),
        "recommended_actor_processes": max(2, min(8, (psutil.cpu_count(logical=True) or 4) - 2)),
        "accelerator": "MLX / Metal" if platform.machine() == "arm64" else "MLX CPU",
    }


def mlx_snapshot() -> dict[str, Any]:
    """Return live Metal information. Call only inside the training process."""

    import importlib.metadata

    import mlx.core as mx

    probe = mx.array([1.0, 2.0, 3.0])
    mx.eval(probe)
    return {
        "version": importlib.metadata.version("mlx"),
        "device": str(mx.default_device()),
        "metal_available": bool(mx.metal.is_available()),
        "active_memory_bytes": int(mx.get_active_memory()),
        "cache_memory_bytes": int(mx.get_cache_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "device_info": dict(mx.device_info()),
    }
