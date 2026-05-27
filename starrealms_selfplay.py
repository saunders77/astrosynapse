"""Resumable self-play training for the Star Realms simulator.

This module trains a lightweight PPO-style actor-critic without external ML
dependencies. The policy learns through the existing `choose(player, options,
state) -> int` interface used by `sim.py`.
"""

from __future__ import annotations

import copy
import concurrent.futures
import json
import math
import multiprocessing
import os
import queue
import random
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from chooser import choose as human_choose
from sim import Game, cardDetails, factions

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

try:
    import torch_directml  # type: ignore
except ImportError as exc:
    torch_directml = None
    TORCH_DIRECTML_IMPORT_ERROR = exc
else:
    TORCH_DIRECTML_IMPORT_ERROR = None


ROOT_DIR = Path(__file__).resolve().parent
RUNS_DIR = ROOT_DIR / "starrealms_policies"
CROSS_RUN_RATINGS_FILE = RUNS_DIR / "cross_run_ratings.json"
LATEST_RUN_NAME = "default"
INITIAL_ELO = 1000.0
CONFIG_DEFAULTS_VERSION = 5
CARD_ACQUIRE_ELO_TEST_GAMES = 500
CARD_ACQUIRE_ELO_K_FACTOR = 24.0
SCRAP_ELO_TEST_GAMES = 1000
SCRAP_ELO_K_FACTOR = 24.0
CROSS_RUN_RATING_VERSION = 2
CROSS_RUN_RATING_MODEL = "glicko_2"
CROSS_RUN_GLICKO2_SCALE = 173.7178
CROSS_RUN_GLICKO2_DEFAULT_RD = 350.0
CROSS_RUN_GLICKO2_MAX_RD = 350.0
CROSS_RUN_GLICKO2_MIN_RD = 1.0
CROSS_RUN_GLICKO2_DEFAULT_VOLATILITY = 0.06
CROSS_RUN_GLICKO2_TAU = 0.5
CROSS_RUN_GLICKO2_INHERITED_RD_FLOOR = 80.0
CROSS_RUN_CONFIDENCE_SIGMAS = 1.96
CROSS_RUN_LEGACY_CONFIDENCE_95_SCALE = 680.0
ELO_LOGISTIC_SCALE = math.log(10.0) / 400.0
CARD_ZONE_SCALE = 15.0
NUMERIC_OPTION_SCALE = 10.0
EPSILON = 1e-12
LEGACY_MODEL_ARCHITECTURE = "legacy_v1"
CURRENT_MODEL_ARCHITECTURE = "deep_v2"
MODEL_TYPE_DEEP = "deep"
MODEL_TYPE_DEFAULT = "default"
DEFAULT_MODEL_HIDDEN_SIZE = 24
DEEP_MODEL_HIDDEN_SIZE = 96
DEVICE_AUTO = "auto"
DEVICE_CPU = "cpu"
DEVICE_DIRECTML = "directml"
DEVICE_CUDA = "cuda"
DEVICE_MPS = "mps"
NOOP_ACTIONS = {"endTurn", "nodiscard", "nokill", "noRowScrap", "noScrapFromHand", "nocopy"}
TRAINING_DECISIONS_PER_GAME_ALL = "ALL"
TRAINING_DECISIONS_PER_GAME_1 = "1"
TRAINING_DECISIONS_PER_GAME_5 = "5"
TRAINING_DECISIONS_PER_GAME_OPTIONS = (
    TRAINING_DECISIONS_PER_GAME_1,
    TRAINING_DECISIONS_PER_GAME_5,
    TRAINING_DECISIONS_PER_GAME_ALL,
)
LIVE_PROGRESS_SAVE_INTERVAL_SECONDS = 0.4
DIRECTML_BENCHMARK_INFERENCE_REPEATS = 10
DIRECTML_BENCHMARK_TRAINING_SAMPLE_COUNT = 256
DIRECTML_BENCHMARK_MINIBATCH_SIZE = 64
DIRECTML_BENCHMARK_MARGIN = 0.95
SIMULATION_WORKERS_AUTO = 0
SIMULATION_TASKS_PER_WORKER = 3
SIMULATION_IN_FLIGHT_TASKS_PER_WORKER = 2
SIMULATION_WORKER_CPU_RESERVE = 2
SIMULATION_WORKER_TORCH_THREADS = 1
SIMULATION_PROGRESS_POLL_SECONDS = 0.25
SIMULATION_WORKER_POLICY_CACHE_LIMIT = 32
DEFAULT_MAX_TRAINING_SAMPLES_PER_GAME = 70
DEFAULT_MAX_TRAINING_SAMPLES_PER_ITERATION = 8192
DEFAULT_MIN_AVAILABLE_MEMORY_MB = 1024
MEMORY_SAFETY_CHECK_MATCH_INTERVAL = 16


def _normalize_ability(raw_ability: Any) -> str:
    if not isinstance(raw_ability, str) or raw_ability == "":
        return "none"
    return _normalize_ability_cached(raw_ability)


@lru_cache(maxsize=None)
def _normalize_ability_cached(raw_ability: str) -> str:
    if raw_ability == "none":
        return "none"
    if raw_ability[0] == "-":
        return raw_ability[1:]
    return raw_ability


ABILITY_FAMILIES = sorted(
    {
        _normalize_ability(card[7])
        for card in cardDetails
    }
    | {
        _normalize_ability(card[8])
        for card in cardDetails
    }
    | {
        _normalize_ability(card[10])
        for card in cardDetails
    }
)
ABILITY_TO_INDEX = {name: idx for idx, name in enumerate(ABILITY_FAMILIES)}
FACTION_TO_INDEX = {name: idx for idx, name in enumerate(factions)}
CARD_TYPE_TO_INDEX = {"ship": 0, "base": 1, "outp": 2}
ACTION_TYPES = [
    "play",
    "abilityOption",
    "scrapFromPlay",
    "attack",
    "attackOpponent",
    "acquire",
    "endTurn",
    "discardNormal",
    "discardDraw",
    "nodiscard",
    "gainattack",
    "draw",
    "trade",
    "authority",
    "nocopy",
    "copyship",
    "freeAcquire",
    "killbase",
    "nokill",
    "rowscrap",
    "noRowScrap",
    "scrapFromHandNormal",
    "scrapFromDiscardNormal",
    "scrapFromHandDraw",
    "scrapFromDiscardDraw",
    "noScrapFromHand",
    "other",
]
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTION_TYPES)}
CARD_NAME_ORDER = [str(card[0]) for card in cardDetails]
CARD_BY_NAME = {str(card[0]): card for card in cardDetails}
CARD_COST_BY_NAME = {str(card[0]): int(card[1]) for card in cardDetails}
SCRAP_ELO_ACTION_SOURCE = {
    "scrapFromHandNormal": "hand",
    "scrapFromHandDraw": "hand",
    "scrapFromDiscardNormal": "discard",
    "scrapFromDiscardDraw": "discard",
}
SCRAP_ELO_NO_SCRAP_KEY = "noscrap"
SCRAP_ELO_SOURCE_LABELS = {
    "hand": "Hand",
    "discard": "Discard",
    "noscrap": "No Scrap",
}

CARD_FEATURE_SIZE = 8 + len(factions) + len(CARD_TYPE_TO_INDEX) + len(ABILITY_FAMILIES) + 2
ZONE_FEATURE_SIZE = CARD_FEATURE_SIZE + 4
STATE_SCALAR_SIZE = 18
STATE_VECTOR_SIZE = ZONE_FEATURE_SIZE * 11 + STATE_SCALAR_SIZE
OPTION_VECTOR_SIZE = len(ACTION_TYPES) + len(factions) + len(ABILITY_FAMILIES) + 7 + CARD_FEATURE_SIZE


def _zero_vector(size: int) -> List[float]:
    return [0.0 for _ in range(size)]


def _zero_matrix(rows: int, cols: int) -> List[List[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def _copy_nested(value: Any) -> Any:
    return copy.deepcopy(value)


def _stable_softmax(logits: Sequence[float], temperature: float = 1.0) -> List[float]:
    if not logits:
        return []
    temp = max(temperature, 1e-3)
    scaled = [logit / temp for logit in logits]
    max_logit = max(scaled)
    exps = [math.exp(logit - max_logit) for logit in scaled]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(logits) for _ in logits]
    return [value / total for value in exps]


def _sample_index(probs: Sequence[float]) -> int:
    draw = random.random()
    cumulative = 0.0
    for idx, prob in enumerate(probs):
        cumulative += prob
        if draw <= cumulative:
            return idx
    return max(0, len(probs) - 1)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    average = _mean(values)
    variance = sum((value - average) * (value - average) for value in values) / len(values)
    return math.sqrt(max(variance, 0.0))


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _lerp(start: float, end: float, progress: float) -> float:
    return start + (end - start) * _clip(progress, 0.0, 1.0)


def _weighted_choice(choices: Sequence[Tuple[Any, float]]) -> Any:
    filtered = [(item, weight) for item, weight in choices if weight > 0]
    if not filtered:
        raise ValueError("At least one positive-weight choice is required.")
    total = sum(weight for _, weight in filtered)
    draw = random.random() * total
    cumulative = 0.0
    for item, weight in filtered:
        cumulative += weight
        if draw <= cumulative:
            return item
    return filtered[-1][0]


def _timestamp() -> float:
    return time.time()


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _format_timestamp(timestamp: Any) -> str:
    try:
        return datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


def _fork_origin_label(forked_from: Any) -> str:
    if not isinstance(forked_from, dict) or not forked_from:
        return "-"
    source_run = str(forked_from.get("run_name") or "").strip()
    source_checkpoint = str(
        forked_from.get("resolved_checkpoint")
        or forked_from.get("checkpoint")
        or ""
    ).strip()
    if source_run and source_checkpoint:
        return f"{source_run}/{source_checkpoint}"
    if source_run:
        return source_run
    return "-"


def _optional_timestamp(value: Any) -> Optional[float]:
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(timestamp) or timestamp <= 0.0:
        return None
    return timestamp


def _last_promotion_timestamp_from_state(state: Dict[str, Any]) -> Optional[float]:
    try:
        promotions = int(state.get("promotions", 0) or 0)
    except (TypeError, ValueError):
        promotions = 0
    if promotions <= 0:
        return None

    existing = _optional_timestamp(state.get("last_promotion_at"))
    if existing is not None:
        return existing

    checkpoints = list(state.get("checkpoints") or [])
    latest_checkpoint = str(state.get("latest_checkpoint") or "").strip()
    try:
        last_iteration = int(state.get("last_promotion_iteration", 0) or 0)
    except (TypeError, ValueError):
        last_iteration = 0

    candidates: List[float] = []
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            continue
        checkpoint_time = _optional_timestamp(checkpoint.get("created_at"))
        if checkpoint_time is None:
            continue
        checkpoint_name = str(checkpoint.get("name") or "").strip()
        try:
            checkpoint_iteration = int(checkpoint.get("iteration", 0) or 0)
        except (TypeError, ValueError):
            checkpoint_iteration = 0
        if checkpoint_name == latest_checkpoint:
            candidates.append(checkpoint_time)
        elif last_iteration > 0 and checkpoint_iteration == last_iteration:
            candidates.append(checkpoint_time)
        elif checkpoint.get("promoted_from"):
            candidates.append(checkpoint_time)

    if candidates:
        return max(candidates)
    return _optional_timestamp(state.get("updated_at"))


def _format_mebibytes(byte_count: int) -> str:
    return f"{float(max(0, int(byte_count))) / (1024.0 * 1024.0):.0f} MiB"


def _available_system_memory_bytes() -> Optional[int]:
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:
            return None
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return pages * page_size
    except (AttributeError, OSError, ValueError):
        return None


def _memory_safety_threshold_bytes(config: Any) -> int:
    try:
        threshold_mb = int(getattr(config, "min_available_memory_mb", DEFAULT_MIN_AVAILABLE_MEMORY_MB))
    except (TypeError, ValueError):
        threshold_mb = DEFAULT_MIN_AVAILABLE_MEMORY_MB
    return max(0, threshold_mb) * 1024 * 1024


def _check_memory_safety(config: Any, context: str) -> None:
    threshold = _memory_safety_threshold_bytes(config)
    if threshold <= 0:
        return
    available = _available_system_memory_bytes()
    if available is None or available >= threshold:
        return
    raise MemoryError(
        f"Available system memory dropped below the safety threshold while {context}: "
        f"{_format_mebibytes(available)} free, threshold {_format_mebibytes(threshold)}. "
        "Training stopped before putting more pressure on the machine."
    )


def default_simulation_workers(cpu_count: Optional[int] = None) -> int:
    logical_cpus = max(1, int(cpu_count or os.cpu_count() or 1))
    if logical_cpus <= 2:
        return 1
    return max(1, logical_cpus - SIMULATION_WORKER_CPU_RESERVE)


_SIMULATION_WORKER_MANAGER_LOCK = threading.RLock()
_ACTIVE_SIMULATION_RUNS: Dict[str, Dict[str, Any]] = {}


def _active_simulation_run_names() -> List[str]:
    with _SIMULATION_WORKER_MANAGER_LOCK:
        return sorted(_ACTIVE_SIMULATION_RUNS.keys())


def _active_simulation_run_snapshot() -> Dict[str, Dict[str, Any]]:
    with _SIMULATION_WORKER_MANAGER_LOCK:
        return {key: dict(value) for key, value in _ACTIVE_SIMULATION_RUNS.items()}


def _register_active_simulation_run(
    run_name: str,
    configured_workers: Any = SIMULATION_WORKERS_AUTO,
) -> str:
    key = str(run_name or LATEST_RUN_NAME)
    with _SIMULATION_WORKER_MANAGER_LOCK:
        _ACTIVE_SIMULATION_RUNS[key] = {
            "started_at": _timestamp(),
            "simulation_workers": normalize_simulation_workers(configured_workers),
        }
    return key


def _unregister_active_simulation_run(key: str) -> None:
    with _SIMULATION_WORKER_MANAGER_LOCK:
        _ACTIVE_SIMULATION_RUNS.pop(str(key), None)


def active_simulation_runs() -> List[str]:
    return _active_simulation_run_names()


def _auto_simulation_workers_for_active_run(
    allocation_key: Optional[str] = None,
    cpu_count: Optional[int] = None,
) -> int:
    budget = default_simulation_workers(cpu_count)
    active_runs = _active_simulation_run_snapshot()
    key = str(allocation_key) if allocation_key else ""
    if key and key not in active_runs:
        active_runs[key] = {
            "started_at": _timestamp(),
            "simulation_workers": SIMULATION_WORKERS_AUTO,
        }
    if not active_runs:
        return budget

    explicit_workers = sum(
        max(0, int(item.get("simulation_workers", SIMULATION_WORKERS_AUTO)))
        for item in active_runs.values()
        if int(item.get("simulation_workers", SIMULATION_WORKERS_AUTO)) > 0
    )
    auto_runs = sorted(
        active_key
        for active_key, item in active_runs.items()
        if int(item.get("simulation_workers", SIMULATION_WORKERS_AUTO)) == SIMULATION_WORKERS_AUTO
    )
    remaining_budget = max(1, budget - explicit_workers)
    if not auto_runs:
        return remaining_budget

    active_count = len(auto_runs)
    base = remaining_budget // active_count
    extra = remaining_budget % active_count
    if not key:
        return max(1, base)
    try:
        index = auto_runs.index(key)
    except ValueError:
        return max(1, base)
    return max(1, base + (1 if index < extra else 0))


def normalize_simulation_workers(value: Any) -> int:
    try:
        workers = int(value)
    except (TypeError, ValueError):
        raise ValueError("simulation_workers must be an integer. Use 0 for auto.")
    if workers < 0:
        raise ValueError("simulation_workers must be 0 for auto or a positive integer.")
    return workers


def resolve_simulation_workers(
    value: Any = SIMULATION_WORKERS_AUTO,
    work_items: int = 0,
    allocation_key: Optional[str] = None,
) -> int:
    requested = normalize_simulation_workers(value)
    workers = (
        _auto_simulation_workers_for_active_run(allocation_key)
        if requested == SIMULATION_WORKERS_AUTO
        else requested
    )
    if work_items > 0:
        workers = min(workers, max(1, int(work_items)))
    return max(1, workers)


def _simulation_chunk_sizes(total_items: int, worker_count: int) -> List[int]:
    total = max(0, int(total_items))
    if total <= 0:
        return []
    workers = max(1, min(int(worker_count), total))
    target_tasks = min(total, max(workers, workers * SIMULATION_TASKS_PER_WORKER))
    chunk_size = max(1, math.ceil(total / max(1, target_tasks)))
    sizes: List[int] = []
    remaining = total
    while remaining > 0:
        size = min(chunk_size, remaining)
        sizes.append(size)
        remaining -= size
    return sizes


def _max_in_flight_simulation_tasks(worker_count: int, task_count: Optional[int] = None) -> int:
    workers = max(1, int(worker_count))
    limit = max(workers, workers * SIMULATION_IN_FLIGHT_TASKS_PER_WORKER)
    if task_count is not None:
        limit = min(limit, max(1, int(task_count)))
    return max(1, limit)


def _random_seed() -> int:
    return random.randrange(1, 2**63 - 1)


def _safe_slug(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "item"
    cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in raw)
    return cleaned.strip("_") or "item"


def _directml_available() -> bool:
    return torch is not None and torch_directml is not None


def _select_torch_device(preference: str = DEVICE_AUTO) -> Tuple[Any, str]:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for starrealms_selfplay.py. "
            "Run this module from the project's training venv where torch is installed."
        ) from TORCH_IMPORT_ERROR

    normalized = str(preference or DEVICE_AUTO).strip().lower()
    if normalized == DEVICE_DIRECTML:
        if torch_directml is None:
            raise RuntimeError(
                "DirectML was requested, but torch_directml is not installed. "
                "Use Python 3.12 in your training venv and install torch-directml."
            ) from TORCH_DIRECTML_IMPORT_ERROR
        return torch_directml.device(), DEVICE_DIRECTML

    if normalized == DEVICE_CPU:
        return torch.device("cpu"), DEVICE_CPU

    if normalized == DEVICE_CUDA and hasattr(torch, "cuda") and torch.cuda.is_available():
        return torch.device("cuda"), DEVICE_CUDA

    if normalized == DEVICE_MPS and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), DEVICE_MPS

    if normalized == DEVICE_AUTO:
        if torch_directml is not None:
            return torch_directml.device(), DEVICE_DIRECTML
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return torch.device("cuda"), DEVICE_CUDA
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), DEVICE_MPS
        return torch.device("cpu"), DEVICE_CPU

    return torch.device("cpu"), DEVICE_CPU


def _device_from_backend(backend: str) -> Any:
    normalized = str(backend or DEVICE_CPU).strip().lower()
    if normalized == DEVICE_DIRECTML:
        if torch_directml is None:
            raise RuntimeError(
                "DirectML was requested, but torch_directml is not installed. "
                "Use Python 3.12 in your training venv and install torch-directml."
            ) from TORCH_DIRECTML_IMPORT_ERROR
        return torch_directml.device()
    if normalized == DEVICE_CPU:
        return torch.device("cpu")
    return _select_torch_device(normalized)[0]


def _synthetic_float(index: int) -> float:
    return ((index % 23) - 11) / 11.0


def _synthetic_state_vectors(batch_size: int, state_size: int) -> List[List[float]]:
    return [
        [_synthetic_float(row_index * 17 + feature_index * 3) for feature_index in range(state_size)]
        for row_index in range(batch_size)
    ]


def _synthetic_option_batches(
    batch_size: int,
    option_size: int,
    option_count: int,
) -> List[List[List[float]]]:
    return [
        [
            [_synthetic_float(row_index * 29 + option_index * 11 + feature_index) for feature_index in range(option_size)]
            for option_index in range(option_count)
        ]
        for row_index in range(batch_size)
    ]


def _build_policy_module(
    state_size: int,
    option_size: int,
    hidden_size: int,
    architecture: str,
) -> "nn.Module":
    if nn is None:
        raise RuntimeError(
            "PyTorch is required for starrealms_selfplay.py. "
            "Run this module from the project's training venv where torch is installed."
        ) from TORCH_IMPORT_ERROR
    if architecture == LEGACY_MODEL_ARCHITECTURE:
        return _LegacyTorchPolicyModule(state_size, option_size, hidden_size)
    if architecture == CURRENT_MODEL_ARCHITECTURE:
        return _DeepTorchPolicyModule(state_size, option_size, hidden_size)
    raise ValueError(f"Unknown policy architecture: {architecture}")


_DIRECTML_POLICY_BENCHMARK_CACHE: Dict[Tuple[int, int, int, str], Dict[str, Any]] = {}


def _benchmark_directml_policy_shape(
    state_size: int,
    option_size: int,
    hidden_size: int,
    architecture: str,
) -> Dict[str, Any]:
    if torch is None or nn is None or torch_directml is None:
        raise RuntimeError("DirectML benchmarking requires torch and torch_directml.")

    cache_key = (int(state_size), int(option_size), int(hidden_size), str(architecture))
    cached = _DIRECTML_POLICY_BENCHMARK_CACHE.get(cache_key)
    if cached is not None:
        return _copy_nested(cached)

    inference_option_count = 12
    training_sample_count = DIRECTML_BENCHMARK_TRAINING_SAMPLE_COUNT
    training_minibatch_size = min(DIRECTML_BENCHMARK_MINIBATCH_SIZE, training_sample_count)
    training_option_count = 12

    inference_state_vecs = _synthetic_state_vectors(1, state_size)
    inference_option_batches = _synthetic_option_batches(1, option_size, inference_option_count)
    training_state_vecs = _synthetic_state_vectors(training_sample_count, state_size)
    training_option_batches = _synthetic_option_batches(training_sample_count, option_size, training_option_count)
    training_actions = [sample_index % training_option_count for sample_index in range(training_sample_count)]
    training_old_log_probs = [_synthetic_float(sample_index) * 0.2 for sample_index in range(training_sample_count)]
    training_returns = [_synthetic_float(sample_index * 5) * 0.4 for sample_index in range(training_sample_count)]
    training_advantages = [_synthetic_float(sample_index * 7) * 0.3 for sample_index in range(training_sample_count)]

    timings: Dict[str, Dict[str, float]] = {}
    for backend in (DEVICE_CPU, DEVICE_DIRECTML):
        device = _device_from_backend(backend)
        model = _build_policy_module(state_size, option_size, hidden_size, architecture).to(device)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The operator 'aten::lerp\\.Scalar_out' is not currently supported on the DML backend.*",
                category=UserWarning,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)

            inference_state_tensor = torch.tensor(inference_state_vecs, dtype=torch.float32, device=device)
            inference_option_tensor = torch.tensor(inference_option_batches, dtype=torch.float32, device=device)
            model.eval()
            with torch.no_grad():
                for _ in range(2):
                    model(inference_state_tensor, inference_option_tensor)
                started_at = time.perf_counter()
                for _ in range(DIRECTML_BENCHMARK_INFERENCE_REPEATS):
                    logits, values = model(inference_state_tensor, inference_option_tensor)
                inference_anchor = float(logits.reshape(-1)[0].item()) + float(values.reshape(-1)[0].item())
                inference_seconds = (time.perf_counter() - started_at) / DIRECTML_BENCHMARK_INFERENCE_REPEATS

            state_tensor = torch.tensor(training_state_vecs, dtype=torch.float32, device=device)
            option_tensor = torch.tensor(training_option_batches, dtype=torch.float32, device=device)
            action_tensor = torch.tensor(training_actions, dtype=torch.int64, device=device)
            old_log_prob_tensor = torch.tensor(training_old_log_probs, dtype=torch.float32, device=device)
            return_tensor = torch.tensor(training_returns, dtype=torch.float32, device=device)
            advantage_tensor = torch.tensor(training_advantages, dtype=torch.float32, device=device)

            model.train()
            started_at = time.perf_counter()
            training_anchor = 0.0
            for _ in range(2):
                for batch_start in range(0, training_sample_count, training_minibatch_size):
                    batch_end = batch_start + training_minibatch_size
                    logits, value = model(
                        state_tensor[batch_start:batch_end],
                        option_tensor[batch_start:batch_end],
                    )
                    log_probs = torch.log_softmax(logits, dim=1)
                    batch_actions = action_tensor[batch_start:batch_end]
                    chosen_log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                    batch_old_log_probs = old_log_prob_tensor[batch_start:batch_end]
                    batch_returns = return_tensor[batch_start:batch_end]
                    batch_advantages = advantage_tensor[batch_start:batch_end]
                    ratio = torch.exp(chosen_log_probs - batch_old_log_probs)
                    clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
                    policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
                    value_loss = 0.5 * torch.square(value - batch_returns).mean()
                    loss = policy_loss + 0.5 * value_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    training_anchor += float(loss.item())
            training_seconds = time.perf_counter() - started_at

        timings[backend] = {
            "inference_seconds": inference_seconds + (inference_anchor * 0.0),
            "training_seconds": training_seconds + (training_anchor * 0.0),
        }

    cpu_inference = float(timings[DEVICE_CPU]["inference_seconds"])
    directml_inference = float(timings[DEVICE_DIRECTML]["inference_seconds"])
    cpu_training = float(timings[DEVICE_CPU]["training_seconds"])
    directml_training = float(timings[DEVICE_DIRECTML]["training_seconds"])
    directml_wins = (
        directml_inference <= cpu_inference * DIRECTML_BENCHMARK_MARGIN
        and directml_training <= cpu_training * DIRECTML_BENCHMARK_MARGIN
    )
    preferred_backend = DEVICE_DIRECTML if directml_wins else DEVICE_CPU
    benchmark = {
        "preferred_backend": preferred_backend,
        "cpu_inference_ms": round(cpu_inference * 1000.0, 3),
        "directml_inference_ms": round(directml_inference * 1000.0, 3),
        "cpu_training_ms": round(cpu_training * 1000.0, 3),
        "directml_training_ms": round(directml_training * 1000.0, 3),
        "sample_count": int(training_sample_count),
        "minibatch_size": int(training_minibatch_size),
        "reason": (
            "DirectML won the synthetic policy-shape benchmark."
            if directml_wins
            else "CPU won the synthetic policy-shape benchmark, so DirectML would likely slow this run down."
        ),
    }
    _DIRECTML_POLICY_BENCHMARK_CACHE[cache_key] = _copy_nested(benchmark)
    return benchmark


def _resolve_policy_device(
    preference: str,
    state_size: int,
    option_size: int,
    hidden_size: int,
    architecture: str,
) -> Dict[str, Any]:
    normalized = str(preference or DEVICE_AUTO).strip().lower()
    if normalized == DEVICE_DIRECTML:
        if torch_directml is None:
            raise RuntimeError(
                "DirectML was requested, but torch_directml is not installed. "
                "Use Python 3.12 in your training venv and install torch-directml."
            ) from TORCH_DIRECTML_IMPORT_ERROR
        benchmark = _benchmark_directml_policy_shape(
            state_size=state_size,
            option_size=option_size,
            hidden_size=hidden_size,
            architecture=architecture,
        )
        backend = str(benchmark.get("preferred_backend", DEVICE_CPU))
        device = _device_from_backend(backend)
        return {
            "requested_backend": DEVICE_DIRECTML,
            "backend": backend,
            "device": device,
            "reason": str(benchmark.get("reason", "")),
            "benchmark": benchmark,
        }

    device, backend = _select_torch_device(normalized)
    return {
        "requested_backend": normalized,
        "backend": backend,
        "device": device,
        "reason": "",
        "benchmark": None,
    }


def runtime_environment() -> Dict[str, Any]:
    directml_error = None
    if TORCH_DIRECTML_IMPORT_ERROR is not None:
        directml_error = f"{type(TORCH_DIRECTML_IMPORT_ERROR).__name__}: {TORCH_DIRECTML_IMPORT_ERROR}"
    training_device, training_backend = _select_torch_device(DEVICE_AUTO) if torch is not None else (None, "unavailable")
    cpu_count = os.cpu_count() or 1
    return {
        "python_version": ".".join(str(part) for part in os.sys.version_info[:3]),
        "interpreter": os.sys.executable,
        "cpu_count": int(cpu_count),
        "default_simulation_workers": default_simulation_workers(cpu_count),
        "torch_available": torch is not None,
        "torch_version": None if torch is None else getattr(torch, "__version__", None),
        "torch_directml_available": _directml_available(),
        "torch_directml_error": directml_error,
        "auto_device": training_backend,
        "auto_device_repr": None if training_device is None else str(training_device),
    }


def resolve_device_backend(preference: str = DEVICE_AUTO) -> str:
    _, backend = _select_torch_device(preference)
    return backend


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2)
    last_error: Optional[Exception] = None
    for attempt in range(8):
        temp_path = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.{attempt}.tmp")
        try:
            temp_path.write_text(content, encoding="utf-8")
            os.replace(temp_path, path)
            _invalidate_json_cache(path)
            return
        except PermissionError as exc:
            last_error = exc
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            time.sleep(0.05 * (attempt + 1))
        except Exception:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            raise
    if last_error is not None:
        raise PermissionError(
            f"Permission error while saving {path}. "
            "Another process may still have the file open; please wait a moment and try again."
        ) from last_error


def _is_card_tuple(value: Any) -> bool:
    return isinstance(value, tuple) and len(value) >= 14 and isinstance(value[0], str)


def _is_play_card(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 5 and _is_card_tuple(value[0])


def _extract_card(value: Any) -> Optional[Tuple[Any, ...]]:
    if _is_card_tuple(value):
        if value[0] == "none":
            return None
        return value
    if _is_play_card(value):
        return value[0]
    return None


def _iter_zone_cards(zone: Any) -> List[Any]:
    if zone is None:
        return []
    if isinstance(zone, dict):
        cards: List[Any] = []
        for items in zone.values():
            cards.extend(items)
        return cards
    if isinstance(zone, list):
        return zone
    return []


@lru_cache(maxsize=None)
def _cached_card_feature_updates(card: Tuple[Any, ...]) -> Tuple[Tuple[int, float], ...]:
    updates: List[Tuple[int, float]] = [
        (0, 1.0),
        (1, card[1] / 8.0),
        (2, card[2] / 8.0),
        (3, card[3] / 8.0),
        (4, card[4] / 8.0),
        (5, card[12] / 8.0),
        (6, card[9] / 5.0),
        (7, card[11] / 5.0),
    ]

    faction_offset = 8
    faction_index = FACTION_TO_INDEX.get(card[5])
    if faction_index is not None:
        updates.append((faction_offset + faction_index, 1.0))

    type_offset = faction_offset + len(factions)
    card_type_index = CARD_TYPE_TO_INDEX.get(card[6])
    if card_type_index is not None:
        updates.append((type_offset + card_type_index, 1.0))

    ability_offset = type_offset + len(CARD_TYPE_TO_INDEX)
    for raw_ability in (card[7], card[8], card[10]):
        ability_index = ABILITY_TO_INDEX.get(_normalize_ability(raw_ability))
        if ability_index is not None:
            updates.append((ability_offset + ability_index, 1.0))

    option_offset = ability_offset + len(ABILITY_FAMILIES)
    if isinstance(card[7], str) and card[7].startswith("-"):
        updates.append((option_offset, 1.0))
    if isinstance(card[8], str) and card[8].startswith("-"):
        updates.append((option_offset + 1, 1.0))
    return tuple(updates)


@lru_cache(maxsize=None)
def _cached_card_feature_vector(card: Tuple[Any, ...]) -> Tuple[float, ...]:
    features = _zero_vector(CARD_FEATURE_SIZE)
    for index, value in _cached_card_feature_updates(card):
        features[index] += value
    return tuple(features)


def _add_single_card_features(features: List[float], card: Tuple[Any, ...], scale: float) -> None:
    if card is None:
        return
    for index, value in _cached_card_feature_updates(card):
        features[index] += value * scale


def _single_card_features(card: Optional[Tuple[Any, ...]]) -> List[float]:
    if card is None:
        return _zero_vector(CARD_FEATURE_SIZE)
    return list(_cached_card_feature_vector(card))


def _zone_features_and_count(zone: Any) -> Tuple[List[float], int]:
    features = _zero_vector(ZONE_FEATURE_SIZE)
    count = 0
    scale = 1.0 / CARD_ZONE_SCALE

    if zone is None:
        return features, count

    if isinstance(zone, dict):
        zone_groups = zone.values()
    elif isinstance(zone, list):
        zone_groups = (zone,)
    else:
        return features, count

    for group in zone_groups:
        for item in group:
            count += 1
            if isinstance(item, tuple) and len(item) >= 14 and isinstance(item[0], str):
                if item[0] != "none":
                    _add_single_card_features(features, item, scale)
                continue
            if isinstance(item, list) and len(item) >= 5:
                card = item[0]
                if isinstance(card, tuple) and len(card) >= 14 and isinstance(card[0], str):
                    if card[0] != "none":
                        _add_single_card_features(features, card, scale)
                    option_state = item[2]
                    if option_state not in (None, "used"):
                        features[CARD_FEATURE_SIZE] += scale
                    if option_state == "used":
                        features[CARD_FEATURE_SIZE + 1] += scale
                    if item[1]:
                        features[CARD_FEATURE_SIZE + 2] += scale
                    if item[4]:
                        features[CARD_FEATURE_SIZE + 3] += scale
    return features, count


def _zone_features(zone: Any) -> List[float]:
    return _zone_features_and_count(zone)[0]


def state_to_vector(state: Dict[str, Any], legal_option_count: int = 0) -> List[float]:
    zone_values = {
        "hand": state.get("hand"),
        "discardPile": state.get("discardPile"),
        "scrambleDeck": state.get("scrambleDeck"),
        "topCards": state.get("topCards"),
        "cardsInPlay": state.get("cardsInPlay"),
        "tradeRow": state.get("tradeRow"),
        "opponentDiscardPile": state.get("opponentDiscardPile"),
        "opponentScrambleDeckAndHand": state.get("opponentScrambleDeckAndHand"),
        "opponentTopCards": state.get("opponentTopCards"),
        "opponentHandCards": state.get("opponentHandCards"),
        "opponentCardsInPlay": state.get("opponentCardsInPlay"),
    }
    zone_order = [
        "hand",
        "discardPile",
        "scrambleDeck",
        "topCards",
        "cardsInPlay",
        "tradeRow",
        "opponentDiscardPile",
        "opponentScrambleDeckAndHand",
        "opponentTopCards",
        "opponentHandCards",
        "opponentCardsInPlay",
    ]
    zone_stats = {name: _zone_features_and_count(zone_values[name]) for name in zone_order}
    features: List[float] = []
    for zone_name in zone_order:
        features.extend(zone_stats[zone_name][0])

    hand_count = zone_stats["hand"][1]
    discard_count = zone_stats["discardPile"][1]
    deck_count = zone_stats["scrambleDeck"][1] + zone_stats["topCards"][1]
    in_play_count = zone_stats["cardsInPlay"][1]
    opponent_known_hand_count = zone_stats["opponentHandCards"][1]
    opponent_unknown_count = zone_stats["opponentScrambleDeckAndHand"][1] + zone_stats["opponentTopCards"][1]
    opponent_discard_count = zone_stats["opponentDiscardPile"][1]
    opponent_in_play_count = zone_stats["opponentCardsInPlay"][1]

    scalars = [
        state.get("authority", 0) / 50.0,
        state.get("opponentAuthority", 0) / 50.0,
        (state.get("authority", 0) - state.get("opponentAuthority", 0)) / 50.0,
        state.get("attack", 0) / 20.0,
        state.get("trade", 0) / 20.0,
        state.get("mustDiscard", 0) / 5.0,
        state.get("opponentMustDiscard", 0) / 5.0,
        1.0 if state.get("nextShipTop") else 0.0,
        state.get("blobPlayCount", 0) / 10.0,
        hand_count / 20.0,
        discard_count / 30.0,
        deck_count / 30.0,
        in_play_count / 15.0,
        opponent_known_hand_count / 10.0,
        opponent_unknown_count / 30.0,
    ]
    scalars.extend(
        [
            opponent_discard_count / 30.0,
            opponent_in_play_count / 15.0,
            legal_option_count / 30.0,
        ]
    )
    return features + scalars


def _option_action_name(option: Sequence[Any]) -> str:
    if not option:
        return "other"
    raw_name = option[0]
    if not isinstance(raw_name, str):
        return "other"
    if raw_name in ACTION_TO_INDEX:
        return raw_name
    return "other"


def _resolve_state_referenced_card(option: Sequence[Any], state: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    action = _option_action_name(option)
    if action in {"abilityOption", "scrapFromPlay"} and len(option) >= 3:
        faction = option[1]
        index = option[2]
        if isinstance(faction, str) and isinstance(index, int):
            cards_in_play = state.get("cardsInPlay", {})
            if faction in cards_in_play and 0 <= index < len(cards_in_play[faction]):
                return _extract_card(cards_in_play[faction][index])
    if action in {"attack", "killbase"} and len(option) >= 3:
        faction = option[1]
        index = option[2]
        if isinstance(faction, str) and isinstance(index, int):
            cards_in_play = state.get("opponentCardsInPlay", {})
            if faction in cards_in_play and 0 <= index < len(cards_in_play[faction]):
                return _extract_card(cards_in_play[faction][index])
    return None


def _resolve_option_card(option: Sequence[Any], state: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    for item in option[1:]:
        card = _extract_card(item)
        if card is not None:
            return card
    return _resolve_state_referenced_card(option, state)


def _resolve_option_ability(option: Sequence[Any]) -> str:
    action = _option_action_name(option)
    if action in {"gainattack", "draw", "trade", "authority"}:
        return action
    for item in option[1:]:
        if isinstance(item, str):
            normalized = _normalize_ability(item)
            if normalized in ABILITY_TO_INDEX and normalized != "none":
                return normalized
    return "none"


def option_to_vector(option: Sequence[Any], state: Dict[str, Any]) -> List[float]:
    features = _zero_vector(OPTION_VECTOR_SIZE)
    offset = 0

    action = _option_action_name(option)
    features[offset + ACTION_TO_INDEX[action]] = 1.0
    offset += len(ACTION_TYPES)

    for item in option[1:]:
        if isinstance(item, str) and item in FACTION_TO_INDEX:
            features[offset + FACTION_TO_INDEX[item]] = 1.0
            break
    offset += len(factions)

    ability_name = _resolve_option_ability(option)
    if ability_name in ABILITY_TO_INDEX:
        features[offset + ABILITY_TO_INDEX[ability_name]] = 1.0
    offset += len(ABILITY_FAMILIES)

    numeric_values: List[float] = []
    for item in option[1:]:
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            numeric_values.append(_clip(float(item) / NUMERIC_OPTION_SCALE, -3.0, 3.0))
    features[offset] = len(option) / 8.0
    features[offset + 1] = 1.0 if action in NOOP_ACTIONS else 0.0
    for idx in range(5):
        if idx < len(numeric_values):
            features[offset + 2 + idx] = numeric_values[idx]
    offset += 7

    resolved_card = _resolve_option_card(option, state)
    if resolved_card is not None:
        features[offset:] = _cached_card_feature_vector(resolved_card)
    return features


@dataclass
class TrainingConfig:
    hidden_size: int = DEEP_MODEL_HIDDEN_SIZE
    model_architecture: str = CURRENT_MODEL_ARCHITECTURE
    device_preference: str = DEVICE_AUTO
    learning_rate: float = 0.0019
    min_learning_rate: float = 0.0004
    ppo_epochs: int = 2
    ppo_minibatch_size: int = 64
    ppo_clip: float = 0.2
    min_ppo_clip: float = 0.1
    value_coef: float = 0.5
    grad_clip: float = 1.0
    epsilon_random: float = 0.07
    min_epsilon_random: float = 0.015
    train_temperature: float = 0.9
    min_train_temperature: float = 0.9
    eval_temperature: float = 0.35
    checkpoint_interval: int = 1
    training_matches_per_iteration: int = 5
    training_games_per_match: int = 16
    training_decisions_per_game: str = TRAINING_DECISIONS_PER_GAME_ALL
    promotion_games: int = 24
    promotion_score_threshold: float = 0.6
    simulation_workers: int = SIMULATION_WORKERS_AUTO
    candidate_patience: int = 8
    candidate_reset_threshold: float = 0.35
    league_recent_window: int = 10
    league_champion_weight: float = 0.45
    league_best_weight: float = 0.2
    league_recent_weight: float = 0.25
    league_historical_weight: float = 0.1
    opponent_normal_weight: float = 0.6
    opponent_learnable_weight: float = 0.3
    opponent_chaotic_weight: float = 0.1
    opponent_learnable_temperature: float = 1.4
    opponent_chaotic_temperature: float = 1.9
    opponent_learnable_epsilon_random: float = 0.1
    opponent_chaotic_epsilon_random: float = 0.14
    anneal_steps: int = 3000
    plateau_start_iterations_without_promotion: int = 12
    plateau_full_iterations_without_promotion: int = 36
    plateau_learning_rate_boost: float = 1.7
    plateau_epsilon_boost: float = 2.4
    plateau_temperature_boost: float = 1.0
    plateau_max_learning_rate: float = 0.0033
    plateau_max_epsilon_random: float = 0.16
    plateau_max_train_temperature: float = 0.9
    elo_k: float = 24.0
    min_available_memory_mb: int = DEFAULT_MIN_AVAILABLE_MEMORY_MB
    max_turns_per_game: int = 400
    max_actions_per_turn: int = 200
    max_iterations: Optional[int] = None

    def merged(self, overrides: Optional[Dict[str, Any]] = None) -> "TrainingConfig":
        if not overrides:
            return self
        data = asdict(self)
        data.update({key: value for key, value in overrides.items() if key in data})
        return TrainingConfig(**data)


if nn is not None and torch is not None:
    class _LegacyTorchPolicyModule(nn.Module):
        def __init__(self, state_size: int, option_size: int, hidden_size: int) -> None:
            super().__init__()
            self.state_linear = nn.Linear(state_size, hidden_size)
            self.option_linear = nn.Linear(option_size, hidden_size)
            self.joint_bias = nn.Parameter(torch.zeros(hidden_size))
            self.policy_head = nn.Linear(hidden_size, 1)
            self.value_head = nn.Linear(hidden_size, 1)

        def _state_hidden(self, state_tensor: "torch.Tensor") -> "torch.Tensor":
            return torch.tanh(self.state_linear(state_tensor))

        def _joint_hidden(self, state_hidden: "torch.Tensor", option_tensor: "torch.Tensor") -> "torch.Tensor":
            option_hidden = torch.tanh(self.option_linear(option_tensor))
            expanded_state_hidden = state_hidden
            while expanded_state_hidden.dim() < option_hidden.dim():
                expanded_state_hidden = expanded_state_hidden.unsqueeze(-2)
            bias_shape = [1] * (option_hidden.dim() - 1) + [-1]
            return torch.tanh(expanded_state_hidden + option_hidden + self.joint_bias.view(*bias_shape))

        def policy_only(self, state_tensor: "torch.Tensor", option_tensor: "torch.Tensor") -> "torch.Tensor":
            state_hidden = self._state_hidden(state_tensor)
            joint_hidden = self._joint_hidden(state_hidden, option_tensor)
            return self.policy_head(joint_hidden).squeeze(-1)

        def forward(self, state_tensor: "torch.Tensor", option_tensor: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            state_hidden = self._state_hidden(state_tensor)
            joint_hidden = self._joint_hidden(state_hidden, option_tensor)
            logits = self.policy_head(joint_hidden).squeeze(-1)
            value = self.value_head(state_hidden).squeeze(-1)
            return logits, value


    class _DeepTorchPolicyModule(nn.Module):
        def __init__(self, state_size: int, option_size: int, hidden_size: int) -> None:
            super().__init__()
            self.state_linear = nn.Linear(state_size, hidden_size)
            self.option_linear = nn.Linear(option_size, hidden_size)
            self.joint_bias = nn.Parameter(torch.zeros(hidden_size))

            self.state_refine_1 = nn.Linear(hidden_size, hidden_size)
            self.state_refine_2 = nn.Linear(hidden_size, hidden_size)
            self.option_refine_1 = nn.Linear(hidden_size, hidden_size)
            self.option_refine_2 = nn.Linear(hidden_size, hidden_size)
            self.joint_mix_1 = nn.Linear(hidden_size * 4, hidden_size)
            self.joint_mix_2 = nn.Linear(hidden_size, hidden_size)
            self.value_refine = nn.Linear(hidden_size, hidden_size)

            self.policy_head = nn.Linear(hidden_size, 1)
            self.value_head = nn.Linear(hidden_size, 1)

        def _state_hidden(self, state_tensor: "torch.Tensor") -> "torch.Tensor":
            state_hidden = torch.tanh(self.state_linear(state_tensor))
            state_hidden = state_hidden + 0.35 * torch.tanh(self.state_refine_1(state_hidden))
            state_hidden = state_hidden + 0.35 * torch.tanh(self.state_refine_2(state_hidden))
            return state_hidden

        def _joint_hidden(self, state_hidden: "torch.Tensor", option_tensor: "torch.Tensor") -> "torch.Tensor":
            option_hidden = torch.tanh(self.option_linear(option_tensor))
            option_hidden = option_hidden + 0.35 * torch.tanh(self.option_refine_1(option_hidden))
            option_hidden = option_hidden + 0.35 * torch.tanh(self.option_refine_2(option_hidden))

            expanded_state_hidden = state_hidden
            while expanded_state_hidden.dim() < option_hidden.dim():
                expanded_state_hidden = expanded_state_hidden.unsqueeze(-2)
            state_expanded = expanded_state_hidden.expand_as(option_hidden)
            bias_shape = [1] * (option_hidden.dim() - 1) + [-1]
            legacy_joint = torch.tanh(state_expanded + option_hidden + self.joint_bias.view(*bias_shape))
            interaction = state_expanded * option_hidden
            joint_input = torch.cat([legacy_joint, state_expanded, option_hidden, interaction], dim=-1)
            joint_hidden = legacy_joint + 0.35 * torch.tanh(self.joint_mix_1(joint_input))
            joint_hidden = joint_hidden + 0.35 * torch.tanh(self.joint_mix_2(joint_hidden))
            return joint_hidden

        def policy_only(self, state_tensor: "torch.Tensor", option_tensor: "torch.Tensor") -> "torch.Tensor":
            state_hidden = self._state_hidden(state_tensor)
            joint_hidden = self._joint_hidden(state_hidden, option_tensor)
            return self.policy_head(joint_hidden).squeeze(-1)

        def forward(self, state_tensor: "torch.Tensor", option_tensor: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            state_hidden = self._state_hidden(state_tensor)
            joint_hidden = self._joint_hidden(state_hidden, option_tensor)
            value_hidden = state_hidden + 0.35 * torch.tanh(self.value_refine(state_hidden))
            logits = self.policy_head(joint_hidden).squeeze(-1)
            value = self.value_head(value_hidden).squeeze(-1)
            return logits, value
else:
    class _LegacyTorchPolicyModule:  # pragma: no cover - only used when torch is unavailable
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for starrealms_selfplay.py. "
                "Run this module from the project's training venv where torch is installed."
            ) from TORCH_IMPORT_ERROR


    class _DeepTorchPolicyModule:  # pragma: no cover - only used when torch is unavailable
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for starrealms_selfplay.py. "
                "Run this module from the project's training venv where torch is installed."
            ) from TORCH_IMPORT_ERROR


class PolicyNetwork:
    def __init__(
        self,
        state_size: int = STATE_VECTOR_SIZE,
        option_size: int = OPTION_VECTOR_SIZE,
        hidden_size: int = 96,
        architecture: str = CURRENT_MODEL_ARCHITECTURE,
        device_preference: str = DEVICE_AUTO,
        init_seed: Optional[int] = None,
    ) -> None:
        if torch is None or nn is None:
            raise RuntimeError(
                "PyTorch is required for starrealms_selfplay.py. "
                "Run this module from the project's training venv where torch is installed."
            ) from TORCH_IMPORT_ERROR
        if init_seed is not None:
            torch.manual_seed(init_seed)
        self.state_size = state_size
        self.option_size = option_size
        self.hidden_size = hidden_size
        self.architecture = architecture
        self.device_preference = str(device_preference or DEVICE_AUTO).strip().lower()
        device_resolution = _resolve_policy_device(
            self.device_preference,
            state_size=state_size,
            option_size=option_size,
            hidden_size=hidden_size,
            architecture=architecture,
        )
        self.device_requested_backend = str(device_resolution.get("requested_backend", self.device_preference))
        self.device_backend = str(device_resolution.get("backend", DEVICE_CPU))
        self.device = device_resolution.get("device")
        self.device_reason = str(device_resolution.get("reason", ""))
        self.device_benchmark = _copy_nested(device_resolution.get("benchmark"))
        self.model = _build_policy_module(state_size, option_size, hidden_size, architecture).to(self.device)
        self._reset_optimizer(0.0015)

    def _reset_optimizer(self, learning_rate: float) -> None:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(learning_rate))

    def device_summary(self) -> Dict[str, Any]:
        return {
            "requested_backend": self.device_requested_backend,
            "backend": self.device_backend,
            "repr": str(self.device),
            "reason": self.device_reason,
            "benchmark": _copy_nested(self.device_benchmark),
        }

    def clone(self) -> "PolicyNetwork":
        clone = PolicyNetwork(
            state_size=self.state_size,
            option_size=self.option_size,
            hidden_size=self.hidden_size,
            architecture=self.architecture,
            device_preference=self.device_preference,
        )
        clone.model.load_state_dict(self.model.state_dict())
        clone._reset_optimizer(self.optimizer.param_groups[0]["lr"])
        return clone

    def to_dict(self, include_optimizer: bool = True) -> Dict[str, Any]:
        state_dict = {
            name: tensor.detach().cpu().tolist()
            for name, tensor in self.model.state_dict().items()
        }
        payload = {
            "backend": "torch",
            "state_size": self.state_size,
            "option_size": self.option_size,
            "hidden_size": self.hidden_size,
            "architecture": self.architecture,
            "device_preference": self.device_preference,
            "state_dict": state_dict,
        }
        if include_optimizer:
            payload["optimizer"] = {
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PolicyNetwork":
        state_size = payload.get("state_size", STATE_VECTOR_SIZE)
        option_size = payload.get("option_size", OPTION_VECTOR_SIZE)
        architecture = payload.get("architecture")
        if architecture is None:
            state_dict = payload.get("state_dict") or {}
            if any(name.startswith("state_refine_") or name.startswith("joint_mix_") or name.startswith("value_refine") for name in state_dict.keys()):
                architecture = CURRENT_MODEL_ARCHITECTURE
            else:
                architecture = LEGACY_MODEL_ARCHITECTURE
        hidden_size = payload.get("hidden_size", 24 if architecture == LEGACY_MODEL_ARCHITECTURE else 96)
        device_preference = payload.get("device_preference", DEVICE_AUTO)
        model = cls(
            state_size=state_size,
            option_size=option_size,
            hidden_size=hidden_size,
            architecture=architecture,
            device_preference=device_preference,
        )

        if payload.get("backend") == "torch" or "state_dict" in payload:
            tensor_state = {
                name: torch.tensor(values, dtype=torch.float32, device=model.device)
                for name, values in payload["state_dict"].items()
            }
            model.model.load_state_dict(tensor_state)
            optimizer_state = payload.get("optimizer") or {}
            learning_rate = float(optimizer_state.get("learning_rate", 0.0015))
            model._reset_optimizer(learning_rate)
            return model

        params = payload["params"]
        model = cls(
            state_size=state_size,
            option_size=option_size,
            hidden_size=hidden_size,
            architecture=LEGACY_MODEL_ARCHITECTURE,
            device_preference=device_preference,
        )
        converted_state = {
            "state_linear.weight": torch.tensor(params["state_w"], dtype=torch.float32, device=model.device),
            "state_linear.bias": torch.tensor(params["state_b"], dtype=torch.float32, device=model.device),
            "option_linear.weight": torch.tensor(params["option_w"], dtype=torch.float32, device=model.device),
            "option_linear.bias": torch.tensor(params["option_b"], dtype=torch.float32, device=model.device),
            "joint_bias": torch.tensor(params["joint_b"], dtype=torch.float32, device=model.device),
            "policy_head.weight": torch.tensor([params["policy_w"]], dtype=torch.float32, device=model.device),
            "policy_head.bias": torch.tensor([params["policy_b"]], dtype=torch.float32, device=model.device),
            "value_head.weight": torch.tensor([params["value_w"]], dtype=torch.float32, device=model.device),
            "value_head.bias": torch.tensor([params["value_b"]], dtype=torch.float32, device=model.device),
        }
        model.model.load_state_dict(converted_state)
        return model

    def _state_tensor(self, state_vec: Sequence[float]) -> "torch.Tensor":
        return torch.tensor(state_vec, dtype=torch.float32, device=self.device)

    def _state_tensor_batch(self, state_vecs: Sequence[Sequence[float]]) -> "torch.Tensor":
        return torch.tensor(state_vecs, dtype=torch.float32, device=self.device)

    def _option_tensor(self, option_vecs: Sequence[Sequence[float]]) -> "torch.Tensor":
        return torch.tensor(option_vecs, dtype=torch.float32, device=self.device)

    def _padded_option_tensor(
        self,
        option_vec_batches: Sequence[Sequence[Sequence[float]]],
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        if not option_vec_batches:
            raise ValueError("At least one option batch is required.")

        batch_size = len(option_vec_batches)
        max_options = max(len(option_vecs) for option_vecs in option_vec_batches)
        if max_options <= 0:
            raise ValueError("Each state requires at least one legal option.")

        option_tensor = torch.zeros(
            (batch_size, max_options, self.option_size),
            dtype=torch.float32,
            device=self.device,
        )
        option_mask = torch.zeros((batch_size, max_options), dtype=torch.bool, device=self.device)

        for batch_index, option_vecs in enumerate(option_vec_batches):
            option_count = len(option_vecs)
            if option_count <= 0:
                raise ValueError("Each state requires at least one legal option.")
            option_tensor[batch_index, :option_count] = torch.tensor(
                option_vecs,
                dtype=torch.float32,
                device=self.device,
            )
            option_mask[batch_index, :option_count] = True

        return option_tensor, option_mask

    def _masked_logits(
        self,
        logits: "torch.Tensor",
        option_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        if option_mask is None:
            return logits
        return logits.masked_fill(~option_mask, -1e9)

    def _policy_logits(self, state_tensor: "torch.Tensor", option_tensor: "torch.Tensor") -> "torch.Tensor":
        return self.model.policy_only(state_tensor, option_tensor)

    def select_action(
        self,
        state_vec: Sequence[float],
        option_vecs: Sequence[Sequence[float]],
        deterministic: bool = False,
        temperature: float = 1.0,
        epsilon_random: float = 0.0,
        need_log_prob: bool = False,
        need_value: bool = False,
    ) -> Dict[str, Any]:
        if self.model.training:
            self.model.eval()
        with torch.no_grad():
            state_tensor = self._state_tensor(state_vec)
            option_tensor = self._option_tensor(option_vecs)
            result: Dict[str, Any] = {}

            if deterministic and not need_log_prob and not need_value:
                logits = self._policy_logits(state_tensor, option_tensor)
                result["action_index"] = int(torch.argmax(logits).item())
                return result

            value_tensor = None
            if need_value:
                logits, value_tensor = self.model(state_tensor, option_tensor)
            else:
                logits = self._policy_logits(state_tensor, option_tensor)

            scaled_logits = logits / max(temperature, 1e-3)
            probs = torch.softmax(scaled_logits, dim=0)
            if deterministic:
                action_index = int(torch.argmax(scaled_logits).item())
            elif random.random() < epsilon_random:
                action_index = random.randrange(len(option_vecs))
            else:
                action_index = int(torch.multinomial(probs, 1).item())

            result["action_index"] = action_index
            if need_log_prob:
                result["log_prob"] = float(torch.log(probs[action_index].clamp_min(EPSILON)).item())
            if need_value and value_tensor is not None:
                result["value"] = float(value_tensor.item())
            return result

    def select_actions_batch(
        self,
        state_vecs: Sequence[Sequence[float]],
        option_vec_batches: Sequence[Sequence[Sequence[float]]],
        deterministic: bool = False,
        temperature: float = 1.0,
        epsilon_random: float = 0.0,
        need_log_prob: bool = False,
        need_value: bool = False,
    ) -> List[Dict[str, Any]]:
        if not state_vecs:
            return []
        if len(state_vecs) != len(option_vec_batches):
            raise ValueError("state_vecs and option_vec_batches must have the same length.")

        if self.model.training:
            self.model.eval()

        with torch.no_grad():
            state_tensor = self._state_tensor_batch(state_vecs)
            option_tensor, option_mask = self._padded_option_tensor(option_vec_batches)

            if deterministic and not need_log_prob and not need_value:
                logits = self._policy_logits(state_tensor, option_tensor)
                masked_logits = self._masked_logits(logits, option_mask)
                action_indices = torch.argmax(masked_logits, dim=1).detach().cpu().tolist()
                return [{"action_index": int(action_index)} for action_index in action_indices]

            value_tensor = None
            if need_value:
                logits, value_tensor = self.model(state_tensor, option_tensor)
            else:
                logits = self._policy_logits(state_tensor, option_tensor)

            masked_logits = self._masked_logits(logits, option_mask)
            scaled_logits = masked_logits / max(temperature, 1e-3)

            if deterministic:
                action_tensor = torch.argmax(scaled_logits, dim=1)
            else:
                probs = torch.softmax(scaled_logits, dim=1)
                action_tensor = torch.multinomial(probs, 1).squeeze(1)
                if epsilon_random > 0.0:
                    action_indices = action_tensor.detach().cpu().tolist()
                    for index, option_vecs in enumerate(option_vec_batches):
                        if random.random() < epsilon_random:
                            action_indices[index] = random.randrange(len(option_vecs))
                    action_tensor = torch.tensor(action_indices, dtype=torch.int64, device=self.device)

            action_indices = [int(value) for value in action_tensor.detach().cpu().tolist()]
            results: List[Dict[str, Any]] = [{"action_index": action_index} for action_index in action_indices]

            if need_log_prob:
                log_probs = torch.log_softmax(scaled_logits, dim=1)
                chosen_log_probs = log_probs.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
                for result, log_prob in zip(results, chosen_log_probs.detach().cpu().tolist()):
                    result["log_prob"] = float(log_prob)

            if need_value and value_tensor is not None:
                for result, value in zip(results, value_tensor.detach().cpu().tolist()):
                    result["value"] = float(value)

            return results

    def _sample_batch_tensors(
        self,
        samples: Sequence[Dict[str, Any]],
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        state_tensor = self._state_tensor_batch([sample["state_vec"] for sample in samples])
        option_tensor, option_mask = self._padded_option_tensor([sample["option_vecs"] for sample in samples])
        action_tensor = torch.tensor(
            [int(sample["action_index"]) for sample in samples],
            dtype=torch.int64,
            device=self.device,
        )
        old_log_prob_tensor = torch.tensor(
            [float(sample["old_log_prob"]) for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        return_tensor = torch.tensor(
            [float(sample["return"]) for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        advantage_tensor = torch.tensor(
            [float(sample["advantage"]) for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        return (
            state_tensor,
            option_tensor,
            option_mask,
            action_tensor,
            old_log_prob_tensor,
            return_tensor,
            advantage_tensor,
        )

    def train_on_samples(self, samples: List[Dict[str, Any]], config: TrainingConfig) -> Dict[str, float]:
        if not samples:
            return {
                "samples": 0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "clip_fraction": 0.0,
                "avg_ratio": 1.0,
                "avg_value_prediction": 0.0,
            }

        self.model.train()
        self.optimizer.param_groups[0]["lr"] = config.learning_rate

        policy_losses: List[float] = []
        value_losses: List[float] = []
        ratios: List[float] = []
        clipped: List[float] = []
        value_predictions: List[float] = []
        minibatch_size = max(1, min(int(config.ppo_minibatch_size), len(samples)))

        for _ in range(config.ppo_epochs):
            random.shuffle(samples)
            for batch_start in range(0, len(samples), minibatch_size):
                sample_batch = samples[batch_start:batch_start + minibatch_size]
                (
                    state_tensor,
                    option_tensor,
                    option_mask,
                    action_tensor,
                    old_log_prob_tensor,
                    return_tensor,
                    advantage_tensor,
                ) = self._sample_batch_tensors(sample_batch)

                logits, value = self.model(state_tensor, option_tensor)
                masked_logits = self._masked_logits(logits, option_mask)
                log_probs = torch.log_softmax(masked_logits, dim=1)
                chosen_log_probs = log_probs.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
                ratio = torch.exp(chosen_log_probs - old_log_prob_tensor)
                clipped_ratio = torch.clamp(ratio, 1.0 - config.ppo_clip, 1.0 + config.ppo_clip)
                surrogate_one = ratio * advantage_tensor
                surrogate_two = clipped_ratio * advantage_tensor
                policy_loss = -torch.min(surrogate_one, surrogate_two).mean()
                value_loss = 0.5 * torch.square(value - return_tensor).mean()
                loss = policy_loss + config.value_coef * value_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip)
                self.optimizer.step()

                ratio_values = ratio.detach().cpu().tolist()
                clipped_flags = (
                    (ratio < (1.0 - config.ppo_clip)) | (ratio > (1.0 + config.ppo_clip))
                ).detach().cpu().tolist()
                value_batch = value.detach().cpu().tolist()

                policy_loss_value = float(policy_loss.item())
                value_loss_value = float(value_loss.item())
                policy_losses.extend([policy_loss_value] * len(sample_batch))
                value_losses.extend([value_loss_value] * len(sample_batch))
                ratios.extend(float(ratio_value) for ratio_value in ratio_values)
                clipped.extend(1.0 if was_clipped else 0.0 for was_clipped in clipped_flags)
                value_predictions.extend(float(prediction) for prediction in value_batch)

        return {
            "samples": float(len(samples)),
            "policy_loss": _mean(policy_losses),
            "value_loss": _mean(value_losses),
            "clip_fraction": _mean(clipped),
            "avg_ratio": _mean(ratios),
            "avg_value_prediction": _mean(value_predictions),
        }


class PolicyActor:
    def __init__(
        self,
        policy: PolicyNetwork,
        deterministic: bool = False,
        temperature: float = 1.0,
        epsilon_random: float = 0.0,
        collector: Optional[List[Dict[str, Any]]] = None,
        decision_callback: Optional[Any] = None,
    ) -> None:
        self.policy = policy
        self.deterministic = deterministic
        self.temperature = temperature
        self.epsilon_random = epsilon_random
        self.collector = collector
        self.decision_callback = decision_callback

    def choose(self, player_name: str, options: Sequence[Sequence[Any]], known_game_state: Dict[str, Any]) -> int:
        state_vec = state_to_vector(known_game_state, legal_option_count=len(options))
        option_vecs = [option_to_vector(option, known_game_state) for option in options]
        collect_details = self.collector is not None
        selection = self.policy.select_action(
            state_vec,
            option_vecs,
            deterministic=self.deterministic,
            temperature=self.temperature,
            epsilon_random=self.epsilon_random,
            need_log_prob=collect_details,
            need_value=collect_details,
        )
        if collect_details:
            self.collector.append(
                {
                    "state_vec": state_vec,
                    "option_vecs": option_vecs,
                    "action_index": selection["action_index"],
                    "old_log_prob": selection["log_prob"],
                    "value": selection["value"],
                    "player_name": player_name,
                }
            )
        if self.decision_callback is not None:
            self.decision_callback(player_name, options, known_game_state, selection["action_index"])
        return selection["action_index"]


_SIMULATION_WORKER_POLICY_A: Optional[PolicyNetwork] = None
_SIMULATION_WORKER_POLICY_B: Optional[PolicyNetwork] = None
_SIMULATION_WORKER_CONFIG: Optional[TrainingConfig] = None
_SIMULATION_WORKER_PROGRESS_QUEUE: Optional[Any] = None
_SIMULATION_WORKER_POLICY_CACHE: Dict[str, PolicyNetwork] = {}


def _configure_simulation_worker_runtime() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(SIMULATION_WORKER_TORCH_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(SIMULATION_WORKER_TORCH_THREADS))
    if torch is None:
        return
    try:
        torch.set_num_threads(SIMULATION_WORKER_TORCH_THREADS)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(SIMULATION_WORKER_TORCH_THREADS)
    except Exception:
        pass


def _policy_payload_for_simulation(policy: PolicyNetwork) -> Dict[str, Any]:
    payload = policy.to_dict(include_optimizer=False)
    payload["device_preference"] = DEVICE_CPU
    return payload


def _config_payload_for_simulation(config: TrainingConfig) -> Dict[str, Any]:
    payload = asdict(config)
    payload["device_preference"] = DEVICE_CPU
    return payload


def _config_from_payload(payload: Dict[str, Any]) -> TrainingConfig:
    return TrainingConfig().merged(payload)


def _actor_temperature(config: TrainingConfig, deterministic: bool, override_temperature: Optional[float] = None) -> float:
    if deterministic:
        return float(config.eval_temperature)
    if override_temperature is None:
        return float(config.train_temperature)
    return max(float(override_temperature), 1e-3)


def _actor_epsilon_random(config: TrainingConfig, deterministic: bool, override_epsilon: Optional[float] = None) -> float:
    if deterministic:
        return 0.0
    if override_epsilon is None:
        return max(0.0, float(config.epsilon_random))
    return max(0.0, float(override_epsilon))


def _training_opponent_exploration_choices(config: TrainingConfig) -> List[Tuple[Dict[str, Any], float]]:
    normal = {
        "style": "normal",
        "temperature": float(config.train_temperature),
        "epsilon_random": max(0.0, float(config.epsilon_random)),
    }
    return [
        (normal, max(0.0, float(config.opponent_normal_weight))),
        (
            {
                "style": "learnable",
                "temperature": max(1e-3, float(config.opponent_learnable_temperature)),
                "epsilon_random": max(0.0, float(config.opponent_learnable_epsilon_random)),
            },
            max(0.0, float(config.opponent_learnable_weight)),
        ),
        (
            {
                "style": "chaotic",
                "temperature": max(1e-3, float(config.opponent_chaotic_temperature)),
                "epsilon_random": max(0.0, float(config.opponent_chaotic_epsilon_random)),
            },
            max(0.0, float(config.opponent_chaotic_weight)),
        ),
    ]


def _sample_training_opponent_exploration(config: TrainingConfig) -> Dict[str, Any]:
    choices = _training_opponent_exploration_choices(config)
    if sum(weight for _, weight in choices) <= 0.0:
        return choices[0][0]
    run_name_a, run_name_b = _weighted_choice(choices)
    if random.random() < 0.5:
        return run_name_b, run_name_a
    return run_name_a, run_name_b


def _training_opponent_exploration_plan(config: TrainingConfig, match_count: int) -> List[Dict[str, Any]]:
    total_matches = max(0, int(match_count))
    if total_matches <= 0:
        return []
    choices = [(style, weight) for style, weight in _training_opponent_exploration_choices(config) if weight > 0.0]
    if not choices:
        return [_training_opponent_exploration_choices(config)[0][0] for _ in range(total_matches)]

    total_weight = sum(weight for _, weight in choices)
    desired_counts = [(style, total_matches * weight / total_weight) for style, weight in choices]
    counts = [int(math.floor(desired)) for _, desired in desired_counts]
    remaining = total_matches - sum(counts)
    remainders = sorted(
        range(len(desired_counts)),
        key=lambda index: desired_counts[index][1] - counts[index],
        reverse=True,
    )
    for index in remainders[:remaining]:
        counts[index] += 1

    plan: List[Dict[str, Any]] = []
    for (style, _), count in zip(desired_counts, counts):
        plan.extend(_copy_nested(style) for _ in range(count))
    random.shuffle(plan)
    return plan


def _policy_from_simulation_payload(task: Dict[str, Any], payload_key: str, cache_key_key: str) -> PolicyNetwork:
    cache_key = str(task.get(cache_key_key) or "")
    if cache_key:
        cached = _SIMULATION_WORKER_POLICY_CACHE.get(cache_key)
        if cached is not None:
            return cached

    policy = PolicyNetwork.from_dict(task[payload_key])
    if cache_key:
        if len(_SIMULATION_WORKER_POLICY_CACHE) >= SIMULATION_WORKER_POLICY_CACHE_LIMIT:
            _SIMULATION_WORKER_POLICY_CACHE.clear()
        _SIMULATION_WORKER_POLICY_CACHE[cache_key] = policy
    return policy


def _simulation_worker_init(
    policy_a_payload: Dict[str, Any],
    policy_b_payload: Dict[str, Any],
    config_payload: Dict[str, Any],
    progress_queue: Optional[Any] = None,
) -> None:
    global _SIMULATION_WORKER_POLICY_A, _SIMULATION_WORKER_POLICY_B, _SIMULATION_WORKER_CONFIG
    global _SIMULATION_WORKER_PROGRESS_QUEUE
    _configure_simulation_worker_runtime()
    _SIMULATION_WORKER_POLICY_A = PolicyNetwork.from_dict(policy_a_payload)
    _SIMULATION_WORKER_POLICY_B = PolicyNetwork.from_dict(policy_b_payload)
    _SIMULATION_WORKER_CONFIG = _config_from_payload(config_payload)
    _SIMULATION_WORKER_PROGRESS_QUEUE = progress_queue


def _simulation_progress_worker_init(progress_queue: Optional[Any] = None) -> None:
    global _SIMULATION_WORKER_PROGRESS_QUEUE
    _configure_simulation_worker_runtime()
    _SIMULATION_WORKER_PROGRESS_QUEUE = progress_queue


def _create_simulation_progress_queue() -> Tuple[Any, Any]:
    context = multiprocessing.get_context("spawn") if os.name == "nt" else multiprocessing.get_context()
    return context, context.Queue()


def _close_simulation_progress_queue(progress_queue: Optional[Any]) -> None:
    if progress_queue is None:
        return
    try:
        progress_queue.close()
    except Exception:
        pass
    try:
        progress_queue.join_thread()
    except Exception:
        pass


def _process_pool_executor(
    max_workers: int,
    *,
    initializer: Optional[Callable[..., None]] = None,
    initargs: Tuple[Any, ...] = (),
    mp_context: Optional[Any] = None,
) -> concurrent.futures.ProcessPoolExecutor:
    kwargs: Dict[str, Any] = {"max_workers": max_workers}
    if initializer is not None:
        kwargs["initializer"] = initializer
        kwargs["initargs"] = initargs
    if mp_context is not None:
        kwargs["mp_context"] = mp_context
    return concurrent.futures.ProcessPoolExecutor(**kwargs)


def _warm_simulation_worker() -> int:
    _configure_simulation_worker_runtime()
    return os.getpid()


class _SimulationPool:
    def __init__(self, worker_count: int) -> None:
        self.worker_count = max(1, int(worker_count))
        self.progress_context, self.progress_queue = _create_simulation_progress_queue()
        self.executor = _process_pool_executor(
            max_workers=self.worker_count,
            initializer=_simulation_progress_worker_init,
            initargs=(self.progress_queue,),
            mp_context=self.progress_context,
        )
        self._warm_futures = [self.executor.submit(_warm_simulation_worker) for _ in range(self.worker_count)]

    def wait_until_ready(self) -> None:
        if not self._warm_futures:
            return
        futures = self._warm_futures
        self._warm_futures = []
        concurrent.futures.wait(futures)
        for future in futures:
            future.result()

    def close(self) -> None:
        try:
            try:
                self.executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                self.executor.shutdown(wait=True)
        finally:
            _close_simulation_progress_queue(self.progress_queue)


def _emit_simulation_game_progress(progress_queue: Optional[Any] = None, games_completed: int = 1) -> None:
    target_queue = progress_queue if progress_queue is not None else _SIMULATION_WORKER_PROGRESS_QUEUE
    if target_queue is None:
        return
    try:
        target_queue.put_nowait(int(games_completed))
    except Exception:
        pass


def _drain_simulation_progress_queue(progress_queue: Optional[Any]) -> int:
    if progress_queue is None:
        return 0
    games_completed = 0
    while True:
        try:
            games_completed += int(progress_queue.get_nowait())
        except queue.Empty:
            break
        except Exception:
            break
    return games_completed


def _consume_simulation_futures(
    futures: Sequence[Any],
    progress_queue: Optional[Any],
    progress_callback: Callable[[int], None],
    result_callback: Callable[[Dict[str, Any]], None],
) -> None:
    pending = set(futures)
    while pending:
        done, pending = concurrent.futures.wait(
            pending,
            timeout=SIMULATION_PROGRESS_POLL_SECONDS,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        progress_delta = _drain_simulation_progress_queue(progress_queue)
        if progress_delta > 0:
            progress_callback(progress_delta)
        for future in done:
            result_callback(future.result())

    progress_delta = _drain_simulation_progress_queue(progress_queue)
    if progress_delta > 0:
        progress_callback(progress_delta)


def _consume_simulation_task_iterator(
    executor: concurrent.futures.Executor,
    worker_func: Callable[[Dict[str, Any]], Dict[str, Any]],
    tasks: Iterable[Dict[str, Any]],
    *,
    max_in_flight: int,
    progress_queue: Optional[Any],
    progress_callback: Callable[[int], None],
    result_callback: Callable[[Dict[str, Any]], None],
) -> None:
    task_iter = iter(tasks)
    pending = set()
    exhausted = False

    def submit_until_full() -> None:
        nonlocal exhausted
        while not exhausted and len(pending) < max(1, int(max_in_flight)):
            try:
                task = next(task_iter)
            except StopIteration:
                exhausted = True
                break
            pending.add(executor.submit(worker_func, task))

    try:
        submit_until_full()
        while pending:
            done, pending = concurrent.futures.wait(
                pending,
                timeout=SIMULATION_PROGRESS_POLL_SECONDS,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            progress_delta = _drain_simulation_progress_queue(progress_queue)
            if progress_delta > 0:
                progress_callback(progress_delta)
            for future in done:
                result_callback(future.result())
            submit_until_full()

        progress_delta = _drain_simulation_progress_queue(progress_queue)
        if progress_delta > 0:
            progress_callback(progress_delta)
    except BaseException:
        for future in pending:
            future.cancel()
        raise


def _seed_simulation(seed: int) -> None:
    random.seed(int(seed))
    if torch is not None:
        try:
            torch.manual_seed(int(seed) % (2**31 - 1))
        except Exception:
            pass


def _simulate_match_games(
    policy_a: PolicyNetwork,
    policy_b: PolicyNetwork,
    config: TrainingConfig,
    *,
    game_count: int,
    seed: int,
    collect_a: bool,
    deterministic_a: bool,
    deterministic_b: bool,
    temperature_a: Optional[float] = None,
    epsilon_random_a: Optional[float] = None,
    temperature_b: Optional[float] = None,
    epsilon_random_b: Optional[float] = None,
    progress_queue: Optional[Any] = None,
) -> Dict[str, Any]:
    _seed_simulation(seed)
    collector: Optional[List[Dict[str, Any]]] = [] if collect_a else None
    trajectory_by_game: Optional[List[List[Dict[str, Any]]]] = [] if collect_a else None
    wins_a = 0
    wins_b = 0
    per_game_winners: List[str] = []
    ended_by_limit_games = 0
    started_at = _timestamp()

    for _ in range(max(0, int(game_count))):
        game_collector: Optional[List[Dict[str, Any]]] = [] if collect_a else None
        actor_a = PolicyActor(
            policy_a,
            deterministic=deterministic_a,
            temperature=_actor_temperature(config, deterministic_a, temperature_a),
            epsilon_random=_actor_epsilon_random(config, deterministic_a, epsilon_random_a),
            collector=game_collector,
        )
        actor_b = PolicyActor(
            policy_b,
            deterministic=deterministic_b,
            temperature=_actor_temperature(config, deterministic_b, temperature_b),
            epsilon_random=_actor_epsilon_random(config, deterministic_b, epsilon_random_b),
            collector=None,
        )
        game = Game(
            "policy_a",
            "policy_b",
            p1_choose=actor_a.choose,
            p2_choose=actor_b.choose,
            verbose=False,
            max_turns=config.max_turns_per_game,
            max_actions_per_turn=config.max_actions_per_turn,
        )
        per_game_winners.append(game.winner.name)
        if game.winner.name == "policy_a":
            wins_a += 1
        else:
            wins_b += 1
        if game.ended_by_limit:
            ended_by_limit_games += 1
        if collect_a and collector is not None and trajectory_by_game is not None and game_collector is not None:
            selected_game_collector = _sample_training_game_trajectory(
                game_collector,
                config.training_decisions_per_game,
            )
            if selected_game_collector:
                collector.extend(selected_game_collector)
                trajectory_by_game.append(selected_game_collector)
        _emit_simulation_game_progress(progress_queue)

    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "games_played": len(per_game_winners),
        "per_game_winners": per_game_winners,
        "trajectory": collector or [],
        "trajectory_by_game": trajectory_by_game or [],
        "ended_by_limit_games": ended_by_limit_games,
        "worker_duration_seconds": _timestamp() - started_at,
    }


def _simulate_initialized_match_chunk_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    if _SIMULATION_WORKER_POLICY_A is None or _SIMULATION_WORKER_POLICY_B is None or _SIMULATION_WORKER_CONFIG is None:
        raise RuntimeError("Simulation worker was not initialized.")
    return _simulate_match_games(
        _SIMULATION_WORKER_POLICY_A,
        _SIMULATION_WORKER_POLICY_B,
        _SIMULATION_WORKER_CONFIG,
        game_count=int(task["game_count"]),
        seed=int(task["seed"]),
        collect_a=bool(task.get("collect_a", False)),
        deterministic_a=bool(task.get("deterministic_a", False)),
        deterministic_b=bool(task.get("deterministic_b", False)),
        temperature_a=task.get("temperature_a"),
        epsilon_random_a=task.get("epsilon_random_a"),
        temperature_b=task.get("temperature_b"),
        epsilon_random_b=task.get("epsilon_random_b"),
    )


def _simulate_initialized_balanced_chunk_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    if _SIMULATION_WORKER_POLICY_A is None or _SIMULATION_WORKER_POLICY_B is None or _SIMULATION_WORKER_CONFIG is None:
        raise RuntimeError("Simulation worker was not initialized.")
    swapped = bool(task.get("swapped", False))
    result = _simulate_match_games(
        _SIMULATION_WORKER_POLICY_B if swapped else _SIMULATION_WORKER_POLICY_A,
        _SIMULATION_WORKER_POLICY_A if swapped else _SIMULATION_WORKER_POLICY_B,
        _SIMULATION_WORKER_CONFIG,
        game_count=int(task["game_count"]),
        seed=int(task["seed"]),
        collect_a=False,
        deterministic_a=True,
        deterministic_b=True,
    )
    result["swapped"] = swapped
    return result


def _simulate_payload_match_chunk_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    _configure_simulation_worker_runtime()
    policy_a = _policy_from_simulation_payload(task, "policy_a_payload", "policy_a_cache_key")
    policy_b = _policy_from_simulation_payload(task, "policy_b_payload", "policy_b_cache_key")
    config = _config_from_payload(task["config_payload"])
    result = _simulate_match_games(
        policy_a,
        policy_b,
        config,
        game_count=int(task["game_count"]),
        seed=int(task["seed"]),
        collect_a=bool(task.get("collect_a", False)),
        deterministic_a=bool(task.get("deterministic_a", False)),
        deterministic_b=bool(task.get("deterministic_b", False)),
        temperature_a=task.get("temperature_a"),
        epsilon_random_a=task.get("epsilon_random_a"),
        temperature_b=task.get("temperature_b"),
        epsilon_random_b=task.get("epsilon_random_b"),
    )
    if "match_index" in task:
        result["match_index"] = int(task["match_index"])
    if "opponent" in task:
        result["opponent"] = _copy_nested(task["opponent"])
    return result


def _simulate_payload_balanced_chunk_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    _configure_simulation_worker_runtime()
    policy_a = _policy_from_simulation_payload(task, "policy_a_payload", "policy_a_cache_key")
    policy_b = _policy_from_simulation_payload(task, "policy_b_payload", "policy_b_cache_key")
    config = _config_from_payload(task["config_payload"])
    swapped = bool(task.get("swapped", False))
    result = _simulate_match_games(
        policy_b if swapped else policy_a,
        policy_a if swapped else policy_b,
        config,
        game_count=int(task["game_count"]),
        seed=int(task["seed"]),
        collect_a=False,
        deterministic_a=True,
        deterministic_b=True,
    )
    result["swapped"] = swapped
    if "pairing_index" in task:
        result["pairing_index"] = int(task["pairing_index"])
    return result


def _combine_match_partials(partials: Sequence[Dict[str, Any]], *, started_at: Optional[float] = None) -> Dict[str, Any]:
    wins_a = sum(int(partial.get("wins_a", 0)) for partial in partials)
    wins_b = sum(int(partial.get("wins_b", 0)) for partial in partials)
    games_played = sum(int(partial.get("games_played", 0)) for partial in partials)
    per_game_winners: List[str] = []
    trajectory: List[Dict[str, Any]] = []
    trajectory_by_game: List[List[Dict[str, Any]]] = []
    ended_by_limit_games = 0
    for partial in partials:
        per_game_winners.extend(str(winner) for winner in partial.get("per_game_winners", []))
        trajectory.extend(list(partial.get("trajectory") or []))
        trajectory_by_game.extend(list(partial.get("trajectory_by_game") or []))
        ended_by_limit_games += int(partial.get("ended_by_limit_games", 0))
    duration_seconds = (
        _timestamp() - float(started_at)
        if started_at is not None
        else sum(float(partial.get("worker_duration_seconds", 0.0)) for partial in partials)
    )
    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "games_played": games_played,
        "return_value": (wins_a - wins_b) / max(games_played, 1),
        "per_game_winners": per_game_winners,
        "trajectory": trajectory,
        "trajectory_by_game": trajectory_by_game,
        "ended_by_limit_games": ended_by_limit_games,
        "duration_seconds": duration_seconds,
    }


def _simulate_balanced_match_serial(
    policy_a: PolicyNetwork,
    policy_b: PolicyNetwork,
    config: TrainingConfig,
    *,
    games_per_match: int,
    seed: int,
    progress_queue: Optional[Any] = None,
) -> Dict[str, Any]:
    games = max(1, int(games_per_match))
    if games <= 1:
        return _combine_match_partials(
            [
                _simulate_match_games(
                    policy_a,
                    policy_b,
                    config,
                    game_count=1,
                    seed=seed,
                    collect_a=False,
                    deterministic_a=True,
                    deterministic_b=True,
                    progress_queue=progress_queue,
                )
            ]
        )

    games_as_first = max(1, games // 2)
    games_as_second = max(1, games - games_as_first)
    started_at = _timestamp()
    first_seat = _simulate_match_games(
        policy_a,
        policy_b,
        config,
        game_count=games_as_first,
        seed=seed,
        collect_a=False,
        deterministic_a=True,
        deterministic_b=True,
        progress_queue=progress_queue,
    )
    second_seat = _simulate_match_games(
        policy_b,
        policy_a,
        config,
        game_count=games_as_second,
        seed=seed + 1,
        collect_a=False,
        deterministic_a=True,
        deterministic_b=True,
        progress_queue=progress_queue,
    )

    wins_a = int(first_seat.get("wins_a", 0)) + int(second_seat.get("wins_b", 0))
    wins_b = int(first_seat.get("wins_b", 0)) + int(second_seat.get("wins_a", 0))
    games_played = int(first_seat.get("games_played", 0)) + int(second_seat.get("games_played", 0))
    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "games_played": games_played,
        "return_value": (wins_a - wins_b) / max(games_played, 1),
        "per_game_winners": list(first_seat.get("per_game_winners", []))
        + [("policy_a" if winner == "policy_b" else "policy_b") for winner in second_seat.get("per_game_winners", [])],
        "trajectory": [],
        "duration_seconds": _timestamp() - started_at,
        "seat_results": [
            {"seat": "a_first", **first_seat},
            {"seat": "a_second", **second_seat},
        ],
    }


def _combine_balanced_partials(partials: Sequence[Dict[str, Any]], *, started_at: Optional[float] = None) -> Dict[str, Any]:
    wins_a = 0
    wins_b = 0
    games_played = 0
    per_game_winners: List[str] = []
    seat_results: List[Dict[str, Any]] = []
    for partial in partials:
        swapped = bool(partial.get("swapped", False))
        games_played += int(partial.get("games_played", 0))
        if swapped:
            wins_a += int(partial.get("wins_b", 0))
            wins_b += int(partial.get("wins_a", 0))
            per_game_winners.extend(
                ("policy_a" if winner == "policy_b" else "policy_b")
                for winner in partial.get("per_game_winners", [])
            )
            seat_results.append({"seat": "a_second", **partial})
        else:
            wins_a += int(partial.get("wins_a", 0))
            wins_b += int(partial.get("wins_b", 0))
            per_game_winners.extend(str(winner) for winner in partial.get("per_game_winners", []))
            seat_results.append({"seat": "a_first", **partial})
    duration_seconds = (
        _timestamp() - float(started_at)
        if started_at is not None
        else sum(float(partial.get("worker_duration_seconds", 0.0)) for partial in partials)
    )
    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "games_played": games_played,
        "return_value": (wins_a - wins_b) / max(games_played, 1),
        "per_game_winners": per_game_winners,
        "trajectory": [],
        "duration_seconds": duration_seconds,
        "seat_results": seat_results,
    }


def _simulate_balanced_match_payload_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    _configure_simulation_worker_runtime()
    policy_a = _policy_from_simulation_payload(task, "policy_a_payload", "policy_a_cache_key")
    policy_b = _policy_from_simulation_payload(task, "policy_b_payload", "policy_b_cache_key")
    config = _config_from_payload(task["config_payload"])
    config.simulation_workers = 1
    result = _simulate_balanced_match_serial(
        policy_a,
        policy_b,
        config,
        games_per_match=int(task["games_per_match"]),
        seed=int(task["seed"]),
        progress_queue=_SIMULATION_WORKER_PROGRESS_QUEUE,
    )
    if "pairing_index" in task:
        result["pairing_index"] = int(task["pairing_index"])
    return result


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _elo_update(rating_a: float, rating_b: float, score_a: float, k_factor: float) -> Tuple[float, float]:
    expected_a = _elo_expected(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    score_b = 1.0 - score_a
    new_a = rating_a + k_factor * (score_a - expected_a)
    new_b = rating_b + k_factor * (score_b - expected_b)
    return new_a, new_b


class RunFiles:
    def __init__(self, run_name: str) -> None:
        self.run_name = run_name
        self.run_dir = RUNS_DIR / run_name
        self.analysis_dir = self.run_dir / "analysis"
        self.policy_file = self.run_dir / "latest_policy.json"
        self.candidate_policy_file = self.run_dir / "candidate_policy.json"
        self.training_state_file = self.run_dir / "training_state.json"
        self.checkpoints_dir = self.run_dir / "checkpoints"


def _default_candidate_state(base_checkpoint: str, iteration: int) -> Dict[str, Any]:
    return {
        "base_checkpoint": base_checkpoint,
        "base_iteration": iteration,
        "attempts_since_reset": 0,
        "total_attempts": 0,
        "resets": 0,
        "promotions": 0,
        "rating_pass_elo": None,
        "last_score": None,
        "last_result": "initialized",
        "last_reset_reason": "initialization",
        "last_opponents": [],
    }


def model_type_overrides(model_type: str) -> Dict[str, Any]:
    normalized = str(model_type or MODEL_TYPE_DEEP).strip().lower()
    if normalized == MODEL_TYPE_DEEP:
        return {
            "model_architecture": CURRENT_MODEL_ARCHITECTURE,
            "hidden_size": DEEP_MODEL_HIDDEN_SIZE,
        }
    if normalized == MODEL_TYPE_DEFAULT:
        return {
            "model_architecture": LEGACY_MODEL_ARCHITECTURE,
            "hidden_size": DEFAULT_MODEL_HIDDEN_SIZE,
        }
    raise ValueError(f"Unknown model type '{model_type}'. Expected 'deep' or 'default'.")


def _convert_policy_architecture(
    source_policy: PolicyNetwork,
    target_architecture: str,
    target_hidden_size: int,
    target_device_preference: str,
) -> Tuple[PolicyNetwork, str]:
    requested_architecture = str(target_architecture or source_policy.architecture).strip()
    requested_hidden_size = int(target_hidden_size)
    requested_device_preference = str(target_device_preference or DEVICE_AUTO).strip().lower()

    if requested_hidden_size <= 0:
        raise ValueError("target_hidden_size must be positive.")

    if (
        source_policy.architecture == requested_architecture
        and int(source_policy.hidden_size) == requested_hidden_size
        and str(source_policy.device_preference).strip().lower() == requested_device_preference
    ):
        return PolicyNetwork.from_dict(source_policy.to_dict(include_optimizer=False) | {"device_preference": requested_device_preference}), "cloned_exact"

    if (
        source_policy.architecture == requested_architecture
        and int(source_policy.hidden_size) == requested_hidden_size
    ):
        payload = source_policy.to_dict(include_optimizer=False)
        payload["device_preference"] = requested_device_preference
        return PolicyNetwork.from_dict(payload), "cloned_device"

    if (
        source_policy.architecture == LEGACY_MODEL_ARCHITECTURE
        and requested_architecture == CURRENT_MODEL_ARCHITECTURE
        and requested_hidden_size >= int(source_policy.hidden_size)
    ):
        converted = PolicyNetwork(
            state_size=source_policy.state_size,
            option_size=source_policy.option_size,
            hidden_size=requested_hidden_size,
            architecture=requested_architecture,
            device_preference=requested_device_preference,
        )
        source_model = source_policy.model
        target_model = converted.model
        source_hidden_size = int(source_policy.hidden_size)

        with torch.no_grad():
            for parameter in target_model.parameters():
                parameter.zero_()

            target_model.state_linear.weight[:source_hidden_size, :] = source_model.state_linear.weight.detach().to(converted.device)
            target_model.state_linear.bias[:source_hidden_size] = source_model.state_linear.bias.detach().to(converted.device)
            target_model.option_linear.weight[:source_hidden_size, :] = source_model.option_linear.weight.detach().to(converted.device)
            target_model.option_linear.bias[:source_hidden_size] = source_model.option_linear.bias.detach().to(converted.device)
            target_model.joint_bias[:source_hidden_size] = source_model.joint_bias.detach().to(converted.device)

            target_model.policy_head.weight[:, :source_hidden_size] = source_model.policy_head.weight.detach().to(converted.device)
            target_model.policy_head.bias.copy_(source_model.policy_head.bias.detach().to(converted.device))
            target_model.value_head.weight[:, :source_hidden_size] = source_model.value_head.weight.detach().to(converted.device)
            target_model.value_head.bias.copy_(source_model.value_head.bias.detach().to(converted.device))

        return converted, "legacy_to_deep_exact_embed"

    raise ValueError(
        "Unsupported checkpoint architecture conversion: "
        f"{source_policy.architecture}/{source_policy.hidden_size} -> "
        f"{requested_architecture}/{requested_hidden_size}. "
        "Currently supported conversions are same-architecture clones and legacy_v1 -> deep_v2 embedding."
    )


def normalize_training_decisions_per_game(value: Any) -> str:
    normalized = str(value if value is not None else TRAINING_DECISIONS_PER_GAME_ALL).strip().upper()
    if normalized not in TRAINING_DECISIONS_PER_GAME_OPTIONS:
        raise ValueError(
            "training_decisions_per_game must be one of: "
            + ", ".join(TRAINING_DECISIONS_PER_GAME_OPTIONS)
            + "."
        )
    return normalized


def training_decisions_per_game_limit(value: Any) -> Optional[int]:
    normalized = normalize_training_decisions_per_game(value)
    if normalized == TRAINING_DECISIONS_PER_GAME_ALL:
        return None
    return int(normalized)


def training_samples_per_game_limit(value: Any = TRAINING_DECISIONS_PER_GAME_ALL) -> int:
    decisions_limit = training_decisions_per_game_limit(value)
    if decisions_limit is None:
        return DEFAULT_MAX_TRAINING_SAMPLES_PER_GAME
    return max(1, min(int(decisions_limit), DEFAULT_MAX_TRAINING_SAMPLES_PER_GAME))


def _sample_training_game_trajectory(
    game_trajectory: Sequence[Dict[str, Any]],
    training_decisions_per_game: Any = TRAINING_DECISIONS_PER_GAME_ALL,
) -> List[Dict[str, Any]]:
    trajectory = list(game_trajectory or [])
    limit = training_samples_per_game_limit(training_decisions_per_game)
    if len(trajectory) <= limit:
        return trajectory
    return random.sample(trajectory, limit)


def normalize_train_temperature(value: Any) -> float:
    try:
        temperature = float(value)
    except (TypeError, ValueError):
        raise ValueError("train_temperature must be a number.")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("train_temperature must be greater than 0.")
    return temperature


def derive_temperature_schedule_overrides(train_temperature: float) -> Dict[str, float]:
    normalized_temperature = round(normalize_train_temperature(train_temperature), 6)
    return {
        "train_temperature": normalized_temperature,
        "min_train_temperature": normalized_temperature,
        "plateau_temperature_boost": 1.0,
        "plateau_max_train_temperature": normalized_temperature,
    }


def normalize_promotion_score_threshold(value: Any) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        raise ValueError("promotion_score_threshold must be a number.")
    if not math.isfinite(threshold) or threshold < 0.5 or threshold > 1.0:
        raise ValueError("promotion_score_threshold must be between 0.5 and 1.0.")
    return threshold


def new_run_overrides(
    model_type: str = MODEL_TYPE_DEEP,
    training_matches_per_iteration: int = 5,
    training_games_per_match: int = 16,
    training_decisions_per_game: str = TRAINING_DECISIONS_PER_GAME_ALL,
    train_temperature: float = 0.9,
    promotion_games: int = 24,
    promotion_score_threshold: float = 0.6,
    simulation_workers: int = SIMULATION_WORKERS_AUTO,
    device_preference: str = DEVICE_AUTO,
) -> Dict[str, Any]:
    if int(training_matches_per_iteration) <= 0:
        raise ValueError("training_matches_per_iteration must be positive.")
    if int(training_games_per_match) <= 0:
        raise ValueError("training_games_per_match must be positive.")
    if int(promotion_games) <= 0:
        raise ValueError("promotion_games must be positive.")

    overrides = model_type_overrides(model_type)
    overrides.update(derive_temperature_schedule_overrides(train_temperature))
    overrides.update(
        {
            "device_preference": str(device_preference or DEVICE_AUTO).strip().lower(),
            "training_matches_per_iteration": int(training_matches_per_iteration),
            "training_games_per_match": int(training_games_per_match),
            "training_decisions_per_game": normalize_training_decisions_per_game(training_decisions_per_game),
            "promotion_games": int(promotion_games),
            "promotion_score_threshold": normalize_promotion_score_threshold(promotion_score_threshold),
            "simulation_workers": normalize_simulation_workers(simulation_workers),
        }
    )
    return overrides


def _default_training_state(config: TrainingConfig) -> Dict[str, Any]:
    created_at = _timestamp()
    return {
        "status": "idle",
        "run_name": LATEST_RUN_NAME,
        "config_defaults_version": CONFIG_DEFAULTS_VERSION,
        "iteration": 0,
        "total_matches": 0,
        "total_games": 0,
        "created_at": created_at,
        "created_datetime": _format_timestamp(created_at),
        "updated_at": created_at,
        "forked_from": None,
        "strategy": "champion_league_v2",
        "strategy_version": 2,
        "current_elo": INITIAL_ELO,
        "best_elo": INITIAL_ELO,
        "best_checkpoint": "checkpoint_000000.json",
        "latest_checkpoint": "checkpoint_000000.json",
        "promotions": 0,
        "last_promotion_iteration": 0,
        "last_promotion_at": None,
        "last_promotion_datetime": "",
        "config": asdict(config),
        "last_match": None,
        "last_update": None,
        "last_eval": None,
        "last_rating_pass": None,
        "live_progress": None,
        "device_plan": None,
        "candidate": _default_candidate_state("checkpoint_000000.json", 0),
        "checkpoints": [
            {
                "name": "checkpoint_000000.json",
                "iteration": 0,
                "elo": INITIAL_ELO,
                "created_at": _timestamp(),
                "note": "initial random policy",
            }
        ],
    }


def _load_json_or_none(path: Path, retries: int = 6, delay_seconds: float = 0.05) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None

    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (PermissionError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(delay_seconds * (attempt + 1))

    if last_error is not None:
        raise last_error
    return None


_JSON_CACHE_LOCK = threading.Lock()
_JSON_CACHE: Dict[str, Tuple[Tuple[int, int], Optional[Dict[str, Any]]]] = {}


def _json_cache_path_key(path: Path) -> str:
    return str(path)


def _invalidate_json_cache(path: Path) -> None:
    with _JSON_CACHE_LOCK:
        _JSON_CACHE.pop(_json_cache_path_key(path), None)


def _load_cached_json_or_none(path: Path, retries: int = 6, delay_seconds: float = 0.05) -> Optional[Dict[str, Any]]:
    cache_key = _json_cache_path_key(path)
    try:
        stat = path.stat()
    except FileNotFoundError:
        with _JSON_CACHE_LOCK:
            _JSON_CACHE.pop(cache_key, None)
        return None

    signature = (stat.st_mtime_ns, stat.st_size)
    with _JSON_CACHE_LOCK:
        cached = _JSON_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature:
            payload = cached[1]
            return None if payload is None else dict(payload)

    payload = _load_json_or_none(path, retries=retries, delay_seconds=delay_seconds)
    try:
        refreshed_stat = path.stat()
    except FileNotFoundError:
        with _JSON_CACHE_LOCK:
            _JSON_CACHE.pop(cache_key, None)
        return None if payload is None else dict(payload)

    refreshed_signature = (refreshed_stat.st_mtime_ns, refreshed_stat.st_size)
    with _JSON_CACHE_LOCK:
        _JSON_CACHE[cache_key] = (refreshed_signature, payload)
    return None if payload is None else dict(payload)


def _load_policy_from_path(path: Path) -> Optional[PolicyNetwork]:
    payload = _load_json_or_none(path)
    if payload is None:
        return None
    return PolicyNetwork.from_dict(payload)


def _ensure_checkpoint_entry(state: Dict[str, Any], checkpoint_name: str, iteration: int, elo: float) -> Dict[str, Any]:
    checkpoints = state.setdefault("checkpoints", [])
    for item in checkpoints:
        if item.get("name") == checkpoint_name:
            item["iteration"] = int(item.get("iteration", iteration))
            item["elo"] = float(item.get("elo", elo))
            return item
    entry = {
        "name": checkpoint_name,
        "iteration": iteration,
        "elo": float(elo),
        "created_at": _timestamp(),
    }
    checkpoints.append(entry)
    return entry


def _migrate_run_state(files: RunFiles, policy: PolicyNetwork, state: Dict[str, Any], config: TrainingConfig) -> Tuple[PolicyNetwork, Dict[str, Any], bool]:
    dirty = False
    state["run_name"] = state.get("run_name", files.run_name)
    if state.get("created_at") is None:
        try:
            state["created_at"] = files.training_state_file.stat().st_mtime
        except OSError:
            state["created_at"] = _timestamp()
        dirty = True
    created_datetime = _format_timestamp(state.get("created_at"))
    if state.get("created_datetime") != created_datetime:
        state["created_datetime"] = created_datetime
        dirty = True
    if "forked_from" not in state:
        state["forked_from"] = None
        dirty = True
    state["checkpoints"] = list(state.get("checkpoints") or [])

    for checkpoint in state["checkpoints"]:
        if "iteration" in checkpoint:
            checkpoint["iteration"] = int(checkpoint["iteration"])
        if "elo" in checkpoint:
            checkpoint["elo"] = float(checkpoint["elo"])

    best_name = state.get("best_checkpoint") or state.get("latest_checkpoint") or "checkpoint_000000.json"
    latest_name = state.get("latest_checkpoint") or best_name
    strategy_version = int(state.get("strategy_version", 0) or 0)
    migrated_to_v2 = state.get("strategy") != "champion_league_v2" or strategy_version < 2

    if migrated_to_v2:
        latest_name = best_name
        champion_policy = _load_policy_from_path(files.checkpoints_dir / latest_name)
        if champion_policy is not None:
            policy = champion_policy
        champion_entry = _ensure_checkpoint_entry(
            state,
            latest_name,
            int(state.get("iteration", 0)),
            float(state.get("best_elo", state.get("current_elo", INITIAL_ELO))),
        )
        champion_elo = float(champion_entry.get("elo", state.get("best_elo", INITIAL_ELO)))
        state["strategy"] = "champion_league_v2"
        state["strategy_version"] = 2
        state["latest_checkpoint"] = latest_name
        state["best_checkpoint"] = best_name or latest_name
        state["current_elo"] = champion_elo
        state["best_elo"] = max(float(state.get("best_elo", champion_elo)), champion_elo)
        state["promotions"] = int(state.get("promotions", 0))
        state["last_eval"] = state.get("last_eval")
        state["candidate"] = _default_candidate_state(latest_name, int(state.get("iteration", 0)))
        dirty = True
    else:
        state["strategy"] = "champion_league_v2"
        state["strategy_version"] = 2
        state["promotions"] = int(state.get("promotions", 0))
        state["last_eval"] = state.get("last_eval")
        default_candidate = _default_candidate_state(latest_name, int(state.get("iteration", 0)))
        candidate_state = dict(default_candidate)
        candidate_state.update(state.get("candidate") or {})
        if candidate_state != state.get("candidate"):
            state["candidate"] = candidate_state
            dirty = True

    champion_entry = _ensure_checkpoint_entry(
        state,
        state.get("latest_checkpoint", latest_name),
        int(state.get("iteration", 0)),
        float(state.get("current_elo", INITIAL_ELO)),
    )
    state["current_elo"] = float(champion_entry.get("elo", state.get("current_elo", INITIAL_ELO)))
    state["best_checkpoint"] = state.get("best_checkpoint", state.get("latest_checkpoint"))
    if state["best_checkpoint"]:
        best_entry = _ensure_checkpoint_entry(
            state,
            state["best_checkpoint"],
            int(champion_entry.get("iteration", state.get("iteration", 0))),
            float(state.get("best_elo", state.get("current_elo", INITIAL_ELO))),
        )
        state["best_elo"] = max(float(state.get("best_elo", INITIAL_ELO)), float(best_entry.get("elo", INITIAL_ELO)))

    if "last_promotion_iteration" not in state or state.get("last_promotion_iteration") is None:
        if int(state.get("promotions", 0)) > 0:
            state["last_promotion_iteration"] = int(champion_entry.get("iteration", state.get("iteration", 0)))
        else:
            state["last_promotion_iteration"] = 0
        dirty = True
    else:
        state["last_promotion_iteration"] = max(0, int(state.get("last_promotion_iteration", 0)))

    last_promotion_at = _last_promotion_timestamp_from_state(state)
    last_promotion_datetime = _format_timestamp(last_promotion_at) if last_promotion_at is not None else ""
    if state.get("last_promotion_at") != last_promotion_at:
        state["last_promotion_at"] = last_promotion_at
        dirty = True
    if state.get("last_promotion_datetime") != last_promotion_datetime:
        state["last_promotion_datetime"] = last_promotion_datetime
        dirty = True

    if not files.policy_file.exists() or migrated_to_v2:
        _atomic_write_json(files.policy_file, policy.to_dict(include_optimizer=True))
        dirty = True

    if not files.candidate_policy_file.exists():
        _atomic_write_json(files.candidate_policy_file, policy.to_dict(include_optimizer=True))
        dirty = True

    return policy, state, dirty


def _ensure_run(run_name: str, config_overrides: Optional[Dict[str, Any]] = None) -> RunFiles:
    files = RunFiles(run_name)
    files.run_dir.mkdir(parents=True, exist_ok=True)
    files.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if files.training_state_file.exists() and files.policy_file.exists():
        return files

    config = TrainingConfig().merged(config_overrides)
    policy = PolicyNetwork(
        hidden_size=config.hidden_size,
        architecture=config.model_architecture,
        device_preference=config.device_preference,
    )
    state = _default_training_state(config)
    state["run_name"] = run_name
    state["device_plan"] = policy.device_summary()
    _atomic_write_json(files.policy_file, policy.to_dict(include_optimizer=True))
    _atomic_write_json(files.candidate_policy_file, policy.to_dict(include_optimizer=True))
    _atomic_write_json(files.training_state_file, state)
    _atomic_write_json(files.checkpoints_dir / "checkpoint_000000.json", policy.to_dict(include_optimizer=False))
    return files


def run_exists(run_name: str) -> bool:
    files = RunFiles(run_name)
    return files.training_state_file.exists() and files.policy_file.exists()


def _load_policy_and_state(run_name: str, config_overrides: Optional[Dict[str, Any]] = None) -> Tuple[RunFiles, PolicyNetwork, Dict[str, Any], TrainingConfig]:
    files = _ensure_run(run_name, config_overrides=config_overrides)
    state_payload = _load_json_or_none(files.training_state_file)
    if state_payload is None:
        state_payload = _default_training_state(TrainingConfig())
        state_payload["run_name"] = run_name
    config_defaults_version = int(state_payload.get("config_defaults_version", 0) or 0)
    policy_payload = _load_json_or_none(files.policy_file)
    original_saved_config = dict(state_payload.get("config", {}) or {})
    saved_config = dict(original_saved_config)
    if "training_games_per_match" not in saved_config:
        saved_config["training_games_per_match"] = 24
    if "training_decisions_per_game" not in saved_config:
        saved_config["training_decisions_per_game"] = TRAINING_DECISIONS_PER_GAME_ALL
    else:
        saved_config["training_decisions_per_game"] = normalize_training_decisions_per_game(
            saved_config.get("training_decisions_per_game")
        )
    if saved_config.get("promotion_games") == 13:
        saved_config["promotion_games"] = 24
    if "simulation_workers" not in saved_config:
        saved_config["simulation_workers"] = SIMULATION_WORKERS_AUTO
    else:
        saved_config["simulation_workers"] = normalize_simulation_workers(saved_config.get("simulation_workers"))
    saved_config.pop("max_training_samples_per_match", None)
    saved_config.pop("max_training_samples_per_iteration", None)
    if isinstance(state_payload.get("last_update"), dict):
        last_update = state_payload["last_update"]
        last_update.pop("max_training_samples_per_match", None)
        if "max_training_samples_per_iteration" in last_update:
            last_update.pop("max_training_samples_per_iteration", None)
            last_update.setdefault("sample_reservoir_limit", DEFAULT_MAX_TRAINING_SAMPLES_PER_ITERATION)
    if "min_available_memory_mb" not in saved_config:
        saved_config["min_available_memory_mb"] = DEFAULT_MIN_AVAILABLE_MEMORY_MB
    else:
        saved_config["min_available_memory_mb"] = max(0, int(saved_config.get("min_available_memory_mb") or 0))
    default_config = TrainingConfig()
    for key in (
        "opponent_normal_weight",
        "opponent_learnable_weight",
        "opponent_chaotic_weight",
        "opponent_learnable_temperature",
        "opponent_chaotic_temperature",
        "opponent_learnable_epsilon_random",
        "opponent_chaotic_epsilon_random",
    ):
        if key not in saved_config:
            saved_config[key] = getattr(default_config, key)
    if config_defaults_version < CONFIG_DEFAULTS_VERSION:
        if config_defaults_version < 2:
            legacy_default_updates = {
                "promotion_score_threshold": ((0.55,), 0.6),
                "learning_rate": ((0.0015,), 0.0019),
                "min_learning_rate": ((0.0003,), 0.0004),
                "epsilon_random": ((0.05,), 0.07),
                "min_epsilon_random": ((0.01,), 0.015),
                "train_temperature": ((1.0, 1.1), 0.9),
                "min_train_temperature": ((0.55, 0.62), 0.5),
                "plateau_learning_rate_boost": ((2.0, 2.15), 1.7),
                "plateau_epsilon_boost": ((3.0, 3.2), 2.4),
                "plateau_temperature_boost": ((1.35, 1.45, 1.28), 1.22),
                "plateau_max_learning_rate": ((0.0035, 0.0042), 0.0033),
                "plateau_max_epsilon_random": ((0.18, 0.22), 0.16),
                "plateau_max_train_temperature": ((1.45, 1.6, 1.35), 1.15),
            }
            for key, (old_values, new_value) in legacy_default_updates.items():
                current_value = saved_config.get(key)
                if current_value is None:
                    saved_config[key] = new_value
                    continue
                try:
                    numeric_value = float(current_value)
                except (TypeError, ValueError):
                    continue
                if any(abs(numeric_value - float(old_value)) <= 1e-12 for old_value in old_values):
                    saved_config[key] = new_value
        if config_defaults_version < 3:
            saved_config.update(derive_temperature_schedule_overrides(saved_config.get("train_temperature", 0.9)))
        if config_defaults_version < 5:
            opponent_mix_default_updates = {
                "opponent_normal_weight": ((0.4,), 0.6),
                "opponent_learnable_weight": ((0.45,), 0.3),
                "opponent_chaotic_weight": ((0.15,), 0.1),
            }
            for key, (old_values, new_value) in opponent_mix_default_updates.items():
                current_value = saved_config.get(key)
                if current_value is None:
                    saved_config[key] = new_value
                    continue
                try:
                    numeric_value = float(current_value)
                except (TypeError, ValueError):
                    continue
                if any(abs(numeric_value - float(old_value)) <= 1e-12 for old_value in old_values):
                    saved_config[key] = new_value
        state_payload["config_defaults_version"] = CONFIG_DEFAULTS_VERSION
    if policy_payload is not None:
        payload_architecture = policy_payload.get("architecture")
        if payload_architecture is None:
            payload_state_dict = policy_payload.get("state_dict") or {}
            if any(name.startswith("state_refine_") or name.startswith("joint_mix_") or name.startswith("value_refine") for name in payload_state_dict.keys()):
                payload_architecture = CURRENT_MODEL_ARCHITECTURE
            else:
                payload_architecture = LEGACY_MODEL_ARCHITECTURE
        saved_config["model_architecture"] = payload_architecture
        if "hidden_size" not in original_saved_config or saved_config.get("hidden_size") != policy_payload.get("hidden_size"):
            saved_config["hidden_size"] = policy_payload.get("hidden_size", saved_config.get("hidden_size", 96))
    state_payload["config"] = saved_config
    config = TrainingConfig().merged(saved_config).merged(config_overrides)
    if policy_payload is None:
        policy = PolicyNetwork(
            hidden_size=config.hidden_size,
            architecture=config.model_architecture,
            device_preference=config.device_preference,
        )
    else:
        policy = PolicyNetwork.from_dict(policy_payload)
    policy, state_payload, dirty = _migrate_run_state(files, policy, state_payload, config)
    device_plan = policy.device_summary()
    if state_payload.get("device_plan") != device_plan:
        state_payload["device_plan"] = device_plan
        dirty = True
    dirty = dirty or (saved_config != original_saved_config)
    if dirty:
        _save_policy_and_state(files, policy, state_payload)
    return files, policy, state_payload, config


def _save_policy_and_state(files: RunFiles, policy: PolicyNetwork, state: Dict[str, Any]) -> None:
    state["updated_at"] = _timestamp()
    state["device_plan"] = policy.device_summary()
    _atomic_write_json(files.policy_file, policy.to_dict(include_optimizer=True))
    _atomic_write_json(files.training_state_file, state)


def _checkpoint_path(files: RunFiles, iteration: int) -> Path:
    return files.checkpoints_dir / f"champion_{iteration:06d}.json"


def load_policy(run_name: str = LATEST_RUN_NAME, checkpoint: str = "latest") -> PolicyNetwork:
    files, policy, state, _ = _load_policy_and_state(run_name)
    if checkpoint == "latest":
        return policy
    if checkpoint == "candidate":
        candidate_policy = _load_policy_from_path(files.candidate_policy_file)
        return candidate_policy or policy.clone()
    if checkpoint == "best":
        checkpoint_name = state.get("best_checkpoint", state.get("latest_checkpoint"))
    else:
        checkpoint_name = checkpoint
    checkpoint_file = files.checkpoints_dir / checkpoint_name
    payload = _load_json_or_none(checkpoint_file)
    if payload is None:
        return policy
    return PolicyNetwork.from_dict(payload)


def list_runs() -> List[Dict[str, Any]]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runs: List[Dict[str, Any]] = []
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        state_path = run_dir / "training_state.json"
        state = _load_cached_json_or_none(state_path) or {}
        checkpoints = list(state.get("checkpoints") or [])
        created_at = state.get("created_at")
        created_datetime = state.get("created_datetime") or _format_timestamp(created_at)
        last_promotion_at = _last_promotion_timestamp_from_state(state)
        last_promotion_datetime = _format_timestamp(last_promotion_at) if last_promotion_at is not None else ""
        forked_from = state.get("forked_from")
        runs.append(
            {
                "run_name": state.get("run_name", run_dir.name),
                "status": state.get("status", "idle"),
                "iteration": int(state.get("iteration", 0)),
                "total_matches": int(state.get("total_matches", 0)),
                "total_games": int(state.get("total_games", 0)),
                "current_elo": float(state.get("current_elo", INITIAL_ELO)),
                "best_elo": float(state.get("best_elo", INITIAL_ELO)),
                "best_checkpoint": state.get("best_checkpoint"),
                "latest_checkpoint": state.get("latest_checkpoint"),
                "promotions": int(state.get("promotions", 0)),
                "last_promotion_iteration": int(state.get("last_promotion_iteration", 0) or 0),
                "last_promotion_at": last_promotion_at,
                "last_promotion_datetime": last_promotion_datetime,
                "checkpoint_count": len(checkpoints),
                "created_at": created_at,
                "created_datetime": created_datetime,
                "forked_from": forked_from,
                "fork_origin": _fork_origin_label(forked_from),
                "updated_at": state.get("updated_at"),
                "last_match": state.get("last_match"),
                "last_update": state.get("last_update"),
                "last_eval": state.get("last_eval"),
                "last_error": state.get("last_error"),
                "run_dir": str(run_dir),
            }
        )
    def run_sort_key(item: Dict[str, Any]) -> Tuple[float, str]:
        last_promotion_at = _optional_timestamp(item.get("last_promotion_at"))
        created_sort_at = _optional_timestamp(item.get("created_at")) or 0.0
        return (
            last_promotion_at if last_promotion_at is not None else created_sort_at,
            str(item["run_name"]),
        )

    runs.sort(key=run_sort_key, reverse=True)
    return runs


def list_checkpoints(run_name: str = LATEST_RUN_NAME) -> List[Dict[str, Any]]:
    files = RunFiles(run_name)
    state = _load_cached_json_or_none(files.training_state_file) or {}
    best_name = state.get("best_checkpoint")
    latest_name = state.get("latest_checkpoint")
    checkpoints = []
    for item in list(state.get("checkpoints") or []):
        checkpoint = dict(item)
        checkpoint["is_best"] = checkpoint.get("name") == best_name
        checkpoint["is_latest"] = checkpoint.get("name") == latest_name
        checkpoint["is_candidate"] = False
        checkpoints.append(checkpoint)
    if files.candidate_policy_file.exists():
        candidate_state = dict(state.get("candidate") or {})
        candidate_elo = candidate_state.get("rating_pass_elo")
        if candidate_elo is None:
            candidate_elo = state.get("current_elo", INITIAL_ELO)
        checkpoints.append(
            {
                "name": "candidate",
                "iteration": int(state.get("iteration", 0)),
                "elo": float(candidate_elo),
                "is_best": False,
                "is_latest": False,
                "is_candidate": True,
                "base_checkpoint": candidate_state.get("base_checkpoint"),
                "attempts_since_reset": int(candidate_state.get("attempts_since_reset", 0)),
                "last_score": candidate_state.get("last_score"),
            }
        )
    checkpoints.sort(key=lambda item: (int(item.get("iteration", 0)), item.get("name", "")))
    return checkpoints


def get_run_state(run_name: str = LATEST_RUN_NAME) -> Dict[str, Any]:
    files = RunFiles(run_name)
    state = _load_cached_json_or_none(files.training_state_file) or {}
    if not state:
        return {}
    state["run_name"] = state.get("run_name", run_name)
    state["created_datetime"] = state.get("created_datetime") or _format_timestamp(state.get("created_at"))
    last_promotion_at = _last_promotion_timestamp_from_state(state)
    state["last_promotion_at"] = last_promotion_at
    state["last_promotion_datetime"] = _format_timestamp(last_promotion_at) if last_promotion_at is not None else ""
    state["fork_origin"] = _fork_origin_label(state.get("forked_from"))
    state["run_dir"] = str(files.run_dir)
    state["candidate_policy_file"] = str(files.candidate_policy_file)
    return state


def _checkpoint_sort_key(item: Dict[str, Any]) -> Tuple[int, str]:
    return int(item.get("iteration", 0)), str(item.get("name", ""))


def _resolve_checkpoint_name(state: Dict[str, Any], checkpoint: str) -> str:
    checkpoint_name = str(checkpoint or "").strip()
    if checkpoint_name == "latest":
        checkpoint_name = str(state.get("latest_checkpoint") or "")
    elif checkpoint_name == "best":
        checkpoint_name = str(state.get("best_checkpoint") or state.get("latest_checkpoint") or "")
    return checkpoint_name


def _pick_latest_checkpoint_entry(checkpoints: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not checkpoints:
        return None
    return max(checkpoints, key=_checkpoint_sort_key)


def _pick_best_checkpoint_entry(checkpoints: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not checkpoints:
        return None
    return max(
        checkpoints,
        key=lambda item: (
            float(item.get("elo", INITIAL_ELO)),
            int(item.get("iteration", 0)),
            str(item.get("name", "")),
        ),
    )


def _scrub_deleted_checkpoint_refs(value: Any, deleted_checkpoint: str, replacement_checkpoint: Optional[str]) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, nested_value in value.items():
            if key == "checkpoint" and nested_value == deleted_checkpoint:
                cleaned[key] = replacement_checkpoint
            elif key in {"base_checkpoint", "best_checkpoint", "latest_checkpoint", "champion_checkpoint"} and nested_value == deleted_checkpoint:
                cleaned[key] = replacement_checkpoint
            elif key == "promoted_checkpoint" and nested_value == deleted_checkpoint:
                cleaned[key] = None
            elif key == "name" and nested_value == deleted_checkpoint and value.get("kind") == "checkpoint":
                continue
            else:
                cleaned[key] = _scrub_deleted_checkpoint_refs(nested_value, deleted_checkpoint, replacement_checkpoint)
        return cleaned
    if isinstance(value, list):
        cleaned_list: List[Any] = []
        for item in value:
            cleaned_item = _scrub_deleted_checkpoint_refs(item, deleted_checkpoint, replacement_checkpoint)
            if isinstance(cleaned_item, dict) and cleaned_item.get("kind") == "checkpoint" and cleaned_item.get("name") is None:
                continue
            cleaned_list.append(cleaned_item)
        return cleaned_list
    return value


def delete_checkpoints(run_name: str = LATEST_RUN_NAME, checkpoints: Sequence[str] = ()) -> Dict[str, Any]:
    files, policy, state, _ = _load_policy_and_state(run_name)
    requested_names = [_resolve_checkpoint_name(state, checkpoint) for checkpoint in checkpoints]
    checkpoint_names = [name for name in requested_names if name]
    if not checkpoint_names:
        raise ValueError("At least one checkpoint name is required.")
    if any(name == "candidate" for name in checkpoint_names):
        raise ValueError("The candidate row is not a checkpoint file. Delete saved checkpoints instead.")

    with _TRAINERS_LOCK:
        trainer = _TRAINERS.get(run_name)
    if trainer is not None and trainer.is_running:
        raise RuntimeError(f"Run '{run_name}' is currently training. Interrupt it before deleting checkpoints.")
    if str(state.get("status", "idle")).lower() in {"training", "stop_requested", "rating"}:
        raise RuntimeError(f"Run '{run_name}' is marked as active. Interrupt training before deleting checkpoints.")

    checkpoint_entries = list(state.get("checkpoints") or [])
    existing_names = {str(item.get("name", "")) for item in checkpoint_entries}
    missing = [name for name in checkpoint_names if name not in existing_names]
    if missing:
        if len(missing) == 1:
            raise FileNotFoundError(f"Checkpoint '{missing[0]}' was not found in run '{run_name}'.")
        raise FileNotFoundError(f"These checkpoints were not found in run '{run_name}': {', '.join(sorted(missing))}.")

    unique_checkpoint_names = sorted(set(checkpoint_names))
    deleted_set = set(unique_checkpoint_names)
    for checkpoint_name in unique_checkpoint_names:
        checkpoint_path = files.checkpoints_dir / checkpoint_name
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    state["checkpoints"] = [
        dict(item)
        for item in checkpoint_entries
        if str(item.get("name", "")) not in deleted_set
    ]

    deleted_latest = str(state.get("latest_checkpoint")) in deleted_set
    deleted_best = str(state.get("best_checkpoint")) in deleted_set

    replacement_checkpoint: Optional[str] = None
    if deleted_latest or not state.get("checkpoints"):
        replacement_checkpoint = "champion_current.json"
        replacement_path = files.checkpoints_dir / replacement_checkpoint
        _atomic_write_json(replacement_path, policy.to_dict(include_optimizer=False))
        replacement_entry = _ensure_checkpoint_entry(
            state,
            replacement_checkpoint,
            int(state.get("iteration", 0)),
            float(state.get("current_elo", INITIAL_ELO)),
        )
        replacement_entry["iteration"] = int(state.get("iteration", 0))
        replacement_entry["elo"] = float(state.get("current_elo", INITIAL_ELO))
        replacement_entry["created_at"] = replacement_entry.get("created_at", _timestamp())
        replacement_entry["note"] = "synthetic current champion snapshot"
        state["latest_checkpoint"] = replacement_checkpoint
    else:
        latest_entry = _pick_latest_checkpoint_entry(list(state.get("checkpoints") or []))
        if latest_entry is not None:
            state["latest_checkpoint"] = latest_entry.get("name")

    latest_entry = next(
        (item for item in state.get("checkpoints", []) if item.get("name") == state.get("latest_checkpoint")),
        None,
    )
    if latest_entry is not None:
        state["current_elo"] = float(latest_entry.get("elo", state.get("current_elo", INITIAL_ELO)))

    if deleted_best or str(state.get("best_checkpoint")) in deleted_set:
        best_entry = _pick_best_checkpoint_entry(list(state.get("checkpoints") or []))
        if best_entry is not None:
            state["best_checkpoint"] = best_entry.get("name")
            state["best_elo"] = float(best_entry.get("elo", state.get("current_elo", INITIAL_ELO)))
        else:
            state["best_checkpoint"] = state.get("latest_checkpoint")
            state["best_elo"] = float(state.get("current_elo", INITIAL_ELO))
    else:
        best_entry = next(
            (item for item in state.get("checkpoints", []) if item.get("name") == state.get("best_checkpoint")),
            None,
        )
        if best_entry is not None:
            state["best_elo"] = float(best_entry.get("elo", state.get("best_elo", INITIAL_ELO)))

    replacement_reference = str(state.get("latest_checkpoint") or state.get("best_checkpoint") or "")
    for checkpoint_name in unique_checkpoint_names:
        state["candidate"] = _scrub_deleted_checkpoint_refs(
            state.get("candidate") or {},
            checkpoint_name,
            replacement_reference,
        )
        state["last_match"] = _scrub_deleted_checkpoint_refs(
            state.get("last_match"),
            checkpoint_name,
            replacement_reference,
        )
        state["last_eval"] = _scrub_deleted_checkpoint_refs(
            state.get("last_eval"),
            checkpoint_name,
            replacement_reference,
        )
        state["last_rating_pass"] = _scrub_deleted_checkpoint_refs(
            state.get("last_rating_pass"),
            checkpoint_name,
            replacement_reference,
        )
        state["last_update"] = _scrub_deleted_checkpoint_refs(
            state.get("last_update"),
            checkpoint_name,
            replacement_reference,
        )
        with _POLICY_CACHE_LOCK:
            _POLICY_CACHE.pop((run_name, checkpoint_name), None)

    with _POLICY_CACHE_LOCK:
        _POLICY_CACHE.pop((run_name, "best"), None)
        _POLICY_CACHE.pop((run_name, "latest"), None)
    with _TRAINERS_LOCK:
        _TRAINERS.pop(run_name, None)

    _save_policy_and_state(files, policy, state)

    return {
        "run_name": run_name,
        "deleted_checkpoints": unique_checkpoint_names,
        "latest_checkpoint": state.get("latest_checkpoint"),
        "best_checkpoint": state.get("best_checkpoint"),
        "current_elo": float(state.get("current_elo", INITIAL_ELO)),
        "best_elo": float(state.get("best_elo", INITIAL_ELO)),
        "checkpoint_count": len(list(state.get("checkpoints") or [])),
        "replacement_checkpoint": replacement_checkpoint,
    }


def delete_checkpoint(run_name: str = LATEST_RUN_NAME, checkpoint: str = "") -> Dict[str, Any]:
    result = delete_checkpoints(run_name=run_name, checkpoints=[checkpoint])
    deleted = list(result.get("deleted_checkpoints") or [])
    result["deleted_checkpoint"] = deleted[0] if deleted else ""
    return result


def _play_match(
    policy_a: PolicyNetwork,
    policy_b: PolicyNetwork,
    config: TrainingConfig,
    collect_a: bool = False,
    deterministic_a: bool = False,
    deterministic_b: bool = False,
    games_per_match: int = 24,
    game_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    simulation_pool: Optional[_SimulationPool] = None,
    cache_key_prefix: str = "",
    temperature_a: Optional[float] = None,
    epsilon_random_a: Optional[float] = None,
    temperature_b: Optional[float] = None,
    epsilon_random_b: Optional[float] = None,
) -> Dict[str, Any]:
    total_games = max(1, int(games_per_match))
    started_at = _timestamp()

    def emit_progress(games_completed: int, wins_a: int, wins_b: int) -> None:
        if game_progress_callback is None:
            return
        try:
            game_progress_callback(
                {
                    "games_completed": int(games_completed),
                    "games_target": int(total_games),
                    "wins_a": int(wins_a),
                    "wins_b": int(wins_b),
                    "duration_seconds": _timestamp() - started_at,
                }
            )
        except Exception:
            pass

    worker_count = resolve_simulation_workers(config.simulation_workers, total_games)
    if simulation_pool is not None:
        worker_count = min(max(1, int(simulation_pool.worker_count)), total_games)
    chunk_sizes = _simulation_chunk_sizes(total_games, worker_count)
    partials: List[Dict[str, Any]] = []
    completed_games = 0
    wins_a = 0
    wins_b = 0

    if worker_count <= 1 or total_games <= 1:
        for chunk_size in chunk_sizes:
            partial = _simulate_match_games(
                policy_a,
                policy_b,
                config,
                game_count=chunk_size,
                seed=_random_seed(),
                collect_a=collect_a,
                deterministic_a=deterministic_a,
                deterministic_b=deterministic_b,
                temperature_a=temperature_a,
                epsilon_random_a=epsilon_random_a,
                temperature_b=temperature_b,
                epsilon_random_b=epsilon_random_b,
            )
            partials.append(partial)
            completed_games += int(partial.get("games_played", 0))
            wins_a += int(partial.get("wins_a", 0))
            wins_b += int(partial.get("wins_b", 0))
            emit_progress(completed_games, wins_a, wins_b)
        return _combine_match_partials(partials, started_at=started_at)

    policy_a_payload = _policy_payload_for_simulation(policy_a)
    policy_b_payload = _policy_payload_for_simulation(policy_b)
    config_payload = _config_payload_for_simulation(config)

    if simulation_pool is not None:
        progress_queue = simulation_pool.progress_queue
        simulation_pool.wait_until_ready()
        _drain_simulation_progress_queue(progress_queue)
        prefix = cache_key_prefix or f"match:{os.getpid()}:{id(policy_a)}:{id(policy_b)}:{started_at:.6f}"
        tasks = [
            {
                "policy_a_payload": policy_a_payload,
                "policy_b_payload": policy_b_payload,
                "policy_a_cache_key": f"{prefix}:a",
                "policy_b_cache_key": f"{prefix}:b",
                "config_payload": config_payload,
                "game_count": chunk_size,
                "seed": _random_seed(),
                "collect_a": collect_a,
                "deterministic_a": deterministic_a,
                "deterministic_b": deterministic_b,
                "temperature_a": temperature_a,
                "epsilon_random_a": epsilon_random_a,
                "temperature_b": temperature_b,
                "epsilon_random_b": epsilon_random_b,
            }
            for chunk_size in chunk_sizes
        ]
        futures = [simulation_pool.executor.submit(_simulate_payload_match_chunk_worker, task) for task in tasks]

        def on_progress(delta: int) -> None:
            nonlocal completed_games
            completed_games = min(total_games, completed_games + int(delta))
            emit_progress(completed_games, wins_a, wins_b)

        def on_result(partial: Dict[str, Any]) -> None:
            nonlocal completed_games, wins_a, wins_b
            partials.append(partial)
            wins_a += int(partial.get("wins_a", 0))
            wins_b += int(partial.get("wins_b", 0))
            completed_games = max(completed_games, min(total_games, sum(int(item.get("games_played", 0)) for item in partials)))
            emit_progress(completed_games, wins_a, wins_b)

        _consume_simulation_futures(futures, progress_queue, on_progress, on_result)
        _drain_simulation_progress_queue(progress_queue)
        return _combine_match_partials(partials, started_at=started_at)

    progress_context = None
    progress_queue = None
    if game_progress_callback is not None:
        progress_context, progress_queue = _create_simulation_progress_queue()
    try:
        with _process_pool_executor(
            max_workers=worker_count,
            initializer=_simulation_worker_init,
            initargs=(policy_a_payload, policy_b_payload, config_payload, progress_queue),
            mp_context=progress_context,
        ) as executor:
            futures = [
                executor.submit(
                    _simulate_initialized_match_chunk_worker,
                    {
                        "game_count": chunk_size,
                        "seed": _random_seed(),
                        "collect_a": collect_a,
                        "deterministic_a": deterministic_a,
                        "deterministic_b": deterministic_b,
                        "temperature_a": temperature_a,
                        "epsilon_random_a": epsilon_random_a,
                        "temperature_b": temperature_b,
                        "epsilon_random_b": epsilon_random_b,
                    },
                )
                for chunk_size in chunk_sizes
            ]

            def on_progress(delta: int) -> None:
                nonlocal completed_games
                completed_games = min(total_games, completed_games + int(delta))
                emit_progress(completed_games, wins_a, wins_b)

            def on_result(partial: Dict[str, Any]) -> None:
                nonlocal completed_games, wins_a, wins_b
                partials.append(partial)
                wins_a += int(partial.get("wins_a", 0))
                wins_b += int(partial.get("wins_b", 0))
                completed_games = max(completed_games, min(total_games, sum(int(item.get("games_played", 0)) for item in partials)))
                emit_progress(completed_games, wins_a, wins_b)

            _consume_simulation_futures(futures, progress_queue, on_progress, on_result)
    finally:
        _close_simulation_progress_queue(progress_queue)

    return _combine_match_partials(partials, started_at=started_at)


def _play_balanced_match(
    policy_a: PolicyNetwork,
    policy_b: PolicyNetwork,
    config: TrainingConfig,
    games_per_match: int,
    game_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    simulation_pool: Optional[_SimulationPool] = None,
    cache_key_prefix: str = "",
) -> Dict[str, Any]:
    if games_per_match <= 1:
        return _play_match(
            policy_a,
            policy_b,
            config,
            collect_a=False,
            deterministic_a=True,
            deterministic_b=True,
            games_per_match=1,
            game_progress_callback=game_progress_callback,
        )

    games_as_first = max(1, games_per_match // 2)
    games_as_second = max(1, games_per_match - games_as_first)
    total_games = games_as_first + games_as_second
    started_at = _timestamp()

    def emit_progress(games_completed: int, wins_a: int, wins_b: int) -> None:
        if game_progress_callback is None:
            return
        try:
            game_progress_callback(
                {
                    "games_completed": int(games_completed),
                    "games_target": int(games_per_match),
                    "wins_a": int(wins_a),
                    "wins_b": int(wins_b),
                    "duration_seconds": _timestamp() - started_at,
                }
            )
        except Exception:
            pass

    worker_count = resolve_simulation_workers(config.simulation_workers, total_games)
    if simulation_pool is not None:
        worker_count = min(max(1, int(simulation_pool.worker_count)), total_games)
    partials: List[Dict[str, Any]] = []
    completed_games = 0
    wins_a = 0
    wins_b = 0
    tasks: List[Dict[str, Any]] = []
    for chunk_size in _simulation_chunk_sizes(games_as_first, worker_count):
        tasks.append({"game_count": chunk_size, "seed": _random_seed(), "swapped": False})
    for chunk_size in _simulation_chunk_sizes(games_as_second, worker_count):
        tasks.append({"game_count": chunk_size, "seed": _random_seed(), "swapped": True})

    if worker_count <= 1 or total_games <= 1:
        for task in tasks:
            partial = _simulate_match_games(
                policy_b if task["swapped"] else policy_a,
                policy_a if task["swapped"] else policy_b,
                config,
                game_count=int(task["game_count"]),
                seed=int(task["seed"]),
                collect_a=False,
                deterministic_a=True,
                deterministic_b=True,
            )
            partial["swapped"] = bool(task["swapped"])
            partials.append(partial)
            completed_games += int(partial.get("games_played", 0))
            if partial["swapped"]:
                wins_a += int(partial.get("wins_b", 0))
                wins_b += int(partial.get("wins_a", 0))
            else:
                wins_a += int(partial.get("wins_a", 0))
                wins_b += int(partial.get("wins_b", 0))
            emit_progress(completed_games, wins_a, wins_b)
        return _combine_balanced_partials(partials, started_at=started_at)

    policy_a_payload = _policy_payload_for_simulation(policy_a)
    policy_b_payload = _policy_payload_for_simulation(policy_b)
    config_payload = _config_payload_for_simulation(config)

    if simulation_pool is not None:
        progress_queue = simulation_pool.progress_queue
        simulation_pool.wait_until_ready()
        _drain_simulation_progress_queue(progress_queue)
        prefix = cache_key_prefix or f"balanced:{os.getpid()}:{id(policy_a)}:{id(policy_b)}:{started_at:.6f}"
        payload_tasks = [
            {
                **task,
                "policy_a_payload": policy_a_payload,
                "policy_b_payload": policy_b_payload,
                "policy_a_cache_key": f"{prefix}:a",
                "policy_b_cache_key": f"{prefix}:b",
                "config_payload": config_payload,
            }
            for task in tasks
        ]
        futures = [simulation_pool.executor.submit(_simulate_payload_balanced_chunk_worker, task) for task in payload_tasks]

        def on_progress(delta: int) -> None:
            nonlocal completed_games
            completed_games = min(total_games, completed_games + int(delta))
            emit_progress(completed_games, wins_a, wins_b)

        def on_result(partial: Dict[str, Any]) -> None:
            nonlocal completed_games, wins_a, wins_b
            partials.append(partial)
            if bool(partial.get("swapped", False)):
                wins_a += int(partial.get("wins_b", 0))
                wins_b += int(partial.get("wins_a", 0))
            else:
                wins_a += int(partial.get("wins_a", 0))
                wins_b += int(partial.get("wins_b", 0))
            completed_games = max(completed_games, min(total_games, sum(int(item.get("games_played", 0)) for item in partials)))
            emit_progress(completed_games, wins_a, wins_b)

        _consume_simulation_futures(futures, progress_queue, on_progress, on_result)
        _drain_simulation_progress_queue(progress_queue)
        return _combine_balanced_partials(partials, started_at=started_at)

    progress_context = None
    progress_queue = None
    if game_progress_callback is not None:
        progress_context, progress_queue = _create_simulation_progress_queue()
    try:
        with _process_pool_executor(
            max_workers=worker_count,
            initializer=_simulation_worker_init,
            initargs=(policy_a_payload, policy_b_payload, config_payload, progress_queue),
            mp_context=progress_context,
        ) as executor:
            futures = [executor.submit(_simulate_initialized_balanced_chunk_worker, task) for task in tasks]

            def on_progress(delta: int) -> None:
                nonlocal completed_games
                completed_games = min(total_games, completed_games + int(delta))
                emit_progress(completed_games, wins_a, wins_b)

            def on_result(partial: Dict[str, Any]) -> None:
                nonlocal completed_games, wins_a, wins_b
                partials.append(partial)
                if bool(partial.get("swapped", False)):
                    wins_a += int(partial.get("wins_b", 0))
                    wins_b += int(partial.get("wins_a", 0))
                else:
                    wins_a += int(partial.get("wins_a", 0))
                    wins_b += int(partial.get("wins_b", 0))
                completed_games = max(completed_games, min(total_games, sum(int(item.get("games_played", 0)) for item in partials)))
                emit_progress(completed_games, wins_a, wins_b)

            _consume_simulation_futures(futures, progress_queue, on_progress, on_result)
    finally:
        _close_simulation_progress_queue(progress_queue)

    return _combine_balanced_partials(partials, started_at=started_at)


def _compact_training_match_summary(match_summary: Dict[str, Any]) -> Dict[str, Any]:
    compact = dict(match_summary)
    compact.pop("trajectory", None)
    compact.pop("trajectory_by_game", None)
    return compact


def _extend_sample_reservoir(
    reservoir: List[Dict[str, Any]],
    incoming_samples: Sequence[Dict[str, Any]],
    *,
    max_samples: int,
    seen_count: int,
) -> int:
    limit = max(1, int(max_samples))
    seen = max(0, int(seen_count))
    for sample in incoming_samples:
        seen += 1
        if len(reservoir) < limit:
            reservoir.append(sample)
            continue
        replacement_index = random.randrange(seen)
        if replacement_index < limit:
            reservoir[replacement_index] = sample
    return seen


class SelfPlayTrainer:
    def __init__(self, run_name: str = LATEST_RUN_NAME, config_overrides: Optional[Dict[str, Any]] = None) -> None:
        files, policy, state, config = _load_policy_and_state(run_name, config_overrides=config_overrides)
        self.files = files
        self.policy = policy
        self.candidate_policy = _load_policy_from_path(files.candidate_policy_file) or policy.clone()
        self.state = state
        self.config = config
        self.run_name = run_name
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._training_active = False
        self._last_live_progress_save_at = 0.0
        self._simulation_pool: Optional[_SimulationPool] = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._training_active or (self._thread is not None and self._thread.is_alive())

    def _get_simulation_pool(self, worker_count: int) -> Optional[_SimulationPool]:
        worker_count = max(1, int(worker_count))
        if worker_count <= 1:
            self._close_simulation_pool()
            return None
        if self._simulation_pool is not None and self._simulation_pool.worker_count == worker_count:
            return self._simulation_pool
        self._close_simulation_pool()
        self._simulation_pool = _SimulationPool(worker_count)
        return self._simulation_pool

    def _close_simulation_pool(self) -> None:
        pool = self._simulation_pool
        self._simulation_pool = None
        if pool is not None:
            pool.close()

    def _save(self) -> None:
        config_payload = asdict(self.config)
        config_payload["max_iterations"] = None
        self.state["config"] = config_payload
        self.state["device_plan"] = self.policy.device_summary()
        _save_policy_and_state(self.files, self.policy, self.state)
        _atomic_write_json(self.files.candidate_policy_file, self.candidate_policy.to_dict(include_optimizer=True))

    def _save_state_only(self) -> None:
        config_payload = asdict(self.config)
        config_payload["max_iterations"] = None
        self.state["config"] = config_payload
        self.state["device_plan"] = self.policy.device_summary()
        self.state["updated_at"] = _timestamp()
        _atomic_write_json(self.files.training_state_file, self.state)

    def _build_training_live_progress(
        self,
        scheduled_config: TrainingConfig,
        *,
        stage: str,
        iteration_number: int,
        training_games_completed: int,
        promotion_games_completed: int,
        training_matches_completed: int,
        training_stage_complete: bool,
        promotion_stage_complete: bool,
    ) -> Dict[str, Any]:
        training_matches_total = max(1, int(scheduled_config.training_matches_per_iteration))
        training_games_per_match = max(1, int(scheduled_config.training_games_per_match))
        training_games_total = training_matches_total * training_games_per_match
        promotion_games_total = max(1, int(scheduled_config.promotion_games))
        return {
            "kind": "training_iteration",
            "stage": str(stage),
            "iteration_number": int(iteration_number),
            "training_matches_completed": int(training_matches_completed),
            "training_matches_total": int(training_matches_total),
            "training_games_completed": max(0, int(training_games_completed)),
            "training_games_total": int(training_games_total),
            "promotion_games_completed": max(0, int(promotion_games_completed)),
            "promotion_games_total": int(promotion_games_total),
            "iteration_games_completed": max(0, int(training_games_completed) + int(promotion_games_completed)),
            "iteration_games_total": int(training_games_total + promotion_games_total),
            "training_stage_complete": bool(training_stage_complete),
            "promotion_stage_complete": bool(promotion_stage_complete),
            "iteration_complete": bool(training_stage_complete and promotion_stage_complete),
            "updated_at": _timestamp(),
        }

    def _set_live_progress(self, payload: Optional[Dict[str, Any]], *, force_persist: bool = False) -> None:
        self.state["live_progress"] = None if payload is None else _copy_nested(payload)
        now = _timestamp()
        if force_persist or (now - self._last_live_progress_save_at) >= LIVE_PROGRESS_SAVE_INTERVAL_SECONDS:
            self._last_live_progress_save_at = now
            self._save_state_only()

    def _candidate_state(self) -> Dict[str, Any]:
        candidate_state = self.state.get("candidate")
        if candidate_state is None:
            candidate_state = _default_candidate_state(
                self.state.get("latest_checkpoint", "checkpoint_000000.json"),
                int(self.state.get("iteration", 0)),
            )
            self.state["candidate"] = candidate_state
        return candidate_state

    def _scheduled_training_config(self) -> TrainingConfig:
        anneal_steps = max(1, int(self.config.anneal_steps))
        progress = _clip(float(self.state.get("iteration", 0)) / float(anneal_steps), 0.0, 1.0)
        return self.config.merged(
            {
                "learning_rate": _lerp(self.config.learning_rate, self.config.min_learning_rate, progress),
                "epsilon_random": _lerp(self.config.epsilon_random, self.config.min_epsilon_random, progress),
                "ppo_clip": _lerp(self.config.ppo_clip, self.config.min_ppo_clip, progress),
            }
        )

    def _last_promotion_iteration(self) -> int:
        latest_name = self.state.get("latest_checkpoint")
        latest_entry = self._find_checkpoint_entry(str(latest_name)) if latest_name else None
        derived_iteration = int(latest_entry.get("iteration", 0)) if latest_entry is not None else 0
        stored_iteration = int(self.state.get("last_promotion_iteration", derived_iteration) or 0)
        return max(0, stored_iteration)

    def _iterations_since_promotion(self) -> int:
        return max(0, int(self.state.get("iteration", 0)) - self._last_promotion_iteration())

    def _rating_pass_pool(self, max_policies: int, include_candidate: bool) -> List[Dict[str, Any]]:
        checkpoints = self._checkpoint_entries_sorted()
        candidate_available = include_candidate and self.files.candidate_policy_file.exists()
        checkpoint_limit = max(0, max_policies - (1 if candidate_available else 0))

        selected_names: List[str] = []
        checkpoint_by_name = {item.get("name"): item for item in checkpoints if item.get("name")}

        if checkpoint_limit >= len(checkpoints):
            selected_names = [str(item["name"]) for item in checkpoints if item.get("name")]
        else:
            for anchor_name in [
                self.state.get("latest_checkpoint"),
                self.state.get("best_checkpoint"),
                checkpoints[0]["name"] if checkpoints else None,
            ]:
                if (
                    checkpoint_limit > 0
                    and anchor_name
                    and anchor_name in checkpoint_by_name
                    and anchor_name not in selected_names
                    and len(selected_names) < checkpoint_limit
                ):
                    selected_names.append(str(anchor_name))

            remaining = [item for item in checkpoints if item.get("name") not in selected_names]
            extra_slots = max(0, checkpoint_limit - len(selected_names))
            if extra_slots >= len(remaining):
                selected_names.extend(str(item["name"]) for item in remaining if item.get("name"))
            elif extra_slots > 0 and remaining:
                chosen_indices = set()
                for index in range(extra_slots):
                    raw_index = int((index + 0.5) * len(remaining) / extra_slots)
                    raw_index = _clip(raw_index, 0, len(remaining) - 1)
                    chosen_indices.add(int(raw_index))
                while len(chosen_indices) < extra_slots:
                    for index in range(len(remaining)):
                        if index not in chosen_indices:
                            chosen_indices.add(index)
                        if len(chosen_indices) >= extra_slots:
                            break
                selected_names.extend(
                    str(remaining[index]["name"])
                    for index in sorted(chosen_indices)
                    if remaining[index].get("name")
                )

        participants: List[Dict[str, Any]] = []
        for checkpoint_name in selected_names:
            checkpoint = checkpoint_by_name.get(checkpoint_name)
            if checkpoint is None:
                continue
            participants.append(
                {
                    "name": checkpoint_name,
                    "kind": "checkpoint",
                    "iteration": int(checkpoint.get("iteration", 0)),
                    "elo": float(checkpoint.get("elo", INITIAL_ELO)),
                }
            )

        if candidate_available and len(participants) < max_policies:
            candidate_state = self._candidate_state()
            candidate_elo = candidate_state.get("rating_pass_elo")
            if candidate_elo is None:
                candidate_elo = self.state.get("current_elo", INITIAL_ELO)
            participants.append(
                {
                    "name": "candidate",
                    "kind": "candidate",
                    "iteration": int(self.state.get("iteration", 0)),
                    "elo": float(candidate_elo),
                    "base_checkpoint": candidate_state.get("base_checkpoint"),
                }
            )

        participants.sort(key=lambda item: (int(item.get("iteration", 0)), item.get("name", "")))
        return participants

    def _policy_for_rating_participant(self, participant: Dict[str, Any]) -> PolicyNetwork:
        if participant.get("kind") == "candidate":
            return self.candidate_policy
        checkpoint_name = participant.get("name")
        if checkpoint_name == self.state.get("latest_checkpoint"):
            return self.policy
        return _cached_policy(self.run_name, str(checkpoint_name))

    def run_rating_pass(
        self,
        max_policies: int = 8,
        games_per_pair: int = 12,
        include_candidate: bool = True,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        ) -> Dict[str, Any]:
        if max_policies < 2:
            raise ValueError("Rating pass needs at least 2 policies.")
        if games_per_pair < 2:
            raise ValueError("Rating pass needs at least 2 games per pairing.")
        status = str(self.state.get("status", "idle"))
        last_updated_at = float(self.state.get("updated_at", 0.0) or 0.0)
        recently_updated = (_timestamp() - last_updated_at) <= 300.0
        if self.is_running or (status == "training" and recently_updated):
            raise RuntimeError("Interrupt training for this run before starting a rating pass.")

        participants = self._rating_pass_pool(max_policies=max_policies, include_candidate=include_candidate)
        if len(participants) < 2:
            raise RuntimeError("Not enough available models in this run to perform a rating pass.")

        rating_map = {item["name"]: float(item.get("elo", INITIAL_ELO)) for item in participants}
        started_at = _timestamp()
        active_simulation_key = _register_active_simulation_run(
            f"{self.run_name}:rating",
            self.config.simulation_workers,
        )
        self.state["status"] = "rating"
        self.state["last_error"] = None
        self.state["last_rating_pass"] = {
            "status": "running",
            "participant_count": len(participants),
            "games_per_pair": int(games_per_pair),
            "include_candidate": bool(include_candidate),
            "simulation_workers": int(self.config.simulation_workers),
            "resolved_simulation_workers": resolve_simulation_workers(
                self.config.simulation_workers,
                max(1, (len(participants) * (len(participants) - 1) // 2) * int(games_per_pair)),
                allocation_key=active_simulation_key,
            ),
            "active_simulation_runs": active_simulation_runs(),
            "started_at": started_at,
        }
        self._save()

        try:
            pairings: List[Dict[str, Any]] = []
            pairing_specs: List[Dict[str, Any]] = []
            for index_a, participant_a in enumerate(participants):
                for index_b in range(index_a + 1, len(participants)):
                    pairing_specs.append(
                        {
                            "pairing_index": len(pairing_specs),
                            "participant_a": participant_a,
                            "participant_b": participants[index_b],
                        }
                    )
            pairings_total = len(pairing_specs)
            games_total = pairings_total * int(games_per_pair)
            last_progress_save_at = 0.0
            active_pairing_games_completed = 0
            progress_games_completed = 0

            def emit_rating_progress(force_persist: bool = False) -> None:
                nonlocal last_progress_save_at
                games_completed = min(
                    games_total,
                    max(
                        int(progress_games_completed),
                        len(pairings) * int(games_per_pair) + int(active_pairing_games_completed),
                    ),
                )
                payload = {
                    "pairings_completed": len(pairings),
                    "pairings_total": pairings_total,
                    "games_completed": games_completed,
                    "games_target": games_total,
                    "duration_seconds": _timestamp() - started_at,
                }
                if progress_callback is not None:
                    try:
                        progress_callback(payload)
                    except Exception:
                        pass
                now = _timestamp()
                if force_persist or (now - last_progress_save_at) >= LIVE_PROGRESS_SAVE_INTERVAL_SECONDS:
                    last_progress_save_at = now
                    self.state["last_rating_pass"] = {
                        **dict(self.state.get("last_rating_pass") or {}),
                        "status": "running",
                        "pairings_played": len(pairings),
                        "pairings_total": pairings_total,
                        "games_completed": games_completed,
                        "games_target": games_total,
                        "duration_seconds": _timestamp() - started_at,
                    }
                    self._save_state_only()

            def record_pairing_result(spec: Dict[str, Any], match_summary: Dict[str, Any]) -> None:
                nonlocal active_pairing_games_completed, progress_games_completed
                participant_a = spec["participant_a"]
                participant_b = spec["participant_b"]
                name_a = str(participant_a["name"])
                name_b = str(participant_b["name"])
                rating_a_before = float(rating_map[name_a])
                rating_b_before = float(rating_map[name_b])
                score_a = float(match_summary.get("wins_a", 0)) / max(int(match_summary.get("games_played", 0)), 1)
                rating_a_after, rating_b_after = _elo_update(
                    rating_a_before,
                    rating_b_before,
                    score_a,
                    self.config.elo_k,
                )
                rating_map[name_a] = rating_a_after
                rating_map[name_b] = rating_b_after
                pairings.append(
                    {
                        "a": name_a,
                        "b": name_b,
                        "wins_a": int(match_summary.get("wins_a", 0)),
                        "wins_b": int(match_summary.get("wins_b", 0)),
                        "games_played": int(match_summary.get("games_played", 0)),
                        "score_a": score_a,
                        "rating_a_before": rating_a_before,
                        "rating_b_before": rating_b_before,
                        "rating_a_after": rating_a_after,
                        "rating_b_after": rating_b_after,
                        "duration_seconds": float(match_summary.get("duration_seconds", 0.0)),
                    }
                )
                active_pairing_games_completed = 0
                progress_games_completed = max(progress_games_completed, len(pairings) * int(games_per_pair))
                emit_rating_progress()

            worker_count = resolve_simulation_workers(
                self.config.simulation_workers,
                max(1, len(pairing_specs) * int(games_per_pair)),
                allocation_key=active_simulation_key,
            )
            emit_rating_progress(force_persist=True)
            parallel_pairings = len(pairing_specs) >= max(2, worker_count // 2)
            if worker_count <= 1:
                for spec in pairing_specs:
                    active_pairing_games_completed = 0

                    def pairing_progress(progress: Dict[str, Any]) -> None:
                        nonlocal active_pairing_games_completed
                        active_pairing_games_completed = int(progress.get("games_completed", 0))
                        emit_rating_progress()

                    match_summary = _play_balanced_match(
                        self._policy_for_rating_participant(spec["participant_a"]),
                        self._policy_for_rating_participant(spec["participant_b"]),
                        self.config,
                        games_per_match=games_per_pair,
                        game_progress_callback=pairing_progress,
                    )
                    record_pairing_result(spec, match_summary)
            elif not parallel_pairings:
                simulation_pool = _SimulationPool(worker_count)
                try:
                    for spec in pairing_specs:
                        active_pairing_games_completed = 0

                        def pairing_progress(progress: Dict[str, Any]) -> None:
                            nonlocal active_pairing_games_completed
                            active_pairing_games_completed = int(progress.get("games_completed", 0))
                            emit_rating_progress()

                        match_summary = _play_balanced_match(
                            self._policy_for_rating_participant(spec["participant_a"]),
                            self._policy_for_rating_participant(spec["participant_b"]),
                            self.config,
                            games_per_match=games_per_pair,
                            game_progress_callback=pairing_progress,
                            simulation_pool=simulation_pool,
                            cache_key_prefix=(
                                f"{self.run_name}:rating:{started_at}:"
                                f"pairing:{spec['pairing_index']}"
                            ),
                        )
                        record_pairing_result(spec, match_summary)
                finally:
                    simulation_pool.close()
            else:
                config_payload = _config_payload_for_simulation(self.config)
                spec_by_index = {int(spec["pairing_index"]): spec for spec in pairing_specs}
                simulation_pool = _SimulationPool(worker_count)
                progress_queue = simulation_pool.progress_queue
                tasks = [
                    {
                        "pairing_index": int(spec["pairing_index"]),
                        "policy_a_payload": _policy_payload_for_simulation(
                            self._policy_for_rating_participant(spec["participant_a"])
                        ),
                        "policy_b_payload": _policy_payload_for_simulation(
                            self._policy_for_rating_participant(spec["participant_b"])
                        ),
                        "policy_a_cache_key": f"{self.run_name}:rating:{started_at}:policy:{spec['participant_a']['name']}",
                        "policy_b_cache_key": f"{self.run_name}:rating:{started_at}:policy:{spec['participant_b']['name']}",
                        "config_payload": config_payload,
                        "games_per_match": int(games_per_pair),
                        "seed": _random_seed(),
                    }
                    for spec in pairing_specs
                ]
                try:
                    simulation_pool.wait_until_ready()
                    _drain_simulation_progress_queue(progress_queue)
                    futures = [
                        simulation_pool.executor.submit(_simulate_balanced_match_payload_worker, task)
                        for task in tasks
                    ]

                    def on_progress(delta: int) -> None:
                        nonlocal progress_games_completed
                        progress_games_completed = min(games_total, progress_games_completed + int(delta))
                        emit_rating_progress()

                    def on_result(match_summary: Dict[str, Any]) -> None:
                        spec = spec_by_index[int(match_summary["pairing_index"])]
                        record_pairing_result(spec, match_summary)

                    _consume_simulation_futures(futures, progress_queue, on_progress, on_result)
                finally:
                    simulation_pool.close()

            candidate_state = self._candidate_state()
            for participant in participants:
                name = str(participant["name"])
                rating = float(rating_map[name])
                if participant.get("kind") == "candidate":
                    candidate_state["rating_pass_elo"] = rating
                    continue
                checkpoint_name = name
                checkpoint_entry = self._find_checkpoint_entry(checkpoint_name)
                if checkpoint_entry is None:
                    checkpoint_entry = _ensure_checkpoint_entry(
                        self.state,
                        checkpoint_name,
                        int(participant.get("iteration", 0)),
                        rating,
                    )
                checkpoint_entry["elo"] = rating

            latest_name = self.state.get("latest_checkpoint")
            latest_entry = self._find_checkpoint_entry(str(latest_name)) if latest_name else None
            if latest_entry is not None:
                self.state["current_elo"] = float(latest_entry.get("elo", self.state.get("current_elo", INITIAL_ELO)))

            checkpoint_entries = self._checkpoint_entries_sorted()
            if checkpoint_entries:
                best_entry = max(
                    checkpoint_entries,
                    key=lambda item: (float(item.get("elo", INITIAL_ELO)), int(item.get("iteration", 0))),
                )
                self.state["best_checkpoint"] = best_entry.get("name")
                self.state["best_elo"] = float(best_entry.get("elo", INITIAL_ELO))

            leaderboard = [
                {
                    "name": str(participant["name"]),
                    "kind": str(participant.get("kind", "checkpoint")),
                    "iteration": int(participant.get("iteration", 0)),
                    "elo": float(rating_map[str(participant["name"])]),
                }
                for participant in participants
            ]
            leaderboard.sort(key=lambda item: (item["elo"], item["iteration"], item["name"]), reverse=True)

            summary = {
                "status": "completed",
                "participant_count": len(participants),
                "participants": [
                    {
                        "name": str(participant["name"]),
                        "kind": str(participant.get("kind", "checkpoint")),
                        "iteration": int(participant.get("iteration", 0)),
                    }
                    for participant in participants
                ],
                "include_candidate": bool(include_candidate),
                "games_per_pair": int(games_per_pair),
                "simulation_workers": int(self.config.simulation_workers),
                "resolved_simulation_workers": worker_count,
                "pairings_played": len(pairings),
                "pairings_total": pairings_total,
                "games_completed": len(pairings) * int(games_per_pair),
                "games_target": games_total,
                "pairings": pairings,
                "leaderboard": leaderboard,
                "duration_seconds": _timestamp() - started_at,
                "completed_at": _timestamp(),
            }
            if progress_callback is not None:
                try:
                    progress_callback(
                        {
                            "pairings_completed": len(pairings),
                            "pairings_total": pairings_total,
                            "games_completed": len(pairings) * int(games_per_pair),
                            "games_target": games_total,
                            "duration_seconds": summary["duration_seconds"],
                        }
                    )
                except Exception:
                    pass
            self.state["last_rating_pass"] = summary
            self.state["status"] = "idle"
            self.state["last_error"] = None
            self._save()
            return summary
        except Exception as exc:
            self.state["status"] = "idle"
            self.state["last_error"] = f"{type(exc).__name__}: {exc}"
            self.state["last_rating_pass"] = {
                "status": "failed",
                "participant_count": len(participants),
                "games_per_pair": int(games_per_pair),
                "include_candidate": bool(include_candidate),
                "simulation_workers": int(self.config.simulation_workers),
                "duration_seconds": _timestamp() - started_at,
                "error": self.state["last_error"],
            }
            self._save()
            raise
        finally:
            _unregister_active_simulation_run(active_simulation_key)

    def _find_checkpoint_entry(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        for item in self.state.get("checkpoints", []):
            if item["name"] == checkpoint_name:
                return item
        return None

    def _checkpoint_entries_sorted(self) -> List[Dict[str, Any]]:
        return sorted(
            list(self.state.get("checkpoints", [])),
            key=lambda item: (int(item.get("iteration", 0)), item.get("name", "")),
        )

    def _sample_league_opponent(self) -> Tuple[PolicyNetwork, Dict[str, Any]]:
        latest_name = self.state.get("latest_checkpoint")
        best_name = self.state.get("best_checkpoint")
        historical_names = [item["name"] for item in self._checkpoint_entries_sorted() if item.get("name")]
        historical_names = [name for name in historical_names if name != latest_name]

        recent_window = max(0, int(self.config.league_recent_window))
        recent_pool = historical_names[-recent_window:] if recent_window > 0 else []
        older_pool = historical_names[:-recent_window] if recent_window > 0 else historical_names

        choices: List[Tuple[Dict[str, Any], float]] = [
            (
                {
                    "label": "champion",
                    "source": "champion",
                    "checkpoint": latest_name,
                },
                float(self.config.league_champion_weight),
            )
        ]

        if best_name and best_name != latest_name:
            choices.append(
                (
                    {
                        "label": "best",
                        "source": "best",
                        "checkpoint": best_name,
                    },
                    float(self.config.league_best_weight),
                )
            )
        if recent_pool:
            choices.append(
                (
                    {
                        "label": "recent",
                        "source": "recent",
                        "checkpoint": random.choice(recent_pool),
                    },
                    float(self.config.league_recent_weight),
                )
            )
        if older_pool:
            choices.append(
                (
                    {
                        "label": "historical",
                        "source": "historical",
                        "checkpoint": random.choice(older_pool),
                    },
                    float(self.config.league_historical_weight),
                )
            )

        selected = _weighted_choice(choices)
        checkpoint_name = selected.get("checkpoint")
        if selected["source"] == "champion" or checkpoint_name in (None, latest_name):
            return self.policy, selected
        return _cached_policy(self.run_name, checkpoint_name), selected

    def _match_to_training_samples(self, match_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        trajectory_by_game = list(match_summary.get("trajectory_by_game") or [])
        if not trajectory_by_game:
            flat_trajectory = list(match_summary.get("trajectory") or [])
            if flat_trajectory:
                trajectory_by_game = [flat_trajectory]
        if not trajectory_by_game:
            return []
        return_value = match_summary["return_value"]
        selected_trajectory: List[Dict[str, Any]] = []

        for game_trajectory in trajectory_by_game:
            if not game_trajectory:
                continue
            sampled_game_trajectory = _sample_training_game_trajectory(
                game_trajectory,
                self.config.training_decisions_per_game,
            )
            for step in sampled_game_trajectory:
                step["return"] = return_value
                step["advantage"] = return_value - step["value"]
            selected_trajectory.extend(sampled_game_trajectory)

        if not selected_trajectory:
            return []

        advantages = [step["advantage"] for step in selected_trajectory]
        advantage_mean = _mean(advantages)
        advantage_std = _std(advantages)
        if advantage_std > 1e-6:
            for step in selected_trajectory:
                step["advantage"] = (step["advantage"] - advantage_mean) / advantage_std
        return selected_trajectory

    def _summarize_training_matches(self, match_summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not match_summaries:
            return {
                "wins": 0,
                "losses": 0,
                "games_played": 0,
                "return_value": 0.0,
                "duration_seconds": 0.0,
                "matches_played": 0,
                "opponents": [],
            }
        return {
            "wins": sum(int(match.get("wins_a", 0)) for match in match_summaries),
            "losses": sum(int(match.get("wins_b", 0)) for match in match_summaries),
            "games_played": sum(int(match.get("games_played", 0)) for match in match_summaries),
            "return_value": _mean([float(match.get("return_value", 0.0)) for match in match_summaries]),
            "duration_seconds": sum(float(match.get("duration_seconds", 0.0)) for match in match_summaries),
            "matches_played": len(match_summaries),
            "opponents": [match.get("opponent") for match in match_summaries],
        }

    def _record_live_match_progress(
        self,
        match_summary: Dict[str, Any],
        kind: str,
        completed_training_matches: Sequence[Dict[str, Any]],
    ) -> None:
        self.state["total_matches"] = int(self.state.get("total_matches", 0)) + 1
        self.state["total_games"] = int(self.state.get("total_games", 0)) + int(match_summary.get("games_played", 0))
        if kind == "training":
            self.state["last_match"] = self._summarize_training_matches(completed_training_matches)
            self._save_state_only()
        else:
            self.state["last_eval"] = {
                "wins": match_summary.get("wins_a", 0),
                "losses": match_summary.get("wins_b", 0),
                "games_played": match_summary.get("games_played", 0),
                "score": float(match_summary.get("wins_a", 0)) / max(int(match_summary.get("games_played", 0)), 1),
                "promoted": False,
                "action": "evaluating",
                "promoted_checkpoint": None,
                "champion_checkpoint": self.state.get("latest_checkpoint"),
                "duration_seconds": match_summary.get("duration_seconds", 0.0),
            }
            self._save()

    def _reseed_candidate(
        self,
        base_iteration: int,
        total_attempts: int,
        total_resets: int,
        total_promotions: int,
        last_score: float,
        last_result: str,
        reason: str,
        opponents: Sequence[Dict[str, Any]],
    ) -> None:
        self.candidate_policy = self.policy.clone()
        candidate_state = _default_candidate_state(
            self.state.get("latest_checkpoint", "checkpoint_000000.json"),
            base_iteration,
        )
        candidate_state["total_attempts"] = total_attempts
        candidate_state["resets"] = total_resets
        candidate_state["promotions"] = total_promotions
        candidate_state["last_score"] = last_score
        candidate_state["last_result"] = last_result
        candidate_state["last_reset_reason"] = reason
        candidate_state["last_opponents"] = list(opponents)
        candidate_state["rating_pass_elo"] = float(self.state.get("current_elo", INITIAL_ELO))
        self.state["candidate"] = candidate_state

    def _promote_candidate(self, iteration: int, eval_summary: Dict[str, Any], training_summary: Dict[str, Any]) -> str:
        previous_checkpoint = self.state.get("latest_checkpoint", "checkpoint_000000.json")
        previous_entry = _ensure_checkpoint_entry(
            self.state,
            previous_checkpoint,
            int(self.state.get("iteration", 0)),
            float(self.state.get("current_elo", INITIAL_ELO)),
        )
        previous_elo = float(previous_entry.get("elo", self.state.get("current_elo", INITIAL_ELO)))
        score = float(eval_summary.get("wins_a", 0)) / max(int(eval_summary.get("games_played", 0)), 1)
        new_candidate_elo, updated_previous_elo = _elo_update(previous_elo, previous_elo, score, self.config.elo_k)
        previous_entry["elo"] = updated_previous_elo
        previous_entry["last_defense"] = eval_summary

        checkpoint_path = _checkpoint_path(self.files, iteration)
        checkpoint_name = checkpoint_path.name
        _atomic_write_json(checkpoint_path, self.candidate_policy.to_dict(include_optimizer=False))

        promoted_at = _timestamp()
        promoted_entry = _ensure_checkpoint_entry(self.state, checkpoint_name, iteration, new_candidate_elo)
        promoted_entry["elo"] = new_candidate_elo
        promoted_entry["created_at"] = promoted_at
        promoted_entry["promoted_from"] = previous_checkpoint
        promoted_entry["promotion_eval"] = eval_summary
        promoted_entry["training_summary"] = training_summary

        self.policy = self.candidate_policy.clone()
        self.state["latest_checkpoint"] = checkpoint_name
        self.state["current_elo"] = new_candidate_elo
        self.state["last_promotion_iteration"] = int(iteration)
        self.state["last_promotion_at"] = promoted_at
        self.state["last_promotion_datetime"] = _format_timestamp(promoted_at)
        if new_candidate_elo >= float(self.state.get("best_elo", INITIAL_ELO)):
            self.state["best_elo"] = new_candidate_elo
            self.state["best_checkpoint"] = checkpoint_name
        return checkpoint_name

    def _play_training_matches(
        self,
        scheduled_config: TrainingConfig,
        next_iteration: int,
        simulation_pool: Optional[_SimulationPool] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
        match_count = max(1, int(scheduled_config.training_matches_per_iteration))
        games_per_match = max(1, int(scheduled_config.training_games_per_match))
        total_games = match_count * games_per_match
        worker_count = resolve_simulation_workers(scheduled_config.simulation_workers, total_games)
        if simulation_pool is not None:
            worker_count = min(max(1, int(simulation_pool.worker_count)), total_games)
        training_matches: List[Dict[str, Any]] = []
        training_samples: List[Dict[str, Any]] = []
        opponent_summaries: List[Dict[str, Any]] = []
        training_games_completed = 0
        training_sample_candidates = 0
        max_iteration_samples = DEFAULT_MAX_TRAINING_SAMPLES_PER_ITERATION
        opponent_exploration_plan = _training_opponent_exploration_plan(scheduled_config, match_count)
        _check_memory_safety(scheduled_config, "starting training match collection")

        def record_completed_match(match_summary: Dict[str, Any], opponent_meta: Dict[str, Any]) -> None:
            nonlocal training_sample_candidates
            match_summary["opponent"] = opponent_meta
            samples = self._match_to_training_samples(match_summary)
            training_sample_candidates = _extend_sample_reservoir(
                training_samples,
                samples,
                max_samples=max_iteration_samples,
                seen_count=training_sample_candidates,
            )
            compact_summary = _compact_training_match_summary(match_summary)
            training_matches.append(compact_summary)
            opponent_summaries.append(opponent_meta)
            if len(training_matches) % MEMORY_SAFETY_CHECK_MATCH_INTERVAL == 0:
                _check_memory_safety(
                    scheduled_config,
                    f"collecting training match {len(training_matches)} of {match_count}",
                )
            self._record_live_match_progress(compact_summary, "training", training_matches)

        if worker_count <= 1:
            for match_index in range(match_count):
                opponent_policy, opponent_meta = self._sample_league_opponent()
                opponent_style = opponent_exploration_plan[match_index]
                opponent_meta = {
                    **opponent_meta,
                    "exploration": _copy_nested(opponent_style),
                }
                completed_before_match = training_games_completed
                match_summary = _play_match(
                    self.candidate_policy,
                    opponent_policy,
                    scheduled_config,
                    collect_a=True,
                    deterministic_a=False,
                    deterministic_b=False,
                    games_per_match=games_per_match,
                    temperature_b=float(opponent_style["temperature"]),
                    epsilon_random_b=float(opponent_style["epsilon_random"]),
                    game_progress_callback=lambda progress, match_index=match_index, completed_before_match=completed_before_match: self._set_live_progress(
                        self._build_training_live_progress(
                            scheduled_config,
                            stage="training",
                            iteration_number=next_iteration,
                            training_games_completed=completed_before_match + int(progress.get("games_completed", 0)),
                            promotion_games_completed=0,
                            training_matches_completed=match_index,
                            training_stage_complete=False,
                            promotion_stage_complete=False,
                        )
                    ),
                )
                training_games_completed += int(match_summary.get("games_played", 0))
                record_completed_match(match_summary, opponent_meta)
            return (
                training_matches,
                training_samples,
                opponent_summaries,
                training_games_completed,
                training_sample_candidates,
            )

        candidate_payload = _policy_payload_for_simulation(self.candidate_policy)
        config_payload = _config_payload_for_simulation(scheduled_config)
        candidate_cache_key = f"{self.run_name}:iteration:{next_iteration}:candidate:training"
        target_tasks = min(total_games, max(worker_count, worker_count * SIMULATION_TASKS_PER_WORKER))
        if match_count >= worker_count:
            chunk_size = games_per_match
        else:
            chunk_size = max(1, math.ceil(total_games / max(1, target_tasks)))
        chunks_per_match = max(1, math.ceil(games_per_match / max(1, chunk_size)))
        total_task_count = max(1, match_count * chunks_per_match)
        max_in_flight = _max_in_flight_simulation_tasks(worker_count, total_task_count)

        expected_chunks_by_match: Dict[int, int] = {}
        opponent_by_match: Dict[int, Dict[str, Any]] = {}
        partials_by_match: Dict[int, List[Dict[str, Any]]] = {}
        completed_matches = 0
        completed_from_results = 0
        progress_queue = simulation_pool.progress_queue if simulation_pool is not None else None
        if progress_queue is not None:
            _drain_simulation_progress_queue(progress_queue)

        def iter_training_tasks() -> Any:
            for match_index in range(match_count):
                opponent_policy, opponent_meta = self._sample_league_opponent()
                opponent_style = opponent_exploration_plan[match_index]
                opponent_meta = {
                    **opponent_meta,
                    "exploration": _copy_nested(opponent_style),
                }
                opponent_payload = _policy_payload_for_simulation(opponent_policy)
                opponent_cache_key = f"{self.run_name}:iteration:{next_iteration}:opponent:{match_index}"
                opponent_by_match[match_index] = opponent_meta
                expected_chunks_by_match[match_index] = chunks_per_match
                remaining_games = games_per_match
                while remaining_games > 0:
                    games_in_chunk = min(chunk_size, remaining_games)
                    remaining_games -= games_in_chunk
                    yield {
                        "match_index": match_index,
                        "policy_a_payload": candidate_payload,
                        "policy_b_payload": opponent_payload,
                        "policy_a_cache_key": candidate_cache_key,
                        "policy_b_cache_key": opponent_cache_key,
                        "config_payload": config_payload,
                        "game_count": games_in_chunk,
                        "seed": _random_seed(),
                        "collect_a": True,
                        "deterministic_a": False,
                        "deterministic_b": False,
                        "temperature_b": float(opponent_style["temperature"]),
                        "epsilon_random_b": float(opponent_style["epsilon_random"]),
                        "opponent": opponent_meta,
                    }

        def persist_training_progress() -> None:
            self._set_live_progress(
                self._build_training_live_progress(
                    scheduled_config,
                    stage="training",
                    iteration_number=next_iteration,
                    training_games_completed=training_games_completed,
                    promotion_games_completed=0,
                    training_matches_completed=completed_matches,
                    training_stage_complete=False,
                    promotion_stage_complete=False,
                )
            )

        def on_progress(delta: int) -> None:
            nonlocal training_games_completed
            training_games_completed = min(total_games, training_games_completed + int(delta))
            persist_training_progress()

        def on_result(partial: Dict[str, Any]) -> None:
            nonlocal training_games_completed, completed_matches, completed_from_results
            match_index = int(partial["match_index"])
            partials = partials_by_match.setdefault(match_index, [])
            partials.append(partial)
            completed_from_results += int(partial.get("games_played", 0))
            training_games_completed = max(training_games_completed, min(total_games, completed_from_results))

            if len(partials) >= expected_chunks_by_match.get(match_index, 0):
                match_summary = _combine_match_partials(partials)
                opponent_meta = opponent_by_match.pop(match_index, {})
                record_completed_match(match_summary, opponent_meta)
                partials.clear()
                partials_by_match.pop(match_index, None)
                expected_chunks_by_match.pop(match_index, None)
                completed_matches += 1

            persist_training_progress()

        if simulation_pool is not None:
            simulation_pool.wait_until_ready()
            _consume_simulation_task_iterator(
                simulation_pool.executor,
                _simulate_payload_match_chunk_worker,
                iter_training_tasks(),
                max_in_flight=max_in_flight,
                progress_queue=progress_queue,
                progress_callback=on_progress,
                result_callback=on_result,
            )
            _drain_simulation_progress_queue(progress_queue)
            return (
                training_matches,
                training_samples,
                opponent_summaries,
                training_games_completed,
                training_sample_candidates,
            )

        progress_context, progress_queue = _create_simulation_progress_queue()
        try:
            with _process_pool_executor(
                max_workers=worker_count,
                initializer=_simulation_progress_worker_init,
                initargs=(progress_queue,),
                mp_context=progress_context,
            ) as executor:
                _consume_simulation_task_iterator(
                    executor,
                    _simulate_payload_match_chunk_worker,
                    iter_training_tasks(),
                    max_in_flight=max_in_flight,
                    progress_queue=progress_queue,
                    progress_callback=on_progress,
                    result_callback=on_result,
                )
        finally:
            _close_simulation_progress_queue(progress_queue)

        return (
            training_matches,
            training_samples,
            opponent_summaries,
            training_games_completed,
            training_sample_candidates,
        )

    def train_iteration(self) -> Dict[str, Any]:
        scheduled_config = self._scheduled_training_config()
        _check_memory_safety(scheduled_config, "starting training iteration")
        candidate_state = self._candidate_state()
        next_iteration = int(self.state.get("iteration", 0)) + 1
        training_games_total = (
            max(1, int(scheduled_config.training_matches_per_iteration))
            * max(1, int(scheduled_config.training_games_per_match))
        )
        promotion_games_total = max(1, int(scheduled_config.promotion_games))
        simulation_pool = self._get_simulation_pool(
            resolve_simulation_workers(
                scheduled_config.simulation_workers,
                training_games_total,
                allocation_key=self.run_name,
            )
        )

        self._set_live_progress(
            self._build_training_live_progress(
                scheduled_config,
                stage="training",
                iteration_number=next_iteration,
                training_games_completed=0,
                promotion_games_completed=0,
                training_matches_completed=0,
                training_stage_complete=False,
                promotion_stage_complete=False,
            ),
            force_persist=True,
        )

        (
            training_matches,
            training_samples,
            opponent_summaries,
            training_games_completed,
            training_sample_candidates,
        ) = self._play_training_matches(
            scheduled_config,
            next_iteration,
            simulation_pool=simulation_pool,
        )

        self._set_live_progress(
            self._build_training_live_progress(
                scheduled_config,
                stage="optimizing",
                iteration_number=next_iteration,
                training_games_completed=training_games_completed,
                promotion_games_completed=0,
                training_matches_completed=len(training_matches),
                training_stage_complete=True,
                promotion_stage_complete=False,
            ),
            force_persist=True,
        )
        _check_memory_safety(scheduled_config, "starting PPO optimization")
        update_stats = self.candidate_policy.train_on_samples(training_samples, scheduled_config)
        training_summary = self._summarize_training_matches(training_matches)
        self._set_live_progress(
            self._build_training_live_progress(
                scheduled_config,
                stage="promotion",
                iteration_number=next_iteration,
                training_games_completed=training_games_completed,
                promotion_games_completed=0,
                training_matches_completed=len(training_matches),
                training_stage_complete=True,
                promotion_stage_complete=False,
            ),
            force_persist=True,
        )
        simulation_pool = self._get_simulation_pool(
            resolve_simulation_workers(
                scheduled_config.simulation_workers,
                promotion_games_total,
                allocation_key=self.run_name,
            )
        )
        _check_memory_safety(scheduled_config, "starting promotion evaluation")
        eval_summary = _play_match(
            self.candidate_policy,
            self.policy,
            scheduled_config,
            collect_a=False,
            deterministic_a=True,
            deterministic_b=True,
            games_per_match=promotion_games_total,
            game_progress_callback=lambda progress: self._set_live_progress(
                self._build_training_live_progress(
                    scheduled_config,
                    stage="promotion",
                    iteration_number=next_iteration,
                    training_games_completed=training_games_completed,
                    promotion_games_completed=int(progress.get("games_completed", 0)),
                    training_matches_completed=len(training_matches),
                    training_stage_complete=True,
                    promotion_stage_complete=False,
                )
            ),
            simulation_pool=simulation_pool,
            cache_key_prefix=f"{self.run_name}:iteration:{next_iteration}:promotion",
        )
        self._record_live_match_progress(eval_summary, "evaluation", training_matches)
        candidate_score = float(eval_summary["wins_a"]) / max(int(eval_summary["games_played"]), 1)
        promoted = (
            eval_summary["wins_a"] > eval_summary["wins_b"]
            and candidate_score >= float(scheduled_config.promotion_score_threshold)
        )
        defending_champion = self.state.get("latest_checkpoint")
        total_attempts = int(candidate_state.get("total_attempts", 0)) + 1
        total_resets = int(candidate_state.get("resets", 0))
        total_promotions = int(candidate_state.get("promotions", 0))
        action = "keep_training"
        promoted_checkpoint: Optional[str] = None

        self.state["iteration"] = next_iteration
        self.state["last_match"] = training_summary

        self.state["last_update"] = {
            **update_stats,
            "learning_rate": scheduled_config.learning_rate,
            "epsilon_random": scheduled_config.epsilon_random,
            "train_temperature": scheduled_config.train_temperature,
            "ppo_clip": scheduled_config.ppo_clip,
            "simulation_workers": int(scheduled_config.simulation_workers),
            "resolved_simulation_workers": resolve_simulation_workers(
                scheduled_config.simulation_workers,
                allocation_key=self.run_name,
            ),
            "active_simulation_runs": active_simulation_runs(),
            "iterations_since_promotion": self._iterations_since_promotion(),
            "matches_collected": len(training_matches),
            "samples_collected": len(training_samples),
            "sample_candidates": int(training_sample_candidates),
            "max_training_samples_per_game": DEFAULT_MAX_TRAINING_SAMPLES_PER_GAME,
            "sample_reservoir_limit": DEFAULT_MAX_TRAINING_SAMPLES_PER_ITERATION,
        }

        if promoted:
            promoted_checkpoint = self._promote_candidate(next_iteration, eval_summary, training_summary)
            total_promotions += 1
            self.state["promotions"] = int(self.state.get("promotions", 0)) + 1
            action = "promoted"
            self._reseed_candidate(
                next_iteration,
                total_attempts,
                total_resets,
                total_promotions,
                candidate_score,
                "promoted",
                f"Promoted to {promoted_checkpoint}",
                opponent_summaries,
            )
        else:
            attempts_since_reset = int(candidate_state.get("attempts_since_reset", 0)) + 1
            should_reset = (
                candidate_score <= float(scheduled_config.candidate_reset_threshold)
                or attempts_since_reset >= max(1, int(scheduled_config.candidate_patience))
            )
            if should_reset:
                total_resets += 1
                reason = (
                    f"Candidate reset after score {candidate_score:.3f} "
                    f"and {attempts_since_reset} attempt(s) since the last reset."
                )
                action = "reset_candidate"
                self._reseed_candidate(
                    next_iteration,
                    total_attempts,
                    total_resets,
                    total_promotions,
                    candidate_score,
                    "reset",
                    reason,
                    opponent_summaries,
                )
            else:
                candidate_state["attempts_since_reset"] = attempts_since_reset
                candidate_state["total_attempts"] = total_attempts
                candidate_state["last_score"] = candidate_score
                candidate_state["last_result"] = "keep_training"
                candidate_state["last_opponents"] = opponent_summaries

        self.state["last_eval"] = {
            "wins": eval_summary["wins_a"],
            "losses": eval_summary["wins_b"],
            "games_played": eval_summary["games_played"],
            "score": candidate_score,
            "promoted": promoted,
            "action": action,
            "promoted_checkpoint": promoted_checkpoint,
            "champion_checkpoint": defending_champion,
            "duration_seconds": eval_summary["duration_seconds"],
        }
        self.state["last_error"] = None
        self._set_live_progress(
            self._build_training_live_progress(
                scheduled_config,
                stage="complete",
                iteration_number=next_iteration,
                training_games_completed=training_games_completed,
                promotion_games_completed=int(eval_summary.get("games_played", 0)),
                training_matches_completed=len(training_matches),
                training_stage_complete=True,
                promotion_stage_complete=True,
            ),
            force_persist=True,
        )
        self._save()
        return {
            "match": training_summary,
            "evaluation": self.state["last_eval"],
            "update": update_stats,
            "iteration": self.state["iteration"],
            "current_elo": self.state.get("current_elo", INITIAL_ELO),
            "action": action,
        }

    def _training_loop(self) -> None:
        active_simulation_key = _register_active_simulation_run(
            self.run_name,
            self.config.simulation_workers,
        )
        with self._lock:
            self.state["status"] = "training"
            self.state["last_error"] = None
            self._save()
        target_iterations = self.config.max_iterations
        try:
            while not self._stop_event.is_set():
                if target_iterations is not None and self.state["iteration"] >= target_iterations:
                    break
                self.train_iteration()
        except Exception as exc:
            with self._lock:
                self.state["status"] = "error"
                self.state["last_error"] = f"{type(exc).__name__}: {exc}"
                self._save()
        finally:
            self._close_simulation_pool()
            _unregister_active_simulation_run(active_simulation_key)
            with self._lock:
                if self.state.get("status") != "error":
                    self.state["status"] = "idle"
                self._training_active = False
                if self._thread is threading.current_thread():
                    self._thread = None
                self._save()

    def start(self, background: bool = True) -> Dict[str, Any]:
        with self._lock:
            if self.is_running:
                return {"started": False, "message": "training already running", "status": self.state["status"]}
            self._stop_event.clear()
            self._training_active = True
            if background:
                self._thread = threading.Thread(
                    target=self._training_loop,
                    name=f"starrealms-selfplay-{self.run_name}",
                    daemon=True,
                )
                self._thread.start()
                return {"started": True, "background": True, "run_name": self.run_name}
        self._training_loop()
        return {"started": True, "background": False, "run_name": self.run_name}

    def interrupt(self) -> Dict[str, Any]:
        thread_to_join: Optional[threading.Thread] = None
        with self._lock:
            if not self.is_running:
                self.state["status"] = "idle"
                self._save()
                return {"stopped": False, "message": "training was not running"}
            self.state["status"] = "stop_requested"
            self._save()
            self._stop_event.set()
            thread_to_join = self._thread
        if thread_to_join is not None and thread_to_join is not threading.current_thread():
            thread_to_join.join()
        with self._lock:
            if thread_to_join is not None and not thread_to_join.is_alive() and self._thread is thread_to_join:
                self._thread = None
        return {"stopped": True, "message": "training stopped after the current match"}

    def progress_summary(self) -> Dict[str, Any]:
        runtime = 0.0
        if "created_at" in self.state and self.state["created_at"] is not None:
            runtime = _timestamp() - float(self.state["created_at"])
        scheduled_config = self._scheduled_training_config()
        device_plan = self.policy.device_summary()
        return {
            "run_name": self.run_name,
            "status": self.state.get("status", "idle"),
            "created_at": self.state.get("created_at"),
            "created_datetime": self.state.get("created_datetime") or _format_timestamp(self.state.get("created_at")),
            "last_promotion_at": self.state.get("last_promotion_at"),
            "last_promotion_datetime": self.state.get("last_promotion_datetime")
            or _format_timestamp(self.state.get("last_promotion_at")),
            "forked_from": self.state.get("forked_from"),
            "fork_origin": _fork_origin_label(self.state.get("forked_from")),
            "iteration": self.state.get("iteration", 0),
            "total_matches": self.state.get("total_matches", 0),
            "total_games": self.state.get("total_games", 0),
            "current_elo": round(float(self.state.get("current_elo", INITIAL_ELO)), 2),
            "best_elo": round(float(self.state.get("best_elo", INITIAL_ELO)), 2),
            "best_checkpoint": self.state.get("best_checkpoint"),
            "latest_checkpoint": self.state.get("latest_checkpoint"),
            "promotions": int(self.state.get("promotions", 0)),
            "candidate": self.state.get("candidate"),
            "last_match": self.state.get("last_match"),
            "last_update": self.state.get("last_update"),
            "last_eval": self.state.get("last_eval"),
            "last_rating_pass": self.state.get("last_rating_pass"),
            "live_progress": self.state.get("live_progress"),
            "last_error": self.state.get("last_error"),
            "runtime": _format_seconds(runtime),
            "run_dir": str(self.files.run_dir),
            "device_backend": str(device_plan.get("backend", DEVICE_CPU)),
            "device_repr": str(device_plan.get("repr", "cpu")),
            "device_preference": self.config.device_preference,
            "device_requested_backend": str(device_plan.get("requested_backend", self.config.device_preference)),
            "device_reason": str(device_plan.get("reason", "")),
            "device_benchmark": _copy_nested(device_plan.get("benchmark")),
            "simulation_workers": int(self.config.simulation_workers),
            "resolved_simulation_workers": resolve_simulation_workers(
                self.config.simulation_workers,
                allocation_key=self.run_name,
            ),
            "active_simulation_runs": active_simulation_runs(),
            "logical_processors": int(os.cpu_count() or 1),
            "last_promotion_iteration": self._last_promotion_iteration(),
            "iterations_since_promotion": self._iterations_since_promotion(),
            "scheduled_learning_rate": float(scheduled_config.learning_rate),
            "scheduled_epsilon_random": float(scheduled_config.epsilon_random),
            "scheduled_train_temperature": float(scheduled_config.train_temperature),
        }


_TRAINERS: Dict[str, SelfPlayTrainer] = {}
_POLICY_CACHE: Dict[Tuple[str, str], Tuple[float, PolicyNetwork]] = {}
_TRAINERS_LOCK = threading.RLock()
_POLICY_CACHE_LOCK = threading.Lock()


def _get_trainer(run_name: str = LATEST_RUN_NAME, config_overrides: Optional[Dict[str, Any]] = None) -> SelfPlayTrainer:
    with _TRAINERS_LOCK:
        trainer = _TRAINERS.get(run_name)
        if trainer is not None:
            if config_overrides and trainer.config != trainer.config.merged(config_overrides):
                if trainer.is_running:
                    raise RuntimeError(
                        f"Run '{run_name}' is currently training. Interrupt it before changing run settings."
                    )
                trainer = SelfPlayTrainer(run_name, config_overrides=config_overrides)
                _TRAINERS[run_name] = trainer
            return trainer
        trainer = SelfPlayTrainer(run_name, config_overrides=config_overrides)
        _TRAINERS[run_name] = trainer
        return trainer


def summarize_start_training(run_name: str = LATEST_RUN_NAME, **config_overrides: Any) -> str:
    trainer = _get_trainer(run_name, config_overrides or None)
    started = trainer.start(background=True)
    summary = trainer.progress_summary()
    if started.get("started") and started.get("background"):
        return (
            f"Started background training for '{run_name}'. "
            f"Iteration {summary['iteration']}, status {summary['status']}, policy dir {summary['run_dir']}."
        )
    return f"Training for '{run_name}' is already running."


def start_training(
    run_name: str = LATEST_RUN_NAME,
    background: bool = True,
    **config_overrides: Any,
) -> Dict[str, Any]:
    trainer = _get_trainer(run_name, config_overrides or None)
    trainer.start(background=background)
    return trainer.progress_summary()


def summarize_training_progress(run_name: str = LATEST_RUN_NAME) -> str:
    trainer = _get_trainer(run_name)
    summary = trainer.progress_summary()
    last_match = summary.get("last_match") or {}
    last_eval = summary.get("last_eval") or {}
    match_text = "no completed matches yet"
    if last_match:
        match_text = (
            f"last match {last_match.get('wins', 0)}-{last_match.get('losses', 0)} "
            f"over {last_match.get('games_played', 0)} games"
        )
    eval_text = ""
    if last_eval:
        eval_text = (
            f" Last eval {last_eval.get('wins', 0)}-{last_eval.get('losses', 0)} "
            f"({last_eval.get('action', 'keep_training')})."
        )
    return (
        f"Run '{summary['run_name']}' is {summary['status']}. "
        f"Iteration {summary['iteration']}, total matches {summary['total_matches']}, "
        f"Elo {summary['current_elo']} (best {summary['best_elo']}), promotions {summary.get('promotions', 0)}, "
        f"device {summary.get('device_backend', DEVICE_CPU)}, "
        f"{match_text}.{eval_text} "
        f"Artifacts: {summary['run_dir']}."
    )


def training_progress(run_name: str = LATEST_RUN_NAME) -> Dict[str, Any]:
    return _get_trainer(run_name).progress_summary()


def interrupt_training(run_name: str = LATEST_RUN_NAME) -> str:
    trainer = _get_trainer(run_name)
    result = trainer.interrupt()
    if result["stopped"]:
        return f"Training for '{run_name}' has stopped. Latest weights are saved in {trainer.files.run_dir}."
    return f"Training for '{run_name}' was not running."


def continue_training(run_name: str = LATEST_RUN_NAME, **config_overrides: Any) -> str:
    trainer = _get_trainer(run_name, config_overrides or None)
    trainer.start(background=True)
    return summarize_training_progress(run_name)


def train_iterations(
    iterations: int,
    run_name: str = LATEST_RUN_NAME,
    **config_overrides: Any,
) -> Dict[str, Any]:
    trainer = _get_trainer(run_name, config_overrides or None)
    original_max = trainer.config.max_iterations
    trainer.config.max_iterations = trainer.state.get("iteration", 0) + max(0, iterations)
    try:
        trainer.start(background=False)
        return trainer.progress_summary()
    finally:
        trainer.config.max_iterations = original_max


def create_run(
    run_name: str,
    model_type: str = MODEL_TYPE_DEEP,
    device_preference: str = DEVICE_AUTO,
    training_matches_per_iteration: int = 5,
    training_games_per_match: int = 16,
    training_decisions_per_game: str = TRAINING_DECISIONS_PER_GAME_ALL,
    train_temperature: float = 0.9,
    promotion_games: int = 24,
    promotion_score_threshold: float = 0.6,
    simulation_workers: int = SIMULATION_WORKERS_AUTO,
) -> Dict[str, Any]:
    if not str(run_name).strip():
        raise ValueError("run_name is required.")
    if run_exists(run_name):
        raise ValueError(f"Run '{run_name}' already exists.")

    overrides = new_run_overrides(
        model_type=model_type,
        device_preference=device_preference,
        training_matches_per_iteration=training_matches_per_iteration,
        training_games_per_match=training_games_per_match,
        training_decisions_per_game=training_decisions_per_game,
        train_temperature=train_temperature,
        promotion_games=promotion_games,
        promotion_score_threshold=promotion_score_threshold,
        simulation_workers=simulation_workers,
    )
    _ensure_run(run_name, config_overrides=overrides)
    trainer = _get_trainer(run_name, overrides)
    return trainer.progress_summary()


def create_run_from_checkpoint(
    run_name: str,
    source_run_name: str,
    source_checkpoint: str = "latest",
    model_type: str = MODEL_TYPE_DEEP,
    device_preference: str = DEVICE_AUTO,
    training_matches_per_iteration: int = 5,
    training_games_per_match: int = 16,
    training_decisions_per_game: str = TRAINING_DECISIONS_PER_GAME_ALL,
    train_temperature: float = 0.9,
    promotion_games: int = 24,
    promotion_score_threshold: float = 0.6,
    simulation_workers: int = SIMULATION_WORKERS_AUTO,
) -> Dict[str, Any]:
    if not str(run_name).strip():
        raise ValueError("run_name is required.")
    if run_exists(run_name):
        raise ValueError(f"Run '{run_name}' already exists.")
    if not run_exists(source_run_name):
        raise ValueError(f"Source run '{source_run_name}' does not exist.")

    source_policy = load_policy(source_run_name, source_checkpoint)
    source_state = get_run_state(source_run_name)
    resolved_source_checkpoint = _resolve_checkpoint_name(source_state, source_checkpoint) or source_checkpoint
    target_model = model_type_overrides(model_type)

    config = TrainingConfig().merged(
        {
            "hidden_size": int(target_model["hidden_size"]),
            "model_architecture": str(target_model["model_architecture"]),
            "device_preference": str(device_preference or DEVICE_AUTO).strip().lower(),
            "training_matches_per_iteration": int(training_matches_per_iteration),
            "training_games_per_match": int(training_games_per_match),
            "training_decisions_per_game": normalize_training_decisions_per_game(training_decisions_per_game),
            "promotion_games": int(promotion_games),
            "promotion_score_threshold": normalize_promotion_score_threshold(promotion_score_threshold),
            "simulation_workers": normalize_simulation_workers(simulation_workers),
        }
    )
    config = config.merged(derive_temperature_schedule_overrides(train_temperature))

    if config.training_matches_per_iteration <= 0:
        raise ValueError("training_matches_per_iteration must be positive.")
    if config.training_games_per_match <= 0:
        raise ValueError("training_games_per_match must be positive.")
    if config.promotion_games <= 0:
        raise ValueError("promotion_games must be positive.")

    files = RunFiles(run_name)
    files.run_dir.mkdir(parents=True, exist_ok=True)
    files.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    fork_policy, conversion_mode = _convert_policy_architecture(
        source_policy,
        target_architecture=config.model_architecture,
        target_hidden_size=config.hidden_size,
        target_device_preference=config.device_preference,
    )
    fork_policy._reset_optimizer(config.learning_rate)

    state = _default_training_state(config)
    state["run_name"] = run_name
    state["forked_from"] = {
        "run_name": source_run_name,
        "checkpoint": source_checkpoint,
        "resolved_checkpoint": resolved_source_checkpoint,
        "source_architecture": source_policy.architecture,
        "source_hidden_size": int(source_policy.hidden_size),
        "target_architecture": config.model_architecture,
        "target_hidden_size": int(config.hidden_size),
        "requested_model_type": str(model_type or MODEL_TYPE_DEEP).strip().lower(),
        "conversion_mode": conversion_mode,
        "created_at": _timestamp(),
    }
    if state.get("checkpoints"):
        state["checkpoints"][0]["note"] = f"forked from {source_run_name}/{source_checkpoint}"
        state["checkpoints"][0]["source_checkpoint"] = resolved_source_checkpoint
    state["device_plan"] = fork_policy.device_summary()

    _atomic_write_json(files.policy_file, fork_policy.to_dict(include_optimizer=True))
    _atomic_write_json(files.candidate_policy_file, fork_policy.to_dict(include_optimizer=True))
    _atomic_write_json(files.training_state_file, state)
    _atomic_write_json(files.checkpoints_dir / "checkpoint_000000.json", fork_policy.to_dict(include_optimizer=False))

    trainer = SelfPlayTrainer(run_name)
    with _TRAINERS_LOCK:
        _TRAINERS[run_name] = trainer
    return trainer.progress_summary()


def update_run_config(
    run_name: str,
    train_temperature: Optional[float] = None,
    promotion_score_threshold: Optional[float] = None,
    simulation_workers: Optional[int] = None,
) -> Dict[str, Any]:
    if not run_exists(run_name):
        raise ValueError(f"Run '{run_name}' does not exist.")

    trainer = _get_trainer(run_name)
    if trainer.is_running or str(trainer.state.get("status", "idle")).lower() in {"training", "stop_requested", "rating"}:
        raise RuntimeError(f"Run '{run_name}' is currently training. Interrupt it before changing run settings.")

    overrides: Dict[str, Any] = {}
    if train_temperature is not None:
        overrides.update(derive_temperature_schedule_overrides(train_temperature))
    if promotion_score_threshold is not None:
        overrides["promotion_score_threshold"] = normalize_promotion_score_threshold(promotion_score_threshold)
    if simulation_workers is not None:
        overrides["simulation_workers"] = normalize_simulation_workers(simulation_workers)
    if not overrides:
        return trainer.progress_summary()

    trainer.config = trainer.config.merged(overrides)
    trainer._save()
    return trainer.progress_summary()


def run_rating_pass(
    run_name: str = LATEST_RUN_NAME,
    max_policies: int = 8,
    games_per_pair: int = 12,
    include_candidate: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    trainer = _get_trainer(run_name)
    return trainer.run_rating_pass(
        max_policies=max_policies,
        games_per_pair=games_per_pair,
        include_candidate=include_candidate,
        progress_callback=progress_callback,
    )


def summarize_rating_pass(
    run_name: str = LATEST_RUN_NAME,
    max_policies: int = 8,
    games_per_pair: int = 12,
    include_candidate: bool = True,
) -> str:
    summary = run_rating_pass(
        run_name=run_name,
        max_policies=max_policies,
        games_per_pair=games_per_pair,
        include_candidate=include_candidate,
    )
    top_entry = (summary.get("leaderboard") or [{}])[0]
    return (
        f"Rating pass for '{run_name}' completed with {summary.get('participant_count', 0)} models "
        f"and {summary.get('pairings_played', 0)} pairings. "
        f"Top rated: {top_entry.get('name', '-')} at {round(float(top_entry.get('elo', INITIAL_ELO)), 2)} Elo."
    )


_CROSS_RUN_RATING_LOCK = threading.Lock()


def _default_cross_run_rating_state() -> Dict[str, Any]:
    now = _timestamp()
    return {
        "version": CROSS_RUN_RATING_VERSION,
        "created_at": now,
        "created_datetime": _format_timestamp(now),
        "updated_at": now,
        "rating_model": CROSS_RUN_RATING_MODEL,
        "total_games": 0,
        "ratings": {},
        "pairings": {},
        "last_calibration": None,
    }


def _load_cross_run_rating_state() -> Dict[str, Any]:
    state = _load_cached_json_or_none(CROSS_RUN_RATINGS_FILE) or _default_cross_run_rating_state()
    state["version"] = int(state.get("version", CROSS_RUN_RATING_VERSION) or CROSS_RUN_RATING_VERSION)
    state.setdefault("created_at", _timestamp())
    state["created_datetime"] = state.get("created_datetime") or _format_timestamp(state.get("created_at"))
    state.setdefault("updated_at", state.get("created_at"))
    state["rating_model"] = CROSS_RUN_RATING_MODEL
    state.setdefault("total_games", 0)
    state.setdefault("ratings", {})
    state.setdefault("pairings", {})
    state.setdefault("last_calibration", None)
    return state


def _save_cross_run_rating_state(state: Dict[str, Any]) -> None:
    state["version"] = CROSS_RUN_RATING_VERSION
    state["rating_model"] = CROSS_RUN_RATING_MODEL
    state["updated_at"] = _timestamp()
    _atomic_write_json(CROSS_RUN_RATINGS_FILE, state)


def _cross_run_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return number


def _clamp_cross_run_rd(rating_deviation: Any) -> float:
    return _clip(
        _cross_run_float(rating_deviation, CROSS_RUN_GLICKO2_DEFAULT_RD),
        CROSS_RUN_GLICKO2_MIN_RD,
        CROSS_RUN_GLICKO2_MAX_RD,
    )


def _clamp_cross_run_volatility(volatility: Any) -> float:
    return _clip(
        _cross_run_float(volatility, CROSS_RUN_GLICKO2_DEFAULT_VOLATILITY),
        1e-6,
        1.2,
    )


def _cross_run_legacy_rating_deviation(existing: Dict[str, Any]) -> float:
    if "rating_deviation" in existing:
        return _clamp_cross_run_rd(existing.get("rating_deviation"))

    confidence = _cross_run_float(existing.get("confidence_radius"), 0.0)
    if confidence <= 0.0:
        try:
            games = int(existing.get("games", 0) or 0)
        except (TypeError, ValueError):
            games = 0
        if games > 0:
            confidence = CROSS_RUN_LEGACY_CONFIDENCE_95_SCALE / math.sqrt(float(games))

    if confidence > 0.0:
        return _clamp_cross_run_rd(confidence / CROSS_RUN_CONFIDENCE_SIGMAS)
    return CROSS_RUN_GLICKO2_DEFAULT_RD


def _cross_run_confidence_radius_from_entry(entry: Dict[str, Any]) -> Optional[float]:
    rating_deviation = _clamp_cross_run_rd(entry.get("rating_deviation"))
    return float(rating_deviation) * CROSS_RUN_CONFIDENCE_SIGMAS


def _cross_run_inherited_rating_deviation(source_entry: Dict[str, Any]) -> float:
    if "rating_deviation" in source_entry:
        source_rd = _clamp_cross_run_rd(source_entry.get("rating_deviation"))
    else:
        source_rd = _cross_run_legacy_rating_deviation(source_entry)
    return _clamp_cross_run_rd(max(source_rd, CROSS_RUN_GLICKO2_INHERITED_RD_FLOOR))


def _run_state_checkpoint_entry(state: Dict[str, Any], checkpoint_name: str) -> Optional[Dict[str, Any]]:
    resolved_name = str(checkpoint_name or "").strip()
    if not resolved_name:
        return None
    for item in list(state.get("checkpoints") or []):
        if str(item.get("name") or "") == resolved_name:
            return dict(item)
    return None


def _cross_run_seed_from_entry(
    source_entry: Dict[str, Any],
    *,
    source_type: str,
    source_run_name: str,
    source_checkpoint: str,
    event_at: Optional[float],
) -> Dict[str, Any]:
    return {
        "event_at": 0.0 if event_at is None else float(event_at),
        "elo": _cross_run_float(source_entry.get("elo"), INITIAL_ELO),
        "rating_deviation": _cross_run_inherited_rating_deviation(source_entry),
        "volatility": _clamp_cross_run_volatility(source_entry.get("volatility")),
        "source": {
            "type": source_type,
            "run_name": str(source_run_name),
            "checkpoint": str(source_checkpoint),
            "event_at": event_at,
            "event_datetime": _format_timestamp(event_at),
        },
    }


def _cross_run_parent_rating_seed(
    state: Dict[str, Any],
    participant: Dict[str, Any],
    checkpoint: str,
    existing: Dict[str, Any],
    previous_checkpoint: str,
) -> Optional[Dict[str, Any]]:
    run_name = str(participant.get("run_name") or "").strip()
    if not run_name:
        return None
    run_state = get_run_state(run_name)
    checkpoint_entry = _run_state_checkpoint_entry(run_state, checkpoint)
    candidates: List[Dict[str, Any]] = []

    promoted_from = str((checkpoint_entry or {}).get("promoted_from") or "").strip()
    if promoted_from and existing and previous_checkpoint == promoted_from:
        candidates.append(
            _cross_run_seed_from_entry(
                existing,
                source_type="promoted_from",
                source_run_name=run_name,
                source_checkpoint=promoted_from,
                event_at=_optional_timestamp((checkpoint_entry or {}).get("created_at")),
            )
        )

    forked_from = run_state.get("forked_from")
    if isinstance(forked_from, dict):
        source_run_name = str(forked_from.get("run_name") or "").strip()
        source_checkpoint = str(
            forked_from.get("resolved_checkpoint")
            or forked_from.get("checkpoint")
            or ""
        ).strip()
        source_entry = dict((state.get("ratings") or {}).get(source_run_name) or {})
        source_entry_checkpoint = str(source_entry.get("best_checkpoint") or "").strip()
        if source_entry and source_run_name and (
            not source_checkpoint
            or not source_entry_checkpoint
            or source_checkpoint == source_entry_checkpoint
        ):
            candidates.append(
                _cross_run_seed_from_entry(
                    source_entry,
                    source_type="forked_from",
                    source_run_name=source_run_name,
                    source_checkpoint=source_checkpoint or source_entry_checkpoint,
                    event_at=_optional_timestamp(forked_from.get("created_at"))
                    or _optional_timestamp(run_state.get("created_at")),
                )
            )

    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item.get("event_at", 0.0)))


def _glicko2_reduce_impact(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + (3.0 * phi * phi) / (math.pi * math.pi))


def _glicko2_expected_score(mu: float, opponent_mu: float, opponent_phi: float) -> float:
    return 1.0 / (1.0 + math.exp(-_glicko2_reduce_impact(opponent_phi) * (mu - opponent_mu)))


def _cross_run_glicko2_update(
    rating: float,
    rating_deviation: float,
    volatility: float,
    outcomes: Sequence[Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    mu = (float(rating) - INITIAL_ELO) / CROSS_RUN_GLICKO2_SCALE
    phi = _clamp_cross_run_rd(rating_deviation) / CROSS_RUN_GLICKO2_SCALE
    sigma = _clamp_cross_run_volatility(volatility)
    matches = list(outcomes)

    if not matches:
        phi_prime = min(
            math.sqrt(phi * phi + sigma * sigma) * CROSS_RUN_GLICKO2_SCALE,
            CROSS_RUN_GLICKO2_MAX_RD,
        )
        return float(rating), phi_prime, sigma

    variance_inverse = 0.0
    score_delta = 0.0
    for opponent_rating, opponent_rd, score in matches:
        opponent_mu = (float(opponent_rating) - INITIAL_ELO) / CROSS_RUN_GLICKO2_SCALE
        opponent_phi = _clamp_cross_run_rd(opponent_rd) / CROSS_RUN_GLICKO2_SCALE
        impact = _glicko2_reduce_impact(opponent_phi)
        expected = _glicko2_expected_score(mu, opponent_mu, opponent_phi)
        variance_inverse += impact * impact * expected * (1.0 - expected)
        score_delta += impact * (float(score) - expected)

    if variance_inverse <= 0.0:
        return float(rating), _clamp_cross_run_rd(rating_deviation), sigma

    variance = 1.0 / variance_inverse
    delta = variance * score_delta
    a = math.log(sigma * sigma)
    tau_squared = CROSS_RUN_GLICKO2_TAU * CROSS_RUN_GLICKO2_TAU

    def volatility_objective(x_value: float) -> float:
        exp_x = math.exp(x_value)
        numerator = exp_x * (delta * delta - phi * phi - variance - exp_x)
        denominator = 2.0 * (phi * phi + variance + exp_x) ** 2
        return (numerator / denominator) - ((x_value - a) / tau_squared)

    lower = a
    if delta * delta > phi * phi + variance:
        upper = math.log(delta * delta - phi * phi - variance)
    else:
        step = 1
        upper = a - step * CROSS_RUN_GLICKO2_TAU
        while volatility_objective(upper) < 0.0:
            step += 1
            upper = a - step * CROSS_RUN_GLICKO2_TAU

    f_lower = volatility_objective(lower)
    f_upper = volatility_objective(upper)
    while abs(upper - lower) > 1e-6:
        candidate = lower + (lower - upper) * f_lower / (f_upper - f_lower)
        f_candidate = volatility_objective(candidate)
        if f_candidate * f_upper < 0.0:
            lower = upper
            f_lower = f_upper
        else:
            f_lower *= 0.5
        upper = candidate
        f_upper = f_candidate

    new_sigma = _clamp_cross_run_volatility(math.exp(lower / 2.0))
    phi_star = math.sqrt(phi * phi + new_sigma * new_sigma)
    new_phi = 1.0 / math.sqrt((1.0 / (phi_star * phi_star)) + (1.0 / variance))
    new_mu = mu + new_phi * new_phi * score_delta
    new_rating = INITIAL_ELO + CROSS_RUN_GLICKO2_SCALE * new_mu
    new_rd = _clamp_cross_run_rd(new_phi * CROSS_RUN_GLICKO2_SCALE)
    return float(new_rating), float(new_rd), float(new_sigma)


def _round_optional_float(value: Any, digits: int = 2) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _cross_run_best_participants(runs: Optional[Sequence[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    participants: List[Dict[str, Any]] = []
    for run in list(runs if runs is not None else list_runs()):
        run_name = str(run.get("run_name") or "").strip()
        if not run_name:
            continue
        checkpoint_name = str(run.get("best_checkpoint") or run.get("latest_checkpoint") or "").strip()
        if not checkpoint_name:
            continue
        participants.append(
            {
                "run_name": run_name,
                "checkpoint": checkpoint_name,
                "best_checkpoint": checkpoint_name,
                "best_iteration": int(run.get("iteration", 0) or 0),
                "created_at": run.get("created_at"),
                "created_datetime": run.get("created_datetime") or _format_timestamp(run.get("created_at")),
                "fork_origin": run.get("fork_origin") or _fork_origin_label(run.get("forked_from")),
            }
        )
    return participants


def _cross_run_parent_seed_order(participants: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(participant: Dict[str, Any]) -> Tuple[float, str]:
        try:
            created_at = float(participant.get("created_at") or 0.0)
        except (TypeError, ValueError):
            created_at = 0.0
        return created_at, str(participant.get("run_name") or "")

    return sorted((dict(participant) for participant in participants), key=sort_key)


def _ensure_cross_run_rating_entry(state: Dict[str, Any], participant: Dict[str, Any]) -> Dict[str, Any]:
    ratings = state.setdefault("ratings", {})
    run_name = str(participant.get("run_name") or "").strip()
    if not run_name:
        raise ValueError("Cross-run participant is missing a run name.")
    checkpoint = str(participant.get("best_checkpoint") or participant.get("checkpoint") or "").strip()
    now = _timestamp()
    existing = dict(ratings.get(run_name) or {})
    previous_checkpoint = str(existing.get("best_checkpoint") or "").strip()
    checkpoint_changed = bool(existing) and previous_checkpoint and previous_checkpoint != checkpoint
    parent_seed = (
        _cross_run_parent_rating_seed(state, participant, checkpoint, existing, previous_checkpoint)
        if (not existing or checkpoint_changed)
        else None
    )
    inherited_rd = (
        float(parent_seed["rating_deviation"])
        if parent_seed is not None
        else (_cross_run_legacy_rating_deviation(existing) if existing else CROSS_RUN_GLICKO2_DEFAULT_RD)
    )
    entry = {
        "run_name": run_name,
        "elo": float(parent_seed["elo"]) if parent_seed is not None else _cross_run_float(existing.get("elo"), INITIAL_ELO),
        "rating_deviation": inherited_rd,
        "volatility": float(parent_seed["volatility"]) if parent_seed is not None else _clamp_cross_run_volatility(existing.get("volatility")),
        "games": int(existing.get("games", 0)),
        "wins": int(existing.get("wins", 0)),
        "losses": int(existing.get("losses", 0)),
        "draws": int(existing.get("draws", 0)),
        "best_checkpoint": checkpoint,
        "best_iteration": int(participant.get("best_iteration", existing.get("best_iteration", 0)) or 0),
        "created_at": participant.get("created_at", existing.get("created_at")),
        "created_datetime": participant.get("created_datetime") or existing.get("created_datetime") or "",
        "fork_origin": participant.get("fork_origin") or existing.get("fork_origin") or "-",
        "first_rated_at": existing.get("first_rated_at") or now,
        "last_played_at": existing.get("last_played_at"),
    }
    if parent_seed is not None:
        entry["rating_seed"] = parent_seed["source"]
    elif existing.get("rating_seed") is not None:
        entry["rating_seed"] = existing.get("rating_seed")
    if checkpoint_changed:
        entry["previous_best_checkpoint"] = previous_checkpoint
        entry["best_changed_at"] = now
        entry["games"] = 0
        entry["wins"] = 0
        entry["losses"] = 0
        entry["draws"] = 0
        entry["last_played_at"] = None
        entry["rating_deviation"] = inherited_rd
        if parent_seed is None:
            entry["rating_deviation"] = _clamp_cross_run_rd(
                max(entry["rating_deviation"], CROSS_RUN_GLICKO2_INHERITED_RD_FLOOR)
            )
    confidence = _cross_run_confidence_radius_from_entry(entry)
    entry["confidence_radius"] = None if confidence is None else float(confidence)
    ratings[run_name] = entry
    return entry


def _cross_run_pair_key(run_name_a: str, run_name_b: str) -> str:
    left, right = sorted((str(run_name_a), str(run_name_b)))
    return f"{left}||{right}"


def _cross_run_pair_entry(state: Dict[str, Any], run_name_a: str, run_name_b: str) -> Tuple[str, Dict[str, Any]]:
    key = _cross_run_pair_key(run_name_a, run_name_b)
    left, right = key.split("||", 1)
    pairings = state.setdefault("pairings", {})
    entry = dict(pairings.get(key) or {})
    entry.setdefault("run_a", left)
    entry.setdefault("run_b", right)
    entry.setdefault("games", 0)
    entry.setdefault("wins_a", 0)
    entry.setdefault("wins_b", 0)
    entry.setdefault("draws", 0)
    pairings[key] = entry
    return key, entry


def _cross_run_participant_weights(
    participants: Sequence[Dict[str, Any]],
    state: Dict[str, Any],
) -> Dict[str, float]:
    created_values: List[Tuple[float, str]] = []
    for participant in participants:
        run_name = str(participant.get("run_name") or "")
        try:
            created_at = float(participant.get("created_at") or 0.0)
        except (TypeError, ValueError):
            created_at = 0.0
        created_values.append((created_at, run_name))
    newest_first = [name for _, name in sorted(created_values, reverse=True)]
    recency_rank = {name: index for index, name in enumerate(newest_first)}
    weights: Dict[str, float] = {}
    ratings = state.get("ratings") or {}
    for participant in participants:
        run_name = str(participant.get("run_name") or "")
        entry = ratings.get(run_name) or {}
        confidence = _cross_run_confidence_radius_from_entry(entry) if entry else None
        uncertainty_weight = 8.0 if confidence is None else _clip(float(confidence) / 100.0, 0.2, 8.0)
        rank = recency_rank.get(run_name, len(participants))
        recency_weight = 1.0 / math.sqrt(float(rank) + 1.0)
        weights[run_name] = max(0.0001, recency_weight * uncertainty_weight)
    return weights


def _sample_cross_run_pair(
    participants: Sequence[Dict[str, Any]],
    state: Dict[str, Any],
    planned_pair_counts: Dict[str, int],
) -> Tuple[str, str]:
    if len(participants) < 2:
        raise ValueError("At least two runs with saved best policies are required for cross-run calibration.")
    weights = _cross_run_participant_weights(participants, state)
    choices: List[Tuple[Tuple[str, str], float]] = []
    for index, participant_a in enumerate(participants):
        run_name_a = str(participant_a.get("run_name") or "")
        for participant_b in participants[index + 1:]:
            run_name_b = str(participant_b.get("run_name") or "")
            key, pair_entry = _cross_run_pair_entry(state, run_name_a, run_name_b)
            existing_games = int(pair_entry.get("games", 0))
            planned_games = int(planned_pair_counts.get(key, 0))
            pair_penalty = 1.0 / math.sqrt(float(existing_games + planned_games) + 1.0)
            choices.append(((run_name_a, run_name_b), weights.get(run_name_a, 1.0) * weights.get(run_name_b, 1.0) * pair_penalty))
    run_name_a, run_name_b = _weighted_choice(choices)
    if random.random() < 0.5:
        return run_name_b, run_name_a
    return run_name_a, run_name_b


def _cross_run_config_for_pair(
    run_name_a: str,
    run_name_b: str,
    simulation_workers: int,
) -> TrainingConfig:
    config_a = _get_trainer(run_name_a).config
    config_b = _get_trainer(run_name_b).config
    return config_a.merged(
        {
            "max_turns_per_game": max(int(config_a.max_turns_per_game), int(config_b.max_turns_per_game)),
            "max_actions_per_turn": max(int(config_a.max_actions_per_turn), int(config_b.max_actions_per_turn)),
            "eval_temperature": min(float(config_a.eval_temperature), float(config_b.eval_temperature)),
            "simulation_workers": int(simulation_workers),
        }
    )


def _cross_run_leaderboard_from_state(
    state: Dict[str, Any],
    participants: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    participant_list = list(participants if participants is not None else _cross_run_best_participants())
    participant_by_name = {
        str(participant.get("run_name") or ""): participant
        for participant in participant_list
    }
    entries: List[Dict[str, Any]] = []
    for run_name, raw_entry in (state.get("ratings") or {}).items():
        participant = participant_by_name.get(str(run_name))
        if participant is None:
            continue
        entry = _ensure_cross_run_rating_entry(state, participant)
        confidence = _cross_run_confidence_radius_from_entry(entry)
        entries.append(
            {
                "run_name": str(run_name),
                "elo": round(float(entry.get("elo", INITIAL_ELO)), 2),
                "rating_deviation": _round_optional_float(entry.get("rating_deviation"), 2),
                "volatility": _round_optional_float(entry.get("volatility"), 6),
                "confidence_radius": _round_optional_float(confidence, 2),
                "games": int(entry.get("games", 0)),
                "wins": int(entry.get("wins", 0)),
                "losses": int(entry.get("losses", 0)),
                "draws": int(entry.get("draws", 0)),
                "best_checkpoint": entry.get("best_checkpoint"),
                "best_iteration": int(entry.get("best_iteration", 0) or 0),
                "created_at": entry.get("created_at"),
                "created_datetime": entry.get("created_datetime") or "",
                "fork_origin": entry.get("fork_origin") or "-",
                "last_played_at": entry.get("last_played_at"),
                "last_played_datetime": _format_timestamp(entry.get("last_played_at")),
            }
        )
    entries.sort(key=lambda item: (float(item.get("elo", INITIAL_ELO)), int(item.get("games", 0)), item.get("run_name", "")), reverse=True)
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank
    return entries


def cross_run_rating_summary() -> Dict[str, Any]:
    with _CROSS_RUN_RATING_LOCK:
        state = _load_cross_run_rating_state()
        participants = _cross_run_best_participants()
        for participant in _cross_run_parent_seed_order(participants):
            _ensure_cross_run_rating_entry(state, participant)
        leaderboard = _cross_run_leaderboard_from_state(state, participants)
        return {
            "status": "available" if len(participants) >= 2 else "needs_two_runs",
            "rating_model": CROSS_RUN_RATING_MODEL,
            "participant_count": len(participants),
            "total_games": int(state.get("total_games", 0)),
            "updated_at": state.get("updated_at"),
            "updated_datetime": _format_timestamp(state.get("updated_at")),
            "confidence_level": "approx_95_percent",
            "confidence_formula": f"+/- {CROSS_RUN_CONFIDENCE_SIGMAS:.2f} * Glicko-2 RD",
            "leaderboard": leaderboard,
            "last_calibration": state.get("last_calibration"),
            "ratings_file": str(CROSS_RUN_RATINGS_FILE),
        }


def _cross_run_pair_tasks(
    pair_counts: Dict[Tuple[str, str], int],
    participant_by_name: Dict[str, Dict[str, Any]],
    resolved_workers: int,
    started_at: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pairings: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []
    for pairing_index, ((run_name_a, run_name_b), game_count) in enumerate(pair_counts.items()):
        participant_a = participant_by_name[run_name_a]
        participant_b = participant_by_name[run_name_b]
        checkpoint_a = str(participant_a["checkpoint"])
        checkpoint_b = str(participant_b["checkpoint"])
        policy_a = _cached_policy(run_name_a, checkpoint_a)
        policy_b = _cached_policy(run_name_b, checkpoint_b)
        config = _cross_run_config_for_pair(run_name_a, run_name_b, 1)
        policy_a_payload = _policy_payload_for_simulation(policy_a)
        policy_b_payload = _policy_payload_for_simulation(policy_b)
        config_payload = _config_payload_for_simulation(config)
        prefix = (
            f"cross-run:{run_name_a}:{checkpoint_a}:"
            f"{run_name_b}:{checkpoint_b}:{started_at:.6f}"
        )
        pairings.append(
            {
                "run_a": run_name_a,
                "checkpoint_a": checkpoint_a,
                "participant_a": participant_a,
                "run_b": run_name_b,
                "checkpoint_b": checkpoint_b,
                "participant_b": participant_b,
                "games_target": int(game_count),
                "pending_chunks": 0,
                "partials": [],
            }
        )
        if int(game_count) <= 1:
            chunk_tasks = [{"game_count": 1, "seed": _random_seed(), "swapped": False}]
        else:
            games_as_first = max(1, int(game_count) // 2)
            games_as_second = max(1, int(game_count) - games_as_first)
            chunk_tasks = [
                {"game_count": chunk_size, "seed": _random_seed(), "swapped": False}
                for chunk_size in _simulation_chunk_sizes(games_as_first, resolved_workers)
            ]
            chunk_tasks.extend(
                {
                    "game_count": chunk_size,
                    "seed": _random_seed(),
                    "swapped": True,
                }
                for chunk_size in _simulation_chunk_sizes(games_as_second, resolved_workers)
            )
        pairings[pairing_index]["pending_chunks"] = len(chunk_tasks)
        for task in chunk_tasks:
            tasks.append(
                {
                    **task,
                    "pairing_index": pairing_index,
                    "policy_a_payload": policy_a_payload,
                    "policy_b_payload": policy_b_payload,
                    "policy_a_cache_key": f"{prefix}:a",
                    "policy_b_cache_key": f"{prefix}:b",
                    "config_payload": config_payload,
                }
            )
    return pairings, tasks


def _apply_cross_run_pair_result(
    state: Dict[str, Any],
    participant_a: Dict[str, Any],
    participant_b: Dict[str, Any],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    run_name_a = str(participant_a["run_name"])
    run_name_b = str(participant_b["run_name"])
    entry_a = _ensure_cross_run_rating_entry(state, participant_a)
    entry_b = _ensure_cross_run_rating_entry(state, participant_b)
    pair_key, pair_entry = _cross_run_pair_entry(state, run_name_a, run_name_b)
    left_name = str(pair_entry["run_a"])
    played = int(summary.get("games_played", 0))
    winners = list(summary.get("per_game_winners") or [])
    if len(winners) < played:
        winners.extend(["policy_a"] * max(0, int(summary.get("wins_a", 0)) - winners.count("policy_a")))
        winners.extend(["policy_b"] * max(0, int(summary.get("wins_b", 0)) - winners.count("policy_b")))
        winners.extend(["draw"] * max(0, int(summary.get("draws", 0)) - winners.count("draw")))
    if len(winners) < played:
        winners.extend(["draw"] * (played - len(winners)))

    for winner in winners[:played]:
        if winner == "policy_a":
            score_a = 1.0
            entry_a["wins"] = int(entry_a.get("wins", 0)) + 1
            entry_b["losses"] = int(entry_b.get("losses", 0)) + 1
            if run_name_a == left_name:
                pair_entry["wins_a"] = int(pair_entry.get("wins_a", 0)) + 1
            else:
                pair_entry["wins_b"] = int(pair_entry.get("wins_b", 0)) + 1
        elif winner == "policy_b":
            score_a = 0.0
            entry_b["wins"] = int(entry_b.get("wins", 0)) + 1
            entry_a["losses"] = int(entry_a.get("losses", 0)) + 1
            if run_name_b == left_name:
                pair_entry["wins_a"] = int(pair_entry.get("wins_a", 0)) + 1
            else:
                pair_entry["wins_b"] = int(pair_entry.get("wins_b", 0)) + 1
        else:
            score_a = 0.5
            entry_a["draws"] = int(entry_a.get("draws", 0)) + 1
            entry_b["draws"] = int(entry_b.get("draws", 0)) + 1
            pair_entry["draws"] = int(pair_entry.get("draws", 0)) + 1

        rating_a_before = _cross_run_float(entry_a.get("elo"), INITIAL_ELO)
        rating_b_before = _cross_run_float(entry_b.get("elo"), INITIAL_ELO)
        rd_a_before = _clamp_cross_run_rd(entry_a.get("rating_deviation"))
        rd_b_before = _clamp_cross_run_rd(entry_b.get("rating_deviation"))
        volatility_a_before = _clamp_cross_run_volatility(entry_a.get("volatility"))
        volatility_b_before = _clamp_cross_run_volatility(entry_b.get("volatility"))
        new_a = _cross_run_glicko2_update(
            rating_a_before,
            rd_a_before,
            volatility_a_before,
            [(rating_b_before, rd_b_before, float(score_a))],
        )
        new_b = _cross_run_glicko2_update(
            rating_b_before,
            rd_b_before,
            volatility_b_before,
            [(rating_a_before, rd_a_before, 1.0 - float(score_a))],
        )
        entry_a["elo"], entry_a["rating_deviation"], entry_a["volatility"] = new_a
        entry_b["elo"], entry_b["rating_deviation"], entry_b["volatility"] = new_b
        entry_a["games"] = int(entry_a.get("games", 0)) + 1
        entry_b["games"] = int(entry_b.get("games", 0)) + 1
        pair_entry["games"] = int(pair_entry.get("games", 0)) + 1
        state["total_games"] = int(state.get("total_games", 0)) + 1
    now = _timestamp()
    entry_a["last_played_at"] = now
    entry_b["last_played_at"] = now
    confidence_a = _cross_run_confidence_radius_from_entry(entry_a)
    confidence_b = _cross_run_confidence_radius_from_entry(entry_b)
    entry_a["confidence_radius"] = None if confidence_a is None else float(confidence_a)
    entry_b["confidence_radius"] = None if confidence_b is None else float(confidence_b)
    state["ratings"][run_name_a] = entry_a
    state["ratings"][run_name_b] = entry_b
    state["pairings"][pair_key] = pair_entry
    return pair_entry


def run_cross_run_calibration_games(
    games: int = 200,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    simulation_workers: int = SIMULATION_WORKERS_AUTO,
    run_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    total_games = int(games)
    if total_games <= 0:
        raise ValueError("Cross-run calibration games must be positive.")
    simulation_workers = normalize_simulation_workers(simulation_workers)
    allocation_key = "cross-run-rating"
    started_at = _timestamp()

    def emit_progress(games_completed: int, pairings_completed: int, pairings_total: int) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(
                {
                    "games_completed": int(games_completed),
                    "games_target": int(total_games),
                    "pairings_completed": int(pairings_completed),
                    "pairings_total": int(pairings_total),
                    "duration_seconds": _timestamp() - started_at,
                }
            )
        except Exception:
            pass

    with _CROSS_RUN_RATING_LOCK:
        state = _load_cross_run_rating_state()
        participants = _cross_run_best_participants()
        selected_run_names: List[str] = []
        if run_names is not None:
            seen_selected = set()
            for run_name in run_names:
                normalized = str(run_name or "").strip()
                if normalized and normalized not in seen_selected:
                    seen_selected.add(normalized)
                    selected_run_names.append(normalized)
            selected_set = set(selected_run_names)
            participants = [
                participant
                for participant in participants
                if str(participant.get("run_name") or "") in selected_set
            ]
        if len(participants) < 2:
            raise ValueError("Select at least two runs with saved best policies for cross-run calibration.")
        participant_by_name = {str(item["run_name"]): item for item in participants}
        for participant in _cross_run_parent_seed_order(participants):
            _ensure_cross_run_rating_entry(state, participant)
        selected_run_names = [str(participant["run_name"]) for participant in participants]

        planned_pair_counts: Dict[str, int] = {}
        pair_counts: Dict[Tuple[str, str], int] = {}
        for _ in range(total_games):
            run_name_a, run_name_b = _sample_cross_run_pair(participants, state, planned_pair_counts)
            pair_key = _cross_run_pair_key(run_name_a, run_name_b)
            planned_pair_counts[pair_key] = planned_pair_counts.get(pair_key, 0) + 1
            pair_counts[(run_name_a, run_name_b)] = pair_counts.get((run_name_a, run_name_b), 0) + 1

    completed_games = 0
    pairings_completed = 0
    pair_results: List[Dict[str, Any]] = []
    resolved_workers = 1
    simulation_pool = None
    _register_active_simulation_run(allocation_key, simulation_workers)
    try:
        resolved_workers = resolve_simulation_workers(
            simulation_workers,
            total_games,
            allocation_key=allocation_key,
        )
        pairings, tasks = _cross_run_pair_tasks(pair_counts, participant_by_name, resolved_workers, started_at)
        simulation_pool = _SimulationPool(resolved_workers) if resolved_workers > 1 and total_games > 1 else None
        finished_games = 0

        def record_completed_pair(pairing: Dict[str, Any], summary: Dict[str, Any]) -> None:
            nonlocal completed_games, pairings_completed, finished_games
            participant_a = dict(pairing["participant_a"])
            participant_b = dict(pairing["participant_b"])
            run_name_a = str(pairing["run_a"])
            run_name_b = str(pairing["run_b"])
            played = int(summary.get("games_played", 0))
            finished_games += played
            completed_games = max(completed_games, min(total_games, finished_games))
            pairings_completed += 1

            with _CROSS_RUN_RATING_LOCK:
                state = _load_cross_run_rating_state()
                _apply_cross_run_pair_result(state, participant_a, participant_b, summary)
                state["last_calibration"] = {
                    "status": "running",
                    "rating_model": CROSS_RUN_RATING_MODEL,
                    "started_at": started_at,
                    "games_completed": completed_games,
                    "games_target": total_games,
                    "pairings_completed": pairings_completed,
                    "pairings_total": len(pairings),
                    "resolved_simulation_workers": resolved_workers,
                    "selected_run_names": list(selected_run_names),
                    "selected_participant_count": len(participants),
                }
                _save_cross_run_rating_state(state)

            pair_results.append(
                {
                    "run_a": run_name_a,
                    "checkpoint_a": pairing["checkpoint_a"],
                    "run_b": run_name_b,
                    "checkpoint_b": pairing["checkpoint_b"],
                    "games_played": played,
                    "wins_a": int(summary.get("wins_a", 0)),
                    "wins_b": int(summary.get("wins_b", 0)),
                }
            )
            emit_progress(completed_games, pairings_completed, len(pairings))

        def on_progress(delta: int) -> None:
            nonlocal completed_games
            completed_games = min(total_games, completed_games + int(delta))
            emit_progress(completed_games, pairings_completed, len(pairings))

        def on_result(partial: Dict[str, Any]) -> None:
            pairing_index = int(partial["pairing_index"])
            pairing = pairings[pairing_index]
            pairing["partials"].append(partial)
            pairing["pending_chunks"] = int(pairing.get("pending_chunks", 0)) - 1
            if int(pairing["pending_chunks"]) > 0:
                return
            summary = _combine_balanced_partials(list(pairing.get("partials") or []))
            record_completed_pair(pairing, summary)

        if simulation_pool is not None:
            progress_queue = simulation_pool.progress_queue
            simulation_pool.wait_until_ready()
            _drain_simulation_progress_queue(progress_queue)
            _consume_simulation_task_iterator(
                simulation_pool.executor,
                _simulate_payload_balanced_chunk_worker,
                tasks,
                max_in_flight=_max_in_flight_simulation_tasks(resolved_workers, len(tasks)),
                progress_queue=progress_queue,
                progress_callback=on_progress,
                result_callback=on_result,
            )
            _drain_simulation_progress_queue(progress_queue)
        else:
            for task in tasks:
                partial = _simulate_payload_balanced_chunk_worker(task)
                on_progress(int(partial.get("games_played", 0)))
                on_result(partial)
    finally:
        if simulation_pool is not None:
            simulation_pool.close()
        _unregister_active_simulation_run(allocation_key)

    with _CROSS_RUN_RATING_LOCK:
        state = _load_cross_run_rating_state()
        for participant in _cross_run_parent_seed_order(participants):
            _ensure_cross_run_rating_entry(state, participant)
        leaderboard = _cross_run_leaderboard_from_state(state, participants)
        state["last_calibration"] = {
            "status": "completed",
            "rating_model": CROSS_RUN_RATING_MODEL,
            "started_at": started_at,
            "completed_at": _timestamp(),
            "duration_seconds": _timestamp() - started_at,
            "games_completed": completed_games,
            "games_target": total_games,
            "pairings_completed": pairings_completed,
            "pairings_total": len(pair_counts),
            "resolved_simulation_workers": resolved_workers,
            "selected_run_names": list(selected_run_names),
            "selected_participant_count": len(participants),
        }
        _save_cross_run_rating_state(state)
        return {
            "status": "completed",
            "games_completed": completed_games,
            "games_target": total_games,
            "pairings_played": pairings_completed,
            "pairings_total": len(pair_counts),
            "participant_count": len(participants),
            "selected_run_names": list(selected_run_names),
            "selected_participant_count": len(participants),
            "resolved_simulation_workers": resolved_workers,
            "duration_seconds": _timestamp() - started_at,
            "rating_model": CROSS_RUN_RATING_MODEL,
            "confidence_level": "approx_95_percent",
            "confidence_formula": f"+/- {CROSS_RUN_CONFIDENCE_SIGMAS:.2f} * Glicko-2 RD",
            "leaderboard": leaderboard,
            "pair_results": pair_results,
            "ratings_file": str(CROSS_RUN_RATINGS_FILE),
        }


def summarize_cross_run_rating(games: int = 200) -> str:
    summary = run_cross_run_calibration_games(games=games)
    top_entry = (summary.get("leaderboard") or [{}])[0]
    confidence = top_entry.get("confidence_radius")
    confidence_text = "-" if confidence is None else f"+/- {float(confidence):.1f}"
    return (
        f"Cross-run rating completed with {summary.get('games_completed', 0)} calibration games. "
        f"Top rated run: {top_entry.get('run_name', '-')} at "
        f"{round(float(top_entry.get('elo', INITIAL_ELO)), 2)} Elo ({confidence_text})."
    )


def _initial_card_choice_rating_state() -> Dict[str, Dict[str, Any]]:
    return {
        "ratings": {card_name: INITIAL_ELO for card_name in CARD_NAME_ORDER},
        "decision_counts": {card_name: 0 for card_name in CARD_NAME_ORDER},
        "pairwise_counts": {card_name: 0 for card_name in CARD_NAME_ORDER},
        "win_counts": {card_name: 0 for card_name in CARD_NAME_ORDER},
        "loss_counts": {card_name: 0 for card_name in CARD_NAME_ORDER},
        "information": {card_name: 0.0 for card_name in CARD_NAME_ORDER},
    }


def _card_choice_probabilities(
    ratings: Dict[str, float],
    card_names: Sequence[str],
) -> Dict[str, float]:
    unique_names = [str(card_name) for card_name in card_names]
    if not unique_names:
        return {}
    max_rating = max(float(ratings.get(card_name, INITIAL_ELO)) for card_name in unique_names)
    weights: Dict[str, float] = {}
    total_weight = 0.0
    for card_name in unique_names:
        weight = math.exp((float(ratings.get(card_name, INITIAL_ELO)) - max_rating) * ELO_LOGISTIC_SCALE)
        weights[card_name] = weight
        total_weight += weight
    if total_weight <= 0.0:
        uniform_probability = 1.0 / len(unique_names)
        return {card_name: uniform_probability for card_name in unique_names}
    return {card_name: weights[card_name] / total_weight for card_name in unique_names}


def _apply_card_acquire_choice_result(
    rating_state: Dict[str, Dict[str, Any]],
    winner_name: str,
    loser_names: Sequence[str],
    k_factor: float,
) -> int:
    winner_name = str(winner_name or "").strip()
    if not winner_name:
        return 0

    unique_losers: List[str] = []
    seen_losers = set()
    for loser_name in loser_names:
        normalized_name = str(loser_name or "").strip()
        if not normalized_name or normalized_name == winner_name or normalized_name in seen_losers:
            continue
        seen_losers.add(normalized_name)
        unique_losers.append(normalized_name)

    if not unique_losers:
        return 0

    participant_names = [winner_name, *unique_losers]
    ratings = rating_state["ratings"]
    decision_counts = rating_state["decision_counts"]
    pairwise_counts = rating_state["pairwise_counts"]
    win_counts = rating_state["win_counts"]
    loss_counts = rating_state["loss_counts"]
    information = rating_state["information"]

    probabilities = _card_choice_probabilities(ratings, participant_names)
    deltas: Dict[str, float] = {}
    participant_count = len(participant_names)
    for card_name in participant_names:
        expected_probability = float(probabilities.get(card_name, 0.0))
        actual_score = 1.0 if card_name == winner_name else 0.0
        deltas[card_name] = float(k_factor) * (actual_score - expected_probability)
        decision_counts[card_name] = int(decision_counts.get(card_name, 0)) + 1
        pairwise_counts[card_name] = int(pairwise_counts.get(card_name, 0)) + max(participant_count - 1, 0)
        information[card_name] = float(information.get(card_name, 0.0)) + expected_probability * (1.0 - expected_probability)
        if card_name == winner_name:
            win_counts[card_name] = int(win_counts.get(card_name, 0)) + 1
        else:
            loss_counts[card_name] = int(loss_counts.get(card_name, 0)) + 1

    for card_name, delta in deltas.items():
        ratings[card_name] = float(ratings.get(card_name, INITIAL_ELO)) + delta

    return len(unique_losers)


def _card_rating_uncertainty_from_information(information_value: float) -> Optional[float]:
    info = float(information_value)
    if not math.isfinite(info) or info <= 0.0:
        return None
    return 1.0 / (ELO_LOGISTIC_SCALE * math.sqrt(info))


def _normalized_card_choice_leaderboard(
    rating_state: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], float, float]:
    ratings = rating_state["ratings"]
    explorer_rating = float(ratings.get("Explorer", INITIAL_ELO))
    if abs(explorer_rating) <= 1e-9:
        normalization_factor = 1.0
    else:
        normalization_factor = explorer_rating / 200.0
    leaderboard: List[Dict[str, Any]] = []
    for card_name in CARD_NAME_ORDER:
        raw_rating = float(ratings.get(card_name, INITIAL_ELO))
        raw_uncertainty = _card_rating_uncertainty_from_information(float(rating_state["information"].get(card_name, 0.0)))
        normalized_rating = raw_rating / normalization_factor
        normalized_uncertainty = None if raw_uncertainty is None else raw_uncertainty / normalization_factor
        leaderboard.append(
            {
                "card_name": card_name,
                "elo": round(normalized_rating, 4),
                "raw_elo": round(raw_rating, 4),
                "uncertainty": None if normalized_uncertainty is None else round(normalized_uncertainty, 4),
                "raw_uncertainty": None if raw_uncertainty is None else round(raw_uncertainty, 4),
                "decision_count": int(rating_state["decision_counts"].get(card_name, 0)),
                "pairwise_comparisons": int(rating_state["pairwise_counts"].get(card_name, 0)),
                "wins": int(rating_state["win_counts"].get(card_name, 0)),
                "losses": int(rating_state["loss_counts"].get(card_name, 0)),
            }
        )
    leaderboard.sort(key=lambda entry: (-float(entry["elo"]), str(entry["card_name"])))
    return leaderboard, normalization_factor, explorer_rating


def _scrap_option_key(source: str, card_name: str) -> str:
    normalized_source = str(source or "").strip().lower()
    if normalized_source == "noscrap":
        return SCRAP_ELO_NO_SCRAP_KEY
    return f"{normalized_source}:{str(card_name or '').strip()}"


def _scrap_option_label(source: str, card_name: str) -> str:
    normalized_source = str(source or "").strip().lower()
    if normalized_source == "noscrap":
        return SCRAP_ELO_SOURCE_LABELS["noscrap"]
    source_label = SCRAP_ELO_SOURCE_LABELS.get(normalized_source, normalized_source.title())
    return f"{source_label}: {str(card_name or '').strip()}"


def _scrap_option_order() -> List[Dict[str, str]]:
    options = [
        {
            "option_key": SCRAP_ELO_NO_SCRAP_KEY,
            "source": "noscrap",
            "card_name": "noscrap",
            "option_label": _scrap_option_label("noscrap", "noscrap"),
        }
    ]
    for source in ("hand", "discard"):
        for card_name in CARD_NAME_ORDER:
            options.append(
                {
                    "option_key": _scrap_option_key(source, card_name),
                    "source": source,
                    "card_name": card_name,
                    "option_label": _scrap_option_label(source, card_name),
                }
            )
    return options


def _initial_scrap_choice_rating_state() -> Dict[str, Dict[str, Any]]:
    options = _scrap_option_order()
    return {
        "ratings": {option["option_key"]: INITIAL_ELO for option in options},
        "decision_counts": {option["option_key"]: 0 for option in options},
        "pairwise_counts": {option["option_key"]: 0 for option in options},
        "win_counts": {option["option_key"]: 0 for option in options},
        "loss_counts": {option["option_key"]: 0 for option in options},
        "information": {option["option_key"]: 0.0 for option in options},
        "option_meta": {option["option_key"]: dict(option) for option in options},
        "option_order": [option["option_key"] for option in options],
    }


def _ensure_scrap_option_rating(
    rating_state: Dict[str, Dict[str, Any]],
    option: Dict[str, str],
) -> None:
    option_key = str(option.get("option_key") or "").strip()
    if not option_key:
        return
    rating_state["ratings"].setdefault(option_key, INITIAL_ELO)
    rating_state["decision_counts"].setdefault(option_key, 0)
    rating_state["pairwise_counts"].setdefault(option_key, 0)
    rating_state["win_counts"].setdefault(option_key, 0)
    rating_state["loss_counts"].setdefault(option_key, 0)
    rating_state["information"].setdefault(option_key, 0.0)
    rating_state["option_meta"].setdefault(option_key, dict(option))
    if option_key not in rating_state["option_order"]:
        rating_state["option_order"].append(option_key)


def _scrap_elo_option_from_sim_option(option: Sequence[Any]) -> Optional[Dict[str, str]]:
    action = _option_action_name(option)
    if action == "noScrapFromHand":
        return {
            "option_key": SCRAP_ELO_NO_SCRAP_KEY,
            "source": "noscrap",
            "card_name": "noscrap",
            "option_label": _scrap_option_label("noscrap", "noscrap"),
        }
    source = SCRAP_ELO_ACTION_SOURCE.get(action)
    if source is None or len(option) < 3:
        return None
    card = _extract_card(option[2])
    if card is None:
        return None
    card_name = str(card[0] or "").strip()
    if not card_name:
        return None
    return {
        "option_key": _scrap_option_key(source, card_name),
        "source": source,
        "card_name": card_name,
        "option_label": _scrap_option_label(source, card_name),
    }


def _extract_scrap_elo_decision_from_choice(
    options: Sequence[Sequence[Any]],
    action_index: int,
) -> Optional[Dict[str, Any]]:
    if action_index < 0 or action_index >= len(options):
        return None
    action_names = [_option_action_name(option) for option in options]
    if "noScrapFromHand" not in action_names:
        return None

    winner_option = _scrap_elo_option_from_sim_option(options[action_index])
    if winner_option is None:
        return None

    unique_options: List[Dict[str, str]] = []
    seen_keys = set()
    for option in options:
        scrap_option = _scrap_elo_option_from_sim_option(option)
        if scrap_option is None:
            continue
        option_key = scrap_option["option_key"]
        if option_key in seen_keys:
            continue
        seen_keys.add(option_key)
        unique_options.append(scrap_option)

    winner_key = winner_option["option_key"]
    if winner_key not in seen_keys:
        unique_options.append(winner_option)
        seen_keys.add(winner_key)

    loser_options = [option for option in unique_options if option["option_key"] != winner_key]
    return {
        "winner_option": winner_option,
        "loser_options": loser_options,
        "unique_options": unique_options,
        "unique_option_count": len(unique_options),
    }


def _apply_scrap_choice_result(
    rating_state: Dict[str, Dict[str, Any]],
    winner_option: Dict[str, str],
    loser_options: Sequence[Dict[str, str]],
    k_factor: float,
) -> int:
    winner_key = str(winner_option.get("option_key") or "").strip()
    if not winner_key:
        return 0

    unique_losers: List[Dict[str, str]] = []
    seen_loser_keys = set()
    for loser_option in loser_options:
        loser_key = str(loser_option.get("option_key") or "").strip()
        if not loser_key or loser_key == winner_key or loser_key in seen_loser_keys:
            continue
        seen_loser_keys.add(loser_key)
        unique_losers.append(dict(loser_option))

    if not unique_losers:
        return 0

    _ensure_scrap_option_rating(rating_state, winner_option)
    for loser_option in unique_losers:
        _ensure_scrap_option_rating(rating_state, loser_option)

    ratings = rating_state["ratings"]
    decision_counts = rating_state["decision_counts"]
    pairwise_counts = rating_state["pairwise_counts"]
    win_counts = rating_state["win_counts"]
    loss_counts = rating_state["loss_counts"]
    information = rating_state["information"]

    participant_keys = [winner_key, *[loser_option["option_key"] for loser_option in unique_losers]]
    for option_key in participant_keys:
        decision_counts[option_key] = int(decision_counts.get(option_key, 0)) + 1

    base_winner_rating = float(ratings.get(winner_key, INITIAL_ELO))
    base_ratings = {
        loser_option["option_key"]: float(ratings.get(loser_option["option_key"], INITIAL_ELO))
        for loser_option in unique_losers
    }
    deltas: Dict[str, float] = {winner_key: 0.0}
    for loser_option in unique_losers:
        loser_key = loser_option["option_key"]
        expected_winner = _elo_expected(base_winner_rating, base_ratings[loser_key])
        delta = float(k_factor) * (1.0 - expected_winner)
        deltas[winner_key] = deltas.get(winner_key, 0.0) + delta
        deltas[loser_key] = deltas.get(loser_key, 0.0) - delta
        pairwise_counts[winner_key] = int(pairwise_counts.get(winner_key, 0)) + 1
        pairwise_counts[loser_key] = int(pairwise_counts.get(loser_key, 0)) + 1
        win_counts[winner_key] = int(win_counts.get(winner_key, 0)) + 1
        loss_counts[loser_key] = int(loss_counts.get(loser_key, 0)) + 1
        fisher = expected_winner * (1.0 - expected_winner)
        information[winner_key] = float(information.get(winner_key, 0.0)) + fisher
        information[loser_key] = float(information.get(loser_key, 0.0)) + fisher

    for option_key, delta in deltas.items():
        ratings[option_key] = float(ratings.get(option_key, INITIAL_ELO)) + delta

    return len(unique_losers)


def _scrap_choice_leaderboard(rating_state: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    leaderboard: List[Dict[str, Any]] = []
    option_keys = list(rating_state.get("option_order") or [])
    for option_key in rating_state.get("ratings", {}).keys():
        if option_key not in option_keys:
            option_keys.append(option_key)
    for option_key in option_keys:
        meta = dict(rating_state["option_meta"].get(option_key) or {})
        raw_rating = float(rating_state["ratings"].get(option_key, INITIAL_ELO))
        raw_uncertainty = _card_rating_uncertainty_from_information(
            float(rating_state["information"].get(option_key, 0.0))
        )
        leaderboard.append(
            {
                "option_key": option_key,
                "source": str(meta.get("source", "")),
                "card_name": str(meta.get("card_name", "")),
                "option_label": str(meta.get("option_label") or option_key),
                "elo": round(raw_rating, 4),
                "uncertainty": None if raw_uncertainty is None else round(raw_uncertainty, 4),
                "decision_count": int(rating_state["decision_counts"].get(option_key, 0)),
                "pairwise_comparisons": int(rating_state["pairwise_counts"].get(option_key, 0)),
                "wins": int(rating_state["win_counts"].get(option_key, 0)),
                "losses": int(rating_state["loss_counts"].get(option_key, 0)),
            }
        )
    leaderboard.sort(key=lambda entry: (-float(entry["elo"]), str(entry["option_label"])))
    return leaderboard


def format_card_acquire_elo_test_report(result: Dict[str, Any]) -> str:
    future_trade = dict(result.get("future_trade") or {})
    immediate_trade = dict(result.get("immediate_trade") or {})
    future_leaderboard = list(future_trade.get("leaderboard") or [])
    immediate_leaderboard = list(immediate_trade.get("leaderboard") or [])
    lines = [
        f"Card Acquire Elo Test for {result.get('run_name', '-')}/{result.get('checkpoint', '-')}",
        f"Resolved checkpoint: {result.get('resolved_checkpoint', '-')}",
        f"Games: {result.get('games', 0)}",
        f"Deterministic: {result.get('deterministic', True)}",
        f"Ended by limit: {result.get('ended_by_limit_games', 0)} game(s)",
        f"Turn summaries: {result.get('turn_summaries', 0)}",
        f"Eligible single-acquire turns: {result.get('eligible_single_acquire_turns', 0)}",
        f"Future-trade scored decisions: {future_trade.get('scored_decisions', 0)}",
        f"Future-trade pairwise comparisons: {future_trade.get('pairwise_comparisons', 0)}",
        f"Immediate-trade scored decisions: {immediate_trade.get('scored_decisions', 0)}",
        f"Immediate-trade pairwise comparisons: {immediate_trade.get('pairwise_comparisons', 0)}",
        f"Future-trade Explorer raw Elo: {round(float(future_trade.get('explorer_raw_elo', INITIAL_ELO)), 4)}",
        f"Future-trade normalization factor: {round(float(future_trade.get('normalization_factor', 1.0)), 6)}",
        f"Immediate-trade Explorer raw Elo: {round(float(immediate_trade.get('explorer_raw_elo', INITIAL_ELO)), 4)}",
        f"Immediate-trade normalization factor: {round(float(immediate_trade.get('normalization_factor', 1.0)), 6)}",
        "Rating update model: multinomial Elo / Plackett-Luce choice update (per decision, updates sum to zero).",
        "Uncertainty: approximate 1-sigma Elo standard error from diagonal Fisher information of the same choice model.",
        f"Duration: {_format_seconds(float(result.get('duration_seconds', 0.0)))}",
        "",
        "Future-Trade-Aware Rankings",
    ]
    for index, entry in enumerate(future_leaderboard, start=1):
        uncertainty = entry.get("uncertainty")
        uncertainty_text = "-" if uncertainty is None else f"{float(uncertainty):.2f}"
        lines.append(
            f"{index:>2}. {entry.get('card_name', '-'):<20} "
            f"Elo {float(entry.get('elo', 0.0)):>8.2f}  "
            f"+/- {uncertainty_text:>8}  "
            f"dec {int(entry.get('decision_count', 0)):>4}  "
            f"cmp {int(entry.get('pairwise_comparisons', 0)):>5}"
        )
    lines.append("")
    lines.append("Immediate-Trade-Only Rankings")
    for index, entry in enumerate(immediate_leaderboard, start=1):
        uncertainty = entry.get("uncertainty")
        uncertainty_text = "-" if uncertainty is None else f"{float(uncertainty):.2f}"
        lines.append(
            f"{index:>2}. {entry.get('card_name', '-'):<20} "
            f"Elo {float(entry.get('elo', 0.0)):>8.2f}  "
            f"+/- {uncertainty_text:>8}  "
            f"dec {int(entry.get('decision_count', 0)):>4}  "
            f"cmp {int(entry.get('pairwise_comparisons', 0)):>5}"
        )
    return "\n".join(lines)


def _extract_card_acquire_decisions(game_turn_summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    turn_summaries = 0
    eligible_single_acquire_turns = 0
    decisions: List[Dict[str, Any]] = []

    for turn_summary in game_turn_summaries:
        turn_summaries += 1
        acquisition_events = list(turn_summary.get("acquisitionEvents") or [])
        total_acquisitions = int(turn_summary.get("totalAcquisitions", len(acquisition_events)))
        if total_acquisitions != 1 or len(acquisition_events) != 1:
            continue

        event = acquisition_events[0]
        if str(event.get("type", "")) != "acquire":
            continue

        eligible_single_acquire_turns += 1
        winner_name = str(event.get("cardName", "")).strip()
        if not winner_name:
            continue

        total_trade_gained = float(turn_summary.get("totalTradeGained", 0.0))
        available_trade = float(event.get("tradeAvailable", 0.0))
        future_trade_loser_names: List[str] = []
        immediate_trade_loser_names: List[str] = []
        seen_future_trade_losers = set()
        seen_immediate_trade_losers = set()
        for trade_card in list(event.get("tradeRowSnapshot") or []):
            if not isinstance(trade_card, (list, tuple)) or len(trade_card) < 2:
                continue
            card_name = str(trade_card[0] or "").strip()
            if not card_name or card_name == winner_name:
                continue
            card_cost = CARD_COST_BY_NAME.get(card_name)
            if card_cost is None:
                try:
                    card_cost = int(trade_card[1])
                except (TypeError, ValueError):
                    continue
            if float(card_cost) <= total_trade_gained and card_name not in seen_future_trade_losers:
                seen_future_trade_losers.add(card_name)
                future_trade_loser_names.append(card_name)
            if float(card_cost) <= available_trade and card_name not in seen_immediate_trade_losers:
                seen_immediate_trade_losers.add(card_name)
                immediate_trade_loser_names.append(card_name)

        decisions.append(
            {
                "winner_name": winner_name,
                "future_trade_loser_names": future_trade_loser_names,
                "immediate_trade_loser_names": immediate_trade_loser_names,
            }
        )

    return {
        "turn_summaries": turn_summaries,
        "eligible_single_acquire_turns": eligible_single_acquire_turns,
        "decisions": decisions,
    }


def _simulate_card_acquire_elo_chunk_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    _configure_simulation_worker_runtime()
    policy = _policy_from_simulation_payload(task, "policy_payload", "policy_cache_key")
    config = _config_from_payload(task["config_payload"])
    deterministic = bool(task.get("deterministic", True))
    temperature = config.eval_temperature if deterministic else config.train_temperature
    epsilon_random = 0.0 if deterministic else config.epsilon_random
    _seed_simulation(int(task["seed"]))

    games_completed = 0
    ended_by_limit_games = 0
    turn_summaries = 0
    eligible_single_acquire_turns = 0
    decisions: List[Dict[str, Any]] = []

    for _ in range(max(0, int(task["game_count"]))):
        actor_a = PolicyActor(
            policy,
            deterministic=deterministic,
            temperature=temperature,
            epsilon_random=epsilon_random,
        )
        actor_b = PolicyActor(
            policy,
            deterministic=deterministic,
            temperature=temperature,
            epsilon_random=epsilon_random,
        )
        game_turn_summaries: List[Dict[str, Any]] = []

        def capture_turn_summary(summary: Dict[str, Any]) -> None:
            game_turn_summaries.append(_copy_nested(summary))

        game = Game(
            "policy_a",
            "policy_b",
            p1_choose=actor_a.choose,
            p2_choose=actor_b.choose,
            verbose=False,
            max_turns=config.max_turns_per_game,
            max_actions_per_turn=config.max_actions_per_turn,
            turn_summary_callback=capture_turn_summary,
        )
        games_completed += 1
        if game.ended_by_limit:
            ended_by_limit_games += 1
        _emit_simulation_game_progress()
        extracted = _extract_card_acquire_decisions(game_turn_summaries)
        turn_summaries += int(extracted.get("turn_summaries", 0))
        eligible_single_acquire_turns += int(extracted.get("eligible_single_acquire_turns", 0))
        decisions.extend(list(extracted.get("decisions") or []))

    return {
        "games_completed": games_completed,
        "ended_by_limit_games": ended_by_limit_games,
        "turn_summaries": turn_summaries,
        "eligible_single_acquire_turns": eligible_single_acquire_turns,
        "decisions": decisions,
    }


def run_card_acquire_elo_test(
    run_name: str = LATEST_RUN_NAME,
    checkpoint: str = "latest",
    games: int = CARD_ACQUIRE_ELO_TEST_GAMES,
    deterministic: bool = True,
    k_factor: float = CARD_ACQUIRE_ELO_K_FACTOR,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    if games <= 0:
        raise ValueError("games must be positive.")
    if not math.isfinite(float(k_factor)) or float(k_factor) <= 0.0:
        raise ValueError("k_factor must be greater than 0.")

    trainer = _get_trainer(run_name)
    config = trainer.config
    state = trainer.state
    resolved_checkpoint = _resolve_checkpoint_name(state, checkpoint) or str(checkpoint or "latest")
    start_time = time.time()
    worker_count = resolve_simulation_workers(config.simulation_workers, int(games))
    simulation_pool = (
        _SimulationPool(worker_count)
        if progress_callback is not None and worker_count > 1 and int(games) > 1
        else None
    )
    try:
        policy = load_policy(run_name, checkpoint)
    except Exception:
        if simulation_pool is not None:
            simulation_pool.close()
        raise
    temperature = config.eval_temperature if deterministic else config.train_temperature
    epsilon_random = 0.0 if deterministic else config.epsilon_random

    future_trade_state = _initial_card_choice_rating_state()
    immediate_trade_state = _initial_card_choice_rating_state()
    turn_summaries = 0
    eligible_single_acquire_turns = 0
    future_trade_scored_decisions = 0
    future_trade_pairwise_comparisons = 0
    immediate_trade_scored_decisions = 0
    immediate_trade_pairwise_comparisons = 0
    ended_by_limit_games = 0

    def apply_chunk_result(chunk_result: Dict[str, Any]) -> None:
        nonlocal turn_summaries
        nonlocal eligible_single_acquire_turns
        nonlocal future_trade_scored_decisions
        nonlocal future_trade_pairwise_comparisons
        nonlocal immediate_trade_scored_decisions
        nonlocal immediate_trade_pairwise_comparisons
        nonlocal ended_by_limit_games

        ended_by_limit_games += int(chunk_result.get("ended_by_limit_games", 0))
        turn_summaries += int(chunk_result.get("turn_summaries", 0))
        eligible_single_acquire_turns += int(chunk_result.get("eligible_single_acquire_turns", 0))
        for decision in list(chunk_result.get("decisions") or []):
            winner_name = str(decision.get("winner_name", "")).strip()
            if not winner_name:
                continue
            future_trade_comparisons = _apply_card_acquire_choice_result(
                future_trade_state,
                winner_name,
                list(decision.get("future_trade_loser_names") or []),
                float(k_factor),
            )
            if future_trade_comparisons > 0:
                future_trade_scored_decisions += 1
                future_trade_pairwise_comparisons += future_trade_comparisons

            immediate_trade_comparisons = _apply_card_acquire_choice_result(
                immediate_trade_state,
                winner_name,
                list(decision.get("immediate_trade_loser_names") or []),
                float(k_factor),
            )
            if immediate_trade_comparisons > 0:
                immediate_trade_scored_decisions += 1
                immediate_trade_pairwise_comparisons += immediate_trade_comparisons

    def emit_progress(games_completed: int) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(
                {
                    "games_completed": int(games_completed),
                    "games_target": int(games),
                    "duration_seconds": time.time() - start_time,
                }
            )
        except Exception:
            pass

    chunk_sizes = _simulation_chunk_sizes(int(games), worker_count)
    games_completed = 0

    if worker_count <= 1 or int(games) <= 1:
        for chunk_size in chunk_sizes:
            chunk_turn_summaries = []
            chunk_ended_by_limit_games = 0
            for _ in range(chunk_size):
                actor_a = PolicyActor(
                    policy,
                    deterministic=deterministic,
                    temperature=temperature,
                    epsilon_random=epsilon_random,
                )
                actor_b = PolicyActor(
                    policy,
                    deterministic=deterministic,
                    temperature=temperature,
                    epsilon_random=epsilon_random,
                )
                game_turn_summaries: List[Dict[str, Any]] = []

                def capture_turn_summary(summary: Dict[str, Any]) -> None:
                    game_turn_summaries.append(_copy_nested(summary))

                game = Game(
                    "policy_a",
                    "policy_b",
                    p1_choose=actor_a.choose,
                    p2_choose=actor_b.choose,
                    verbose=False,
                    max_turns=config.max_turns_per_game,
                    max_actions_per_turn=config.max_actions_per_turn,
                    turn_summary_callback=capture_turn_summary,
                )
                if game.ended_by_limit:
                    chunk_ended_by_limit_games += 1
                chunk_turn_summaries.extend(game_turn_summaries)
                games_completed += 1
                emit_progress(games_completed)
            extracted = _extract_card_acquire_decisions(chunk_turn_summaries)
            extracted["games_completed"] = chunk_size
            extracted["ended_by_limit_games"] = chunk_ended_by_limit_games
            apply_chunk_result(extracted)
    else:
        policy_payload = _policy_payload_for_simulation(policy)
        config_payload = _config_payload_for_simulation(config)
        tasks = [
            {
                "policy_payload": policy_payload,
                "policy_cache_key": f"{run_name}:acquire:{resolved_checkpoint}:{start_time}:policy",
                "config_payload": config_payload,
                "game_count": chunk_size,
                "deterministic": bool(deterministic),
                "seed": _random_seed(),
            }
            for chunk_size in chunk_sizes
        ]
        completed_from_results = 0
        if progress_callback is not None and simulation_pool is not None:
            progress_queue = simulation_pool.progress_queue
            try:
                simulation_pool.wait_until_ready()
                _drain_simulation_progress_queue(progress_queue)
                futures = [
                    simulation_pool.executor.submit(_simulate_card_acquire_elo_chunk_worker, task)
                    for task in tasks
                ]

                def on_progress(delta: int) -> None:
                    nonlocal games_completed
                    games_completed = min(int(games), games_completed + int(delta))
                    emit_progress(games_completed)

                def on_result(chunk_result: Dict[str, Any]) -> None:
                    nonlocal games_completed, completed_from_results
                    completed_from_results += int(chunk_result.get("games_completed", 0))
                    games_completed = max(games_completed, min(int(games), completed_from_results))
                    apply_chunk_result(chunk_result)
                    emit_progress(games_completed)

                _consume_simulation_futures(futures, progress_queue, on_progress, on_result)
            finally:
                simulation_pool.close()
                simulation_pool = None
        else:
            with _process_pool_executor(
                max_workers=worker_count,
                initializer=_simulation_progress_worker_init,
                initargs=(None,),
            ) as executor:
                futures = [executor.submit(_simulate_card_acquire_elo_chunk_worker, task) for task in tasks]
                for future in concurrent.futures.as_completed(futures):
                    chunk_result = future.result()
                    completed_from_results += int(chunk_result.get("games_completed", 0))
                    games_completed = max(games_completed, min(int(games), completed_from_results))
                    apply_chunk_result(chunk_result)

    future_trade_leaderboard, future_trade_normalization_factor, future_trade_explorer_raw_elo = _normalized_card_choice_leaderboard(
        future_trade_state
    )
    immediate_trade_leaderboard, immediate_trade_normalization_factor, immediate_trade_explorer_raw_elo = _normalized_card_choice_leaderboard(
        immediate_trade_state
    )

    duration_seconds = time.time() - start_time
    result: Dict[str, Any] = {
        "run_name": run_name,
        "checkpoint": checkpoint,
        "resolved_checkpoint": resolved_checkpoint,
        "games": int(games),
        "deterministic": bool(deterministic),
        "k_factor": float(k_factor),
        "simulation_workers": int(config.simulation_workers),
        "resolved_simulation_workers": resolve_simulation_workers(config.simulation_workers, int(games)),
        "ended_by_limit_games": ended_by_limit_games,
        "turn_summaries": turn_summaries,
        "eligible_single_acquire_turns": eligible_single_acquire_turns,
        "rating_model": "multinomial_elo_plackett_luce",
        "future_trade": {
            "scored_decisions": future_trade_scored_decisions,
            "pairwise_comparisons": future_trade_pairwise_comparisons,
            "normalization_factor": future_trade_normalization_factor,
            "explorer_raw_elo": future_trade_explorer_raw_elo,
            "leaderboard": future_trade_leaderboard,
        },
        "immediate_trade": {
            "scored_decisions": immediate_trade_scored_decisions,
            "pairwise_comparisons": immediate_trade_pairwise_comparisons,
            "normalization_factor": immediate_trade_normalization_factor,
            "explorer_raw_elo": immediate_trade_explorer_raw_elo,
            "leaderboard": immediate_trade_leaderboard,
        },
        "scored_decisions": future_trade_scored_decisions,
        "pairwise_comparisons": future_trade_pairwise_comparisons,
        "normalization_factor": future_trade_normalization_factor,
        "explorer_raw_elo": future_trade_explorer_raw_elo,
        "leaderboard": future_trade_leaderboard,
        "duration_seconds": duration_seconds,
    }
    report_text = format_card_acquire_elo_test_report(result)
    result["report_text"] = report_text

    files = trainer.files
    files.analysis_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_slug = _safe_slug(resolved_checkpoint)
    timestamp_slug = time.strftime("%Y%m%d-%H%M%S")
    report_stem = f"card_acquire_elo_{checkpoint_slug}_{timestamp_slug}"
    report_path = files.analysis_dir / f"{report_stem}.txt"
    json_path = files.analysis_dir / f"{report_stem}.json"
    result["report_path"] = str(report_path.resolve())
    result["json_path"] = str(json_path.resolve())
    report_path.write_text(report_text, encoding="utf-8")
    _atomic_write_json(json_path, result)
    return result


def format_scrap_elo_test_report(result: Dict[str, Any]) -> str:
    leaderboard = list(result.get("leaderboard") or [])
    lines = [
        f"Scrap Elo Test for {result.get('run_name', '-')}/{result.get('checkpoint', '-')}",
        f"Resolved checkpoint: {result.get('resolved_checkpoint', '-')}",
        f"Games: {result.get('games', 0)}",
        f"Deterministic: {result.get('deterministic', True)}",
        f"Ended by limit: {result.get('ended_by_limit_games', 0)} game(s)",
        f"Scrapany decision points: {result.get('scrapany_decisions', 0)}",
        f"Eligible scrapany decisions: {result.get('eligible_scrapany_decisions', 0)}",
        f"Scored decisions: {result.get('scored_decisions', 0)}",
        f"Pairwise comparisons: {result.get('pairwise_comparisons', 0)}",
        "Rating update model: pairwise Elo; the selected option wins one match against each unselected unique option.",
        "Options are unique by source plus exact card title; noscrap is its own option.",
        "Higher Elo means the policy is more likely to choose that option when resolving scrapany.",
        "Uncertainty: approximate 1-sigma Elo standard error from diagonal Fisher information of the same pairwise model.",
        f"Duration: {_format_seconds(float(result.get('duration_seconds', 0.0)))}",
        "",
        "Option Rankings",
    ]
    for index, entry in enumerate(leaderboard, start=1):
        uncertainty = entry.get("uncertainty")
        uncertainty_text = "-" if uncertainty is None else f"{float(uncertainty):.2f}"
        lines.append(
            f"{index:>2}. {entry.get('option_label', '-'):<26} "
            f"Elo {float(entry.get('elo', 0.0)):>8.2f}  "
            f"+/- {uncertainty_text:>8}  "
            f"dec {int(entry.get('decision_count', 0)):>4}  "
            f"cmp {int(entry.get('pairwise_comparisons', 0)):>5}"
        )
    return "\n".join(lines)


def _simulate_scrap_elo_games(
    policy: PolicyNetwork,
    config: TrainingConfig,
    *,
    game_count: int,
    deterministic: bool,
    seed: Optional[int] = None,
    game_progress_callback: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    if seed is not None:
        _seed_simulation(int(seed))
    temperature = config.eval_temperature if deterministic else config.train_temperature
    epsilon_random = 0.0 if deterministic else config.epsilon_random

    games_completed = 0
    ended_by_limit_games = 0
    scrapany_decisions = 0
    eligible_scrapany_decisions = 0
    decisions: List[Dict[str, Any]] = []

    def capture_scrap_decision(
        _player_name: str,
        options: Sequence[Sequence[Any]],
        _known_game_state: Dict[str, Any],
        action_index: int,
    ) -> None:
        nonlocal scrapany_decisions, eligible_scrapany_decisions
        decision = _extract_scrap_elo_decision_from_choice(options, int(action_index))
        if decision is None:
            return
        scrapany_decisions += 1
        if int(decision.get("unique_option_count", 0)) <= 1:
            return
        eligible_scrapany_decisions += 1
        decisions.append(decision)

    for _ in range(max(0, int(game_count))):
        actor_a = PolicyActor(
            policy,
            deterministic=deterministic,
            temperature=temperature,
            epsilon_random=epsilon_random,
            decision_callback=capture_scrap_decision,
        )
        actor_b = PolicyActor(
            policy,
            deterministic=deterministic,
            temperature=temperature,
            epsilon_random=epsilon_random,
            decision_callback=capture_scrap_decision,
        )
        game = Game(
            "policy_a",
            "policy_b",
            p1_choose=actor_a.choose,
            p2_choose=actor_b.choose,
            verbose=False,
            max_turns=config.max_turns_per_game,
            max_actions_per_turn=config.max_actions_per_turn,
        )
        games_completed += 1
        if game.ended_by_limit:
            ended_by_limit_games += 1
        _emit_simulation_game_progress()
        if game_progress_callback is not None:
            try:
                game_progress_callback()
            except Exception:
                pass

    return {
        "games_completed": games_completed,
        "ended_by_limit_games": ended_by_limit_games,
        "scrapany_decisions": scrapany_decisions,
        "eligible_scrapany_decisions": eligible_scrapany_decisions,
        "decisions": decisions,
    }


def _simulate_scrap_elo_chunk_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    _configure_simulation_worker_runtime()
    policy = _policy_from_simulation_payload(task, "policy_payload", "policy_cache_key")
    config = _config_from_payload(task["config_payload"])
    deterministic = bool(task.get("deterministic", True))
    return _simulate_scrap_elo_games(
        policy,
        config,
        game_count=max(0, int(task["game_count"])),
        deterministic=deterministic,
        seed=int(task["seed"]),
    )


def run_scrap_elo_test(
    run_name: str = LATEST_RUN_NAME,
    checkpoint: str = "latest",
    games: int = SCRAP_ELO_TEST_GAMES,
    deterministic: bool = True,
    k_factor: float = SCRAP_ELO_K_FACTOR,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    if games <= 0:
        raise ValueError("games must be positive.")
    if not math.isfinite(float(k_factor)) or float(k_factor) <= 0.0:
        raise ValueError("k_factor must be greater than 0.")

    trainer = _get_trainer(run_name)
    config = trainer.config
    state = trainer.state
    resolved_checkpoint = _resolve_checkpoint_name(state, checkpoint) or str(checkpoint or "latest")
    start_time = time.time()
    worker_count = resolve_simulation_workers(config.simulation_workers, int(games))
    simulation_pool = (
        _SimulationPool(worker_count)
        if progress_callback is not None and worker_count > 1 and int(games) > 1
        else None
    )
    try:
        policy = load_policy(run_name, checkpoint)
    except Exception:
        if simulation_pool is not None:
            simulation_pool.close()
        raise

    rating_state = _initial_scrap_choice_rating_state()
    scrapany_decisions = 0
    eligible_scrapany_decisions = 0
    scored_decisions = 0
    pairwise_comparisons = 0
    ended_by_limit_games = 0

    def apply_chunk_result(chunk_result: Dict[str, Any]) -> None:
        nonlocal scrapany_decisions
        nonlocal eligible_scrapany_decisions
        nonlocal scored_decisions
        nonlocal pairwise_comparisons
        nonlocal ended_by_limit_games

        ended_by_limit_games += int(chunk_result.get("ended_by_limit_games", 0))
        scrapany_decisions += int(chunk_result.get("scrapany_decisions", 0))
        eligible_scrapany_decisions += int(chunk_result.get("eligible_scrapany_decisions", 0))
        for decision in list(chunk_result.get("decisions") or []):
            comparisons = _apply_scrap_choice_result(
                rating_state,
                dict(decision.get("winner_option") or {}),
                list(decision.get("loser_options") or []),
                float(k_factor),
            )
            if comparisons > 0:
                scored_decisions += 1
                pairwise_comparisons += comparisons

    def emit_progress(games_completed: int) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(
                {
                    "games_completed": int(games_completed),
                    "games_target": int(games),
                    "duration_seconds": time.time() - start_time,
                }
            )
        except Exception:
            pass

    chunk_sizes = _simulation_chunk_sizes(int(games), worker_count)
    games_completed = 0

    if worker_count <= 1 or int(games) <= 1:
        for chunk_size in chunk_sizes:
            def on_game_complete() -> None:
                nonlocal games_completed
                games_completed = min(int(games), games_completed + 1)
                emit_progress(games_completed)

            chunk_result = _simulate_scrap_elo_games(
                policy,
                config,
                game_count=chunk_size,
                deterministic=bool(deterministic),
                game_progress_callback=on_game_complete,
            )
            apply_chunk_result(chunk_result)
    else:
        policy_payload = _policy_payload_for_simulation(policy)
        config_payload = _config_payload_for_simulation(config)
        tasks = [
            {
                "policy_payload": policy_payload,
                "policy_cache_key": f"{run_name}:scrap:{resolved_checkpoint}:{start_time}:policy",
                "config_payload": config_payload,
                "game_count": chunk_size,
                "deterministic": bool(deterministic),
                "seed": _random_seed(),
            }
            for chunk_size in chunk_sizes
        ]
        completed_from_results = 0
        if progress_callback is not None and simulation_pool is not None:
            progress_queue = simulation_pool.progress_queue
            try:
                simulation_pool.wait_until_ready()
                _drain_simulation_progress_queue(progress_queue)
                futures = [
                    simulation_pool.executor.submit(_simulate_scrap_elo_chunk_worker, task)
                    for task in tasks
                ]

                def on_progress(delta: int) -> None:
                    nonlocal games_completed
                    games_completed = min(int(games), games_completed + int(delta))
                    emit_progress(games_completed)

                def on_result(chunk_result: Dict[str, Any]) -> None:
                    nonlocal games_completed, completed_from_results
                    completed_from_results += int(chunk_result.get("games_completed", 0))
                    games_completed = max(games_completed, min(int(games), completed_from_results))
                    apply_chunk_result(chunk_result)
                    emit_progress(games_completed)

                _consume_simulation_futures(futures, progress_queue, on_progress, on_result)
            finally:
                simulation_pool.close()
                simulation_pool = None
        else:
            with _process_pool_executor(
                max_workers=worker_count,
                initializer=_simulation_progress_worker_init,
                initargs=(None,),
            ) as executor:
                futures = [executor.submit(_simulate_scrap_elo_chunk_worker, task) for task in tasks]
                for future in concurrent.futures.as_completed(futures):
                    chunk_result = future.result()
                    completed_from_results += int(chunk_result.get("games_completed", 0))
                    games_completed = max(games_completed, min(int(games), completed_from_results))
                    apply_chunk_result(chunk_result)

    leaderboard = _scrap_choice_leaderboard(rating_state)
    duration_seconds = time.time() - start_time
    result: Dict[str, Any] = {
        "run_name": run_name,
        "checkpoint": checkpoint,
        "resolved_checkpoint": resolved_checkpoint,
        "games": int(games),
        "deterministic": bool(deterministic),
        "k_factor": float(k_factor),
        "simulation_workers": int(config.simulation_workers),
        "resolved_simulation_workers": worker_count,
        "ended_by_limit_games": ended_by_limit_games,
        "scrapany_decisions": scrapany_decisions,
        "eligible_scrapany_decisions": eligible_scrapany_decisions,
        "scored_decisions": scored_decisions,
        "pairwise_comparisons": pairwise_comparisons,
        "rating_model": "pairwise_elo",
        "leaderboard": leaderboard,
        "duration_seconds": duration_seconds,
    }
    report_text = format_scrap_elo_test_report(result)
    result["report_text"] = report_text

    files = trainer.files
    files.analysis_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_slug = _safe_slug(resolved_checkpoint)
    timestamp_slug = time.strftime("%Y%m%d-%H%M%S")
    report_stem = f"scrap_elo_{checkpoint_slug}_{timestamp_slug}"
    report_path = files.analysis_dir / f"{report_stem}.txt"
    json_path = files.analysis_dir / f"{report_stem}.json"
    result["report_path"] = str(report_path.resolve())
    result["json_path"] = str(json_path.resolve())
    report_path.write_text(report_text, encoding="utf-8")
    _atomic_write_json(json_path, result)
    return result


def play_self_game(
    run_name: str = LATEST_RUN_NAME,
    checkpoint: str = "latest",
    deterministic: bool = True,
    verbose: bool = True,
    ui_observer: Optional[Any] = None,
) -> Dict[str, Any]:
    policy_a = load_policy(run_name, checkpoint)
    policy_b = load_policy(run_name, checkpoint)
    config = _get_trainer(run_name).config
    if ui_observer is not None:
        ui_observer.start_session(f"Star Realms Self-Play: {run_name} / {checkpoint}")
    actor_a = PolicyActor(
        policy_a,
        deterministic=deterministic,
        temperature=config.eval_temperature if deterministic else config.train_temperature,
        epsilon_random=0.0 if deterministic else config.epsilon_random,
        decision_callback=None if ui_observer is None else ui_observer.policy_choice,
    )
    actor_b = PolicyActor(
        policy_b,
        deterministic=deterministic,
        temperature=config.eval_temperature if deterministic else config.train_temperature,
        epsilon_random=0.0 if deterministic else config.epsilon_random,
        decision_callback=None if ui_observer is None else ui_observer.policy_choice,
    )
    game = Game(
        "policy_a",
        "policy_b",
        p1_choose=actor_a.choose,
        p2_choose=actor_b.choose,
        verbose=verbose,
        max_turns=config.max_turns_per_game,
        max_actions_per_turn=config.max_actions_per_turn,
    )
    result = {
        "winner": game.winner.name,
        "checkpoint": checkpoint,
        "run_name": run_name,
        "ended_by_limit": game.ended_by_limit,
        "turns_taken": game.turnsTaken,
    }
    if ui_observer is not None:
        ui_observer.finish_session(result)
    return result


def play_policy_match(
    run_name_a: str,
    checkpoint_a: str = "latest",
    run_name_b: Optional[str] = None,
    checkpoint_b: str = "latest",
    games_per_match: int = 24,
    deterministic: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    if not str(run_name_a).strip():
        raise ValueError("run_name_a is required.")
    resolved_run_b = str(run_name_b or run_name_a).strip() or run_name_a
    if games_per_match <= 0:
        raise ValueError("games_per_match must be positive.")

    config_a = _get_trainer(run_name_a).config
    config_b = _get_trainer(resolved_run_b).config
    config = config_a.merged(
        {
            "max_turns_per_game": max(int(config_a.max_turns_per_game), int(config_b.max_turns_per_game)),
            "max_actions_per_turn": max(int(config_a.max_actions_per_turn), int(config_b.max_actions_per_turn)),
            "eval_temperature": min(float(config_a.eval_temperature), float(config_b.eval_temperature)),
        }
    )
    total_games = max(1, int(games_per_match))
    worker_count = resolve_simulation_workers(config.simulation_workers, total_games)
    simulation_pool = (
        _SimulationPool(worker_count)
        if progress_callback is not None and worker_count > 1 and total_games > 1
        else None
    )
    try:
        policy_a = load_policy(run_name_a, checkpoint_a)
        policy_b = load_policy(resolved_run_b, checkpoint_b)
        summary = _play_balanced_match(
            policy_a,
            policy_b,
            config,
            games_per_match=total_games,
            game_progress_callback=progress_callback,
            simulation_pool=simulation_pool,
            cache_key_prefix=(
                f"policy-match:{run_name_a}:{checkpoint_a}:"
                f"{resolved_run_b}:{checkpoint_b}:{time.time()}"
            ),
        )
    finally:
        if simulation_pool is not None:
            simulation_pool.close()
    wins_a = int(summary.get("wins_a", 0))
    wins_b = int(summary.get("wins_b", 0))
    if wins_a > wins_b:
        winner = "policy_a"
    elif wins_b > wins_a:
        winner = "policy_b"
    else:
        winner = "draw"
    return {
        "policy_a": {"run_name": run_name_a, "checkpoint": checkpoint_a},
        "policy_b": {"run_name": resolved_run_b, "checkpoint": checkpoint_b},
        "games_per_match": int(games_per_match),
        "deterministic": bool(deterministic),
        "simulation_workers": int(config.simulation_workers),
        "resolved_simulation_workers": resolve_simulation_workers(config.simulation_workers, int(games_per_match)),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "games_played": int(summary.get("games_played", 0)),
        "score_a": wins_a / max(int(summary.get("games_played", 0)), 1),
        "score_b": wins_b / max(int(summary.get("games_played", 0)), 1),
        "winner": winner,
        "duration_seconds": float(summary.get("duration_seconds", 0.0)),
        "seat_results": list(summary.get("seat_results", [])),
    }


def play_human_game(
    run_name: str = LATEST_RUN_NAME,
    checkpoint: str = "latest",
    human_name: str = "Human",
    policy_name: str = "Policy",
    human_choose_fn: Optional[Any] = None,
    verbose: bool = True,
    ui_observer: Optional[Any] = None,
) -> Dict[str, Any]:
    policy = load_policy(run_name, checkpoint)
    config = _get_trainer(run_name).config
    if ui_observer is not None:
        ui_observer.start_session(f"Star Realms Human Game: {run_name} / {checkpoint}")
    actor = PolicyActor(
        policy,
        deterministic=True,
        temperature=config.eval_temperature,
        epsilon_random=0.0,
        decision_callback=None if ui_observer is None else ui_observer.policy_choice,
    )
    chooser_fn = human_choose if human_choose_fn is None else human_choose_fn
    game = Game(
        human_name,
        policy_name,
        p1_choose=chooser_fn,
        p2_choose=actor.choose,
        verbose=verbose,
        max_turns=config.max_turns_per_game,
        max_actions_per_turn=config.max_actions_per_turn,
    )
    result = {
        "winner": game.winner.name,
        "checkpoint": checkpoint,
        "run_name": run_name,
        "ended_by_limit": game.ended_by_limit,
        "turns_taken": game.turnsTaken,
    }
    if ui_observer is not None:
        ui_observer.finish_session(result)
    return result


def _cached_policy(run_name: str, checkpoint: str) -> PolicyNetwork:
    files, _, state, _ = _load_policy_and_state(run_name)
    if checkpoint == "latest":
        path = files.policy_file
    elif checkpoint == "candidate":
        path = files.candidate_policy_file
    elif checkpoint == "best":
        path = files.checkpoints_dir / state.get("best_checkpoint", state.get("latest_checkpoint"))
    else:
        path = files.checkpoints_dir / checkpoint
    mtime = path.stat().st_mtime if path.exists() else 0.0
    cache_key = (run_name, checkpoint)
    with _POLICY_CACHE_LOCK:
        cached = _POLICY_CACHE.get(cache_key)
    if cached and cached[0] == mtime:
        return cached[1]
    policy = load_policy(run_name, checkpoint)
    with _POLICY_CACHE_LOCK:
        _POLICY_CACHE[cache_key] = (mtime, policy)
    return policy


def choose_with_saved_policy(
    player_name: str,
    options: Sequence[Sequence[Any]],
    known_game_state: Dict[str, Any],
    run_name: str = LATEST_RUN_NAME,
    checkpoint: str = "latest",
    deterministic: bool = True,
) -> int:
    policy = _cached_policy(run_name, checkpoint)
    config = _get_trainer(run_name).config
    actor = PolicyActor(
        policy,
        deterministic=deterministic,
        temperature=config.eval_temperature if deterministic else config.train_temperature,
        epsilon_random=0.0 if deterministic else config.epsilon_random,
        collector=None,
    )
    return actor.choose(player_name, options, known_game_state)
