"""M4-oriented asynchronous self-play and MLX learner loop.

CPU actor processes run the deterministic engine and lightweight NumPy model
while the parent process performs replay updates on Metal.  The overlap keeps
the M4 CPU and GPU useful at the same time without giving every worker its own
large accelerator context.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import time
import uuid
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import SafetensorError, safe_open

from .config import MINIMUM_PROMOTION_PAIRS, RunConfig
from .encoding import FAMILY_COUNT, DecisionFamily, Encoder
from .hardware import RateMeter, mlx_snapshot, system_snapshot
from .league import League, Opponent
from .model import (
    ModelSpec,
    NumpyActor,
    actor_critic_policy_loss,
    bootstrap_bce_loss,
    build_model,
    export_actor,
    load_model,
    load_optimizer_state,
    preference_ranking_loss,
    save_model,
    save_optimizer_state,
)
from .replay import (
    NATURAL_SAMPLING_WEIGHTS,
    POLICY_ROLLOUT_SOURCE_IDS,
    POLICY_ROLLOUT_SOURCE_NAMES,
    GameBalancedPolicyReplayBuffer,
    PreferenceReplayBuffer,
    ReplayBuffer,
    policy_opponent_key,
)
from .retention import RetentionSafetyError, cleanup_previous_checkpoint_npz
from .selfplay import ActorPolicy, WorkerResult, collect_worker_batch
from .storage import Store

_TRAINING_STATE_SCHEMA_VERSION = 2
_EVALUATION_RETRY_BASE_SECONDS = 30.0
_EVALUATION_RETRY_MAX_SECONDS = 15.0 * 60.0
_EVALUATION_RETRY_SEED_STRIDE = 1_000_003
_FINAL_EVALUATION_MAX_ATTEMPTS = 3
_GIB = 1 << 30
_MIB = 1 << 20
# Policy rows contain a state vector, a ragged legal-action matrix, several
# small arrays, and Python/NumPy object overhead.  Budgeting only their float16
# payload substantially understated real resident memory in the first Astro5
# runs.  This deliberately conservative estimate is used only to cap capacity.
_POLICY_REPLAY_BUDGET_BYTES_PER_DECISION = 8_192
_TRAINING_STATE_REQUIRED_COUNTERS = frozenset(
    {
        "games",
        "decisions",
        "updates",
        "samples",
        "player_0_wins",
        "player_1_wins",
        "draws",
        "truncations",
        "turns",
        "forced_choices",
        "rollout_games",
        "seed_cursor",
    }
)


@dataclass(frozen=True, slots=True)
class _RolloutPlan:
    actor_paths: tuple[str | None, str | None]
    baseline_names: tuple[str, str]
    collect_players: tuple[bool, bool]
    epsilons: tuple[float, float]
    seed: int
    games: int
    kind: str
    opponent_id: str | None
    current_player: int | None
    deployment_policy: tuple[bool, bool] = (False, False)


@dataclass(slots=True)
class _Totals:
    games: int
    decisions: int
    updates: int
    samples: int = 0
    player_wins: tuple[int, int] = (0, 0)
    draws: int = 0
    truncated: int = 0
    turns: int = 0
    forced_choices: int = 0
    counterfactual_preferences: int = 0
    reanalysis_positions: int = 0
    search_repeatability_positions: int = 0
    search_top_action_agreements: int = 0
    search_policy_js_sum: float = 0.0
    search_value_abs_delta_sum: float = 0.0
    rollout_games: dict[str, int] = field(default_factory=dict)
    rollout_completed_games: dict[str, int] = field(default_factory=dict)
    rollout_scores: dict[str, float] = field(default_factory=dict)
    opponent_games: dict[str, int] = field(default_factory=dict)
    opponent_scores: dict[str, float] = field(default_factory=dict)


def _record_opponent_result(
    totals: _Totals,
    league: League,
    plan: _RolloutPlan,
    result: WorkerResult,
) -> None:
    """Track every named opponent while updating PFSP only for league members."""

    if plan.opponent_id is None or plan.current_player is None:
        return
    completed_games = result.games - result.truncated
    if completed_games <= 0:
        return
    score = result.wins[plan.current_player] + 0.5 * result.draws
    totals.rollout_completed_games[plan.kind] = (
        totals.rollout_completed_games.get(plan.kind, 0) + completed_games
    )
    totals.rollout_scores[plan.kind] = totals.rollout_scores.get(plan.kind, 0.0) + score
    totals.opponent_games[plan.opponent_id] = (
        totals.opponent_games.get(plan.opponent_id, 0) + completed_games
    )
    totals.opponent_scores[plan.opponent_id] = (
        totals.opponent_scores.get(plan.opponent_id, 0.0) + score
    )
    if plan.kind in {"league", "baseline"}:
        league.record(plan.opponent_id, score, completed_games)


@dataclass(frozen=True, slots=True)
class _EvaluationPlan:
    tier: str
    cadence_games: int
    pairs: int
    automatic_promotion: bool


class _ActiveElapsedClock:
    """Monotonic active-training time that excludes an in-progress pause."""

    def __init__(
        self,
        previous_seconds: float = 0.0,
        *,
        now: Callable[[], float] = time.monotonic,
    ):
        self._previous_seconds = max(0.0, float(previous_seconds))
        self._now = now
        self._session_started = float(now())
        self._paused_at: float | None = None
        self._paused_seconds = 0.0

    def pause(self) -> None:
        if self._paused_at is None:
            self._paused_at = float(self._now())

    def resume(self) -> None:
        if self._paused_at is None:
            return
        now = float(self._now())
        self._paused_seconds += max(0.0, now - self._paused_at)
        self._paused_at = None

    def value(self) -> float:
        now = float(self._now())
        current_pause = max(0.0, now - self._paused_at) if self._paused_at is not None else 0.0
        session_active = max(
            0.0,
            now - self._session_started - self._paused_seconds - current_pause,
        )
        return self._previous_seconds + session_active


def _evaluation_plan(config: RunConfig, games: int) -> _EvaluationPlan:
    """Scale evaluation cost as a run moves from bootstrap to mature play."""

    full_automatic = config.evaluation_pairs >= 2_000
    if config.training_generation >= 5 and config.canary_every_games:
        return _EvaluationPlan(
            tier="canary",
            cadence_games=config.canary_every_games,
            pairs=config.canary_pairs,
            automatic_promotion=False,
        )
    if not config.adaptive_evaluation or not full_automatic:
        return _EvaluationPlan(
            tier="full" if full_automatic else "diagnostic",
            cadence_games=config.evaluate_every_games,
            pairs=config.evaluation_pairs,
            automatic_promotion=full_automatic,
        )

    # The configured interval marks the end of bootstrap; twice that interval
    # marks the point where every promotion uses the full conservative gate.
    if games < config.evaluate_every_games:
        return _EvaluationPlan(
            tier="provisional",
            cadence_games=config.checkpoint_every_games,
            pairs=min(
                config.evaluation_pairs,
                max(MINIMUM_PROMOTION_PAIRS, config.evaluation_pairs // 25),
            ),
            automatic_promotion=True,
        )
    if games < 2 * config.evaluate_every_games:
        return _EvaluationPlan(
            tier="development",
            cadence_games=max(config.checkpoint_every_games, config.evaluate_every_games // 2),
            pairs=min(config.evaluation_pairs, max(1_000, config.evaluation_pairs // 5)),
            automatic_promotion=True,
        )
    return _EvaluationPlan(
        tier="full",
        cadence_games=config.evaluate_every_games,
        pairs=config.evaluation_pairs,
        automatic_promotion=True,
    )


def _epsilon(config: RunConfig, games: int, exploration_multiplier: float = 1.0) -> float:
    if games >= config.epsilon_decay_games:
        scheduled = config.epsilon_end
    else:
        progress = min(1.0, games / max(1, config.epsilon_decay_games))
        scheduled = config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)
    return min(1.0, scheduled * config.exploration_decision_scale * exploration_multiplier)


def _learning_rate(
    config: RunConfig,
    completed_updates: int,
    updates_since_optimizer_reset: int,
) -> float:
    if config.learning_rate_schedule == "cosine_restarts":
        cycle = max(0, completed_updates) // config.learning_rate_restart_updates
        cycle_updates = max(0, completed_updates) % config.learning_rate_restart_updates
        progress = cycle_updates / config.learning_rate_restart_updates
        peak = max(
            config.min_learning_rate,
            config.learning_rate * config.learning_rate_restart_decay**cycle,
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduled = config.min_learning_rate + (peak - config.min_learning_rate) * cosine
    else:
        # The decay horizon is immutable for a run. Extending wall-clock
        # duration therefore cannot rewind optimization to a larger rate.
        progress = min(1.0, max(0.0, completed_updates / config.learning_rate_decay_updates))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduled = (
            config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine
        )
    # A brief optimizer warmup protects a fresh randomly initialized value net
    # from the unusually correlated first replay batches.
    warmup = min(1.0, (updates_since_optimizer_reset + 1) / 500.0)
    return max(config.min_learning_rate * warmup, scheduled * warmup)


def _memory_safety_limits(total_memory_bytes: int) -> dict[str, int]:
    """Return a conservative unified-memory envelope for the local learner.

    Metal allocations, Python replay, actor processes, WindowServer, and the
    filesystem cache all share physical memory on Apple silicon.  MLX's default
    cache can otherwise retain most of a 16 GB machine even when only a few
    megabytes are actively used by the model.
    """

    total = max(4 * _GIB, int(total_memory_bytes))
    # Keep at least a quarter of RAM outside the learner.  This is especially
    # important on 16 GB systems, where WindowServer watchdog failures occur
    # before ordinary allocation errors provide a useful process boundary.
    minimum_available = max(3 * _GIB, total // 4)
    critical_available = max(2 * _GIB, total // 8)
    mlx_cache = max(128 * _MIB, min(512 * _MIB, total // 32))
    mlx_memory = max(1 * _GIB, min(4 * _GIB, total // 4))
    replay_budget = max(384 * _MIB, total // 8)
    raw_capacity = replay_budget // _POLICY_REPLAY_BUDGET_BYTES_PER_DECISION
    # Round down to make the effective limit stable and easy to explain in
    # telemetry.  Even an 8 GB machine retains a useful 100k+ decision window.
    policy_capacity = max(50_000, (raw_capacity // 50_000) * 50_000)
    # macOS has no dependable per-process swap quota. Bound system swap here
    # so retained actor heaps cannot silently consume tens of gigabytes while
    # the ordinary available-RAM signal still looks non-critical.
    maximum_swap_used = max(2 * _GIB, min(8 * _GIB, total // 2))
    minimum_swap_free = max(1 * _GIB, total // 8)
    return {
        "total_memory_bytes": total,
        "minimum_available_bytes": minimum_available,
        "critical_available_bytes": critical_available,
        "mlx_cache_limit_bytes": mlx_cache,
        "mlx_memory_limit_bytes": mlx_memory,
        "policy_replay_capacity": policy_capacity,
        "maximum_swap_used_bytes": maximum_swap_used,
        "minimum_swap_free_bytes": minimum_swap_free,
    }


def _swap_pressure_is_critical(
    system: dict[str, Any],
    limits: dict[str, int],
) -> bool:
    """Return whether swap pressure justifies protective intervention.

    macOS grows its swap pool dynamically, so a small reported free remainder
    is not by itself evidence that the machine is in danger. Treat that signal
    as critical only when physical-memory headroom is also critical. A large
    amount of swap already in use remains an independent backstop.
    """

    if int(system["swap_total_bytes"]) <= 0:
        return False
    if int(system["swap_used_bytes"]) > limits["maximum_swap_used_bytes"]:
        return True
    return (
        int(system["swap_free_bytes"]) < limits["minimum_swap_free_bytes"]
        and int(system["memory_available_bytes"]) < limits["critical_available_bytes"]
    )


def _plateau_status(store: Store, run_id: str, config: RunConfig) -> dict[str, Any]:
    """Summarize recent promotion evidence and choose a bounded response."""

    consecutive = 0
    recent_scores: list[float] = []
    for job in reversed(_completed_trainer_evaluations(store, run_id)):
        promotion = (job.get("result") or {}).get("promotion") or {}
        if bool(promotion.get("promoted")):
            break
        consecutive += 1
        result = job.get("result") or {}
        if "model_a_score" in result:
            recent_scores.append(float(result["model_a_score"]))
    recent_scores.reverse()
    level = consecutive // config.plateau_patience_evaluations
    multiplier = (
        min(config.plateau_max_exploration_multiplier, 2.0**level)
        if config.adaptive_training
        else 1.0
    )
    branch_requested = consecutive >= config.governor_branch_after_failures
    return {
        "consecutive_non_promotions": consecutive,
        "level": level,
        "exploration_multiplier": multiplier,
        "active": bool(level > 0),
        "adaptive_exploration_active": bool(config.adaptive_training and level > 0),
        "branch_requested": branch_requested,
        "recent_full_scores": recent_scores[-6:],
        "recent_full_mean": (
            sum(recent_scores[-6:]) / len(recent_scores[-6:]) if recent_scores else None
        ),
    }


def _governor_status(
    store: Store,
    run_id: str,
    config: RunConfig,
    *,
    games: int,
    diagnostics: dict[str, float],
    plateau: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Choose bounded live controls from canaries and optimization health."""

    previous = store.controller_state(run_id)
    if not config.realtime_governor:
        return {
            "enabled": False,
            "learning_rate_multiplier": 1.0,
            "updates_multiplier": 1.0,
            "reanalysis_multiplier": 1.0,
            "entropy_weight": config.policy_entropy_weight,
            "branch_requested": False,
        }
    last_games = int(previous.get("decision_games", -config.governor_interval_games))
    if games - last_games < config.governor_interval_games:
        return previous

    canary_scores: list[float] = []
    for job in _terminal_trainer_evaluations(store, run_id):
        if (job.get("config") or {}).get("promotion_tier") != "canary":
            continue
        result = job.get("result") or {}
        if "model_a_score" in result and not int(result.get("truncated_games", 0) or 0):
            canary_scores.append(float(result["model_a_score"]))
    canary_count = len(canary_scores)
    previous_canary_count = int(previous.get("canary_count", -1))
    canary_evidence_changed = canary_count != previous_canary_count
    if canary_evidence_changed:
        recent_scores = canary_scores[-6:]
        consecutive_regressions = next(
            (index for index, score in enumerate(reversed(recent_scores)) if score >= 0.5),
            len(recent_scores),
        )
    else:
        recent_scores = [float(value) for value in previous.get("recent_canary_scores", [])]
        consecutive_regressions = int(previous.get("consecutive_canary_regressions", 0))

    if not diagnostics and not recent_scores:
        state = {
            "enabled": True,
            "decision_games": int(games),
            "optimization_decision_games": int(games),
            "canary_decision_games": previous.get("canary_decision_games"),
            "canary_count": canary_count,
            "learning_rate_multiplier": 1.0,
            "updates_multiplier": 1.0,
            "reanalysis_multiplier": 1.0,
            "entropy_weight": config.policy_entropy_weight,
            "target_normalized_entropy": config.governor_target_normalized_entropy,
            "observed_normalized_entropy": None,
            "gradient_clip_fraction": 0.0,
            "consecutive_canary_regressions": 0,
            "recent_canary_scores": [],
            "branch_requested": False,
            "reasons": ["waiting for the first learner and canary evidence"],
        }
        store.set_controller_state(run_id, state)
        return state

    entropy = float(diagnostics.get("normalized_policy_entropy", 1.0))
    clip_fraction = float(diagnostics.get("gradient_clip_fraction", 0.0))
    searched_fraction = float(diagnostics.get("searched_fraction", 0.0))
    importance_ratio = float(diagnostics.get("mean_importance_ratio", 1.0))
    collection_policy_drift = float(diagnostics.get("collection_policy_abs_log_drift", 0.0))
    collection_policy_samples = int(diagnostics.get("collection_policy_samples", 0.0))
    lr_multiplier = 1.0
    updates_multiplier = 1.0
    reanalysis_multiplier = 1.0
    entropy_weight = config.policy_entropy_weight
    reasons: list[str] = []
    if config.governor_strategy == "mature":
        # Mature policies need small, reversible local moves. In particular,
        # do not respond to a losing canary by adding entropy: that prolonged
        # the diffuse losing regime observed in the Low-LR branch.
        if entropy < config.governor_target_normalized_entropy * 0.80:
            entropy_weight *= 1.20
            reanalysis_multiplier *= 1.15
            reasons.append("mature policy entropy is below its local-search band")
        elif entropy > min(0.98, config.governor_target_normalized_entropy * 1.12):
            entropy_weight *= 0.75
            reanalysis_multiplier *= 1.20
            reasons.append("mature policy is more diffuse than its local-search band")
        if collection_policy_samples >= 32 and collection_policy_drift > 0.35:
            updates_multiplier *= 0.90
            reasons.append("verified collection-policy drift favors fewer local updates")
        if canary_evidence_changed and len(recent_scores) >= 6:
            older_mean = sum(recent_scores[-6:-3]) / 3
            newer_mean = sum(recent_scores[-3:]) / 3
            canary_delta = newer_mean - older_mean
            if canary_delta < -0.015:
                lr_multiplier *= 0.80
                updates_multiplier *= 0.85
                reanalysis_multiplier *= 1.20
                reasons.append("three-canary trend regressed; local steps were reduced")
            elif canary_delta > 0.015 and newer_mean > 0.50:
                updates_multiplier *= 1.05
                reasons.append("three-canary trend supports a restrained exploitation push")
        search_target = max(0.002, config.reanalysis_fraction)
        if config.reanalysis_fraction > 0 and searched_fraction < search_target:
            reanalysis_multiplier *= 1.20
            reasons.append("searched supervision is below the mature-mode target")
    else:
        if entropy < config.governor_target_normalized_entropy * 0.80:
            lr_multiplier *= 0.75
            entropy_weight *= 1.6
            reanalysis_multiplier *= 1.5
            reasons.append("policy entropy is below target")
        elif entropy > min(0.98, config.governor_target_normalized_entropy * 1.35):
            entropy_weight *= 0.7
            updates_multiplier *= 1.15
            reasons.append("policy remains too diffuse")
        if consecutive_regressions >= 2:
            lr_multiplier *= 0.75
            entropy_weight *= 1.25
            reanalysis_multiplier *= 1.5
            reasons.append("successive canaries regressed")
        elif len(recent_scores) >= 2 and recent_scores[-1] > recent_scores[-2] + 0.02:
            updates_multiplier *= 1.2
            reasons.append("canary trend supports a short exploitation push")
        search_target = max(0.001, config.reanalysis_fraction * 0.8)
        if config.reanalysis_fraction > 0 and searched_fraction < search_target and recent_scores:
            reanalysis_multiplier *= 1.25
            reasons.append("too few learner batches contain search targets")

    lr_multiplier = min(
        config.governor_max_learning_rate_multiplier,
        max(config.governor_min_learning_rate_multiplier, lr_multiplier),
    )
    updates_multiplier = min(config.governor_max_updates_multiplier, max(0.5, updates_multiplier))
    reanalysis_multiplier = min(4.0, max(0.5, reanalysis_multiplier))
    plateau = plateau or {}
    full_non_promotions = int(plateau.get("consecutive_non_promotions", 0))
    full_plateau_active = bool(plateau.get("active", False))
    if full_plateau_active:
        lr_multiplier *= 0.80
        updates_multiplier *= 0.80
        reanalysis_multiplier = min(1.0, reanalysis_multiplier)
        reasons.append(
            "full-evaluation plateau reduced local steps and blocked reanalysis expansion"
        )
    lr_multiplier = min(
        config.governor_max_learning_rate_multiplier,
        max(config.governor_min_learning_rate_multiplier, lr_multiplier),
    )
    updates_multiplier = min(config.governor_max_updates_multiplier, max(0.5, updates_multiplier))
    reanalysis_multiplier = min(4.0, max(0.5, reanalysis_multiplier))
    branch_requested = consecutive_regressions >= config.governor_branch_after_failures or bool(
        plateau.get("branch_requested", False)
    )
    state = {
        "enabled": True,
        "strategy": config.governor_strategy,
        "decision_games": int(games),
        "optimization_decision_games": int(games),
        "canary_decision_games": (
            int(games) if canary_evidence_changed else previous.get("canary_decision_games")
        ),
        "canary_count": canary_count,
        "canary_evidence_changed": canary_evidence_changed,
        "learning_rate_multiplier": lr_multiplier,
        "updates_multiplier": updates_multiplier,
        "reanalysis_multiplier": reanalysis_multiplier,
        "entropy_weight": min(1.0, max(0.0, entropy_weight)),
        "target_normalized_entropy": config.governor_target_normalized_entropy,
        "observed_normalized_entropy": entropy,
        "gradient_clip_fraction": clip_fraction,
        "mean_importance_ratio": importance_ratio,
        "collection_policy_abs_log_drift": collection_policy_drift,
        "collection_policy_samples": collection_policy_samples,
        "consecutive_canary_regressions": consecutive_regressions,
        "consecutive_full_non_promotions": full_non_promotions,
        "full_evaluation_plateau_active": full_plateau_active,
        "recent_canary_scores": recent_scores,
        "branch_requested": branch_requested,
        "reasons": reasons,
    }
    store.set_controller_state(run_id, state)
    store.event(
        run_id,
        "governor_adjusted",
        "Realtime governor updated bounded training controls",
        state,
    )
    return state


def _pending_trainer_evaluation_job(store: Store, run_id: str) -> dict[str, Any] | None:
    jobs = store.arena_jobs(
        limit=1,
        include_internal=True,
        run_id=run_id,
        statuses=("queued", "running"),
        trainer_scheduled=True,
    )
    return jobs[0] if jobs else None


def _pending_trainer_evaluation(store: Store, run_id: str) -> bool:
    return _pending_trainer_evaluation_job(store, run_id) is not None


def _atomic_actor_export(model: Any, spec: ModelSpec, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.stem}.partial{target.suffix}")
    export_actor(model, spec, temporary, compressed=False)
    temporary.replace(target)
    return target


def _readable_npz(path: str | Path | None, *, required: set[str]) -> bool:
    if not path:
        return False
    target = Path(path)
    if not target.is_file():
        return False
    try:
        with np.load(target, allow_pickle=False) as archive:
            return required.issubset(archive.files)
    except (OSError, ValueError, KeyError):
        return False


def _readable_policy_replay(path: str | Path | None, *, required: set[str]) -> bool:
    if not path:
        return False
    target = Path(path)
    if target.suffix.lower() != ".json":
        return _readable_npz(target, required=required)
    try:
        manifest = json.loads(target.read_text(encoding="utf-8"))
        if manifest.get("format") == "hybrid_game_reservoir_v3":
            hot = manifest.get("hot")
            cold = manifest.get("cold")
            if not isinstance(hot, dict) or not isinstance(cold, dict):
                return False
            cold_shards = cold.get("shards")
            if cold.get("format") != "disk_policy_replay_v1" or not isinstance(cold_shards, list):
                return False
            if any(
                not isinstance(item, dict)
                or not isinstance(item.get("path"), str)
                or not Path(item["path"]).is_dir()
                or not (Path(item["path"]) / "shard.json").is_file()
                for item in cold_shards
            ):
                return False
            manifest = hot
        segments = manifest.get("segments")
        return bool(
            manifest.get("format") == "game_reservoir_incremental_v2"
            and isinstance(segments, list)
            and bool(segments)
            and all(_readable_npz(segment, required=required) for segment in segments)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _expected_model_weight_shapes(spec: ModelSpec) -> dict[str, tuple[int, ...]]:
    """Return the exact tensor contract consumed by the current model loader."""

    hidden = spec.hidden_size
    action_hidden = spec.action_hidden_size
    shapes: dict[str, tuple[int, ...]] = {
        "state_in.weight": (hidden, spec.state_size),
        "state_in.bias": (hidden,),
        "state_norm.weight": (hidden,),
        "state_norm.bias": (hidden,),
        "action_in.weight": (action_hidden, spec.action_size),
        "action_in.bias": (action_hidden,),
        "action_norm.weight": (action_hidden,),
        "action_norm.bias": (action_hidden,),
        "fusion_in.weight": (hidden, hidden + action_hidden),
        "fusion_in.bias": (hidden,),
        "fusion_norm.weight": (hidden,),
        "fusion_norm.bias": (hidden,),
    }

    if spec.objective_version >= 2:
        shapes.update(
            {
                "value_output.weight": (spec.families * spec.bootstrap_heads, hidden),
                "value_output.bias": (spec.families * spec.bootstrap_heads,),
            }
        )
        for index in range(spec.bootstrap_heads):
            shapes.update(
                {
                    f"head_outputs.{index}.weight": (spec.families, hidden),
                    f"head_outputs.{index}.bias": (spec.families,),
                }
            )
    else:
        shapes.update(
            {
                "output.weight": (spec.families * spec.bootstrap_heads, hidden),
                "output.bias": (spec.families * spec.bootstrap_heads,),
            }
        )

    def add_residual(prefix: str, width: int) -> None:
        shapes.update(
            {
                f"{prefix}.norm.weight": (width,),
                f"{prefix}.norm.bias": (width,),
                f"{prefix}.fc1.weight": (width * 2, width),
                f"{prefix}.fc1.bias": (width * 2,),
                f"{prefix}.fc2.weight": (width, width * 2),
                f"{prefix}.fc2.bias": (width,),
            }
        )

    for index in range(spec.residual_blocks):
        add_residual(f"state_blocks.{index}", hidden)
        add_residual(f"fusion_blocks.{index}", hidden)
    add_residual("action_blocks.0", action_hidden)
    if spec.objective_version >= 2:
        for index in range(spec.bootstrap_heads):
            add_residual(f"head_blocks.{index}", hidden)
    return shapes


def _checkpoint_model_is_loadable(path: str | Path, config: RunConfig | None) -> bool:
    """Validate a model/spec pair before selecting it as a resume boundary.

    Existence alone is insufficient after an interrupted copy or disk error: a
    malformed JSON sidecar or truncated safetensors archive would otherwise be
    selected and fail only after the fallback scan had already ended.
    """

    target = Path(path)
    sidecar = target.with_suffix(target.suffix + ".json")
    if not target.is_file() or not sidecar.is_file():
        return False
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return False
        spec = ModelSpec.from_dict(payload)
        positive_integer_fields = (
            "state_size",
            "action_size",
            "families",
            "hidden_size",
            "action_hidden_size",
            "bootstrap_heads",
        )
        if any(
            not isinstance(getattr(spec, field), int)
            or isinstance(getattr(spec, field), bool)
            or getattr(spec, field) < 1
            for field in positive_integer_fields
        ):
            return False
        if (
            not isinstance(spec.residual_blocks, int)
            or isinstance(spec.residual_blocks, bool)
            or spec.residual_blocks < 0
        ):
            return False
        if (
            not isinstance(spec.layer_norm_eps, (int, float))
            or isinstance(spec.layer_norm_eps, bool)
            or not math.isfinite(spec.layer_norm_eps)
            or spec.layer_norm_eps <= 0
        ):
            return False
        if (
            not isinstance(spec.encoder_version, int)
            or isinstance(spec.encoder_version, bool)
            or spec.encoder_version not in {1, 2}
        ):
            return False
        if config is not None:
            encoder = Encoder(version=2 if config.training_generation >= 3 else 1)
            if (
                spec.encoder_version != encoder.version
                or spec.state_size != encoder.state_size
                or spec.action_size != encoder.action_size
                or spec.families != FAMILY_COUNT
                or spec.hidden_size != config.hidden_size
                or spec.action_hidden_size != max(64, config.hidden_size // 2)
                or spec.residual_blocks != config.residual_blocks
                or spec.bootstrap_heads != config.bootstrap_heads
            ):
                return False

        expected_shapes = _expected_model_weight_shapes(spec)
        with safe_open(str(target), framework="numpy") as archive:
            if set(archive.keys()) != set(expected_shapes):
                return False
            for name, expected_shape in expected_shapes.items():
                tensor = archive.get_slice(name)
                if tuple(tensor.get_shape()) != expected_shape or tensor.get_dtype() not in {
                    "BF16",
                    "F16",
                    "F32",
                }:
                    return False
        return True
    except (
        OSError,
        TypeError,
        ValueError,
        KeyError,
        SafetensorError,
    ):
        return False


def _checkpoint_has_required_artifacts(
    checkpoint: dict[str, Any],
    config: RunConfig,
) -> bool:
    """Check the durable artifacts required by this run's resume contract."""

    artifacts = (checkpoint.get("evaluation") or {}).get("artifacts") or {}
    if (
        config.persist_optimizer_state
        and not _readable_npz(
            artifacts.get("optimizer_path"),
            required={"__paths_json__"},
        )
        and (checkpoint.get("evaluation") or {}).get("reason") != "branch import"
    ):
        return False
    if config.resume_replay_items <= 0:
        return True

    if config.training_generation >= 4:
        policy_items = artifacts.get("policy_replay_items")
        if policy_items is None:
            # Gen4 checkpoints predate durable policy replay. They remain safe
            # weight-only branch roots, but not exact mature-run resumes.
            return int(checkpoint.get("games", 0)) == 0
        try:
            policy_items = max(0, int(policy_items))
        except (TypeError, ValueError):
            return False
        if policy_items == 0:
            return True
        return _readable_policy_replay(
            artifacts.get("policy_replay_path"),
            required={
                "format_version",
                "state_size",
                "action_size",
                "bootstrap_heads",
                "episode_game_ids",
                "episode_players",
                "episode_lengths",
                "states",
                "legal_actions",
                "action_offsets",
                "selected_indices",
                "families",
                "targets",
                "behavior_probabilities",
                "bootstrap_masks",
                "steps",
                "search_policy",
                "search_mask",
                "search_values",
                "search_valid",
            },
        )

    replay_items = artifacts.get("replay_items")
    if replay_items is None:
        # Old initial checkpoints legitimately had no replay archive, but a
        # mature checkpoint without a manifest is not an exact Astro3 resume.
        return int(checkpoint.get("games", 0)) == 0
    try:
        replay_items = max(0, int(replay_items))
    except (TypeError, ValueError):
        return False
    if replay_items == 0:
        return True
    replay_format = str(artifacts.get("replay_format") or "recent_v1")
    if replay_format == "full_v2":
        required = {
            "format_version",
            "capacity",
            "state_size",
            "action_size",
            "bootstrap_heads",
            "sequence_cursor",
            "sample_calls",
            "samples_drawn",
            "last_recent_sample_items",
            "last_sample_batch_size",
            "last_importance_weight_min",
            "last_importance_weight_max",
            "last_importance_effective_sample_size",
            "family_samples_drawn",
        }
        for family in range(FAMILY_COUNT):
            prefix = f"family_{family}"
            required.update(
                {
                    f"{prefix}_size",
                    f"{prefix}_write_index",
                    f"{prefix}_writes",
                    f"{prefix}_overwrites",
                    f"{prefix}_states",
                    f"{prefix}_actions",
                    f"{prefix}_targets",
                    f"{prefix}_bootstrap_masks",
                    f"{prefix}_game_ids",
                    f"{prefix}_players",
                    f"{prefix}_steps",
                    f"{prefix}_heads",
                    f"{prefix}_epsilons",
                    f"{prefix}_td_targets",
                    f"{prefix}_td_valid",
                    f"{prefix}_sequences",
                }
            )
        return _readable_npz(artifacts.get("replay_path"), required=required)
    return _readable_npz(
        artifacts.get("replay_path"),
        required={
            "states",
            "actions",
            "families",
            "targets",
            "bootstrap_masks",
            "game_ids",
            "players",
            "steps",
            "heads",
            "epsilons",
            "td_targets",
            "td_valid",
            "sequences",
            "sequence_cursor",
        },
    )


def _latest_loadable_checkpoint(
    store: Store,
    run_id: str,
    config: RunConfig | None = None,
) -> dict[str, Any] | None:
    checkpoints = store.checkpoints(run_id)
    tainted_ids = {
        checkpoint["id"]
        for checkpoint in checkpoints
        if (checkpoint.get("evaluation") or {}).get("reason") == "initial random model"
        and int(checkpoint["games"]) > 0
    }
    changed = True
    while changed:
        changed = False
        for checkpoint in checkpoints:
            if checkpoint.get("parent_id") in tainted_ids and checkpoint["id"] not in tainted_ids:
                tainted_ids.add(checkpoint["id"])
                changed = True

    model_candidates: list[dict[str, Any]] = []
    skipped: list[str] = []
    for checkpoint in checkpoints:
        if checkpoint["id"] in tainted_ids:
            # Older builds could stamp restored counters onto a fresh random
            # model after incorrectly rejecting every Astro4 checkpoint. Its
            # descendants are equally unsafe even when saved as candidates.
            skipped.append(checkpoint["id"])
            continue
        path = Path(checkpoint["path"])
        if not _checkpoint_model_is_loadable(path, config):
            skipped.append(checkpoint["id"])
            continue
        model_candidates.append(checkpoint)
        if config is None or _checkpoint_has_required_artifacts(checkpoint, config):
            selected = dict(checkpoint)
            selected["_resume_artifacts_complete"] = True
            selected["_resume_skipped_checkpoint_ids"] = skipped
            return selected
        skipped.append(checkpoint["id"])

    if not model_candidates:
        return None
    # Keeping the newest usable weights is safer than silently restarting a
    # mature run. Mark the deliberate weight-only degradation so restore can
    # reset optimizer/replay state and surface it in telemetry.
    selected = dict(model_candidates[0])
    selected["_resume_artifacts_complete"] = False
    selected["_resume_skipped_checkpoint_ids"] = skipped
    return selected


def _learner_resume_checkpoint(
    store: Store,
    run_id: str,
    config: RunConfig,
) -> dict[str, Any] | None:
    """Resolve the durable learner head, never an already-rolled-back artifact."""

    latest = _latest_loadable_checkpoint(store, run_id, config)
    if latest is None or not config.rollback_rejected_candidates:
        return latest
    for job in reversed(_completed_trainer_evaluations(store, run_id)):
        result = job.get("result") or {}
        if job["model_a"] == latest["id"] and result.get("_trainer_disposition") == "rolled_back":
            champion_id = store.get_run(run_id).get("champion_id")
            if champion_id:
                champion = store.checkpoint(champion_id)
                if Path(champion["path"]).is_file():
                    return champion
    gate = (latest.get("evaluation") or {}).get("quality_gate") or {}
    if gate.get("rollback_applied"):
        champion_id = store.get_run(run_id).get("champion_id")
        if champion_id:
            return store.checkpoint(champion_id)
    return latest


def _champion_actor_path(store: Store, run_id: str, fallback: Path) -> str:
    run = store.get_run(run_id)
    champion_id = run.get("champion_id")
    if champion_id:
        try:
            actor_path = store.checkpoint(champion_id).get("actor_path")
        except KeyError:
            actor_path = None
        if actor_path and Path(actor_path).is_file():
            return str(Path(actor_path))
    return str(fallback)


def _repair_anomalous_deployment_anchor(store: Store, run_id: str) -> dict[str, Any] | None:
    """Undo the legacy nonzero-game random-anchor corruption on resume."""

    run = store.get_run(run_id)
    champion_id = run.get("champion_id")
    if not champion_id:
        return None
    champion = store.checkpoint(champion_id)
    evaluation = champion.get("evaluation") or {}
    if evaluation.get("reason") != "initial random model" or int(champion["games"]) == 0:
        return champion

    checkpoints = store.checkpoints(run_id)
    evaluated = [
        item
        for item in checkpoints
        if item["id"] != champion_id
        and bool(((item.get("evaluation") or {}).get("latest_arena") or {}).get("promoted"))
    ]
    roots = [
        item
        for item in checkpoints
        if item["id"] != champion_id
        and int(item["games"]) == 0
        and (item.get("evaluation") or {}).get("reason") == "initial random model"
    ]
    replacement = evaluated[0] if evaluated else (roots[-1] if roots else None)
    if replacement is None:
        raise RuntimeError(
            "the deployment model is an invalid nonzero-game random anchor and no safe anchor exists"
        )
    restored = store.set_run_champion(run_id, replacement["id"])
    store.event(
        run_id,
        "deployment_anchor_repaired",
        f"Restored deployment model {restored['label']}",
        {"invalid_checkpoint_id": champion_id, "restored_checkpoint_id": restored["id"]},
    )
    return restored


def _completed_trainer_evaluations(store: Store, run_id: str) -> list[dict[str, Any]]:
    """Return only completed arenas that are valid evidence of playing strength."""

    return [
        job
        for job in _terminal_trainer_evaluations(store, run_id)
        if bool((job.get("config") or {}).get("automatic_promotion"))
        and _trainer_evaluation_outcome(job) in {"promoted", "not_promoted"}
    ]


def _completed_full_evaluation_count(store: Store, run_id: str) -> int:
    """Count valid, terminal full promotion evaluations for a run."""

    return sum(
        1
        for job in _completed_trainer_evaluations(store, run_id)
        if (job.get("config") or {}).get("promotion_tier", "full") == "full"
    )


def _training_budget_reached(
    config: RunConfig,
    *,
    active_elapsed: float,
    games: int,
    full_evaluations: int,
) -> tuple[bool, str]:
    if config.budget_type == "games":
        reached = games >= int(config.budget_games or 0)
        return reached, "game budget complete"
    if config.budget_type == "full_evaluations":
        reached = full_evaluations >= int(config.budget_full_evaluations or 0)
        return reached, "full-evaluation budget complete"
    reached = active_elapsed >= config.duration_minutes * 60.0
    return reached, "duration complete"


def _is_trainer_evaluation(job: dict[str, Any]) -> bool:
    config = job.get("config") or {}
    # Older automatic jobs predate the explicit trainer_scheduled marker. No
    # public/manual path can create an automatic-promotion job, so retaining
    # them here preserves restart compatibility.
    return bool(config.get("trainer_scheduled") or config.get("automatic_promotion"))


def _trainer_evaluation_outcome(job: dict[str, Any]) -> str:
    """Classify whether an arena is skill evidence or retryable infrastructure.

    A completed SQLite row is not automatically a valid comparison. A truncated
    arena can remain valid when an exceptionally small truncation rate was
    already conservatively scored as candidate losses. Larger truncation rates,
    a stale champion opponent, or a structurally incomplete result must be
    retried and must not look like a model regression.
    """

    status = str(job.get("status", ""))
    if status in {"queued", "running"}:
        return "pending"
    if status == "failed":
        return "infrastructure_failed"
    if status == "cancelled":
        return "cancelled"
    if status != "complete":
        return "infrastructure_invalid"

    result = job.get("result") or {}
    promotion = result.get("promotion") or {}
    truncated_games = max(0, int(result.get("truncated_games", 0) or 0))
    games_completed = max(0, int(result.get("games_completed", 0) or 0))
    # One bad worker result in a very large arena should not erase the other
    # games. The arena's promotion calculation has already charged each such
    # game as a candidate loss. Re-run when the rate exceeds 0.01%, or when an
    # old/incomplete result does not report a denominator.
    low_rate_conservative_truncation = bool(
        truncated_games
        and games_completed
        and truncated_games <= int(games_completed * 0.0001)
        and (promotion.get("truncation_adjustment") or {}).get("applied")
    )
    if (
        truncated_games
        and not bool(promotion.get("promoted"))
        and not low_rate_conservative_truncation
    ):
        return "truncated"
    if bool(promotion.get("stale_opponent") or result.get("stale_opponent")):
        return "stale"

    config = job.get("config") or {}
    if not bool(config.get("automatic_promotion")):
        # Diagnostic jobs have no promotion payload by design. Reaching the
        # complete state without truncation is their validity contract.
        return "diagnostic_complete"

    if bool(result.get("early_stopped")):
        early_outcome = result.get("early_stop_outcome")
        latest_look = (result.get("early_rejection") or {}).get("latest_look") or {}
        if early_outcome == "accepted":
            return "promoted" if bool(promotion.get("promoted")) else "infrastructure_invalid"
        if early_outcome == "rejected" or bool(latest_look.get("reject")):
            return "not_promoted"
        return "infrastructure_invalid"
    if "promoted" not in promotion:
        return "infrastructure_invalid"
    if promotion.get("eligible") is False:
        return "infrastructure_invalid"
    return "promoted" if bool(promotion.get("promoted")) else "not_promoted"


def _terminal_trainer_evaluations(store: Store, run_id: str) -> list[dict[str, Any]]:
    return sorted(
        (
            job
            for job in store.arena_jobs(
                limit=20_000,
                include_internal=True,
                run_id=run_id,
                statuses=("complete", "failed", "cancelled"),
                trainer_scheduled=True,
            )
            if _is_trainer_evaluation(job)
        ),
        key=lambda job: float(job["created_at"]),
    )


def _mark_evaluation_disposition(
    store: Store,
    job: dict[str, Any],
    disposition: str,
) -> None:
    """Persist trainer handling so a restart neither misses nor repeats it."""

    result = dict(job.get("result") or {})
    result["_trainer_disposition_processed"] = True
    result["_trainer_disposition"] = disposition
    store.update_arena_job(job["id"], result=result)


def _save_checkpoint(
    *,
    store: Store,
    run_id: str,
    model: Any,
    spec: ModelSpec,
    checkpoint_dir: Path,
    games: int,
    parent_id: str | None,
    champion: bool,
    reason: str,
    optimizer: Any | None = None,
    replay: ReplayBuffer | None = None,
    policy_replay: GameBalancedPolicyReplayBuffer | None = None,
    preference_replay: PreferenceReplayBuffer | None = None,
    resume_replay_items: int = 0,
    full_replay: bool = False,
    training_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Manual/pause/duration checkpoints can share both the game counter and a
    # wall-clock instant.  A nanosecond timestamp keeps names sortable while a
    # full random UUID makes each immutable artifact family independent.
    stem = f"g{games:010d}-{time.time_ns()}-{uuid.uuid4().hex}"
    model_path = checkpoint_dir / f"{stem}.safetensors"
    actor_path = checkpoint_dir / f"{stem}.actor.npz"
    save_model(model, spec, model_path)
    export_actor(model, spec, actor_path)
    artifacts: dict[str, Any] = {"schema_version": 2}
    if optimizer is not None:
        optimizer_path = checkpoint_dir / f"{stem}.optimizer.npz"
        save_optimizer_state(optimizer, optimizer_path)
        artifacts["optimizer_path"] = str(optimizer_path)
    if replay is not None and resume_replay_items > 0:
        replay_path = checkpoint_dir / f"{stem}.replay.npz"
        replay_items = (
            replay.snapshot_full(replay_path)
            if full_replay
            else replay.snapshot(replay_path, max_items=resume_replay_items)
        )
        artifacts["replay_items"] = int(replay_items)
        artifacts["replay_capacity"] = int(replay.capacity)
        artifacts["replay_format"] = "full_v2" if full_replay else "recent_v1"
        if replay_items:
            artifacts["replay_path"] = str(replay_path)
    if policy_replay is not None and resume_replay_items > 0:
        if policy_replay.incremental_snapshots_enabled:
            policy_replay_path = checkpoint_dir / f"{stem}.policy-replay.json"
            policy_items = policy_replay.snapshot_incremental(
                policy_replay_path,
                max_items=0 if full_replay else resume_replay_items,
                force_compact=full_replay,
            )
            policy_replay_format = (
                "hybrid_game_reservoir_v3"
                if policy_replay.disk_capacity
                else "game_reservoir_incremental_v2"
            )
        else:
            policy_replay_path = checkpoint_dir / f"{stem}.policy-replay.npz"
            policy_items = policy_replay.snapshot(
                policy_replay_path,
                max_items=0 if full_replay else resume_replay_items,
            )
            policy_replay_format = "game_reservoir_v1_uncompressed"
        artifacts.update(
            policy_replay_items=int(policy_items),
            policy_replay_capacity=int(policy_replay.capacity),
            policy_replay_disk_capacity=int(policy_replay.disk_capacity),
            policy_replay_total_capacity=int(policy_replay.total_capacity),
            policy_replay_format=policy_replay_format,
        )
        if policy_items:
            artifacts["policy_replay_path"] = str(policy_replay_path)
    if preference_replay is not None and resume_replay_items > 0 and len(preference_replay):
        preference_path = checkpoint_dir / f"{stem}.preference-replay.npz"
        preference_items = preference_replay.snapshot(
            preference_path,
            max_items=0 if full_replay else min(resume_replay_items, preference_replay.capacity),
        )
        artifacts.update(
            preference_replay_items=int(preference_items),
            preference_replay_path=str(preference_path),
        )
    checkpoint = store.add_checkpoint(
        run_id=run_id,
        parent_id=parent_id,
        label=f"{'Champion' if champion else 'Candidate'} · {games:,} games",
        path=str(model_path),
        actor_path=str(actor_path),
        games=games,
        champion=champion,
        evaluation={
            "reason": reason,
            "evaluated": False,
            "artifacts": artifacts,
            "training_state": training_state or {},
        },
    )
    if champion:
        store.update_run(run_id, champion_id=checkpoint["id"])
    if policy_replay is not None and policy_replay.incremental_snapshots_enabled:
        policy_replay.commit_incremental_snapshot()
    store.event(
        run_id,
        "checkpoint_saved",
        f"Saved checkpoint at {games:,} games",
        {"checkpoint_id": checkpoint["id"], "reason": reason},
    )
    return checkpoint


def _compatible_external_actor_path(actor_path: str | None) -> str | None:
    """Resolve and validate a frozen actor before admitting it to self-play."""

    if not actor_path:
        return None
    path = Path(actor_path).expanduser().resolve()
    if not path.is_file():
        return None
    try:
        actor = NumpyActor.load(path)
        encoder = Encoder(version=actor.spec.encoder_version)
        ActorPolicy(actor, encoder)
        # Loading the archive and checking dimensions is not enough to catch a
        # truncated/custom archive with missing tensors. Exercise one complete
        # forward pass before a process worker can receive it.
        actor.predict_options(
            np.zeros(encoder.state_size, dtype=np.float32),
            np.zeros((1, encoder.action_size), dtype=np.float32),
            0,
        )
    except Exception:
        return None
    return str(path)


def _sync_league(
    league: League,
    store: Store,
    run_id: str,
    *,
    include_external_anchors: bool = False,
) -> None:
    existing = {opponent.id for opponent in league.opponents}
    eligible_local_ids: set[str] = set()
    for checkpoint in store.checkpoints(run_id):
        actor_path = checkpoint.get("actor_path")
        evaluation = checkpoint.get("evaluation") or {}
        latest_arena = evaluation.get("latest_arena") or {}
        accepted = bool(checkpoint["is_champion"] or latest_arena.get("promoted"))
        # The zero-game checkpoint is a persistence anchor, not a useful
        # opponent. Unevaluated and rejected candidates are excluded too: the
        # league is an archive of accepted behavior, not every learner wobble.
        if (
            not actor_path
            or int(checkpoint["games"]) == 0
            or not accepted
            or not Path(actor_path).exists()
        ):
            continue
        eligible_local_ids.add(checkpoint["id"])
        if checkpoint["id"] in existing:
            continue
        league.upsert(
            Opponent(
                id=checkpoint["id"],
                actor_path=actor_path,
                kind="champion" if checkpoint["is_champion"] else "checkpoint",
                label=checkpoint["label"],
            )
        )

    # NPZ cleanup intentionally makes former local league actors unavailable.
    # Drop them before another rollout plan can select a path that no longer
    # exists, while leaving baselines and other non-checkpoint opponents alone.
    league.opponents = [
        opponent
        for opponent in league.opponents
        if opponent.kind not in {"checkpoint", "champion"} or opponent.id in eligible_local_ids
    ]

    if not include_external_anchors:
        return

    eligible_external_ids: set[str] = set()
    existing = {opponent.id for opponent in league.opponents}
    for checkpoint in store.checkpoints():
        if checkpoint["run_id"] == run_id or int(checkpoint["games"]) == 0:
            continue
        if not (checkpoint["is_champion"] or checkpoint["is_pinned"]):
            continue
        actor_path = checkpoint.get("actor_path")
        if not actor_path or not Path(actor_path).is_file():
            continue
        if checkpoint["id"] not in existing:
            actor_path = _compatible_external_actor_path(actor_path)
            if actor_path is None:
                continue
            league.upsert(
                Opponent(
                    id=checkpoint["id"],
                    actor_path=actor_path,
                    kind="anchor",
                    label=f"Frozen anchor · {checkpoint['label']}",
                    pinned=bool(checkpoint["is_pinned"]),
                )
            )
            existing.add(checkpoint["id"])
        eligible_external_ids.add(checkpoint["id"])

    # Explicitly unpinning a non-champion external model takes effect without
    # restarting the trainer. Current champions remain stable external anchors.
    league.opponents = [
        opponent
        for opponent in league.opponents
        if opponent.kind != "anchor" or opponent.id in eligible_external_ids
    ]


def _last_scheduled_evaluation_games(store: Store, run_id: str, *, tier: str | None = None) -> int:
    """Return the latest checkpoint with a valid completed trainer arena.

    The historical name is kept for callers, but merely creating a job no
    longer consumes cadence. Failed, cancelled, truncated, stale, and malformed
    comparisons remain retryable.
    """

    checkpoint_games = {item["id"]: int(item["games"]) for item in store.checkpoints(run_id)}
    return max(
        (
            checkpoint_games.get(job["model_a"], 0)
            for job in store.arena_jobs(
                limit=20_000,
                include_internal=True,
                run_id=run_id,
                promotion_tier=tier,
                trainer_scheduled=True,
            )
            if job["model_a"] in checkpoint_games
            and _is_trainer_evaluation(job)
            and _trainer_evaluation_outcome(job)
            in {"diagnostic_complete", "promoted", "not_promoted"}
            and (
                tier is None
                or (job.get("config") or {}).get(
                    "promotion_tier",
                    "full"
                    if (job.get("config") or {}).get("automatic_promotion")
                    else "diagnostic",
                )
                == tier
            )
        ),
        default=0,
    )


def _evaluation_retry_state(
    store: Store,
    run_id: str,
    checkpoint_id: str,
    tier: str,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """Describe bounded exponential backoff after invalid arena attempts."""

    checkpoint_ids = {item["id"] for item in store.checkpoints(run_id)}
    if checkpoint_id not in checkpoint_ids:
        return {"ready": False, "attempts": 0, "reason": "unknown_checkpoint"}
    invalid: list[tuple[dict[str, Any], str]] = []
    for job in store.arena_jobs(
        limit=20_000,
        include_internal=True,
        run_id=run_id,
        promotion_tier=tier,
        trainer_scheduled=True,
    ):
        config = job.get("config") or {}
        job_tier = config.get(
            "promotion_tier",
            "full" if config.get("automatic_promotion") else "diagnostic",
        )
        if job["model_a"] != checkpoint_id or not _is_trainer_evaluation(job) or job_tier != tier:
            continue
        outcome = _trainer_evaluation_outcome(job)
        if outcome == "pending":
            return {"ready": False, "attempts": len(invalid), "reason": "pending"}
        if outcome in {"diagnostic_complete", "promoted", "not_promoted"}:
            break
        invalid.append((job, outcome))

    if not invalid:
        return {"ready": True, "attempts": 0, "reason": None, "retry_at": None}
    latest_job, reason = invalid[0]
    # Staleness is not an infrastructure instability: the old comparison was
    # validly computed, but must immediately be repeated against the champion
    # that replaced its opponent.
    if reason == "stale":
        delay = 0.0
    else:
        exponent = min(len(invalid) - 1, 8)
        delay = min(_EVALUATION_RETRY_MAX_SECONDS, _EVALUATION_RETRY_BASE_SECONDS * 2**exponent)
    retry_at = float(latest_job.get("updated_at") or latest_job.get("created_at") or 0.0) + delay
    current = time.time() if now is None else float(now)
    return {
        "ready": current >= retry_at,
        "attempts": len(invalid),
        "reason": reason,
        "retry_at": retry_at,
        "retry_after_seconds": max(0.0, retry_at - current),
        "last_job_id": latest_job["id"],
    }


def _next_evaluation_candidate(
    store: Store,
    run_id: str,
    config: RunConfig,
    checkpoint: dict[str, Any] | None = None,
    *,
    ignore_retry_backoff: bool = False,
    force_full: bool = False,
) -> tuple[dict[str, Any], _EvaluationPlan] | None:
    """Return the newest due checkpoint when no trainer arena is active.

    Scheduling is keyed to the immutable checkpoint's game count, not the
    learner's moving total. That makes a completed job release exactly the
    newest eligible checkpoint without re-queuing an already tested artifact.
    """

    if _pending_trainer_evaluation(store, run_id):
        return None
    if checkpoint is None:
        checkpoints = store.checkpoints(run_id)
        if not checkpoints:
            return None
        checkpoint = max(
            checkpoints,
            key=lambda item: (int(item["games"]), float(item["created_at"])),
        )
    champion_id = store.get_run(run_id).get("champion_id")
    if not champion_id or champion_id == checkpoint["id"]:
        return None
    quality_gate = (checkpoint.get("evaluation") or {}).get("quality_gate") or {}
    if "passed" in quality_gate and not bool(quality_gate["passed"]):
        # A deterministic gate result belongs to the immutable checkpoint.
        # Do not emit the same blocked-evaluation event on every trainer loop.
        return None
    checkpoint_games = int(checkpoint["games"])
    if force_full:
        plans = [
            _EvaluationPlan(
                tier="full",
                cadence_games=config.evaluate_every_games,
                pairs=config.evaluation_pairs,
                automatic_promotion=config.evaluation_pairs >= MINIMUM_PROMOTION_PAIRS,
            )
        ]
    elif config.training_generation >= 5 and config.canary_every_games:
        plans = [
            _EvaluationPlan(
                tier="full",
                cadence_games=config.evaluate_every_games,
                pairs=config.evaluation_pairs,
                automatic_promotion=config.evaluation_pairs >= MINIMUM_PROMOTION_PAIRS,
            ),
            _EvaluationPlan(
                tier="canary",
                cadence_games=config.canary_every_games,
                pairs=config.canary_pairs,
                automatic_promotion=False,
            ),
        ]
    else:
        plans = [_evaluation_plan(config, checkpoint_games)]
    plan: _EvaluationPlan | None = None
    for candidate_plan in plans:
        last_evaluation_games = _last_scheduled_evaluation_games(
            store,
            run_id,
            tier=candidate_plan.tier,
        )
        if force_full:
            if last_evaluation_games < checkpoint_games:
                plan = candidate_plan
                break
        elif checkpoint_games - last_evaluation_games >= candidate_plan.cadence_games:
            plan = candidate_plan
            break
    if plan is None:
        return None
    retry = _evaluation_retry_state(store, run_id, checkpoint["id"], plan.tier)
    if not ignore_retry_backoff and not bool(retry["ready"]):
        return None
    return checkpoint, plan


def _persisted_active_elapsed(
    store: Store,
    run_id: str,
    checkpoint: dict[str, Any] | None = None,
) -> float:
    """Recover consumed training time without counting backend downtime."""

    checkpoint_state = ((checkpoint or {}).get("evaluation") or {}).get("training_state") or {}
    if "active_elapsed_seconds" in checkpoint_state:
        try:
            elapsed = float(checkpoint_state["active_elapsed_seconds"])
        except (TypeError, ValueError):
            return 0.0
        return elapsed if math.isfinite(elapsed) and elapsed >= 0.0 else 0.0
    metrics = store.metrics(run_id, after=-1, limit=1)
    if not metrics:
        return 0.0
    value = metrics[-1].get("active_elapsed_seconds", 0.0)
    try:
        elapsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return elapsed if math.isfinite(elapsed) and elapsed >= 0.0 else 0.0


def _checkpoint_training_state(
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a complete counter boundary, ignoring legacy partial payloads."""

    state = ((checkpoint or {}).get("evaluation") or {}).get("training_state") or {}
    if not isinstance(state, dict):
        return {}
    version = state.get("schema_version")
    if version is not None:
        try:
            version = int(version)
        except (TypeError, ValueError):
            return {}
        if version < 1 or version > _TRAINING_STATE_SCHEMA_VERSION:
            return {}
    required = set(_TRAINING_STATE_REQUIRED_COUNTERS)
    if version == _TRAINING_STATE_SCHEMA_VERSION:
        required.add("optimizer_updates_at_start")
    if not required.issubset(state):
        return {}
    return state


def _restore_totals(
    store: Store,
    run: dict[str, Any],
    checkpoint: dict[str, Any] | None = None,
) -> _Totals:
    """Restore cumulative telemetry counters alongside checkpointed progress."""

    metrics = store.metrics(run["id"], after=-1, limit=1)
    latest = metrics[-1] if metrics else {}
    checkpoint_state = _checkpoint_training_state(checkpoint)
    if checkpoint_state:
        latest = checkpoint_state
    games = max(int(run["games"]), int(latest.get("games", 0)))
    if checkpoint is not None and checkpoint_state:
        # The model is authoritative. Never resume older weights with counters
        # from games and updates that were never durably checkpointed.
        games = int(checkpoint_state.get("games", checkpoint["games"]))
    metric_games = max(0, int(latest.get("games", games)))
    mean_turns = max(0.0, float(latest.get("mean_turns", 0.0)))
    # A durable Astro3 checkpoint describes the exact learner/replay/optimizer
    # boundary.  Run-table counters may be newer because metrics are committed
    # between checkpoints; using their maximum would silently skip seeds and LR
    # schedule steps that the restored weights never saw.
    decisions = int(latest.get("decisions", 0))
    updates = int(latest.get("updates", 0))
    if not checkpoint_state:
        decisions = max(int(run["decisions"]), decisions)
        updates = max(int(run["updates"]), updates)
    return _Totals(
        games=games,
        decisions=decisions,
        updates=updates,
        samples=max(0, int(latest.get("samples", 0))),
        player_wins=(
            max(0, int(latest.get("player_0_wins", 0))),
            max(0, int(latest.get("player_1_wins", 0))),
        ),
        draws=max(0, int(latest.get("draws", 0))),
        truncated=max(0, int(latest.get("truncations", 0))),
        turns=max(
            0,
            int(latest["turns"]) if "turns" in latest else round(mean_turns * metric_games),
        ),
        forced_choices=max(0, int(latest.get("forced_choices", 0))),
        counterfactual_preferences=max(0, int(latest.get("counterfactual_preferences", 0))),
        reanalysis_positions=max(0, int(latest.get("reanalysis_positions", 0))),
        search_repeatability_positions=max(0, int(latest.get("search_repeatability_positions", 0))),
        search_top_action_agreements=max(0, int(latest.get("search_top_action_agreements", 0))),
        search_policy_js_sum=max(0.0, float(latest.get("search_policy_js_sum", 0.0))),
        search_value_abs_delta_sum=max(0.0, float(latest.get("search_value_abs_delta_sum", 0.0))),
        rollout_games={
            str(key): max(0, int(value))
            for key, value in (latest.get("rollout_games") or {}).items()
        },
        rollout_completed_games={
            str(key): max(0, int(value))
            for key, value in (latest.get("rollout_completed_games") or {}).items()
        },
        rollout_scores={
            str(key): max(0.0, float(value))
            for key, value in (latest.get("rollout_scores") or {}).items()
        },
        opponent_games={
            str(key): max(0, int(value))
            for key, value in (latest.get("opponent_games") or {}).items()
        },
        opponent_scores={
            str(key): max(0.0, float(value))
            for key, value in (latest.get("opponent_scores") or {}).items()
        },
    )


def _training_state(
    totals: _Totals,
    *,
    seed_cursor: int,
    optimizer_updates_at_start: int,
    schedule_games_origin: int = 0,
    active_elapsed_seconds: float | None = None,
    rollout_rng_state: dict[str, Any] | None = None,
    replay_rng_state: dict[str, Any] | None = None,
    league_state: list[dict[str, object]] | None = None,
    policy_reference_kl_controller: dict[str, float] | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "schema_version": _TRAINING_STATE_SCHEMA_VERSION,
        "games": totals.games,
        "decisions": totals.decisions,
        "updates": totals.updates,
        "samples": totals.samples,
        "player_0_wins": totals.player_wins[0],
        "player_1_wins": totals.player_wins[1],
        "draws": totals.draws,
        "truncations": totals.truncated,
        "turns": totals.turns,
        "mean_turns": totals.turns / max(1, totals.games),
        "forced_choices": totals.forced_choices,
        "counterfactual_preferences": totals.counterfactual_preferences,
        "reanalysis_positions": totals.reanalysis_positions,
        "search_repeatability_positions": totals.search_repeatability_positions,
        "search_top_action_agreements": totals.search_top_action_agreements,
        "search_policy_js_sum": totals.search_policy_js_sum,
        "search_value_abs_delta_sum": totals.search_value_abs_delta_sum,
        "rollout_games": dict(totals.rollout_games),
        "rollout_completed_games": dict(totals.rollout_completed_games),
        "rollout_scores": dict(totals.rollout_scores),
        "opponent_games": dict(totals.opponent_games),
        "opponent_scores": dict(totals.opponent_scores),
        "seed_cursor": int(seed_cursor),
        "optimizer_updates_at_start": max(0, int(optimizer_updates_at_start)),
        "schedule_games_origin": max(0, int(schedule_games_origin)),
    }
    if active_elapsed_seconds is not None:
        state["active_elapsed_seconds"] = max(0.0, float(active_elapsed_seconds))
    if rollout_rng_state is not None:
        state["rollout_rng_state"] = rollout_rng_state
    if replay_rng_state is not None:
        state["replay_rng_state"] = replay_rng_state
    if league_state is not None:
        state["league_state"] = league_state
    if policy_reference_kl_controller is not None:
        state["policy_reference_kl_controller"] = {
            "weight": float(policy_reference_kl_controller.get("weight", 0.0)),
            "last_kl": float(policy_reference_kl_controller.get("last_kl", 0.0)),
            "ema_kl": float(policy_reference_kl_controller.get("ema_kl", 0.0)),
            "ema_initialized": bool(policy_reference_kl_controller.get("ema_initialized", False)),
            "next_adjustment_update": int(
                policy_reference_kl_controller.get("next_adjustment_update", 0)
            ),
        }
    return state


def _optimizer_schedule_origin(
    *,
    completed_updates: int,
    training_state: dict[str, Any],
    optimizer_restored: bool,
) -> int:
    """Recover LR warmup origin, or begin a fresh warmup after state loss."""

    completed = max(0, int(completed_updates))
    if not optimizer_restored:
        return completed
    try:
        origin = int(training_state.get("optimizer_updates_at_start", 0))
    except (TypeError, ValueError):
        return 0
    return origin if 0 <= origin <= completed else 0


def _schedule_evaluation(
    *,
    manager: Any,
    store: Store,
    run_id: str,
    checkpoint: dict[str, Any],
    config: RunConfig,
    plan: _EvaluationPlan | None = None,
    cancellation_hook: Callable[[], bool] | None = None,
) -> dict[str, Any] | None:
    champion_id = store.get_run(run_id).get("champion_id")
    if not champion_id or champion_id == checkpoint["id"]:
        return None
    plan = plan or _evaluation_plan(config, int(checkpoint["games"]))
    retry = _evaluation_retry_state(store, run_id, checkpoint["id"], plan.tier)
    evaluation_seed = (
        config.seed
        + int(checkpoint["games"])
        + int(retry["attempts"]) * _EVALUATION_RETRY_SEED_STRIDE
    )
    if plan.automatic_promotion:
        # Preserve the configured statistical design exactly. A provisional
        # tier smaller than the first planned look simply has no early looks;
        # silently changing 512 to 100 would produce a different test.
        early_look_ceiling = (
            config.evaluation_extension_max_pairs
            if config.evaluation_early_look_interval_pairs and config.evaluation_extension_enabled
            else plan.pairs
        )
        early_rejection = bool(
            config.evaluation_early_rejection
            and config.evaluation_early_rejection_min_pairs < early_look_ceiling
        )
        early_acceptance = bool(
            config.evaluation_early_acceptance
            and config.evaluation_early_acceptance_min_pairs < early_look_ceiling
        )
        job = manager.create_automatic(
            checkpoint["id"],
            champion_id,
            pairs=plan.pairs,
            seed=evaluation_seed,
            max_turns=config.max_turns,
            max_actions_per_turn=config.max_actions_per_turn,
            confidence=config.promotion_confidence,
            promotion_margin=config.promotion_margin,
            minimum_promotion_pairs=plan.pairs,
            promotion_tier=plan.tier,
            early_rejection=early_rejection,
            early_rejection_min_pairs=config.evaluation_early_rejection_min_pairs,
            early_rejection_confidence=config.evaluation_early_rejection_confidence,
            early_acceptance=early_acceptance,
            early_acceptance_min_pairs=config.evaluation_early_acceptance_min_pairs,
            early_acceptance_confidence=config.evaluation_early_acceptance_confidence,
            early_look_interval_pairs=config.evaluation_early_look_interval_pairs,
            extension_enabled=config.evaluation_extension_enabled,
            extension_max_pairs=config.evaluation_extension_max_pairs,
            extension_block_pairs=config.evaluation_extension_block_pairs,
            extension_min_score=config.evaluation_extension_min_score,
            extension_min_lower_bound=config.evaluation_extension_min_lower_bound,
            cancellation_hook=cancellation_hook,
        )
    else:
        # Quick validation runs still exercise the paired evaluator, but their
        # deliberately tiny sample can never change champion state.
        from .arena import ArenaConfig

        job = manager.create(
            checkpoint["id"],
            champion_id,
            ArenaConfig(
                pairs=plan.pairs,
                seed=evaluation_seed,
                max_turns=config.max_turns,
                max_actions_per_turn=config.max_actions_per_turn,
                confidence=config.promotion_confidence,
                promotion_tier=plan.tier,
                automatic_promotion=False,
                trainer_scheduled=True,
            ),
            cancellation_hook=cancellation_hook,
        )
    store.event(
        run_id,
        "automatic_evaluation_started",
        f"Started paired evaluation for {checkpoint['label']}",
        {
            "job_id": job["id"],
            "pairs": plan.pairs,
            "tier": plan.tier,
            "attempt": int(retry["attempts"]) + 1,
            "retrying_after": retry.get("reason"),
        },
    )
    return job


def _is_exclusive_full_promotion_job(job: dict[str, Any] | None) -> bool:
    """Return whether training must yield the machine to this arena job."""

    if job is None or job.get("status") not in {"queued", "running"}:
        return False
    config = job.get("config") or {}
    return bool(
        config.get("trainer_scheduled")
        and config.get("automatic_promotion")
        and config.get("promotion_tier", "full") == "full"
    )


def _finish_final_evaluations(
    *,
    store: Store,
    run_id: str,
    manager_getter: Callable[[], Any | None],
    schedule_latest: Callable[[], dict[str, Any] | None],
    process_completed: Callable[[], None],
    interrupt_reason: Callable[[], str | None] | None = None,
    service_checkpoint: Callable[[], bool] | None = None,
    max_attempts: int = _FINAL_EVALUATION_MAX_ATTEMPTS,
) -> dict[str, Any]:
    """Drain old work, evaluate the newest due checkpoint, and drain it too.

    This runs only after natural duration completion. Arena jobs themselves are
    finite (bounded games, turns, and actions), so waiting for the manager does
    not introduce an unbounded statistical workload. Infrastructure-invalid
    results are retried a small number of times; exhaustion remains visibly due
    and will be retried when the run is resumed instead of consuming cadence.
    """

    scheduled_job_ids: list[str] = []
    invalid_outcomes: list[str] = []
    checkpoints_serviced = 0

    def interruption() -> str | None:
        return interrupt_reason() if interrupt_reason is not None else None

    def service_pending_checkpoint() -> None:
        nonlocal checkpoints_serviced
        if service_checkpoint is not None and service_checkpoint():
            checkpoints_serviced += 1

    def wait_for_trainer_job(manager: Any, job_id: str) -> str | None:
        while not manager.wait_for_job(job_id, timeout=0.25):
            service_pending_checkpoint()
            reason = interruption()
            if reason is not None:
                if manager.cancel(job_id) is False:
                    raise RuntimeError("trainer evaluation did not stop after cancellation")
                process_completed()
                return reason
        service_pending_checkpoint()
        process_completed()
        return interruption()

    while True:
        service_pending_checkpoint()
        reason = interruption()
        if reason is not None:
            return {
                "status": "interrupted",
                "interrupt_reason": reason,
                "scheduled_job_ids": scheduled_job_ids,
                "invalid_outcomes": invalid_outcomes,
                "checkpoints_serviced": checkpoints_serviced,
            }

        pending = _pending_trainer_evaluation_job(store, run_id)
        if pending is not None:
            manager = manager_getter()
            if manager is None:
                raise RuntimeError("a pending trainer evaluation has no owning ArenaManager")
            reason = wait_for_trainer_job(manager, str(pending["id"]))
            if reason is not None:
                return {
                    "status": "interrupted",
                    "interrupt_reason": reason,
                    "scheduled_job_ids": scheduled_job_ids,
                    "invalid_outcomes": invalid_outcomes,
                    "checkpoints_serviced": checkpoints_serviced,
                }
            if _pending_trainer_evaluation(store, run_id):
                raise RuntimeError("trainer evaluation remains pending after its worker stopped")

        job = schedule_latest()
        if job is None:
            reason = interruption()
            if reason is not None:
                return {
                    "status": "interrupted",
                    "interrupt_reason": reason,
                    "scheduled_job_ids": scheduled_job_ids,
                    "invalid_outcomes": invalid_outcomes,
                    "checkpoints_serviced": checkpoints_serviced,
                }
            return {
                "status": "complete" if not invalid_outcomes else "retry_exhausted",
                "scheduled_job_ids": scheduled_job_ids,
                "invalid_outcomes": invalid_outcomes,
                "checkpoints_serviced": checkpoints_serviced,
            }

        scheduled_job_ids.append(str(job["id"]))
        manager = manager_getter()
        if manager is None:
            raise RuntimeError("scheduled trainer evaluation has no owning ArenaManager")
        reason = wait_for_trainer_job(manager, str(job["id"]))
        if reason is not None:
            return {
                "status": "interrupted",
                "interrupt_reason": reason,
                "scheduled_job_ids": scheduled_job_ids,
                "invalid_outcomes": invalid_outcomes,
                "checkpoints_serviced": checkpoints_serviced,
            }
        persisted = store.arena_job(str(job["id"]), include_internal=True)
        outcome = _trainer_evaluation_outcome(persisted)
        if outcome in {"diagnostic_complete", "promoted", "not_promoted"}:
            # Re-enter once so cadence/champion state proves the immutable
            # checkpoint no longer has unfinished evaluation intent.
            continue
        invalid_outcomes.append(outcome)
        if len(scheduled_job_ids) >= max(1, max_attempts):
            return {
                "status": "retry_exhausted",
                "scheduled_job_ids": scheduled_job_ids,
                "invalid_outcomes": invalid_outcomes,
                "checkpoints_serviced": checkpoints_serviced,
            }


def _checkpoint_quality_gate(
    *,
    store: Store,
    run_id: str,
    checkpoint: dict[str, Any],
    config: RunConfig,
) -> dict[str, Any]:
    """Run general ensemble, calibration, and fixed-opponent checks."""

    existing = (checkpoint.get("evaluation") or {}).get("quality_gate")
    if existing:
        return existing
    from .diagnostics import checkpoint_diagnostics

    champion_id = store.get_run(run_id).get("champion_id")
    reference_actor_path: str | None = None
    if champion_id and champion_id != checkpoint["id"]:
        reference_actor_path = store.checkpoint(champion_id).get("actor_path")
    diagnostics = checkpoint_diagnostics(
        checkpoint["actor_path"],
        seed=config.seed,
        games=config.checkpoint_diagnostic_games,
        baseline_pairs=config.checkpoint_baseline_pairs,
        natural_positions=config.natural_diagnostic_positions,
        reference_actor_path=reference_actor_path,
    )
    ensemble = diagnostics.get("ensemble") or {}
    heldout = diagnostics["heldout"]
    baselines = diagnostics["baselines"]
    champion_score: float | None = None
    champion_brier: float | None = None
    if champion_id and champion_id != checkpoint["id"]:
        champion_evaluation = store.checkpoint(champion_id).get("evaluation") or {}
        champion_gate = champion_evaluation.get("quality_gate") or {}
        champion_diagnostics = champion_gate.get("diagnostics") or {}
        champion_baselines = champion_diagnostics.get("baselines") or {}
        champion_heldout = champion_diagnostics.get("heldout") or {}
        if "mean_score" in champion_baselines:
            champion_score = float(champion_baselines["mean_score"])
        if "game_grouped_brier" in champion_heldout:
            champion_brier = float(champion_heldout["game_grouped_brier"])
    baseline_regression = (
        champion_score is not None
        and champion_score - float(baselines["mean_score"]) > config.baseline_regression_tolerance
    )
    heldout_regression = (
        champion_brier is not None
        and float(heldout["game_grouped_brier"]) - champion_brier
        > config.heldout_brier_regression_tolerance
    )
    reasons: list[str] = []
    if bool(ensemble.get("natural_diagnostics_fallback")):
        reasons.append(
            "natural-state diagnostics failed: "
            f"{ensemble.get('natural_diagnostics_error', 'unknown error')}"
        )
    if (
        config.minimum_head_disagreement_rate > 0
        and float(ensemble.get("head_argmax_disagreement_rate", 0.0))
        < config.minimum_head_disagreement_rate
    ):
        reasons.append("bootstrap heads collapsed below the required action-disagreement rate")
    if int(baselines["truncated_games"]) > 0:
        reasons.append("fixed-opponent diagnostics contained truncated games")
    if config.gate_baseline_regression and baseline_regression:
        reasons.append("fixed-opponent score regressed beyond the configured tolerance")
    if config.gate_heldout_brier_regression and heldout_regression:
        reasons.append("held-out Brier score regressed beyond the configured tolerance")
    if float(heldout.get("game_grouped_brier", 1.0)) > config.maximum_heldout_brier:
        reasons.append("held-out value calibration exceeded the absolute Brier limit")
    policy_kl = ensemble.get("reference_policy_kl")
    if (
        policy_kl is not None
        and config.checkpoint_kl_limit > 0
        and float(policy_kl) > config.checkpoint_kl_limit
    ):
        reasons.append("candidate policy moved beyond the checkpoint KL safety limit")
    gate = {
        "passed": not reasons,
        "reasons": reasons,
        "champion_baseline_score": champion_score,
        "champion_heldout_brier": champion_brier,
        "baseline_regression_tolerance": config.baseline_regression_tolerance,
        "baseline_regression_gate_enabled": config.gate_baseline_regression,
        "baseline_regression_detected": baseline_regression,
        "heldout_brier_regression_tolerance": config.heldout_brier_regression_tolerance,
        "heldout_brier_gate_enabled": config.gate_heldout_brier_regression,
        "heldout_brier_regression_detected": heldout_regression,
        "maximum_heldout_brier": config.maximum_heldout_brier,
        "minimum_head_disagreement_rate": config.minimum_head_disagreement_rate,
        "checkpoint_kl_limit": config.checkpoint_kl_limit,
        "diagnostics": diagnostics,
    }
    store.update_checkpoint_evaluation(checkpoint["id"], {"quality_gate": gate})
    store.event(
        run_id,
        "checkpoint_quality_gate",
        f"Checkpoint quality gate {'passed' if gate['passed'] else 'failed'}",
        {"checkpoint_id": checkpoint["id"], **gate},
    )
    return gate


def _make_plan(
    *,
    config: RunConfig,
    rng: np.random.Generator,
    league: League,
    current_actor: str,
    fixed_champion_id: str | None = None,
    fixed_champion_actor: str | None = None,
    epsilon: float,
    seed: int,
) -> _RolloutPlan:
    roll = float(rng.random())
    if roll < config.current_selfplay_fraction:
        deployment_policy = (
            config.training_generation >= 3
            and float(rng.random()) < config.deployment_policy_selfplay_fraction
        )
        return _RolloutPlan(
            actor_paths=(current_actor, current_actor),
            baseline_names=("balanced", "balanced"),
            collect_players=(True, True),
            epsilons=(0.0, 0.0) if deployment_policy else (epsilon, epsilon),
            seed=seed,
            games=config.games_per_actor_batch,
            kind="deployment_self_play" if deployment_policy else "self_play",
            opponent_id=None,
            current_player=None,
            deployment_policy=(deployment_policy, deployment_policy),
        )

    fixed_champion_cutoff = config.current_selfplay_fraction + config.fixed_champion_fraction
    if roll < fixed_champion_cutoff:
        if not fixed_champion_id or not fixed_champion_actor:
            raise RuntimeError("fixed champion rollouts require an immutable branch-root actor")
        current_player = int(rng.integers(0, 2))
        actors: list[str | None] = [fixed_champion_actor, fixed_champion_actor]
        actors[current_player] = current_actor
        collect = [False, False]
        collect[current_player] = True
        epsilons = [0.0, 0.0]
        epsilons[current_player] = epsilon
        deployment = [True, True]
        deployment[current_player] = False
        kind = "fixed_champion"
        if float(rng.random()) < config.fixed_champion_probe_fraction:
            collect[current_player] = False
            if float(rng.random()) < 0.5:
                # Exact deployed mean policy on both seats.
                kind = "fixed_champion_probe_deployment"
                epsilons[current_player] = 0.0
                deployment[current_player] = True
            else:
                # Constant exploration makes this rolling score comparable
                # across checkpoints even while the training epsilon decays.
                kind = "fixed_champion_probe_exploration"
                epsilons[current_player] = config.fixed_champion_probe_epsilon
        return _RolloutPlan(
            actor_paths=(actors[0], actors[1]),
            baseline_names=("balanced", "balanced"),
            collect_players=(collect[0], collect[1]),
            epsilons=(epsilons[0], epsilons[1]),
            seed=seed,
            games=config.games_per_actor_batch,
            kind=kind,
            opponent_id=fixed_champion_id,
            current_player=current_player,
            deployment_policy=(deployment[0], deployment[1]),
        )

    checkpoint_opponents = [
        item
        for item in league.opponents
        if item.kind in {"checkpoint", "champion", "anchor"} and item.actor_path
    ]
    league_cutoff = fixed_champion_cutoff + config.league_fraction
    if roll < league_cutoff and checkpoint_opponents:
        opponent = league.select(
            rng,
            mode="pfsp",
            kinds={"checkpoint", "champion", "anchor"},
        )
        current_player = int(rng.integers(0, 2))
        actors: list[str | None] = [opponent.actor_path, opponent.actor_path]
        actors[current_player] = current_actor
        collect = [False, False]
        collect[current_player] = True
        epsilons = [0.0, 0.0]
        epsilons[current_player] = epsilon
        return _RolloutPlan(
            actor_paths=(actors[0], actors[1]),
            baseline_names=("balanced", "balanced"),
            collect_players=(collect[0], collect[1]),
            epsilons=(epsilons[0], epsilons[1]),
            seed=seed,
            games=config.games_per_actor_batch,
            kind="league",
            opponent_id=opponent.id,
            current_player=current_player,
        )

    baseline = str(rng.choice(("balanced", "economy", "aggressive")))
    current_player = int(rng.integers(0, 2))
    actors = [None, None]
    actors[current_player] = current_actor
    baselines = [baseline, baseline]
    collect = [False, False]
    collect[current_player] = True
    epsilons = [0.0, 0.0]
    epsilons[current_player] = epsilon
    return _RolloutPlan(
        actor_paths=(actors[0], actors[1]),
        baseline_names=(baselines[0], baselines[1]),
        collect_players=(collect[0], collect[1]),
        epsilons=(epsilons[0], epsilons[1]),
        seed=seed,
        games=config.games_per_actor_batch,
        kind="baseline",
        opponent_id=f"baseline:{baseline}",
        current_player=current_player,
    )


def _make_bootstrap_plan(
    *,
    config: RunConfig,
    rng: np.random.Generator,
    seed: int,
) -> _RolloutPlan:
    """Collect terminally labelled heuristic play only during replay warmup."""

    styles = ("balanced", "economy", "aggressive")
    first = str(rng.choice(styles))
    second = str(rng.choice(styles))
    return _RolloutPlan(
        actor_paths=(None, None),
        baseline_names=(first, second),
        collect_players=(True, True),
        epsilons=(0.0, 0.0),
        seed=seed,
        games=config.games_per_actor_batch,
        kind="heuristic_bootstrap",
        opponent_id=None,
        current_player=None,
    )


def _submit_rollout(
    executor: ProcessPoolExecutor,
    plan: _RolloutPlan,
    config: RunConfig,
) -> Future[WorkerResult]:
    return executor.submit(
        collect_worker_batch,
        plan.actor_paths,
        games=plan.games,
        seed=plan.seed,
        epsilons=plan.epsilons,
        baseline_names=plan.baseline_names,
        bootstrap_heads=config.bootstrap_heads,
        collect_players=plan.collect_players,
        max_turns=config.max_turns,
        max_actions_per_turn=config.max_actions_per_turn,
        exploration_top_k=config.exploration_top_k,
        bootstrap_probability=config.bootstrap_inclusion_probability,
        randomized_prior_scale=config.randomized_prior_scale,
        deployment_policy=plan.deployment_policy,
        use_bootstrap_targets=config.use_bootstrap_targets,
        collect_preferences=config.tactical_preference_training,
        collect_policy_decisions=config.training_generation >= 4,
        collect_outcome_decisions=config.training_generation < 4,
        policy_actor_advantage=config.policy_actor_advantage,
        policy_actor_gae_lambda=config.policy_actor_gae_lambda,
        counterfactual_fraction=(
            config.counterfactual_fraction if config.training_generation >= 4 else 0.0
        ),
        counterfactual_max_per_game=(
            config.counterfactual_max_per_game if config.training_generation >= 4 else 0
        ),
        reanalysis_fraction=(
            config.reanalysis_fraction if config.training_generation >= 5 else 0.0
        ),
        reanalysis_max_per_game=(
            config.reanalysis_max_per_game if config.training_generation >= 5 else 0
        ),
        reanalysis_max_actions=config.reanalysis_max_actions,
        reanalysis_rollouts_per_action=config.reanalysis_rollouts_per_action,
        reanalysis_horizon_turns=config.reanalysis_horizon_turns,
        reanalysis_policy_temperature=config.reanalysis_policy_temperature,
        policy_replay_decisions_per_player_game=(
            config.policy_replay_decisions_per_player_game if config.training_generation >= 5 else 0
        ),
        encoder_version=2 if config.training_generation >= 3 else 1,
    )


def _train_updates(
    *,
    model: Any,
    optimizer: Any,
    replay: ReplayBuffer,
    policy_replay: GameBalancedPolicyReplayBuffer,
    preference_replay: PreferenceReplayBuffer,
    config: RunConfig,
    count: int,
    totals: _Totals,
    optimizer_updates_at_start: int,
    control: Any,
    reference_model: Any | None = None,
    fixed_champion_key: int = 0,
    learning_rate_multiplier: float = 1.0,
    promotion_direction: dict[str, Any] | None = None,
    policy_reference_kl_controller: dict[str, float] | None = None,
) -> dict[str, float]:
    active_replay_size = len(policy_replay) if config.training_generation >= 4 else len(replay)
    if count <= 0 or active_replay_size < config.replay_warmup:
        return {}

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    reference_kl_weight = float(
        (policy_reference_kl_controller or {}).get("weight", config.policy_reference_kl_weight)
    )

    def resolved_actor_sample_weights(deployment_policy: Any, rollout_sources: Any) -> Any:
        deployment_weights = mx.where(
            deployment_policy > 0,
            mx.array(float(config.deployment_policy_actor_weight)),
            mx.array(1.0),
        )
        source_weights = {
            "unknown": 1.0,
            "self_play": float(config.self_play_actor_weight),
            "deployment_self_play": float(config.self_play_actor_weight),
            "fixed_champion": float(config.fixed_champion_actor_weight),
            "league": float(config.league_actor_weight),
            "baseline": float(config.baseline_actor_weight),
        }
        if not config.source_stratified_actor_loss:
            resolved = mx.zeros_like(deployment_policy)
            for source_name, configured_weight in source_weights.items():
                source_id = POLICY_ROLLOUT_SOURCE_IDS[source_name]
                resolved = mx.where(
                    rollout_sources == source_id,
                    mx.array(configured_weight),
                    resolved,
                )
            return deployment_weights * resolved
        resolved = mx.zeros_like(deployment_policy)
        for source_name, configured_weight in source_weights.items():
            source_id = POLICY_ROLLOUT_SOURCE_IDS[source_name]
            mask = (rollout_sources == source_id).astype(deployment_policy.dtype)
            eligible = mask * deployment_weights
            per_sample = float(configured_weight) / mx.maximum(mx.sum(eligible), mx.array(1.0))
            resolved = resolved + eligible * per_sample
        return resolved

    if config.training_generation >= 4:

        def policy_loss_function(
            states,
            legal_actions,
            legal_mask,
            selected_indices,
            families,
            targets,
            behavior_probabilities,
            masks,
            weights,
            search_policy,
            search_mask,
            search_values,
            search_valid,
            deployment_policy,
            actor_advantages,
            actor_advantage_valid,
            rollout_sources,
            preference_states,
            preferred_actions,
            disfavored_actions,
            preference_families,
            preference_masks,
            preference_weight,
        ):
            policy = actor_critic_policy_loss(
                model,
                states,
                legal_actions,
                legal_mask,
                selected_indices,
                families,
                targets,
                behavior_probabilities,
                masks,
                weights,
                value_loss_weight=config.policy_value_loss_weight,
                entropy_weight=config.policy_entropy_weight,
                importance_clip=config.policy_importance_clip,
                search_policy_targets=search_policy,
                search_mask=search_mask,
                search_values=search_values,
                search_valid=search_valid,
                search_policy_loss_weight=config.reanalysis_policy_loss_weight,
                search_value_loss_weight=config.reanalysis_value_loss_weight,
                search_loss_reference_positions=config.reanalysis_loss_reference_positions,
                actor_sample_weights=resolved_actor_sample_weights(
                    deployment_policy, rollout_sources
                ),
                actor_advantages=actor_advantages,
                actor_advantage_valid=actor_advantage_valid,
                reference_model=reference_model,
                reference_policy_kl_weight=reference_kl_weight,
            )[0]
            counterfactual = preference_ranking_loss(
                model,
                preference_states,
                preferred_actions,
                disfavored_actions,
                preference_families,
                margin=config.preference_margin,
                bootstrap_mask=preference_masks,
            )[0]
            return policy + preference_weight * counterfactual

        loss_and_grad = nn.value_and_grad(model, policy_loss_function)
        last_policy_arrays: tuple[Any, ...] | None = None
        last_policy_batch: Any | None = None
        last_preference_arrays: tuple[Any, ...] | None = None
        loss_value = gradient_norm = learning_rate = 0.0
        completed = 0
        clipped_updates = 0
        gradient_norms: list[float] = []
        clipping_scales: list[float] = []
        parameter_update_norms: list[float] = []
        relative_parameter_update_norms: list[float] = []
        parameter_norm = 0.0
        for _ in range(count):
            if control.should_stop() or control.pause_requested.is_set():
                break
            batch = policy_replay.sample(config.batch_size)
            arrays = (
                mx.array(batch.states),
                mx.array(batch.legal_actions),
                mx.array(batch.legal_mask),
                mx.array(batch.selected_indices),
                mx.array(batch.families),
                mx.array(batch.targets),
                mx.array(batch.behavior_probabilities),
                mx.array(batch.bootstrap_mask),
                mx.array(batch.sample_weights),
                mx.array(batch.search_policy),
                mx.array(batch.search_mask),
                mx.array(batch.search_values),
                mx.array(batch.search_valid),
                mx.array(batch.deployment_policy),
                mx.array(batch.actor_advantages),
                mx.array(batch.actor_advantage_valid),
                mx.array(batch.rollout_sources),
            )
            if len(preference_replay):
                preference_batch = preference_replay.sample(config.preference_batch_size)
                preference_arrays = (
                    mx.array(preference_batch.states),
                    mx.array(preference_batch.preferred_actions),
                    mx.array(preference_batch.disfavored_actions),
                    mx.array(preference_batch.families),
                    mx.array(preference_batch.bootstrap_mask),
                    mx.array(config.counterfactual_loss_weight),
                )
            else:
                preference_arrays = (
                    arrays[0][:1],
                    arrays[1][:1, 0],
                    arrays[1][:1, 0],
                    arrays[4][:1],
                    arrays[7][:1],
                    mx.array(0.0),
                )
            relative_updates = totals.updates - optimizer_updates_at_start
            learning_rate = _learning_rate(
                config,
                relative_updates if config.training_generation >= 5 else totals.updates,
                relative_updates,
            )
            learning_rate = min(
                config.learning_rate * config.governor_max_learning_rate_multiplier,
                max(
                    config.min_learning_rate * config.governor_min_learning_rate_multiplier,
                    learning_rate * learning_rate_multiplier,
                ),
            )
            optimizer.learning_rate = learning_rate
            loss, gradients = loss_and_grad(*arrays, *preference_arrays)
            if promotion_direction and config.promotion_direction_strength > 0:
                from mlx.utils import tree_flatten, tree_unflatten

                guided = []
                for name, gradient in tree_flatten(gradients):
                    direction = promotion_direction.get(name)
                    if direction is None or tuple(direction.shape) != tuple(gradient.shape):
                        guided.append((name, gradient))
                        continue
                    gradient_rms = mx.sqrt(mx.mean(mx.square(gradient)) + 1e-12)
                    guided.append(
                        (
                            name,
                            gradient
                            - config.promotion_direction_strength * gradient_rms * direction,
                        )
                    )
                gradients = tree_unflatten(guided)
            gradients, norm = optim.clip_grad_norm(gradients, config.gradient_clip)
            parameters_before = [
                value for _name, value in tree_flatten(model.trainable_parameters())
            ]
            optimizer.update(model, gradients)
            parameters_after = [
                value for _name, value in tree_flatten(model.trainable_parameters())
            ]
            update_norm_array = mx.sqrt(
                sum(
                    mx.sum(mx.square(after - before))
                    for before, after in zip(parameters_before, parameters_after, strict=True)
                )
            )
            parameter_norm_array = mx.sqrt(
                sum(mx.sum(mx.square(value)) for value in parameters_after)
            )
            mx.eval(
                model.parameters(),
                optimizer.state,
                loss,
                norm,
                update_norm_array,
                parameter_norm_array,
            )
            loss_value = float(loss.item())
            gradient_norm = float(norm.item())
            clipped_updates += int(gradient_norm > config.gradient_clip)
            gradient_norms.append(gradient_norm)
            clipping_scales.append(min(1.0, config.gradient_clip / max(gradient_norm, 1e-12)))
            update_norm = float(update_norm_array.item())
            parameter_norm = float(parameter_norm_array.item())
            parameter_update_norms.append(update_norm)
            relative_parameter_update_norms.append(update_norm / max(parameter_norm, 1e-12))
            totals.updates += 1
            completed += 1
            last_policy_arrays = arrays
            last_policy_batch = batch
            last_preference_arrays = preference_arrays
        metrics: dict[str, float] = {
            "loss": loss_value,
            "gradient_norm": gradient_norm,
            "learning_rate": learning_rate,
            "learner_updates": float(completed),
            "gradient_clip_fraction": clipped_updates / max(1, completed),
            "gradient_norm_p50": float(np.median(gradient_norms)) if gradient_norms else 0.0,
            "gradient_norm_p90": (
                float(np.percentile(gradient_norms, 90)) if gradient_norms else 0.0
            ),
            "gradient_clipping_scale_mean": (
                float(np.mean(clipping_scales)) if clipping_scales else 1.0
            ),
            "parameter_update_norm_mean": (
                float(np.mean(parameter_update_norms)) if parameter_update_norms else 0.0
            ),
            "parameter_update_norm_p90": (
                float(np.percentile(parameter_update_norms, 90)) if parameter_update_norms else 0.0
            ),
            "relative_parameter_update_norm_mean": (
                float(np.mean(relative_parameter_update_norms))
                if relative_parameter_update_norms
                else 0.0
            ),
            "parameter_norm": parameter_norm,
            "learning_rate_multiplier": float(learning_rate_multiplier),
            "promotion_direction_strength": float(
                config.promotion_direction_strength if promotion_direction else 0.0
            ),
            "promotion_direction_tensors": float(len(promotion_direction or {})),
        }
        if last_policy_arrays is not None:
            assert last_policy_batch is not None
            collection_games = np.asarray(last_policy_batch.collected_at_games, dtype=np.int64)
            known_age = collection_games > 0
            sample_ages = np.maximum(0, int(totals.games) - collection_games)
            importance_groups: dict[str, Any] = {
                "tier_hot": mx.array(last_policy_batch.sample_tiers == 0),
                "tier_cold": mx.array(last_policy_batch.sample_tiers == 1),
                "age_unknown": mx.array(~known_age),
                "age_0_50k": mx.array(known_age & (sample_ages <= 50_000)),
                "age_50k_250k": mx.array(
                    known_age & (sample_ages > 50_000) & (sample_ages <= 250_000)
                ),
                "age_over_250k": mx.array(known_age & (sample_ages > 250_000)),
                "searched": mx.array(last_policy_batch.search_valid > 0),
                "unsearched": mx.array(last_policy_batch.search_valid <= 0),
                "behavior_deployment": mx.array(last_policy_batch.deployment_policy > 0),
                "behavior_exploration": mx.array(last_policy_batch.behavior_heads >= 0),
                "behavior_metadata_unknown": mx.array(
                    last_policy_batch.collection_policy_probabilities <= 0
                ),
            }
            for head in range(config.bootstrap_heads):
                importance_groups[f"behavior_head_{head}"] = mx.array(
                    last_policy_batch.behavior_heads == head
                )
            for source_id, source_name in POLICY_ROLLOUT_SOURCE_NAMES.items():
                importance_groups[f"source_{source_name}"] = mx.array(
                    last_policy_batch.rollout_sources == source_id
                )
            if fixed_champion_key:
                importance_groups["opponent_fixed_champion"] = mx.array(
                    last_policy_batch.opponent_keys == fixed_champion_key
                )
            opponent_keys, opponent_counts = np.unique(
                last_policy_batch.opponent_keys[last_policy_batch.opponent_keys > 0],
                return_counts=True,
            )
            for index in np.argsort(opponent_counts)[-8:]:
                opponent_key = int(opponent_keys[index])
                importance_groups[f"opponent_{opponent_key:016x}"] = mx.array(
                    last_policy_batch.opponent_keys == opponent_key
                )
            for family in DecisionFamily:
                importance_groups[f"family_{family.name.lower()}"] = mx.array(
                    last_policy_batch.families == int(family)
                )
            diagnostic_loss, diagnostics = actor_critic_policy_loss(
                model,
                *last_policy_arrays[:9],
                value_loss_weight=config.policy_value_loss_weight,
                entropy_weight=config.policy_entropy_weight,
                importance_clip=config.policy_importance_clip,
                search_policy_targets=last_policy_arrays[9],
                search_mask=last_policy_arrays[10],
                search_values=last_policy_arrays[11],
                search_valid=last_policy_arrays[12],
                search_policy_loss_weight=config.reanalysis_policy_loss_weight,
                search_value_loss_weight=config.reanalysis_value_loss_weight,
                search_loss_reference_positions=config.reanalysis_loss_reference_positions,
                actor_sample_weights=resolved_actor_sample_weights(
                    last_policy_arrays[13], last_policy_arrays[16]
                ),
                actor_advantages=last_policy_arrays[14],
                actor_advantage_valid=last_policy_arrays[15],
                reference_model=reference_model,
                reference_policy_kl_weight=reference_kl_weight,
                collection_policy_probabilities=mx.array(
                    last_policy_batch.collection_policy_probabilities
                ),
                behavior_heads=mx.array(last_policy_batch.behavior_heads),
                importance_groups=importance_groups,
            )
            mx.eval(diagnostic_loss, *diagnostics.values())
            metrics.update(
                {
                    "loss": float(diagnostic_loss.item()),
                    **{name: float(value.item()) for name, value in diagnostics.items()},
                    "brier": float(diagnostics["value_brier"].item()),
                    "accuracy": float(diagnostics["value_accuracy"].item()),
                }
            )
            observed_reference_kl = float(diagnostics["reference_policy_kl"].item())
            relative_update_count = max(1, totals.updates - optimizer_updates_at_start)
            metrics.update(
                policy_reference_kl_effective_weight=reference_kl_weight,
                policy_reference_kl_target=float(config.policy_reference_kl_target),
                reference_policy_kl_per_1000_updates=(
                    1_000.0 * observed_reference_kl / relative_update_count
                ),
            )
            if policy_reference_kl_controller is not None:
                next_weight = reference_kl_weight
                target = float(config.policy_reference_kl_target)
                ema_initialized = bool(policy_reference_kl_controller.get("ema_initialized", False))
                previous_ema = float(
                    policy_reference_kl_controller.get("ema_kl", observed_reference_kl)
                )
                decay = float(config.policy_reference_kl_ema_decay)
                smoothed_reference_kl = (
                    decay * previous_ema + (1.0 - decay) * observed_reference_kl
                    if ema_initialized
                    else observed_reference_kl
                )
                next_adjustment_update = int(
                    policy_reference_kl_controller.get("next_adjustment_update", 0)
                )
                adjustment_due = totals.updates >= next_adjustment_update
                if target > 0 and adjustment_due:
                    if smoothed_reference_kl > target * 1.25:
                        next_weight *= config.policy_reference_kl_adjustment
                    elif smoothed_reference_kl < target / 1.25:
                        next_weight /= config.policy_reference_kl_adjustment
                    next_weight = min(
                        config.policy_reference_kl_max_weight,
                        max(config.policy_reference_kl_min_weight, next_weight),
                    )
                    next_adjustment_update = (
                        totals.updates + config.policy_reference_kl_adjust_interval_updates
                    )
                policy_reference_kl_controller.update(
                    weight=float(next_weight),
                    last_kl=observed_reference_kl,
                    ema_kl=float(smoothed_reference_kl),
                    ema_initialized=True,
                    next_adjustment_update=int(next_adjustment_update),
                )
                metrics["policy_reference_kl_next_weight"] = float(next_weight)
                metrics["policy_reference_kl_ema"] = float(smoothed_reference_kl)
                metrics["policy_reference_kl_adjustment_due"] = float(adjustment_due)
            if last_preference_arrays is not None:
                preference_loss, preference_diagnostics = preference_ranking_loss(
                    model,
                    *last_preference_arrays[:4],
                    margin=config.preference_margin,
                    bootstrap_mask=last_preference_arrays[4],
                )
                mx.eval(preference_loss, *preference_diagnostics.values())
                counterfactual_loss = float(preference_loss.item())
                weighted_counterfactual_loss = (
                    float(last_preference_arrays[5].item()) * counterfactual_loss
                )
                metrics.update(
                    {
                        "loss": float(diagnostic_loss.item()) + weighted_counterfactual_loss,
                        "actor_critic_loss": float(diagnostic_loss.item()),
                        "counterfactual_loss": counterfactual_loss,
                        "weighted_counterfactual_loss": weighted_counterfactual_loss,
                        **{
                            f"counterfactual_{name}": float(value.item())
                            for name, value in preference_diagnostics.items()
                        },
                    }
                )
            # Objective-specific norms are expensive enough to sample rather
            # than compute every update. They expose which loss is consuming
            # the shared trunk when the aggregate gradient is clipped.
            crossed_gradient_probe = (
                completed > 0
                and totals.updates // config.objective_gradient_probe_interval_updates
                != max(0, totals.updates - completed)
                // config.objective_gradient_probe_interval_updates
            )
            if crossed_gradient_probe:

                def component_gradients(
                    kind: str, source_mask: Any | None = None
                ) -> tuple[list[Any], Any]:
                    def component_loss(
                        states,
                        legal_actions,
                        legal_mask,
                        selected_indices,
                        families,
                        targets,
                        behavior_probabilities,
                        masks,
                        weights,
                        search_policy,
                        search_mask,
                        search_values,
                        search_valid,
                        deployment_policy,
                        actor_advantages,
                        actor_advantage_valid,
                        rollout_sources,
                    ):
                        return actor_critic_policy_loss(
                            model,
                            states,
                            legal_actions,
                            legal_mask,
                            selected_indices,
                            families,
                            targets,
                            behavior_probabilities,
                            masks,
                            weights,
                            value_loss_weight=1.0 if kind == "value" else 0.0,
                            entropy_weight=1.0 if kind == "entropy" else 0.0,
                            importance_clip=config.policy_importance_clip,
                            search_policy_targets=search_policy,
                            search_mask=search_mask,
                            search_values=search_values,
                            search_valid=search_valid,
                            search_policy_loss_weight=(1.0 if kind == "search_policy" else 0.0),
                            search_value_loss_weight=(1.0 if kind == "search_value" else 0.0),
                            behavior_policy_loss_weight=1.0 if kind == "actor" else 0.0,
                            search_loss_reference_positions=(
                                config.reanalysis_loss_reference_positions
                            ),
                            actor_sample_weights=(
                                resolved_actor_sample_weights(deployment_policy, rollout_sources)
                                * (
                                    mx.ones_like(deployment_policy)
                                    if source_mask is None
                                    else source_mask
                                )
                            ),
                            actor_advantages=actor_advantages,
                            actor_advantage_valid=actor_advantage_valid,
                            reference_model=(reference_model if kind == "reference_kl" else None),
                            reference_policy_kl_weight=(1.0 if kind == "reference_kl" else 0.0),
                        )[0]

                    _value, gradients = nn.value_and_grad(model, component_loss)(
                        *last_policy_arrays
                    )
                    flattened = [gradient for _name, gradient in tree_flatten(gradients)]
                    norm = mx.sqrt(sum(mx.sum(mx.square(gradient)) for gradient in flattened))
                    return flattened, norm

                component_weights = {
                    "actor": 1.0,
                    "value": float(config.policy_value_loss_weight),
                    "search_policy": float(config.reanalysis_policy_loss_weight),
                    "search_value": float(config.reanalysis_value_loss_weight),
                    # actor_critic_policy_loss differentiates -entropy when
                    # entropy_weight is positive, matching the applied sign.
                    "entropy": float(config.policy_entropy_weight),
                    "reference_kl": reference_kl_weight,
                }
                component_results = {kind: component_gradients(kind) for kind in component_weights}
                source_actor_results = {
                    source_name: component_gradients(
                        "actor",
                        mx.array(last_policy_batch.rollout_sources == source_id),
                    )
                    for source_id, source_name in POLICY_ROLLOUT_SOURCE_NAMES.items()
                }
                gradient_split_results: dict[str, list[tuple[list[Any], Any]]] = {}
                split_count = int(config.objective_gradient_probe_splits)
                if split_count > 1:
                    row_splits = np.arange(len(last_policy_batch.rollout_sources)) % split_count
                    deployment_eligible = (last_policy_batch.deployment_policy <= 0) | (
                        config.deployment_policy_actor_weight > 0
                    )
                    fixed_rows = (
                        last_policy_batch.rollout_sources
                        == POLICY_ROLLOUT_SOURCE_IDS["fixed_champion"]
                    ) & deployment_eligible
                    enabled_nonfixed_sources: set[int] = set()
                    if config.self_play_actor_weight > 0:
                        enabled_nonfixed_sources.add(POLICY_ROLLOUT_SOURCE_IDS["self_play"])
                        enabled_nonfixed_sources.add(
                            POLICY_ROLLOUT_SOURCE_IDS["deployment_self_play"]
                        )
                    if config.league_actor_weight > 0:
                        enabled_nonfixed_sources.add(POLICY_ROLLOUT_SOURCE_IDS["league"])
                    if config.baseline_actor_weight > 0:
                        enabled_nonfixed_sources.add(POLICY_ROLLOUT_SOURCE_IDS["baseline"])
                    nonfixed_rows = (
                        np.isin(
                            last_policy_batch.rollout_sources,
                            tuple(enabled_nonfixed_sources),
                        )
                        & deployment_eligible
                    )
                    for group_name, group_rows in (
                        ("fixed", fixed_rows),
                        ("nonfixed", nonfixed_rows),
                    ):
                        splits = [
                            component_gradients(
                                "actor", mx.array(group_rows & (row_splits == split))
                            )
                            for split in range(split_count)
                            if np.count_nonzero(group_rows & (row_splits == split)) >= 2
                        ]
                        if len(splits) >= 2:
                            gradient_split_results[group_name] = splits
                resolved_actor_weights = resolved_actor_sample_weights(
                    last_policy_arrays[13], last_policy_arrays[16]
                )
                all_actor_weights = (
                    last_policy_arrays[7]
                    * last_policy_arrays[8][:, None]
                    * resolved_actor_weights[:, None]
                )
                all_actor_weight = mx.maximum(mx.sum(all_actor_weights), mx.array(1e-12))
                source_actor_shares = {
                    source_name: mx.sum(
                        all_actor_weights
                        * mx.array(last_policy_batch.rollout_sources == source_id)[:, None]
                    )
                    / all_actor_weight
                    for source_id, source_name in POLICY_ROLLOUT_SOURCE_NAMES.items()
                }
                raw_norms = {kind: result[1] for kind, result in component_results.items()}
                search_gradients = [
                    policy_gradient + value_gradient
                    for policy_gradient, value_gradient in zip(
                        component_results["search_policy"][0],
                        component_results["search_value"][0],
                        strict=True,
                    )
                ]
                search_norm = mx.sqrt(
                    sum(mx.sum(mx.square(gradient)) for gradient in search_gradients)
                )
                weighted_search_gradients = [
                    component_weights["search_policy"] * policy_gradient
                    + component_weights["search_value"] * value_gradient
                    for policy_gradient, value_gradient in zip(
                        component_results["search_policy"][0],
                        component_results["search_value"][0],
                        strict=True,
                    )
                ]
                weighted_search_norm = mx.sqrt(
                    sum(mx.sum(mx.square(gradient)) for gradient in weighted_search_gradients)
                )

                def cosine(left: str, right: str) -> Any:
                    left_gradients, left_norm = component_results[left]
                    right_gradients, right_norm = component_results[right]
                    dot = sum(
                        mx.sum(left_gradient * right_gradient)
                        for left_gradient, right_gradient in zip(
                            left_gradients, right_gradients, strict=True
                        )
                    )
                    return dot / mx.maximum(left_norm * right_norm, mx.array(1e-12))

                cosine_pairs = (
                    ("actor", "value"),
                    ("actor", "search_policy"),
                    ("actor", "search_value"),
                    ("value", "search_policy"),
                    ("value", "search_value"),
                    ("search_policy", "search_value"),
                    ("actor", "entropy"),
                    ("actor", "reference_kl"),
                    ("value", "reference_kl"),
                )
                cosines = {
                    f"objective_{left}_{right}_gradient_cosine": cosine(left, right)
                    for left, right in cosine_pairs
                }
                actor_gradients, actor_norm = component_results["actor"]
                source_actor_cosines = {}
                for source_name, (source_gradients, source_norm) in source_actor_results.items():
                    source_dot = sum(
                        mx.sum(actor_gradient * source_gradient)
                        for actor_gradient, source_gradient in zip(
                            actor_gradients, source_gradients, strict=True
                        )
                    )
                    source_actor_cosines[source_name] = source_dot / mx.maximum(
                        actor_norm * source_norm, mx.array(1e-12)
                    )
                source_pair_cosines = {}
                source_names = tuple(source_actor_results)
                for left_index, left in enumerate(source_names):
                    left_gradients, left_norm = source_actor_results[left]
                    for right in source_names[left_index + 1 :]:
                        right_gradients, right_norm = source_actor_results[right]
                        dot = sum(
                            mx.sum(left_gradient * right_gradient)
                            for left_gradient, right_gradient in zip(
                                left_gradients, right_gradients, strict=True
                            )
                        )
                        source_pair_cosines[f"{left}_{right}"] = dot / mx.maximum(
                            left_norm * right_norm, mx.array(1e-12)
                        )
                gradient_stability_metrics: dict[str, Any] = {}
                for group_name, split_results in gradient_split_results.items():
                    mean_gradients = [
                        sum(split[0][tensor_index] for split in split_results) / len(split_results)
                        for tensor_index in range(len(split_results[0][0]))
                    ]
                    signal_norm = mx.sqrt(
                        sum(mx.sum(mx.square(gradient)) for gradient in mean_gradients)
                    )
                    noise_power = sum(
                        sum(
                            mx.sum(mx.square(gradient - mean_gradient))
                            for gradient, mean_gradient in zip(
                                split_gradients, mean_gradients, strict=True
                            )
                        )
                        for split_gradients, _split_norm in split_results
                    ) / len(split_results)
                    noise_norm = mx.sqrt(noise_power)
                    pair_cosines = []
                    for left_index, (left_gradients, left_norm) in enumerate(split_results):
                        for right_gradients, right_norm in split_results[left_index + 1 :]:
                            dot = sum(
                                mx.sum(left * right)
                                for left, right in zip(left_gradients, right_gradients, strict=True)
                            )
                            pair_cosines.append(
                                dot / mx.maximum(left_norm * right_norm, mx.array(1e-12))
                            )
                    gradient_stability_metrics.update(
                        {
                            f"objective_actor_{group_name}_gradient_signal_norm": signal_norm,
                            f"objective_actor_{group_name}_gradient_noise_norm": noise_norm,
                            f"objective_actor_{group_name}_gradient_snr": (
                                signal_norm / mx.maximum(noise_norm, mx.array(1e-12))
                            ),
                            f"objective_actor_{group_name}_split_cosine_mean": (
                                sum(pair_cosines) / len(pair_cosines)
                            ),
                        }
                    )
                mx.eval(
                    search_norm,
                    weighted_search_norm,
                    *raw_norms.values(),
                    *cosines.values(),
                    *(result[1] for result in source_actor_results.values()),
                    *source_actor_cosines.values(),
                    *source_actor_shares.values(),
                    *source_pair_cosines.values(),
                    *gradient_stability_metrics.values(),
                )

                metrics.update(
                    objective_actor_gradient_norm=float(raw_norms["actor"].item()),
                    objective_value_gradient_norm=float(raw_norms["value"].item()),
                    objective_search_policy_gradient_norm=float(raw_norms["search_policy"].item()),
                    objective_search_value_gradient_norm=float(raw_norms["search_value"].item()),
                    objective_entropy_gradient_norm=float(raw_norms["entropy"].item()),
                    objective_reference_kl_gradient_norm=float(raw_norms["reference_kl"].item()),
                    **{
                        f"objective_actor_source_{source_name}_gradient_norm": float(
                            result[1].item()
                        )
                        for source_name, result in source_actor_results.items()
                    },
                    **{
                        f"objective_actor_source_{source_name}_total_cosine": float(cosine.item())
                        for source_name, cosine in source_actor_cosines.items()
                    },
                    **{
                        f"objective_actor_source_{source_name}_sample_share": float(share.item())
                        for source_name, share in source_actor_shares.items()
                    },
                    **{
                        f"objective_actor_source_{source_name}_weighted_gradient_norm": (
                            float(source_actor_results[source_name][1].item()) * float(share.item())
                        )
                        for source_name, share in source_actor_shares.items()
                    },
                    **{
                        f"objective_actor_source_{pair}_gradient_cosine": float(cosine.item())
                        for pair, cosine in source_pair_cosines.items()
                    },
                    **{
                        name: float(value.item())
                        for name, value in gradient_stability_metrics.items()
                    },
                    objective_search_gradient_norm=float(search_norm.item()),
                    objective_search_weighted_gradient_norm=float(weighted_search_norm.item()),
                    objective_gradient_probe_update=float(totals.updates),
                    **{
                        f"objective_{kind}_weighted_gradient_norm": (
                            float(raw_norm.item()) * abs(component_weights[kind])
                        )
                        for kind, raw_norm in raw_norms.items()
                    },
                    **{name: float(value.item()) for name, value in cosines.items()},
                )
        return metrics

    def loss_function(
        states,
        actions,
        families,
        targets,
        masks,
        weights,
        preference_states,
        preferred_actions,
        disfavored_actions,
        preference_families,
        preference_weight,
    ):
        outcome = bootstrap_bce_loss(model, states, actions, families, targets, masks, weights)[0]
        preference = preference_ranking_loss(
            model,
            preference_states,
            preferred_actions,
            disfavored_actions,
            preference_families,
            margin=config.preference_margin,
        )[0]
        return outcome + preference_weight * preference

    loss_and_grad = nn.value_and_grad(model, loss_function)
    last_arrays: tuple[Any, ...] | None = None
    last_preference_arrays: tuple[Any, ...] | None = None
    loss_value = gradient_norm = learning_rate = 0.0
    completed = 0
    clipped_updates = 0
    for _ in range(count):
        if control.should_stop() or control.pause_requested.is_set():
            break
        batch = replay.sample(config.batch_size)
        if config.use_bootstrap_targets:
            effective_targets = np.where(
                batch.td_valid > 0,
                config.terminal_target_weight * batch.targets
                + (1.0 - config.terminal_target_weight) * batch.td_targets,
                batch.targets,
            ).astype(np.float32)
        else:
            effective_targets = batch.targets.astype(np.float32, copy=False)
        arrays = (
            mx.array(batch.states),
            mx.array(batch.actions),
            mx.array(batch.families),
            mx.array(effective_targets),
            mx.array(batch.bootstrap_mask),
            mx.array(batch.sample_weights),
        )
        if len(preference_replay):
            preference_batch = preference_replay.sample(config.preference_batch_size)
            preference_arrays = (
                mx.array(preference_batch.states),
                mx.array(preference_batch.preferred_actions),
                mx.array(preference_batch.disfavored_actions),
                mx.array(preference_batch.families),
                mx.array(config.preference_loss_weight),
            )
        else:
            # Keep a stable differentiated signature before the preference
            # warmup has produced its first exact pair.
            preference_arrays = (
                arrays[0][:1],
                arrays[1][:1],
                arrays[1][:1],
                arrays[2][:1],
                mx.array(0.0),
            )
        relative_updates = totals.updates - optimizer_updates_at_start
        learning_rate = _learning_rate(
            config,
            relative_updates if config.training_generation >= 5 else totals.updates,
            relative_updates,
        )
        learning_rate = min(
            config.learning_rate * config.governor_max_learning_rate_multiplier,
            max(
                config.min_learning_rate * config.governor_min_learning_rate_multiplier,
                learning_rate * learning_rate_multiplier,
            ),
        )
        optimizer.learning_rate = learning_rate
        loss, gradients = loss_and_grad(*arrays, *preference_arrays)
        gradients, norm = optim.clip_grad_norm(gradients, config.gradient_clip)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss, norm)
        loss_value = float(loss.item())
        gradient_norm = float(norm.item())
        clipped_updates += int(gradient_norm > config.gradient_clip)
        totals.updates += 1
        completed += 1
        last_arrays = arrays
        last_preference_arrays = preference_arrays

    metrics: dict[str, float] = {
        "loss": loss_value,
        "gradient_norm": gradient_norm,
        "learning_rate": learning_rate,
        "learner_updates": float(completed),
        "gradient_clip_fraction": clipped_updates / max(1, completed),
        "learning_rate_multiplier": float(learning_rate_multiplier),
    }
    if last_arrays is not None:
        diagnostic_loss, diagnostics = bootstrap_bce_loss(model, *last_arrays)
        mx.eval(diagnostic_loss, *diagnostics.values())
        metrics.update(
            {
                "loss": float(diagnostic_loss.item()),
                **{name: float(value.item()) for name, value in diagnostics.items()},
                "td_target_fraction": float(batch.td_valid.mean()),
            }
        )
        if last_preference_arrays is not None and len(preference_replay):
            preference_loss, preference_diagnostics = preference_ranking_loss(
                model,
                *last_preference_arrays[:4],
                margin=config.preference_margin,
            )
            mx.eval(preference_loss, *preference_diagnostics.values())
            metrics.update(
                {
                    "preference_loss": float(preference_loss.item()),
                    **{name: float(value.item()) for name, value in preference_diagnostics.items()},
                }
            )
    return metrics


def _configure_mlx(device: str, *, total_memory_bytes: int) -> dict[str, Any]:
    import mlx.core as mx

    if device == "cpu":
        mx.set_default_device(mx.cpu)
    elif device == "gpu":
        if not mx.metal.is_available():
            raise RuntimeError("the requested MLX/Metal GPU is unavailable")
        mx.set_default_device(mx.gpu)
    elif mx.metal.is_available():
        mx.set_default_device(mx.gpu)
    limits = _memory_safety_limits(total_memory_bytes)
    if mx.metal.is_available() and device != "cpu":
        # MLX defaults the free cache to its much larger memory limit.  The
        # observed workload retained 10.6 GB while only ~24 MB was active.
        # Bound both knobs before the first graph is evaluated and discard any
        # allocations made while probing the device.
        mx.set_memory_limit(limits["mlx_memory_limit_bytes"])
        mx.set_cache_limit(limits["mlx_cache_limit_bytes"])
        mx.clear_cache()
    snapshot = mlx_snapshot()
    snapshot["safety_limits"] = limits
    return snapshot


def run_training(
    *,
    run_id: str,
    store: Store,
    project_root: str | Path,
    control: Any,
    publish: Callable[[dict[str, Any]], None],
    evaluation_manager: Any | None = None,
) -> None:
    """Run until the persisted duration expires or a safe stop is requested."""

    # Tiny single-decision matrices are faster with one BLAS thread per actor;
    # process-level parallelism supplies the throughput.
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    run = store.get_run(run_id)
    config = RunConfig.model_validate_persisted(run["config"])
    initial_system = system_snapshot()
    memory_limits = _memory_safety_limits(initial_system["memory_total_bytes"])
    hardware = _configure_mlx(
        config.device,
        total_memory_bytes=initial_system["memory_total_bytes"],
    )
    store.event(
        run_id,
        "training_hardware",
        "Initialized MLX learner and hardware",
        {"mlx": hardware, "system": initial_system, "memory_safety": memory_limits},
    )

    import mlx.core as mx
    import mlx.optimizers as optim

    encoder = Encoder(version=2 if config.training_generation >= 3 else 1)
    # Keep artifacts beside the configured SQLite store so ASTRO2_DATA_DIR and
    # the CLI's --data-dir remain self-contained.
    checkpoint_root = store.path.parent / "checkpoints" / run_id
    runtime_actor = checkpoint_root / "runtime" / "current.actor.npz"
    policy_replay_journal = (
        checkpoint_root
        / "runtime"
        / f"policy-replay-journal-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
    )
    policy_replay_cold = checkpoint_root / "runtime" / "policy-replay-cold"
    latest = _learner_resume_checkpoint(store, run_id, config)
    _repair_anomalous_deployment_anchor(store, run_id)
    if latest is not None:
        model, spec = load_model(latest["path"])
        parent_checkpoint_id = latest["id"]
    else:
        if int(run.get("games", 0)) > 0 or store.checkpoints(run_id):
            raise RuntimeError(
                "refusing to initialize a random model for a non-empty run: "
                "no compatible checkpoint could be loaded"
            )
        spec = ModelSpec(
            state_size=encoder.state_size,
            action_size=encoder.action_size,
            families=FAMILY_COUNT,
            encoder_version=encoder.version,
            hidden_size=config.hidden_size,
            action_hidden_size=max(64, config.hidden_size // 2),
            residual_blocks=config.residual_blocks,
            bootstrap_heads=config.bootstrap_heads,
            objective_version=2 if config.training_generation >= 4 else 1,
        )
        model = build_model(spec)
        parent_checkpoint_id = None
    if (
        spec.state_size != encoder.state_size
        or spec.action_size != encoder.action_size
        or spec.families != FAMILY_COUNT
        or spec.encoder_version != encoder.version
        or spec.objective_version != (2 if config.training_generation >= 4 else 1)
    ):
        raise RuntimeError("checkpoint training/encoder contract does not match this run")

    branch_root_checkpoint = next(
        (
            checkpoint
            for checkpoint in store.checkpoints(run_id)
            if int(checkpoint.get("games", -1)) == 0
            and str((checkpoint.get("evaluation") or {}).get("reason") or "") == "branch import"
        ),
        None,
    )
    if branch_root_checkpoint is None and config.initial_checkpoint_id:
        branch_root_checkpoint = store.checkpoint(config.initial_checkpoint_id)
    fixed_champion_id = (
        str(branch_root_checkpoint["id"]) if branch_root_checkpoint is not None else None
    )
    fixed_champion_actor = (
        str(branch_root_checkpoint.get("actor_path") or "")
        if branch_root_checkpoint is not None
        else None
    )
    if fixed_champion_actor and not Path(fixed_champion_actor).is_file():
        fixed_champion_actor = None
    reference_model = None
    if config.policy_reference_kl_weight > 0:
        if branch_root_checkpoint is None or not Path(branch_root_checkpoint["path"]).is_file():
            raise RuntimeError("policy reference KL requires retained branch-root weights")
        reference_model, reference_spec = load_model(branch_root_checkpoint["path"])
        if reference_spec != spec:
            raise RuntimeError("policy reference checkpoint architecture does not match learner")
        reference_model.eval()
        mx.eval(reference_model.parameters())
        store.event(
            run_id,
            "policy_reference_loaded",
            "Loaded immutable branch-root policy reference",
            {
                "checkpoint_id": fixed_champion_id,
                "weight": config.policy_reference_kl_weight,
            },
        )
    if config.fixed_champion_fraction > 0 and (not fixed_champion_id or not fixed_champion_actor):
        raise RuntimeError("fixed champion quota requires a retained branch-root actor")

    model.train()
    mx.eval(model.parameters())
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        betas=[0.9, 0.95],
        weight_decay=config.weight_decay,
        bias_correction=True,
    )
    if config.persist_optimizer_state:
        optimizer.init(model.trainable_parameters())
        mx.eval(optimizer.state)
    replay = ReplayBuffer(
        capacity=(config.replay_capacity if config.training_generation < 4 else 10_000),
        state_size=encoder.state_size,
        action_size=encoder.action_size,
        bootstrap_heads=config.bootstrap_heads,
        recent_sample_fraction=config.recent_sample_fraction,
        family_sampling_weights=(
            NATURAL_SAMPLING_WEIGHTS if config.replay_sampling_profile == "natural" else None
        ),
        importance_correct_sampling=config.importance_correct_replay,
        seed=config.seed + 41,
    )
    preference_collection_enabled = bool(
        config.tactical_preference_training
        or (config.training_generation >= 4 and config.counterfactual_fraction > 0)
    )
    preference_replay = PreferenceReplayBuffer(
        # Avoid reserving ~173 MB for a buffer that Astro5 never writes.
        capacity=config.preference_replay_capacity if preference_collection_enabled else 1,
        state_size=encoder.state_size,
        action_size=encoder.action_size,
        bootstrap_heads=config.bootstrap_heads,
        seed=config.seed + 43,
    )
    effective_policy_replay_capacity = min(
        config.policy_replay_capacity,
        memory_limits["policy_replay_capacity"],
    )
    policy_replay = GameBalancedPolicyReplayBuffer(
        capacity=effective_policy_replay_capacity,
        state_size=encoder.state_size,
        action_size=encoder.action_size,
        bootstrap_heads=config.bootstrap_heads,
        max_decisions_per_player_game=config.policy_replay_decisions_per_player_game,
        family_balanced=config.policy_replay_family_balanced,
        family_balanced_fraction=config.policy_replay_family_balanced_fraction,
        disk_directory=policy_replay_cold,
        disk_capacity=(
            config.policy_replay_disk_capacity if config.training_generation >= 5 else 0
        ),
        disk_sample_fraction=config.policy_replay_disk_sample_fraction,
        disk_shard_items=config.policy_replay_disk_shard_items,
        seed=config.seed + 47,
    )
    store.event(
        run_id,
        "memory_safety_configured",
        "Applied bounded replay and Metal memory limits",
        {
            **memory_limits,
            "configured_policy_replay_capacity": config.policy_replay_capacity,
            "effective_policy_replay_capacity": effective_policy_replay_capacity,
            "policy_replay_disk_capacity": policy_replay.disk_capacity,
            "policy_replay_total_capacity": policy_replay.total_capacity,
            "policy_replay_disk_sample_fraction": policy_replay.disk_sample_fraction,
            "configured_preference_replay_capacity": config.preference_replay_capacity,
            "effective_preference_replay_capacity": preference_replay.capacity,
        },
    )
    totals = _restore_totals(store, run, latest)
    restored_training_state = _checkpoint_training_state(latest)
    restored_kl_controller = restored_training_state.get("policy_reference_kl_controller") or {}
    policy_reference_kl_controller: dict[str, float] = {
        "weight": float(restored_kl_controller.get("weight", config.policy_reference_kl_weight)),
        "last_kl": float(restored_kl_controller.get("last_kl", 0.0)),
        "ema_kl": float(restored_kl_controller.get("ema_kl", 0.0)),
        "ema_initialized": bool(restored_kl_controller.get("ema_initialized", False)),
        "next_adjustment_update": int(restored_kl_controller.get("next_adjustment_update", 0)),
    }
    schedule_games_origin = max(0, int(restored_training_state.get("schedule_games_origin", 0)))
    artifacts = ((latest or {}).get("evaluation") or {}).get("artifacts") or {}
    promotion_direction: dict[str, Any] = {}
    promotion_direction_metadata: dict[str, Any] = {}
    if config.promotion_direction_enabled:
        from .promotion_direction import load_promotion_direction

        direction_artifact: str | None = (
            config.promotion_direction_path
            if config.promotion_direction_path and Path(config.promotion_direction_path).is_file()
            else None
        )
        for checkpoint in store.checkpoints(run_id):
            if direction_artifact is not None:
                break
            checkpoint_artifacts = (checkpoint.get("evaluation") or {}).get("artifacts") or {}
            candidate_path = checkpoint_artifacts.get("promotion_direction_path")
            if isinstance(candidate_path, str) and Path(candidate_path).is_file():
                direction_artifact = candidate_path
                break
        if direction_artifact is None:
            raise RuntimeError(
                "promotion-direction mode requires its branch-root direction artifact"
            )
        direction_arrays, promotion_direction_metadata = load_promotion_direction(
            direction_artifact
        )
        promotion_direction = {name: mx.array(value) for name, value in direction_arrays.items()}
        mx.eval(*promotion_direction.values())
        store.event(
            run_id,
            "promotion_direction_loaded",
            "Loaded verified promotion-direction guidance",
            promotion_direction_metadata,
        )
    latest_reason = str(((latest or {}).get("evaluation") or {}).get("reason") or "")
    initial_branch_import = bool(
        latest is not None and int(latest.get("games", 0)) == 0 and latest_reason == "branch import"
    )
    try:
        replay_items_persisted = max(
            0,
            int(
                artifacts.get(
                    "policy_replay_items" if config.training_generation >= 4 else "replay_items",
                    0,
                )
            ),
        )
    except (TypeError, ValueError):
        replay_items_persisted = 0
    artifacts_complete = bool(
        latest is None
        or latest.get(
            "_resume_artifacts_complete",
            _checkpoint_has_required_artifacts(latest, config),
        )
    )
    durable_resume: dict[str, Any] = {
        "optimizer_restored": False,
        "replay_items_restored": 0,
        "policy_replay_items_restored": 0,
        "preference_replay_items_restored": 0,
        "replay_rng_restored": False,
        "rollout_rng_restored": False,
        "league_opponents_restored": 0,
        "optimizer_persisted": bool(artifacts.get("optimizer_path")),
        "replay_items_persisted": replay_items_persisted,
        "replay_capacity_at_snapshot": max(
            0,
            int(
                artifacts.get(
                    "policy_replay_total_capacity"
                    if config.training_generation >= 4
                    else "replay_capacity"
                )
                or (
                    policy_replay.total_capacity
                    if config.training_generation >= 4
                    else config.replay_capacity
                )
            ),
        ),
        "replay_snapshot_mode": str(
            artifacts.get(
                "policy_replay_format" if config.training_generation >= 4 else "replay_format"
            )
            or "none"
        ),
        "latest_checkpoint_id": (latest or {}).get("id"),
        "latest_checkpoint_games": int((latest or {}).get("games", 0)),
        "latest_checkpoint_reason": str(
            ((latest or {}).get("evaluation") or {}).get("reason") or ""
        ),
        "checkpoint_artifacts_complete": artifacts_complete,
        "fallback_checkpoint_ids": list((latest or {}).get("_resume_skipped_checkpoint_ids", [])),
        "degraded_reasons": [],
        "branch_optimizer_reset": bool(
            initial_branch_import and config.reset_optimizer_on_branch_start
        ),
        "branch_replay_reset": bool(initial_branch_import and config.reset_replay_on_branch_start),
        "promotion_direction": promotion_direction_metadata,
    }
    if latest is not None and artifacts_complete:
        try:
            if (
                config.training_generation >= 4
                and config.resume_replay_items
                and not (initial_branch_import and config.reset_replay_on_branch_start)
                and artifacts.get("policy_replay_path")
            ):
                durable_resume["policy_replay_items_restored"] = policy_replay.restore(
                    artifacts["policy_replay_path"]
                )
            elif (
                config.resume_replay_items
                and not (initial_branch_import and config.reset_replay_on_branch_start)
                and artifacts.get("replay_path")
            ):
                durable_resume["replay_items_restored"] = replay.restore(artifacts["replay_path"])
            if (
                config.resume_replay_items
                and not (initial_branch_import and config.reset_replay_on_branch_start)
                and artifacts.get("preference_replay_path")
            ):
                durable_resume["preference_replay_items_restored"] = preference_replay.restore(
                    artifacts["preference_replay_path"]
                )
        except (OSError, ValueError, KeyError) as error:
            replay.clear()
            policy_replay.clear()
            preference_replay.clear()
            artifacts_complete = False
            durable_resume["checkpoint_artifacts_complete"] = False
            durable_resume["degraded_reasons"].append(
                f"replay restore failed: {type(error).__name__}"
            )
        try:
            if (
                artifacts_complete
                and config.persist_optimizer_state
                and not (initial_branch_import and config.reset_optimizer_on_branch_start)
                and artifacts.get("optimizer_path")
            ):
                durable_resume["optimizer_restored"] = load_optimizer_state(
                    optimizer, artifacts["optimizer_path"]
                )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            artifacts_complete = False
            durable_resume["checkpoint_artifacts_complete"] = False
            durable_resume["degraded_reasons"].append(
                f"optimizer restore failed: {type(error).__name__}"
            )
        if artifacts_complete and restored_training_state.get("replay_rng_state"):
            replay.restore_rng_state(restored_training_state["replay_rng_state"])
            durable_resume["replay_rng_restored"] = True
    elif latest is not None:
        durable_resume["degraded_reasons"].append(
            "required checkpoint artifacts unavailable; resumed weights with fresh optimizer/replay"
        )
    optimizer_updates_at_start = _optimizer_schedule_origin(
        completed_updates=totals.updates,
        training_state=restored_training_state,
        optimizer_restored=bool(durable_resume["optimizer_restored"]),
    )
    if latest is None:
        latest = _save_checkpoint(
            store=store,
            run_id=run_id,
            model=model,
            spec=spec,
            checkpoint_dir=checkpoint_root,
            games=totals.games,
            parent_id=None,
            champion=True,
            reason="initial random model",
            optimizer=optimizer if config.persist_optimizer_state else None,
            replay=replay,
            policy_replay=policy_replay if config.training_generation >= 4 else None,
            preference_replay=preference_replay,
            resume_replay_items=config.resume_replay_items,
            training_state=_training_state(
                totals,
                seed_cursor=config.seed,
                optimizer_updates_at_start=optimizer_updates_at_start,
                schedule_games_origin=schedule_games_origin,
                active_elapsed_seconds=0.0,
                policy_reference_kl_controller=policy_reference_kl_controller,
            ),
        )
        parent_checkpoint_id = latest["id"]

    if config.training_generation >= 4 and config.resume_replay_items > 0:
        restored_policy_path = artifacts.get("policy_replay_path")
        source_manifest = (
            restored_policy_path
            if artifacts_complete
            and restored_policy_path
            and Path(restored_policy_path).suffix.lower() == ".json"
            else None
        )
        policy_replay.enable_incremental_snapshots(
            policy_replay_journal,
            max_items=config.resume_replay_items,
            source_manifest=source_manifest,
        )

    if durable_resume["degraded_reasons"]:
        store.event(
            run_id,
            "resume_degraded",
            "Resumed checkpoint weights without the complete optimizer/replay boundary",
            dict(durable_resume),
        )

    league = League()
    for name in ("balanced", "economy", "aggressive"):
        league.upsert(
            Opponent(
                id=f"baseline:{name}",
                actor_path=None,
                kind="baseline",
                label=f"{name.title()} baseline",
            )
        )
    _sync_league(
        league,
        store,
        run_id,
        include_external_anchors=config.training_generation >= 3,
    )
    if restored_training_state.get("league_state"):
        durable_resume["league_opponents_restored"] = league.restore(
            restored_training_state["league_state"]
        )

    rng = np.random.default_rng(config.seed + totals.games + 73)
    if restored_training_state.get("rollout_rng_state"):
        rng.bit_generator.state = restored_training_state["rollout_rng_state"]
        durable_resume["rollout_rng_restored"] = True
    rate = RateMeter.start()
    rate.last_games = totals.games
    rate.last_decisions = totals.decisions
    active_clock = _ActiveElapsedClock(_persisted_active_elapsed(store, run_id, latest))
    last_metric_at = 0.0
    last_memory_pressure_event_at = 0.0
    consecutive_critical_memory_samples = 0
    last_diagnostics: dict[str, float] = {}
    last_reported_rollout_games = dict(totals.rollout_completed_games)
    last_reported_rollout_scores = dict(totals.rollout_scores)
    metric_seq = int(time.time() * 1_000)
    seed_cursor = int(
        restored_training_state.get("seed_cursor", config.seed + totals.games * 10_007)
    )
    last_checkpoint_games = int(latest["games"] if latest else totals.games)
    processed_evaluation_jobs = {
        job["id"]
        for job in _terminal_trainer_evaluations(store, run_id)
        if bool((job.get("result") or {}).get("_trainer_disposition_processed"))
    }
    plateau = _plateau_status(store, run_id, config)
    governor: dict[str, Any] = {
        "enabled": bool(config.realtime_governor),
        "learning_rate_multiplier": 1.0,
        "updates_multiplier": 1.0,
        "reanalysis_multiplier": 1.0,
        "entropy_weight": config.policy_entropy_weight,
        "branch_requested": False,
        "reasons": [],
    }
    final_reason = "duration complete"

    def restore_champion(
        champion_id: str,
        *,
        rejected_checkpoint_id: str,
        source: str,
        detail: dict[str, Any] | None = None,
    ) -> bool:
        nonlocal model, optimizer, optimizer_updates_at_start, parent_checkpoint_id
        nonlocal schedule_games_origin
        restore_enabled = config.rollback_rejected_candidates or (
            config.rejected_candidate_action == "restore_lineage"
        )
        if not restore_enabled:
            return False
        champion = store.checkpoint(champion_id)
        restored_model, restored_spec = load_model(champion["path"])
        if restored_spec != spec:
            raise RuntimeError("champion architecture changed during training")
        model = restored_model
        model.train()
        mx.eval(model.parameters())
        optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            betas=[0.9, 0.95],
            weight_decay=config.weight_decay,
            bias_correction=True,
        )
        if config.persist_optimizer_state:
            optimizer.init(model.trainable_parameters())
            mx.eval(optimizer.state)
        champion_artifacts = (champion.get("evaluation") or {}).get("artifacts") or {}
        optimizer_restored = False
        if champion_artifacts.get("optimizer_path"):
            optimizer_restored = load_optimizer_state(
                optimizer, champion_artifacts["optimizer_path"]
            )
        # Replay is part of model lineage. Mixing post-rejection trajectories
        # into restored weights recreates the rejected policy through the next
        # few thousand updates, so restore the champion snapshot or clear it.
        replay.clear()
        policy_replay.clear()
        preference_replay.clear()
        if champion_artifacts.get("replay_path") and config.training_generation < 4:
            replay.restore(champion_artifacts["replay_path"])
        if champion_artifacts.get("policy_replay_path") and config.training_generation >= 4:
            policy_replay.restore(champion_artifacts["policy_replay_path"])
        if champion_artifacts.get("preference_replay_path"):
            preference_replay.restore(champion_artifacts["preference_replay_path"])
        if policy_replay.incremental_snapshots_enabled:
            policy_replay.enable_incremental_snapshots(
                policy_replay_journal,
                max_items=config.resume_replay_items,
                reset=True,
            )
        optimizer_updates_at_start = totals.updates
        schedule_games_origin = totals.games
        parent_checkpoint_id = champion["id"]
        payload = {
            "source": source,
            "rejected_checkpoint_id": rejected_checkpoint_id,
            "champion_id": champion["id"],
            "optimizer_restored": optimizer_restored,
            "policy_replay_items": len(policy_replay),
            "schedule_games_origin": schedule_games_origin,
            **(detail or {}),
        }
        store.event(
            run_id,
            "candidate_rollback",
            f"Restored {champion['label']} after a rejected candidate",
            payload,
        )
        return True

    def process_completed_evaluations() -> str | None:
        """Apply each terminal arena disposition exactly once."""

        evaluation_boundary_checkpoint_id: str | None = None
        for job in _terminal_trainer_evaluations(store, run_id):
            if job["id"] in processed_evaluation_jobs:
                continue
            evaluation_boundary_checkpoint_id = job["model_a"]
            outcome = _trainer_evaluation_outcome(job)
            if outcome == "diagnostic_complete":
                _mark_evaluation_disposition(store, job, outcome)
                processed_evaluation_jobs.add(job["id"])
                continue
            if outcome not in {"promoted", "not_promoted"}:
                _mark_evaluation_disposition(store, job, f"retryable_{outcome}")
                processed_evaluation_jobs.add(job["id"])
                continue
            if outcome == "promoted":
                _mark_evaluation_disposition(store, job, "promoted")
                processed_evaluation_jobs.add(job["id"])
                continue
            restore_enabled = config.rollback_rejected_candidates or (
                config.rejected_candidate_action == "restore_lineage"
            )
            if not restore_enabled:
                disposition = (
                    "branch_requested"
                    if config.rejected_candidate_action == "queue_branch"
                    else "quarantined_continuing"
                )
                _mark_evaluation_disposition(store, job, disposition)
                state = store.controller_state(run_id)
                store.set_controller_state(
                    run_id,
                    {
                        **state,
                        "last_rejected_checkpoint_id": job["model_a"],
                        "last_rejected_at_games": totals.games,
                        "branch_requested": config.rejected_candidate_action == "queue_branch",
                        "deployment_quarantined": True,
                    },
                )
                processed_evaluation_jobs.add(job["id"])
                continue
            current_run = store.get_run(run_id)
            if current_run.get("champion_id") != job["model_b"]:
                # This guard also protects legacy jobs whose result predates
                # the explicit stale_opponent field.
                _mark_evaluation_disposition(store, job, "retryable_stale")
                processed_evaluation_jobs.add(job["id"])
                continue
            restore_champion(
                job["model_b"],
                rejected_checkpoint_id=job["model_a"],
                source="paired_arena",
                detail={"job_id": job["id"]},
            )
            _mark_evaluation_disposition(store, job, "rolled_back")
            processed_evaluation_jobs.add(job["id"])
        return evaluation_boundary_checkpoint_id

    def maybe_schedule_evaluation(
        checkpoint: dict[str, Any] | None = None,
        *,
        ignore_retry_backoff: bool = False,
        force_full: bool = False,
    ) -> dict[str, Any] | None:
        nonlocal evaluation_manager
        due = _next_evaluation_candidate(
            store,
            run_id,
            config,
            checkpoint,
            ignore_retry_backoff=ignore_retry_backoff,
            force_full=force_full,
        )
        if due is None:
            return None
        checkpoint, plan = due
        champion_id = store.get_run(run_id).get("champion_id")
        if not champion_id:
            return None
        quality_gate = _checkpoint_quality_gate(
            store=store,
            run_id=run_id,
            checkpoint=checkpoint,
            config=config,
        )
        if not quality_gate["passed"]:
            if config.training_generation >= 5:
                # A safety gate may quarantine deployment, but it must not
                # erase measurement. Run the same paired workload without
                # promotion authority so final and canary curves stay complete.
                store.event(
                    run_id,
                    "automatic_evaluation_quarantined",
                    f"Measuring gated checkpoint {checkpoint['label']} without promotion",
                    {
                        "checkpoint_id": checkpoint["id"],
                        "reasons": quality_gate["reasons"],
                        "tier": plan.tier,
                    },
                )
                if evaluation_manager is None:
                    from .arena import ArenaManager

                    evaluation_manager = ArenaManager(
                        store, maximum_concurrent_jobs=1, recover=False
                    )
                return _schedule_evaluation(
                    manager=evaluation_manager,
                    store=store,
                    run_id=run_id,
                    checkpoint=checkpoint,
                    config=config,
                    plan=_EvaluationPlan(
                        tier=plan.tier,
                        cadence_games=plan.cadence_games,
                        pairs=plan.pairs,
                        automatic_promotion=False,
                    ),
                    cancellation_hook=lambda: (
                        control.should_stop() or control.pause_requested.is_set()
                    ),
                )
            store.event(
                run_id,
                "automatic_evaluation_blocked",
                f"Skipped arena evaluation for {checkpoint['label']}",
                {
                    "checkpoint_id": checkpoint["id"],
                    "reasons": quality_gate["reasons"],
                },
            )
            rolled_back = restore_champion(
                champion_id,
                rejected_checkpoint_id=checkpoint["id"],
                source="checkpoint_quality_gate",
                detail={"reasons": quality_gate["reasons"]},
            )
            store.update_checkpoint_evaluation(
                checkpoint["id"],
                {
                    "quality_gate": {
                        **quality_gate,
                        "rollback_applied": rolled_back,
                    }
                },
            )
            return None
        if evaluation_manager is None:
            from .arena import ArenaManager

            # The evaluator owns one CPU thread and persists everything. Its
            # daemon thread may outlive learning so a final comparison can
            # finish while the local backend remains open.
            evaluation_manager = ArenaManager(store, maximum_concurrent_jobs=1, recover=False)
        return _schedule_evaluation(
            manager=evaluation_manager,
            store=store,
            run_id=run_id,
            checkpoint=checkpoint,
            config=config,
            plan=plan,
            cancellation_hook=lambda: control.should_stop() or control.pause_requested.is_set(),
        )

    def set_phase(phase: str) -> None:
        status = "paused" if phase == "paused" else "running"
        store.update_run(run_id, status=status, phase=phase)

    def current_active_elapsed() -> float:
        return active_clock.value()

    def cleanup_checkpoint_npz() -> None:
        """Clean prior NPZ snapshots once a durable checkpoint exists."""

        checkpoints = store.checkpoints(run_id)
        if not checkpoints:
            return
        latest_checkpoint_id = str(checkpoints[0]["id"])

        try:
            cleanup_previous_checkpoint_npz(
                store,
                run_id,
                latest_checkpoint_id=latest_checkpoint_id,
            )
        except (OSError, RetentionSafetyError) as error:
            # Cleanup is never a reason to lose training progress.  Stop this
            # retention pass, persist the exact failure, and wait for a future
            # durable boundary rather than retrying with broader semantics.
            store.event(
                run_id,
                "checkpoint_npz_cleanup_failed",
                "Checkpoint NPZ cleanup stopped without broadening its targets",
                {
                    "boundary_checkpoint_id": latest_checkpoint_id,
                    "error": f"{type(error).__name__}: {error}",
                },
            )

    def emit(force: bool = False, phase: str = "self_play+learning") -> None:
        nonlocal consecutive_critical_memory_samples, last_memory_pressure_event_at
        nonlocal last_metric_at, metric_seq
        now = time.monotonic()
        if not force and now - last_metric_at < config.metrics_interval_seconds:
            return
        rate_values = rate.sample(totals.games, totals.decisions)
        outcome_replay_metrics = replay.metrics()
        policy_replay_metrics = policy_replay.metrics()
        replay_metrics = (
            {
                **policy_replay_metrics,
                "policy": policy_replay_metrics,
                "outcome": outcome_replay_metrics,
            }
            if config.training_generation >= 4
            else {**outcome_replay_metrics, "policy": policy_replay_metrics}
        )
        replay_metrics["preferences"] = preference_replay.metrics()
        active_elapsed = current_active_elapsed()
        duration_seconds = config.duration_minutes * 60.0
        full_system = system_snapshot()
        pressure_release: dict[str, int | bool] = {
            "triggered": False,
            "swap_triggered": False,
            "cache_released_bytes": 0,
        }
        swap_unsafe = _swap_pressure_is_critical(full_system, memory_limits)
        if (
            full_system["memory_available_bytes"] < memory_limits["minimum_available_bytes"]
            or swap_unsafe
        ):
            # Cached Metal pages are the cheapest memory to return.  Do this at
            # the same safe learner boundary used for metrics; active arrays
            # remain valid and MLX will rebuild only what the next update uses.
            try:
                cached_before = int(mx.get_cache_memory())
                mx.clear_cache()
                cached_after = int(mx.get_cache_memory())
                pressure_release = {
                    "triggered": True,
                    "swap_triggered": swap_unsafe,
                    "cache_released_bytes": max(0, cached_before - cached_after),
                }
                full_system = system_snapshot()
                if now - last_memory_pressure_event_at >= 60.0:
                    store.event(
                        run_id,
                        "memory_pressure_relieved",
                        "Released the Metal cache to protect the desktop memory reserve",
                        {
                            **pressure_release,
                            "memory_available_bytes": full_system["memory_available_bytes"],
                            "minimum_available_bytes": memory_limits["minimum_available_bytes"],
                            "swap_used_bytes": full_system["swap_used_bytes"],
                            "swap_free_bytes": full_system["swap_free_bytes"],
                            "maximum_swap_used_bytes": memory_limits["maximum_swap_used_bytes"],
                            "minimum_swap_free_bytes": memory_limits["minimum_swap_free_bytes"],
                        },
                    )
                    last_memory_pressure_event_at = now
            except Exception:
                # Telemetry and the hard allocator limits still protect the
                # run if a future MLX build does not expose cache controls.
                pressure_release = {
                    "triggered": True,
                    "swap_triggered": swap_unsafe,
                    "cache_released_bytes": 0,
                }
        if (
            full_system["memory_available_bytes"] < memory_limits["critical_available_bytes"]
            or swap_unsafe
        ):
            consecutive_critical_memory_samples += 1
        else:
            consecutive_critical_memory_samples = 0
        if (
            consecutive_critical_memory_samples >= 3
            and not control.pause_requested.is_set()
            and not control.should_stop()
        ):
            control.pause_requested.set()
            store.event(
                run_id,
                "memory_safety_pause",
                "Automatically paused training after sustained critical memory pressure",
                {
                    "memory_available_bytes": full_system["memory_available_bytes"],
                    "critical_available_bytes": memory_limits["critical_available_bytes"],
                    "swap_used_bytes": full_system["swap_used_bytes"],
                    "swap_free_bytes": full_system["swap_free_bytes"],
                    "maximum_swap_used_bytes": memory_limits["maximum_swap_used_bytes"],
                    "minimum_swap_free_bytes": memory_limits["minimum_swap_free_bytes"],
                    "consecutive_samples": consecutive_critical_memory_samples,
                },
            )
        snapshot = {
            key: full_system[key]
            for key in (
                "cpu_percent",
                "memory_total_bytes",
                "memory_available_bytes",
                "memory_percent",
                "process_rss_bytes",
                "swap_total_bytes",
                "swap_used_bytes",
                "swap_free_bytes",
                "swap_percent",
                "swap_in_bytes",
                "swap_out_bytes",
            )
        }
        try:
            full_metal = mlx_snapshot()
            metal = {
                key: full_metal[key]
                for key in (
                    "device",
                    "metal_available",
                    "active_memory_bytes",
                    "cache_memory_bytes",
                    "peak_memory_bytes",
                )
            }
        except Exception as error:  # pragma: no cover - hardware telemetry only
            metal = {"error": f"{type(error).__name__}: {error}"}
        metric_seq += 1
        rollout_total = max(1, sum(totals.rollout_games.values()))
        rollout_mix = {key: value / rollout_total for key, value in totals.rollout_games.items()}
        rollout_interval_results = {}
        for kind, completed_games in totals.rollout_completed_games.items():
            interval_games = completed_games - last_reported_rollout_games.get(kind, 0)
            if interval_games <= 0:
                continue
            interval_score = totals.rollout_scores.get(
                kind, 0.0
            ) - last_reported_rollout_scores.get(kind, 0.0)
            rollout_interval_results[kind] = {
                "games": interval_games,
                "score": interval_score / interval_games,
            }
        uncertainty = float(last_diagnostics.get("uncertainty", 0.0))
        payload: dict[str, Any] = {
            "seq": metric_seq,
            "run_id": run_id,
            "phase": phase,
            "games": totals.games,
            "decisions": totals.decisions,
            "updates": totals.updates,
            "samples": totals.samples,
            **rate_values,
            "active_elapsed_seconds": active_elapsed,
            "eta_seconds": max(0.0, duration_seconds - active_elapsed),
            "progress": min(1.0, active_elapsed / max(1.0, duration_seconds)),
            "epsilon": _epsilon(
                config,
                max(0, totals.games - schedule_games_origin),
                float(plateau["exploration_multiplier"]),
            ),
            "epsilon_scheduled": _epsilon(config, max(0, totals.games - schedule_games_origin)),
            "schedule_games_origin": schedule_games_origin,
            "learner_games_since_reset": max(0, totals.games - schedule_games_origin),
            "training_generation": config.training_generation,
            "behavior_policy": config.behavior_policy,
            "deployment_policy_selfplay_fraction": (config.deployment_policy_selfplay_fraction),
            "deployment_policy_scheduled_fraction": (
                config.current_selfplay_fraction * config.deployment_policy_selfplay_fraction
            ),
            "deployment_policy_actor_weight": config.deployment_policy_actor_weight,
            "fixed_champion_fraction": config.fixed_champion_fraction,
            "fixed_champion_probe_fraction": config.fixed_champion_probe_fraction,
            "policy_reference_kl_weight": float(
                last_diagnostics.get(
                    "policy_reference_kl_effective_weight",
                    config.policy_reference_kl_weight,
                )
            ),
            "policy_reference_kl_initial_weight": config.policy_reference_kl_weight,
            "policy_reference_kl_target": config.policy_reference_kl_target,
            "fixed_champion_id": fixed_champion_id,
            "target_mode": (
                "legal_set_actor_critic"
                if config.training_generation >= 4
                else "mixed_bootstrap"
                if config.use_bootstrap_targets
                else "monte_carlo"
            ),
            "plateau": dict(plateau),
            "governor": dict(governor),
            "exploration_health": {
                "uncertainty": uncertainty,
                "collapse_warning": bool(
                    config.bootstrap_heads > 1 and totals.updates > 1_000 and uncertainty < 0.005
                ),
            },
            "rollout_games": dict(totals.rollout_games),
            "rollout_results": {
                kind: {
                    "games": games,
                    "score": totals.rollout_scores.get(kind, 0.0) / max(1, games),
                }
                for kind, games in totals.rollout_completed_games.items()
            },
            "rollout_interval_results": rollout_interval_results,
            "opponent_results": {
                opponent_id: {
                    "games": games,
                    "score": totals.opponent_scores.get(opponent_id, 0.0) / max(1, games),
                }
                for opponent_id, games in totals.opponent_games.items()
            },
            "rollout_mix": rollout_mix,
            "durable_resume": dict(durable_resume),
            "curriculum_phase": (
                "heuristic_bootstrap"
                if totals.updates < config.heuristic_bootstrap_updates
                else "self_play"
            ),
            "heuristic_bootstrap_updates_remaining": max(
                0, config.heuristic_bootstrap_updates - totals.updates
            ),
            "player_0_wins": totals.player_wins[0],
            "player_1_wins": totals.player_wins[1],
            "draws": totals.draws,
            "truncations": totals.truncated,
            "truncation_rate": totals.truncated / max(1, totals.games),
            "mean_turns": totals.turns / max(1, totals.games),
            "forced_choices": totals.forced_choices,
            "counterfactual_preferences": totals.counterfactual_preferences,
            "reanalysis_positions": totals.reanalysis_positions,
            "search_repeatability": {
                "positions": totals.search_repeatability_positions,
                "top_action_agreement": (
                    totals.search_top_action_agreements
                    / max(1, totals.search_repeatability_positions)
                ),
                "mean_policy_js": (
                    totals.search_policy_js_sum / max(1, totals.search_repeatability_positions)
                ),
                "mean_value_abs_delta": (
                    totals.search_value_abs_delta_sum
                    / max(1, totals.search_repeatability_positions)
                ),
            },
            "replay": replay_metrics,
            "system": snapshot,
            "metal": metal,
            "memory_safety": {
                **memory_limits,
                **pressure_release,
                "consecutive_critical_samples": consecutive_critical_memory_samples,
                "effective_policy_replay_capacity": policy_replay.capacity,
                "policy_replay_disk_capacity": policy_replay.disk_capacity,
                "policy_replay_total_capacity": policy_replay.total_capacity,
            },
            **last_diagnostics,
        }
        store.append_metric(run_id, metric_seq, payload)
        persisted_status = (
            phase
            if phase in {"pausing", "paused", "stopping"}
            else "stopping"
            if phase == "finalizing" and control.should_stop()
            else "running"
        )
        store.update_run(
            run_id,
            status=persisted_status,
            phase=phase,
            games=totals.games,
            decisions=totals.decisions,
            updates=totals.updates,
        )
        publish(payload)
        last_reported_rollout_games.update(totals.rollout_completed_games)
        last_reported_rollout_scores.update(totals.rollout_scores)
        last_metric_at = now

    def persist_checkpoint(reason: str, *, schedule_evaluation: bool) -> dict[str, Any]:
        """Write one complete learner boundary and update its parent cursor."""

        nonlocal parent_checkpoint_id, last_checkpoint_games
        emit(force=True, phase="checkpointing")
        # Checkpoint serialization temporarily duplicates replay arrays on the
        # CPU.  Free reusable GPU pages first so that durable saves cannot push
        # WindowServer into compressor exhaustion.
        mx.clear_cache()
        save_started = time.monotonic()
        checkpoint = _save_checkpoint(
            store=store,
            run_id=run_id,
            model=model,
            spec=spec,
            checkpoint_dir=checkpoint_root,
            games=totals.games,
            parent_id=parent_checkpoint_id,
            champion=False,
            reason=reason,
            optimizer=optimizer if config.persist_optimizer_state else None,
            replay=replay,
            policy_replay=policy_replay if config.training_generation >= 4 else None,
            preference_replay=preference_replay,
            resume_replay_items=config.resume_replay_items,
            full_replay=reason in {"pause", "final"} and config.training_generation >= 3,
            training_state=_training_state(
                totals,
                seed_cursor=seed_cursor,
                optimizer_updates_at_start=optimizer_updates_at_start,
                schedule_games_origin=schedule_games_origin,
                active_elapsed_seconds=current_active_elapsed(),
                rollout_rng_state=rng.bit_generator.state,
                replay_rng_state=replay.rng_state(),
                league_state=league.snapshot(),
                policy_reference_kl_controller=policy_reference_kl_controller,
            ),
        )
        parent_checkpoint_id = checkpoint["id"]
        last_checkpoint_games = totals.games
        checkpoint_artifacts = (checkpoint.get("evaluation") or {}).get("artifacts") or {}
        durable_resume.update(
            {
                "optimizer_persisted": bool(checkpoint_artifacts.get("optimizer_path")),
                "replay_items_persisted": max(
                    0,
                    int(
                        checkpoint_artifacts.get(
                            "policy_replay_items"
                            if config.training_generation >= 4
                            else "replay_items",
                            0,
                        )
                    ),
                ),
                "replay_capacity_at_snapshot": max(
                    0,
                    int(
                        checkpoint_artifacts.get(
                            "policy_replay_total_capacity"
                            if config.training_generation >= 4
                            else "replay_capacity",
                            policy_replay.total_capacity
                            if config.training_generation >= 4
                            else replay.capacity,
                        )
                    ),
                ),
                "replay_snapshot_mode": str(
                    checkpoint_artifacts.get(
                        "policy_replay_format"
                        if config.training_generation >= 4
                        else "replay_format"
                    )
                    or "none"
                ),
                "checkpoint_artifacts_complete": True,
                "latest_checkpoint_id": checkpoint["id"],
                "latest_checkpoint_games": int(checkpoint["games"]),
                "latest_checkpoint_reason": reason,
                "latest_checkpoint_save_seconds": time.monotonic() - save_started,
            }
        )
        if schedule_evaluation:
            _sync_league(
                league,
                store,
                run_id,
                include_external_anchors=config.training_generation >= 3,
            )
            maybe_schedule_evaluation(checkpoint)
        cleanup_checkpoint_npz()
        emit(
            force=True,
            phase="pausing" if control.pause_requested.is_set() else "self_play+learning",
        )
        return checkpoint

    def current_evaluation_manager() -> Any | None:
        return evaluation_manager

    def finish_exclusive_full_evaluation() -> bool:
        """Wait at an actor-batch boundary while a full promotion gate runs."""

        pending = _pending_trainer_evaluation_job(store, run_id)
        if not _is_exclusive_full_promotion_job(pending):
            return False
        manager = current_evaluation_manager()
        if manager is None:
            raise RuntimeError("a full promotion evaluation has no owning ArenaManager")

        job_id = str(pending["id"])
        active_clock.pause()
        emit(force=True, phase="promotion_evaluation")
        store.event(
            run_id,
            "exclusive_promotion_evaluation_started",
            "Paused rollout and learning for the full promotion evaluation",
            {"job_id": job_id},
        )
        try:
            while not manager.wait_for_job(job_id, timeout=0.25):
                if control.should_stop() or control.pause_requested.is_set():
                    manager.cancel(job_id)
                service_final_checkpoint()
            process_completed_evaluations()
            cleanup_checkpoint_npz()
        finally:
            if not control.should_stop() and not control.pause_requested.is_set():
                active_clock.resume()
        store.event(
            run_id,
            "exclusive_promotion_evaluation_finished",
            "Full promotion evaluation finished; rollout and learning may resume",
            {"job_id": job_id},
        )
        return True

    def service_paused_work() -> bool:
        """Honor pause/manual checkpoint requests before reporting paused."""

        if not control.consume_checkpoint():
            return False
        persist_checkpoint("pause", schedule_evaluation=False)
        return True

    def service_final_checkpoint() -> bool:
        """Persist manual requests while a finite final arena is still running."""

        if not control.consume_checkpoint():
            return False
        persist_checkpoint("manual", schedule_evaluation=False)
        emit(
            force=True,
            phase=(
                "stopping"
                if control.should_stop()
                else "pausing"
                if control.pause_requested.is_set()
                else "finalizing_evaluation"
            ),
        )
        return True

    # A run may resume just after a checkpoint was written but before its
    # evaluation was scheduled (for example after upgrading the backend).
    # Reconsider the latest immutable candidate without duplicating an
    # already-persisted trainer job, unless its evaluation budget is complete.
    if not (
        config.budget_type == "full_evaluations"
        and _completed_full_evaluation_count(store, run_id)
        >= int(config.budget_full_evaluations or 0)
    ):
        maybe_schedule_evaluation()

    # macOS uses spawn, avoiding unsafe post-Metal forks. Actors never import
    # MLX, so each process remains a small engine/NumPy worker.
    context = mp.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=config.actor_processes,
        mp_context=context,
        # A worker is cheap to hydrate from the current 8 MB actor archive.
        # Recycling provides a hard backstop for native allocator high-water
        # marks that Python garbage collection cannot return on macOS.
        max_tasks_per_child=256,
    )
    try:
        emit(force=True, phase="initializing")
        while not control.should_stop():
            if control.wait_if_paused(
                set_phase,
                service_paused_work,
                active_clock.pause,
                active_clock.resume,
            ):
                final_reason = "safe stop requested"
                break

            run = store.get_run(run_id)
            config = RunConfig.model_validate_persisted(run["config"])

            # A rejected learner never becomes the behavior policy. Invalid
            # arenas receive retryable dispositions rather than looking like
            # genuine skill regressions.
            evaluation_boundary_checkpoint_id = process_completed_evaluations()
            active_elapsed = current_active_elapsed()
            budget_reached, budget_reason = _training_budget_reached(
                config,
                active_elapsed=active_elapsed,
                games=totals.games,
                full_evaluations=_completed_full_evaluation_count(store, run_id),
            )
            if budget_reached:
                final_reason = budget_reason
                break
            # The arena deliberately has no stale-job queue. Rechecking once
            # per iteration releases the newest due checkpoint after either an
            # automatic promotion job or a diagnostic trainer job finishes.
            maybe_schedule_evaluation()
            # This point follows a completely drained rollout batch. Full
            # promotion gates run exclusively so their immutable comparison
            # finishes quickly and before the learner advances further.
            if finish_exclusive_full_evaluation():
                continue
            if evaluation_boundary_checkpoint_id is not None:
                cleanup_checkpoint_npz()
            plateau = _plateau_status(store, run_id, config)
            governor = _governor_status(
                store,
                run_id,
                config,
                games=totals.games,
                diagnostics=last_diagnostics,
                plateau=plateau,
            )
            effective_config = config.model_copy(
                update={
                    "policy_entropy_weight": float(
                        governor.get("entropy_weight", config.policy_entropy_weight)
                    ),
                    "reanalysis_fraction": min(
                        1.0,
                        config.reanalysis_fraction
                        * float(governor.get("reanalysis_multiplier", 1.0)),
                    ),
                }
            )
            _sync_league(
                league,
                store,
                run_id,
                include_external_anchors=config.training_generation >= 3,
            )
            if config.behavior_policy == "learner":
                _atomic_actor_export(model, spec, runtime_actor)
                rollout_actor = str(runtime_actor)
            else:
                rollout_actor = _champion_actor_path(store, run_id, runtime_actor)
                if rollout_actor == str(runtime_actor) and not runtime_actor.is_file():
                    _atomic_actor_export(model, spec, runtime_actor)
            epsilon = _epsilon(
                effective_config,
                max(0, totals.games - schedule_games_origin),
                float(plateau["exploration_multiplier"]),
            )
            futures: dict[Future[WorkerResult], _RolloutPlan] = {}
            # More tasks than workers turns the process pool into a small
            # work-stealing queue. Long games/searches no longer strand CPU
            # cores while the learner waits for one oversized actor batch.
            for _ in range(config.actor_processes * config.rollout_tasks_per_actor):
                active_replay_size = (
                    len(policy_replay) if config.training_generation >= 4 else len(replay)
                )
                needs_labeled_warmup = (
                    config.training_generation < 4 and active_replay_size < config.replay_warmup
                )
                if needs_labeled_warmup or totals.updates < config.heuristic_bootstrap_updates:
                    plan = _make_bootstrap_plan(config=effective_config, rng=rng, seed=seed_cursor)
                else:
                    plan = _make_plan(
                        config=effective_config,
                        rng=rng,
                        league=league,
                        current_actor=rollout_actor,
                        fixed_champion_id=fixed_champion_id,
                        fixed_champion_actor=fixed_champion_actor,
                        epsilon=epsilon,
                        seed=seed_cursor,
                    )
                seed_cursor += plan.games + 1_009
                futures[_submit_rollout(executor, plan, effective_config)] = plan

            store.update_run(run_id, phase="self_play+learning")
            last_diagnostics = _train_updates(
                model=model,
                optimizer=optimizer,
                replay=replay,
                policy_replay=policy_replay,
                preference_replay=preference_replay,
                config=effective_config,
                count=max(
                    1,
                    int(
                        round(
                            config.updates_per_iteration
                            * float(governor.get("updates_multiplier", 1.0))
                        )
                    ),
                ),
                totals=totals,
                optimizer_updates_at_start=optimizer_updates_at_start,
                control=control,
                reference_model=reference_model,
                fixed_champion_key=policy_opponent_key(fixed_champion_id),
                learning_rate_multiplier=float(governor.get("learning_rate_multiplier", 1.0)),
                promotion_direction=promotion_direction,
                policy_reference_kl_controller=policy_reference_kl_controller,
            )
            emit(
                phase=(
                    "stopping"
                    if control.should_stop()
                    else "pausing"
                    if control.pause_requested.is_set()
                    else "self_play+learning"
                )
            )

            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for future in done:
                    if future.cancelled():
                        continue
                    plan = futures.pop(future)
                    result = future.result()
                    replay.extend_compact(result.samples)
                    policy_replay.extend_compact(
                        result.policy_samples,
                        rollout_source=plan.kind,
                        opponent_key=policy_opponent_key(plan.opponent_id),
                        collected_at_game=totals.games + result.games,
                    )
                    preference_replay.extend_compact(result.preferences)
                    totals.games += result.games
                    totals.decisions += result.decisions
                    totals.samples += (
                        len(result.policy_samples)
                        if config.training_generation >= 4
                        else len(result.samples)
                    )
                    totals.player_wins = (
                        totals.player_wins[0] + result.wins[0],
                        totals.player_wins[1] + result.wins[1],
                    )
                    totals.draws += result.draws
                    totals.truncated += result.truncated
                    totals.turns += result.turns
                    totals.forced_choices += result.forced_choices
                    totals.counterfactual_preferences += result.counterfactual_preferences
                    totals.reanalysis_positions += result.reanalysis_positions
                    totals.search_repeatability_positions += result.search_repeatability_positions
                    totals.search_top_action_agreements += result.search_top_action_agreements
                    totals.search_policy_js_sum += result.search_policy_js_sum
                    totals.search_value_abs_delta_sum += result.search_value_abs_delta_sum
                    totals.rollout_games[plan.kind] = (
                        totals.rollout_games.get(plan.kind, 0) + result.games
                    )
                    _record_opponent_result(totals, league, plan, result)
                boundary_phase = (
                    "stopping"
                    if control.should_stop()
                    else "pausing"
                    if control.pause_requested.is_set()
                    else "self_play+learning"
                )
                emit(phase=boundary_phase)

            checkpoint_due = totals.games - last_checkpoint_games >= config.checkpoint_every_games
            requested_checkpoint = control.consume_checkpoint()
            if checkpoint_due or requested_checkpoint:
                pausing = control.pause_requested.is_set()
                reason = (
                    "pause"
                    if pausing
                    else "final"
                    if control.should_stop()
                    else "scheduled"
                    if checkpoint_due
                    else "manual"
                )
                persist_checkpoint(
                    reason,
                    schedule_evaluation=not pausing and not control.should_stop(),
                )

            if control.should_stop():
                final_reason = "safe stop requested"
                break

        if totals.games > last_checkpoint_games or control.checkpoint_due():
            emit(force=True, phase="checkpointing")
            mx.clear_cache()
            checkpoint = _save_checkpoint(
                store=store,
                run_id=run_id,
                model=model,
                spec=spec,
                checkpoint_dir=checkpoint_root,
                games=totals.games,
                parent_id=parent_checkpoint_id,
                champion=False,
                reason="final",
                optimizer=optimizer if config.persist_optimizer_state else None,
                replay=replay,
                policy_replay=policy_replay if config.training_generation >= 4 else None,
                preference_replay=preference_replay,
                resume_replay_items=config.resume_replay_items,
                full_replay=control.should_stop() and config.training_generation >= 3,
                training_state=_training_state(
                    totals,
                    seed_cursor=seed_cursor,
                    optimizer_updates_at_start=optimizer_updates_at_start,
                    schedule_games_origin=schedule_games_origin,
                    active_elapsed_seconds=current_active_elapsed(),
                    rollout_rng_state=rng.bit_generator.state,
                    replay_rng_state=replay.rng_state(),
                    league_state=league.snapshot(),
                    policy_reference_kl_controller=policy_reference_kl_controller,
                ),
            )
            parent_checkpoint_id = checkpoint["id"]
            if not control.should_stop():
                maybe_schedule_evaluation(checkpoint)
            cleanup_checkpoint_npz()
        final_evaluation_done = False
        while not control.should_stop() and not final_evaluation_done:
            if control.pause_requested.is_set():
                if control.wait_if_paused(
                    set_phase,
                    service_paused_work,
                    active_clock.pause,
                    active_clock.resume,
                ):
                    final_reason = "safe stop requested"
                    break
                continue
            # No learner mutation occurs during final evaluation, so a requested
            # snapshot is immediately safe and becomes the immutable candidate
            # considered below.
            service_final_checkpoint()

            # Natural completion is an evaluation boundary, not merely a
            # learner boundary. Drain only this run's trainer comparison, then
            # re-resolve the champion and evaluate the newest due snapshot.
            emit(force=True, phase="finalizing_evaluation")
            store.event(
                run_id,
                "final_evaluation_started",
                "Draining trainer arenas and checking the newest checkpoint",
            )
            if evaluation_manager is None and _pending_trainer_evaluation(store, run_id):
                from .arena import ArenaManager

                evaluation_manager = ArenaManager(
                    store,
                    maximum_concurrent_jobs=1,
                    recover=True,
                )
            final_evaluation = _finish_final_evaluations(
                store=store,
                run_id=run_id,
                manager_getter=current_evaluation_manager,
                schedule_latest=lambda: (
                    None
                    if control.should_stop()
                    or control.pause_requested.is_set()
                    or (
                        config.budget_type == "full_evaluations"
                        and _completed_full_evaluation_count(store, run_id)
                        >= int(config.budget_full_evaluations or 0)
                    )
                    else maybe_schedule_evaluation(
                        ignore_retry_backoff=True,
                        force_full=True,
                    )
                ),
                process_completed=lambda: process_completed_evaluations(),
                interrupt_reason=lambda: (
                    "stop_requested"
                    if control.should_stop()
                    else "pause_requested"
                    if control.pause_requested.is_set()
                    else None
                ),
                service_checkpoint=service_final_checkpoint,
            )
            interrupted = final_evaluation.get("status") == "interrupted"
            interrupt_reason = final_evaluation.get("interrupt_reason")
            store.event(
                run_id,
                "final_evaluation_interrupted" if interrupted else "final_evaluation_finished",
                (
                    "Final trainer evaluation stopped at a safe game boundary"
                    if interrupt_reason == "stop_requested"
                    else "Final trainer evaluation paused at a safe game boundary"
                    if interrupt_reason == "pause_requested"
                    else "Final trainer evaluation lifecycle finished"
                ),
                final_evaluation,
            )
            if interrupted:
                if control.should_stop():
                    final_reason = "safe stop requested"
                    break
                continue
            # Close the small gap between the helper's last poll and its return.
            # If a manual request arrived there, persist it and run the final
            # evaluation lifecycle once more for the new immutable checkpoint.
            if service_final_checkpoint():
                continue
            final_evaluation_done = True
        emit(force=True, phase="finalizing")
        store.event(
            run_id,
            "training_loop_finished",
            f"Training loop finished: {final_reason}",
            {"games": totals.games, "decisions": totals.decisions, "updates": totals.updates},
        )
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        mx.clear_cache()


__all__ = ["run_training"]
