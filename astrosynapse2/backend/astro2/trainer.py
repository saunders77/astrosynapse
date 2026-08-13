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

from .config import RunConfig
from .encoding import FAMILY_COUNT, Encoder
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
    GameBalancedPolicyReplayBuffer,
    PreferenceReplayBuffer,
    ReplayBuffer,
)
from .retention import RetentionSafetyError, prune_checkpoint_artifacts
from .selfplay import ActorPolicy, WorkerResult, collect_worker_batch
from .storage import Store

_TRAINING_STATE_SCHEMA_VERSION = 2
_EVALUATION_RETRY_BASE_SECONDS = 30.0
_EVALUATION_RETRY_MAX_SECONDS = 15.0 * 60.0
_EVALUATION_RETRY_SEED_STRIDE = 1_000_003
_FINAL_EVALUATION_MAX_ATTEMPTS = 3
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
    rollout_games: dict[str, int] = field(default_factory=dict)


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

    full_automatic = config.evaluation_pairs >= 5_000
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
            pairs=min(config.evaluation_pairs, max(200, config.evaluation_pairs // 25)),
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


def _plateau_status(store: Store, run_id: str, config: RunConfig) -> dict[str, Any]:
    """Summarize recent promotion evidence and choose a bounded response."""

    consecutive = 0
    for job in reversed(_completed_trainer_evaluations(store, run_id)):
        promotion = (job.get("result") or {}).get("promotion") or {}
        if bool(promotion.get("promoted")):
            break
        consecutive += 1
    level = consecutive // config.plateau_patience_evaluations if config.adaptive_training else 0
    multiplier = min(
        config.plateau_max_exploration_multiplier,
        2.0**level,
    )
    return {
        "consecutive_non_promotions": consecutive,
        "level": level,
        "exploration_multiplier": multiplier,
        "active": bool(config.adaptive_training and level > 0),
    }


def _pending_trainer_evaluation_job(store: Store, run_id: str) -> dict[str, Any] | None:
    checkpoint_ids = {checkpoint["id"] for checkpoint in store.checkpoints(run_id)}
    return next(
        (
            job
            for job in store.arena_jobs(limit=20_000, include_internal=True)
            if job["status"] in {"queued", "running"}
            and job["model_a"] in checkpoint_ids
            and bool((job.get("config") or {}).get("trainer_scheduled"))
        ),
        None,
    )


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
        "output.weight": (spec.families * spec.bootstrap_heads, hidden),
        "output.bias": (spec.families * spec.bootstrap_heads,),
    }

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
    if config.persist_optimizer_state and not _readable_npz(
        artifacts.get("optimizer_path"),
        required={"__paths_json__"},
    ):
        return False
    if config.resume_replay_items <= 0:
        return True

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
    model_candidates: list[dict[str, Any]] = []
    skipped: list[str] = []
    for checkpoint in store.checkpoints(run_id):
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


def _completed_trainer_evaluations(store: Store, run_id: str) -> list[dict[str, Any]]:
    """Return only completed arenas that are valid evidence of playing strength."""

    return [
        job
        for job in _terminal_trainer_evaluations(store, run_id)
        if bool((job.get("config") or {}).get("automatic_promotion"))
        and _trainer_evaluation_outcome(job) in {"promoted", "not_promoted"}
    ]


def _is_trainer_evaluation(job: dict[str, Any]) -> bool:
    config = job.get("config") or {}
    # Older automatic jobs predate the explicit trainer_scheduled marker. No
    # public/manual path can create an automatic-promotion job, so retaining
    # them here preserves restart compatibility.
    return bool(config.get("trainer_scheduled") or config.get("automatic_promotion"))


def _trainer_evaluation_outcome(job: dict[str, Any]) -> str:
    """Classify whether an arena is skill evidence or retryable infrastructure.

    A completed SQLite row is not automatically a valid comparison. A truncated
    arena is valid only when it promoted after conservatively scoring every
    truncation as a candidate loss. Other truncations, a stale champion
    opponent, or a structurally incomplete result must be retried and must not
    look like a model regression.
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
    if int(result.get("truncated_games", 0) or 0) > 0 and not bool(promotion.get("promoted")):
        return "truncated"
    if bool(promotion.get("stale_opponent") or result.get("stale_opponent")):
        return "stale"

    config = job.get("config") or {}
    if not bool(config.get("automatic_promotion")):
        # Diagnostic jobs have no promotion payload by design. Reaching the
        # complete state without truncation is their validity contract.
        return "diagnostic_complete"

    if bool(result.get("early_stopped")):
        latest_look = (result.get("early_rejection") or {}).get("latest_look") or {}
        if bool(latest_look.get("reject")) or result.get("early_stop_reason"):
            return "not_promoted"
        return "infrastructure_invalid"
    if "promoted" not in promotion:
        return "infrastructure_invalid"
    if promotion.get("eligible") is False:
        return "infrastructure_invalid"
    return "promoted" if bool(promotion.get("promoted")) else "not_promoted"


def _terminal_trainer_evaluations(store: Store, run_id: str) -> list[dict[str, Any]]:
    checkpoint_ids = {checkpoint["id"] for checkpoint in store.checkpoints(run_id)}
    return sorted(
        (
            job
            for job in store.arena_jobs(limit=20_000, include_internal=True)
            if job["status"] in {"complete", "failed", "cancelled"}
            and job["model_a"] in checkpoint_ids
            and _is_trainer_evaluation(job)
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
    artifacts: dict[str, Any] = {"schema_version": 1}
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
            or checkpoint["id"] in existing
        ):
            continue
        league.upsert(
            Opponent(
                id=checkpoint["id"],
                actor_path=actor_path,
                kind="champion" if checkpoint["is_champion"] else "checkpoint",
                label=checkpoint["label"],
            )
        )

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
            for job in store.arena_jobs(limit=20_000, include_internal=True)
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
    for job in store.arena_jobs(limit=20_000, include_internal=True):
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
    plan = _evaluation_plan(config, checkpoint_games)
    last_evaluation_games = _last_scheduled_evaluation_games(
        store,
        run_id,
        tier=plan.tier,
    )
    if checkpoint_games - last_evaluation_games < plan.cadence_games:
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
        rollout_games={
            str(key): max(0, int(value))
            for key, value in (latest.get("rollout_games") or {}).items()
        },
    )


def _training_state(
    totals: _Totals,
    *,
    seed_cursor: int,
    optimizer_updates_at_start: int,
    active_elapsed_seconds: float | None = None,
    rollout_rng_state: dict[str, Any] | None = None,
    replay_rng_state: dict[str, Any] | None = None,
    league_state: list[dict[str, object]] | None = None,
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
        "rollout_games": dict(totals.rollout_games),
        "seed_cursor": int(seed_cursor),
        "optimizer_updates_at_start": max(0, int(optimizer_updates_at_start)),
    }
    if active_elapsed_seconds is not None:
        state["active_elapsed_seconds"] = max(0.0, float(active_elapsed_seconds))
    if rollout_rng_state is not None:
        state["rollout_rng_state"] = rollout_rng_state
    if replay_rng_state is not None:
        state["replay_rng_state"] = replay_rng_state
    if league_state is not None:
        state["league_state"] = league_state
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
        early_rejection = bool(
            config.evaluation_early_rejection
            and config.evaluation_early_rejection_min_pairs < plan.pairs
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
    """Run deterministic tactical, held-out, and fixed-opponent checks."""

    existing = (checkpoint.get("evaluation") or {}).get("quality_gate")
    if existing:
        return existing
    from .diagnostics import checkpoint_diagnostics

    diagnostics = checkpoint_diagnostics(
        checkpoint["actor_path"],
        seed=config.seed,
        games=config.checkpoint_diagnostic_games,
        baseline_pairs=config.checkpoint_baseline_pairs,
    )
    tactical = diagnostics["tactical"]
    strategic = diagnostics.get("strategic") or {}
    resource_efficiency = diagnostics.get("resource_efficiency") or {}
    ensemble = diagnostics.get("ensemble") or {}
    heldout = diagnostics["heldout"]
    baselines = diagnostics["baselines"]
    champion_id = store.get_run(run_id).get("champion_id")
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
    tactical_metric = (
        "raw_end_turn_violations"
        if config.gate_raw_tactical_preferences
        else "masked_end_turn_violations"
    )
    if int(tactical[tactical_metric]) > config.max_tactical_violations:
        reasons.append(
            "raw model logits failed the tactical dominance suite"
            if config.gate_raw_tactical_preferences
            else "deployed action policy failed the tactical dominance suite"
        )
    if config.require_early_high_cost_retention and not bool(
        strategic.get("early_high_cost_passed")
    ):
        reasons.append("model preferred premature high-cost scraps over retaining the card")
    if config.require_resource_efficiency and not bool(resource_efficiency.get("passed")):
        reasons.append("model ended early turns while useful trade could be spent")
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
        "tactical_gate_metric": tactical_metric,
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

    checkpoint_opponents = [
        item
        for item in league.opponents
        if item.kind in {"checkpoint", "champion", "anchor"} and item.actor_path
    ]
    league_cutoff = config.current_selfplay_fraction + config.league_fraction
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
        counterfactual_fraction=(
            config.counterfactual_fraction if config.training_generation >= 4 else 0.0
        ),
        counterfactual_max_per_game=(
            config.counterfactual_max_per_game if config.training_generation >= 4 else 0
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
) -> dict[str, float]:
    active_replay_size = len(policy_replay) if config.training_generation >= 4 else len(replay)
    if count <= 0 or active_replay_size < config.replay_warmup:
        return {}

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

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
            preference_states,
            preferred_actions,
            disfavored_actions,
            preference_families,
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
            )[0]
            counterfactual = preference_ranking_loss(
                model,
                preference_states,
                preferred_actions,
                disfavored_actions,
                preference_families,
                margin=config.preference_margin,
            )[0]
            return policy + preference_weight * counterfactual

        loss_and_grad = nn.value_and_grad(model, policy_loss_function)
        last_policy_arrays: tuple[Any, ...] | None = None
        last_preference_arrays: tuple[Any, ...] | None = None
        loss_value = gradient_norm = learning_rate = 0.0
        completed = 0
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
            )
            if len(preference_replay):
                preference_batch = preference_replay.sample(config.preference_batch_size)
                preference_arrays = (
                    mx.array(preference_batch.states),
                    mx.array(preference_batch.preferred_actions),
                    mx.array(preference_batch.disfavored_actions),
                    mx.array(preference_batch.families),
                    mx.array(config.counterfactual_loss_weight),
                )
            else:
                preference_arrays = (
                    arrays[0][:1],
                    arrays[1][:1, 0],
                    arrays[1][:1, 0],
                    arrays[4][:1],
                    mx.array(0.0),
                )
            learning_rate = _learning_rate(
                config, totals.updates, totals.updates - optimizer_updates_at_start
            )
            optimizer.learning_rate = learning_rate
            loss, gradients = loss_and_grad(*arrays, *preference_arrays)
            gradients, norm = optim.clip_grad_norm(gradients, config.gradient_clip)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss, norm)
            loss_value = float(loss.item())
            gradient_norm = float(norm.item())
            totals.updates += 1
            completed += 1
            last_policy_arrays = arrays
            last_preference_arrays = preference_arrays
        metrics: dict[str, float] = {
            "loss": loss_value,
            "gradient_norm": gradient_norm,
            "learning_rate": learning_rate,
            "learner_updates": float(completed),
        }
        if last_policy_arrays is not None:
            diagnostic_loss, diagnostics = actor_critic_policy_loss(
                model,
                *last_policy_arrays,
                value_loss_weight=config.policy_value_loss_weight,
                entropy_weight=config.policy_entropy_weight,
                importance_clip=config.policy_importance_clip,
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
        learning_rate = _learning_rate(
            config,
            totals.updates,
            totals.updates - optimizer_updates_at_start,
        )
        optimizer.learning_rate = learning_rate
        loss, gradients = loss_and_grad(*arrays, *preference_arrays)
        gradients, norm = optim.clip_grad_norm(gradients, config.gradient_clip)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss, norm)
        loss_value = float(loss.item())
        gradient_norm = float(norm.item())
        totals.updates += 1
        completed += 1
        last_arrays = arrays
        last_preference_arrays = preference_arrays

    metrics: dict[str, float] = {
        "loss": loss_value,
        "gradient_norm": gradient_norm,
        "learning_rate": learning_rate,
        "learner_updates": float(completed),
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


def _configure_mlx(device: str) -> dict[str, Any]:
    import mlx.core as mx

    if device == "cpu":
        mx.set_default_device(mx.cpu)
    elif device == "gpu":
        if not mx.metal.is_available():
            raise RuntimeError("the requested MLX/Metal GPU is unavailable")
        mx.set_default_device(mx.gpu)
    elif mx.metal.is_available():
        mx.set_default_device(mx.gpu)
    return mlx_snapshot()


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
    config = RunConfig.model_validate(run["config"])
    hardware = _configure_mlx(config.device)
    store.event(
        run_id,
        "training_hardware",
        "Initialized MLX learner and hardware",
        {"mlx": hardware, "system": system_snapshot()},
    )

    import mlx.core as mx
    import mlx.optimizers as optim

    encoder = Encoder(version=2 if config.training_generation >= 3 else 1)
    # Keep artifacts beside the configured SQLite store so ASTRO2_DATA_DIR and
    # the CLI's --data-dir remain self-contained.
    checkpoint_root = store.path.parent / "checkpoints" / run_id
    runtime_actor = checkpoint_root / "runtime" / "current.actor.npz"
    latest = _learner_resume_checkpoint(store, run_id, config)
    if latest is not None:
        model, spec = load_model(latest["path"])
        parent_checkpoint_id = latest["id"]
    else:
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
    preference_replay = PreferenceReplayBuffer(
        capacity=config.preference_replay_capacity,
        state_size=encoder.state_size,
        action_size=encoder.action_size,
        seed=config.seed + 43,
    )
    policy_replay = GameBalancedPolicyReplayBuffer(
        capacity=config.policy_replay_capacity,
        state_size=encoder.state_size,
        action_size=encoder.action_size,
        bootstrap_heads=config.bootstrap_heads,
        seed=config.seed + 47,
    )
    totals = _restore_totals(store, run, latest)
    restored_training_state = _checkpoint_training_state(latest)
    artifacts = ((latest or {}).get("evaluation") or {}).get("artifacts") or {}
    try:
        replay_items_persisted = max(0, int(artifacts.get("replay_items", 0)))
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
        "replay_rng_restored": False,
        "rollout_rng_restored": False,
        "league_opponents_restored": 0,
        "optimizer_persisted": bool(artifacts.get("optimizer_path")),
        "replay_items_persisted": replay_items_persisted,
        "replay_capacity_at_snapshot": max(
            0,
            int(artifacts.get("replay_capacity") or config.replay_capacity),
        ),
        "replay_snapshot_mode": str(artifacts.get("replay_format") or "none"),
        "latest_checkpoint_id": (latest or {}).get("id"),
        "latest_checkpoint_games": int((latest or {}).get("games", 0)),
        "latest_checkpoint_reason": str(
            ((latest or {}).get("evaluation") or {}).get("reason") or ""
        ),
        "checkpoint_artifacts_complete": artifacts_complete,
        "fallback_checkpoint_ids": list((latest or {}).get("_resume_skipped_checkpoint_ids", [])),
        "degraded_reasons": [],
    }
    if latest is not None and artifacts_complete:
        try:
            if config.resume_replay_items and artifacts.get("replay_path"):
                durable_resume["replay_items_restored"] = replay.restore(artifacts["replay_path"])
        except (OSError, ValueError, KeyError) as error:
            replay.clear()
            artifacts_complete = False
            durable_resume["checkpoint_artifacts_complete"] = False
            durable_resume["degraded_reasons"].append(
                f"replay restore failed: {type(error).__name__}"
            )
        try:
            if (
                artifacts_complete
                and config.persist_optimizer_state
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
            resume_replay_items=config.resume_replay_items,
            training_state=_training_state(
                totals,
                seed_cursor=config.seed,
                optimizer_updates_at_start=optimizer_updates_at_start,
                active_elapsed_seconds=0.0,
            ),
        )
        parent_checkpoint_id = latest["id"]

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
    last_diagnostics: dict[str, float] = {}
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
    final_reason = "duration complete"

    def restore_champion(
        champion_id: str,
        *,
        rejected_checkpoint_id: str,
        source: str,
        detail: dict[str, Any] | None = None,
    ) -> bool:
        nonlocal model, optimizer, optimizer_updates_at_start, parent_checkpoint_id
        if not config.rollback_rejected_candidates:
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
        optimizer_updates_at_start = totals.updates
        parent_checkpoint_id = champion["id"]
        payload = {
            "source": source,
            "rejected_checkpoint_id": rejected_checkpoint_id,
            "champion_id": champion["id"],
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
            if not config.rollback_rejected_candidates:
                _mark_evaluation_disposition(store, job, "rollback_disabled")
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
    ) -> dict[str, Any] | None:
        nonlocal evaluation_manager
        due = _next_evaluation_candidate(
            store,
            run_id,
            config,
            checkpoint,
            ignore_retry_backoff=ignore_retry_backoff,
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

    def apply_artifact_retention(boundary_checkpoint_id: str | None) -> None:
        """Apply configured retention once, without risking the training loop."""

        try:
            prune_checkpoint_artifacts(
                store,
                run_id,
                keep_checkpoints=config.keep_checkpoints,
                boundary_checkpoint_id=boundary_checkpoint_id,
            )
        except (OSError, RetentionSafetyError) as error:
            # Cleanup is never a reason to lose training progress.  Stop this
            # retention pass, persist the exact failure, and wait for a future
            # durable boundary rather than retrying with broader semantics.
            store.event(
                run_id,
                "checkpoint_retention_failed",
                "Checkpoint retention stopped without broadening its targets",
                {
                    "boundary_checkpoint_id": boundary_checkpoint_id,
                    "error": f"{type(error).__name__}: {error}",
                },
            )

    def emit(force: bool = False, phase: str = "self_play+learning") -> None:
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
        snapshot = {
            key: full_system[key]
            for key in (
                "cpu_percent",
                "memory_total_bytes",
                "memory_available_bytes",
                "memory_percent",
                "process_rss_bytes",
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
                totals.games,
                float(plateau["exploration_multiplier"]),
            ),
            "epsilon_scheduled": _epsilon(config, totals.games),
            "training_generation": config.training_generation,
            "behavior_policy": config.behavior_policy,
            "deployment_policy_selfplay_fraction": (config.deployment_policy_selfplay_fraction),
            "deployment_policy_scheduled_fraction": (
                config.current_selfplay_fraction * config.deployment_policy_selfplay_fraction
            ),
            "target_mode": (
                "legal_set_actor_critic"
                if config.training_generation >= 4
                else "mixed_bootstrap"
                if config.use_bootstrap_targets
                else "monte_carlo"
            ),
            "plateau": dict(plateau),
            "exploration_health": {
                "uncertainty": uncertainty,
                "collapse_warning": bool(
                    config.bootstrap_heads > 1 and totals.updates > 1_000 and uncertainty < 0.005
                ),
            },
            "rollout_games": dict(totals.rollout_games),
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
            "replay": replay_metrics,
            "system": snapshot,
            "metal": metal,
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
        last_metric_at = now

    def persist_checkpoint(reason: str, *, schedule_evaluation: bool) -> dict[str, Any]:
        """Write one complete learner boundary and update its parent cursor."""

        nonlocal parent_checkpoint_id, last_checkpoint_games
        emit(force=True, phase="checkpointing")
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
            resume_replay_items=config.resume_replay_items,
            full_replay=reason in {"pause", "final"} and config.training_generation >= 3,
            training_state=_training_state(
                totals,
                seed_cursor=seed_cursor,
                optimizer_updates_at_start=optimizer_updates_at_start,
                active_elapsed_seconds=current_active_elapsed(),
                rollout_rng_state=rng.bit_generator.state,
                replay_rng_state=replay.rng_state(),
                league_state=league.snapshot(),
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
                    int(checkpoint_artifacts.get("replay_items", 0)),
                ),
                "replay_capacity_at_snapshot": max(
                    0,
                    int(checkpoint_artifacts.get("replay_capacity", replay.capacity)),
                ),
                "replay_snapshot_mode": str(checkpoint_artifacts.get("replay_format") or "none"),
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
        apply_artifact_retention(checkpoint["id"])
        emit(
            force=True,
            phase="pausing" if control.pause_requested.is_set() else "self_play+learning",
        )
        return checkpoint

    def current_evaluation_manager() -> Any | None:
        return evaluation_manager

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
    # already-persisted trainer job.
    maybe_schedule_evaluation()

    # macOS uses spawn, avoiding unsafe post-Metal forks. Actors never import
    # MLX, so each process remains a small engine/NumPy worker.
    context = mp.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=config.actor_processes, mp_context=context)
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
            config = RunConfig.model_validate(run["config"])

            # A rejected learner never becomes the behavior policy. Invalid
            # arenas receive retryable dispositions rather than looking like
            # genuine skill regressions.
            evaluation_boundary_checkpoint_id = process_completed_evaluations()
            # The arena deliberately has no stale-job queue. Rechecking once
            # per iteration releases the newest due checkpoint after either an
            # automatic promotion job or a diagnostic trainer job finishes.
            maybe_schedule_evaluation()
            if evaluation_boundary_checkpoint_id is not None:
                apply_artifact_retention(evaluation_boundary_checkpoint_id)
            plateau = _plateau_status(store, run_id, config)
            active_elapsed = current_active_elapsed()
            duration_seconds = config.duration_minutes * 60.0
            if active_elapsed >= duration_seconds:
                break

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
                config,
                totals.games,
                float(plateau["exploration_multiplier"]),
            )
            futures: dict[Future[WorkerResult], _RolloutPlan] = {}
            for _ in range(config.actor_processes):
                if (
                    len(policy_replay) if config.training_generation >= 4 else len(replay)
                ) < config.replay_warmup or totals.updates < config.heuristic_bootstrap_updates:
                    plan = _make_bootstrap_plan(config=config, rng=rng, seed=seed_cursor)
                else:
                    plan = _make_plan(
                        config=config,
                        rng=rng,
                        league=league,
                        current_actor=rollout_actor,
                        epsilon=epsilon,
                        seed=seed_cursor,
                    )
                seed_cursor += plan.games + 1_009
                futures[_submit_rollout(executor, plan, config)] = plan

            store.update_run(run_id, phase="self_play+learning")
            last_diagnostics = _train_updates(
                model=model,
                optimizer=optimizer,
                replay=replay,
                policy_replay=policy_replay,
                preference_replay=preference_replay,
                config=config,
                count=config.updates_per_iteration,
                totals=totals,
                optimizer_updates_at_start=optimizer_updates_at_start,
                control=control,
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
                    policy_replay.extend_compact(result.policy_samples)
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
                    totals.rollout_games[plan.kind] = (
                        totals.rollout_games.get(plan.kind, 0) + result.games
                    )
                    if plan.opponent_id is not None and plan.current_player is not None:
                        completed_games = result.games - result.truncated
                        score = result.wins[plan.current_player] + 0.5 * result.draws
                        if completed_games > 0:
                            league.record(plan.opponent_id, score, completed_games)
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
                resume_replay_items=config.resume_replay_items,
                full_replay=control.should_stop() and config.training_generation >= 3,
                training_state=_training_state(
                    totals,
                    seed_cursor=seed_cursor,
                    optimizer_updates_at_start=optimizer_updates_at_start,
                    active_elapsed_seconds=current_active_elapsed(),
                    rollout_rng_state=rng.bit_generator.state,
                    replay_rng_state=replay.rng_state(),
                    league_state=league.snapshot(),
                ),
            )
            parent_checkpoint_id = checkpoint["id"]
            if not control.should_stop():
                maybe_schedule_evaluation(checkpoint)
            apply_artifact_retention(checkpoint["id"])
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
                    if control.should_stop() or control.pause_requested.is_set()
                    else maybe_schedule_evaluation(ignore_retry_backoff=True)
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
