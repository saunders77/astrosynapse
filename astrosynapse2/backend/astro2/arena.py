"""Persistent, bounded, paired arena evaluation jobs.

Every seed is played twice with exact seat reversal.  This removes most first-
player and deal noise. Short diagnostics can run beside training on the 16 GB
M4 target; full promotion jobs exclusively use the evaluator pool. Jobs write
progress to SQLite, so a browser can disconnect and reconnect without owning
the evaluator process.
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from .baselines import BASELINE_NAMES, make_baseline
from .config import MINIMUM_PROMOTION_PAIRS
from .encoding import DecisionFamily, Encoder
from .engine import (
    Action,
    Decision,
    Game,
    GameConfig,
    GameResult,
    Seating,
    model_action_indices,
)
from .engine_encoding import EngineEncoder
from .model import NumpyActor
from .stats import elo_delta, wilson_interval
from .storage import Store

RECOMMENDED_PAIRS = 2_000
MAX_PAIRS = 2_000
MAX_AUTOMATIC_PAIRS = 4_000
PROMOTION_EXTENSION_PAIRS = 2_000
PROMOTION_EXTENSION_LOWER_MIN = 0.48
PROMOTION_EXTENSION_LOWER_MAX = 0.50


class ModelResolutionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    pairs: int = RECOMMENDED_PAIRS
    seed: int = 20260807
    max_turns: int = 180
    max_actions_per_turn: int = 160
    confidence: float = 0.95
    minimum_promotion_pairs: int = RECOMMENDED_PAIRS
    promotion_margin: float = 0.0
    promotion_tier: str = "full"
    automatic_promotion: bool = False
    trainer_scheduled: bool = False
    early_rejection: bool = False
    early_rejection_min_pairs: int = 512
    early_rejection_confidence: float = 0.995
    early_acceptance: bool = False
    early_acceptance_min_pairs: int = MINIMUM_PROMOTION_PAIRS
    early_acceptance_confidence: float = 0.995

    def __post_init__(self) -> None:
        pair_limit = (
            MAX_AUTOMATIC_PAIRS
            if self.automatic_promotion and self.trainer_scheduled
            else MAX_PAIRS
        )
        if not 1 <= self.pairs <= pair_limit:
            raise ValueError(f"pairs must be between 1 and {pair_limit:,}")
        if not 20 <= self.max_turns <= 500:
            raise ValueError("max_turns must be between 20 and 500")
        if not 20 <= self.max_actions_per_turn <= 500:
            raise ValueError("max_actions_per_turn must be between 20 and 500")
        if not 0.80 <= self.confidence <= 0.999:
            raise ValueError("confidence must be between 0.80 and 0.999")
        if not 0.0 <= self.promotion_margin <= 0.25:
            raise ValueError("promotion_margin must be between 0.0 and 0.25")
        if self.promotion_tier not in {
            "diagnostic",
            "canary",
            "provisional",
            "development",
            "full",
        }:
            raise ValueError(
                "promotion_tier must be diagnostic, canary, provisional, development, or full"
            )
        if not 8 <= self.minimum_promotion_pairs <= MAX_PAIRS:
            raise ValueError(f"minimum_promotion_pairs must be between 8 and {MAX_PAIRS:,}")
        if self.automatic_promotion and self.pairs < self.minimum_promotion_pairs:
            raise ValueError("automatic promotion jobs must run the full minimum paired evaluation")
        if self.automatic_promotion and self.minimum_promotion_pairs < MINIMUM_PROMOTION_PAIRS:
            raise ValueError(
                f"automatic promotion jobs require at least {MINIMUM_PROMOTION_PAIRS:,} pairs"
            )
        if not 8 <= self.early_rejection_min_pairs <= MAX_PAIRS:
            raise ValueError(f"early_rejection_min_pairs must be between 8 and {MAX_PAIRS:,}")
        if not 0.80 <= self.early_rejection_confidence < 1.0:
            raise ValueError("early_rejection_confidence must be between 0.80 and 1.0")
        if self.early_rejection and not self.automatic_promotion:
            raise ValueError("early rejection is available only for automatic promotion jobs")
        if self.early_rejection and self.early_rejection_min_pairs >= self.pairs:
            raise ValueError("early_rejection_min_pairs must be smaller than requested pairs")
        if not MINIMUM_PROMOTION_PAIRS <= self.early_acceptance_min_pairs <= MAX_PAIRS:
            raise ValueError(
                "early_acceptance_min_pairs must be between "
                f"{MINIMUM_PROMOTION_PAIRS:,} and {MAX_PAIRS:,}"
            )
        if not 0.80 <= self.early_acceptance_confidence < 1.0:
            raise ValueError("early_acceptance_confidence must be between 0.80 and 1.0")
        if self.early_acceptance and not self.automatic_promotion:
            raise ValueError("early acceptance is available only for automatic promotion jobs")
        if self.early_acceptance and self.early_acceptance_min_pairs >= self.pairs:
            raise ValueError("early_acceptance_min_pairs must be smaller than requested pairs")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    ref: str
    label: str
    kind: str
    actor_path: str | None = None
    checkpoint_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _ActorChooser:
    def __init__(self, actor: NumpyActor, encoder: Encoder, seed: int):
        self.actor = actor
        self.encoder = encoder
        self.rng = np.random.default_rng(seed)

    def __call__(self, _player_id: int, decision: Decision) -> Action:
        encoded = self.encoder.encode_decision(decision.observation, decision)
        eligible = np.asarray(model_action_indices(decision), dtype=np.int64)
        local_index, _values = self.actor.choose(
            encoded.state,
            encoded.actions[eligible],
            int(encoded.family),
            epsilon=0.0,
            head=None,
            rng=self.rng,
        )
        return decision.actions[int(eligible[local_index])]


class _LoadedModel:
    def __init__(self, resolved: ResolvedModel):
        self.resolved = resolved
        self.actor: NumpyActor | None = None
        self.encoder: Encoder | None = None
        if resolved.kind == "checkpoint":
            if resolved.actor_path is None:
                raise ModelResolutionError("checkpoint has no actor snapshot")
            self.actor = NumpyActor.load(resolved.actor_path)
            self.encoder = EngineEncoder(version=self.actor.spec.encoder_version)
            if self.actor.spec.state_size != self.encoder.state_size:
                raise ModelResolutionError("checkpoint state encoder is incompatible")
            if self.actor.spec.action_size != self.encoder.action_size:
                raise ModelResolutionError("checkpoint action encoder is incompatible")
            if self.actor.spec.families != len(DecisionFamily):
                raise ModelResolutionError("checkpoint decision families are incompatible")

    def chooser(self, seed: int):
        if self.resolved.kind == "baseline":
            return make_baseline(self.resolved.ref.removeprefix("baseline:"), seed=seed)
        assert self.actor is not None and self.encoder is not None
        return _ActorChooser(self.actor, self.encoder, seed)


@lru_cache(maxsize=16)
def _cached_loaded_model(resolved: ResolvedModel) -> _LoadedModel:
    """Load each frozen actor only once in an arena worker process."""

    return _LoadedModel(resolved)


_WORKER_CANCEL_EVENT: Any | None = None


def _initialize_arena_worker(cancel_event: Any) -> None:
    global _WORKER_CANCEL_EVENT
    _WORKER_CANCEL_EVENT = cancel_event


def _arena_worker_cancelled() -> bool:
    return bool(_WORKER_CANCEL_EVENT is not None and _WORKER_CANCEL_EVENT.is_set())


def _play_pair(
    resolved_a: ResolvedModel,
    resolved_b: ResolvedModel,
    config: ArenaConfig,
    pair_index: int,
    cancel_hook: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Play one deterministic seat-swapped pair in the calling process."""

    model_a = _cached_loaded_model(resolved_a)
    model_b = _cached_loaded_model(resolved_b)
    game_seed = _derived_seed(config.seed, pair_index, "game")
    policy_a_seed = _derived_seed(config.seed, pair_index, "model_a")
    policy_b_seed = _derived_seed(config.seed, pair_index, "model_b")
    cancelled = cancel_hook or _arena_worker_cancelled
    common = dict(
        config=GameConfig(
            seed=game_seed,
            seating=Seating.FIXED,
            starting_player=0,
            max_turns=config.max_turns,
            max_actions_per_turn=config.max_actions_per_turn,
        ),
        cancel_hook=cancelled,
    )
    first = Game(
        player_names=(resolved_a.label, resolved_b.label),
        choosers=(model_a.chooser(policy_a_seed), model_b.chooser(policy_b_seed)),
        **common,
    ).run()
    if cancelled():
        raise _ArenaCancelled()
    second = Game(
        player_names=(resolved_b.label, resolved_a.label),
        choosers=(model_b.chooser(policy_b_seed), model_a.chooser(policy_a_seed)),
        **common,
    ).run()
    if cancelled():
        raise _ArenaCancelled()
    first_score = _score(first, 0)
    second_score = _score(second, 1)
    return {
        "pair_index": pair_index,
        "first_score": first_score,
        "second_score": second_score,
        "turns": first.turns + second.turns,
        "decisions": first.decisions + second.decisions,
        "truncated_games": int(first.truncated) + int(second.truncated),
        "record": {
            "pair": pair_index + 1,
            "seed": game_seed,
            "first_game_seed": game_seed,
            "second_game_seed": game_seed,
            "model_a_first_seat_score": first_score,
            "model_a_second_seat_score": second_score,
            "first_game_starting_player": first.starting_player,
            "second_game_starting_player": second.starting_player,
            "first_game_truncated": first.truncated,
            "second_game_truncated": second.truncated,
            "first_game_truncation_reason": first.truncation_reason,
            "second_game_truncation_reason": second.truncation_reason,
        },
    }


def _default_worker_processes() -> int:
    configured = os.environ.get("ASTRO2_ARENA_WORKERS")
    if configured is not None:
        try:
            return max(1, min(16, int(configured)))
        except ValueError as exc:
            raise ValueError("ASTRO2_ARENA_WORKERS must be an integer") from exc
    return max(1, min(16, os.cpu_count() or 4))


def _trainer_arena_worker_processes(available: int) -> int:
    """Reserve CPU capacity for self-play while a scheduled arena runs."""

    configured = os.environ.get("ASTRO2_TRAINER_ARENA_WORKERS")
    if configured is not None:
        try:
            return max(1, min(available, int(configured)))
        except ValueError as exc:
            raise ValueError("ASTRO2_TRAINER_ARENA_WORKERS must be an integer") from exc
    # The target M4 has ten CPU cores and normally runs eight rollout actors.
    # Two evaluators fill the otherwise reserved capacity for short trainer
    # diagnostics. Exclusive full promotions and manual arenas use the full
    # configured quota.
    return max(1, min(2, available))


def _arena_worker_processes(config: ArenaConfig, available: int) -> int:
    """Give exclusive full promotion gates the complete evaluator pool."""

    if (
        config.trainer_scheduled
        and config.automatic_promotion
        and config.promotion_tier == "full"
    ):
        return max(1, available)
    if config.trainer_scheduled:
        return _trainer_arena_worker_processes(available)
    return max(1, available)


def resolve_model(store: Store, reference: str) -> ResolvedModel:
    ref = reference.strip()
    baseline = ref.lower().removeprefix("baseline:")
    if baseline in BASELINE_NAMES:
        return ResolvedModel(
            ref=f"baseline:{baseline}",
            label=f"{baseline.replace('_', ' ').title()} baseline",
            kind="baseline",
        )
    try:
        checkpoint = store.checkpoint(ref)
    except KeyError as exc:
        choices = ", ".join(BASELINE_NAMES)
        raise ModelResolutionError(
            f"unknown model reference {reference!r}; use a checkpoint ID or baseline: {choices}"
        ) from exc
    actor_path = checkpoint.get("actor_path")
    if not actor_path:
        raise ModelResolutionError("checkpoint actor snapshot is unavailable")
    path = Path(actor_path).expanduser().resolve()
    if not path.is_file():
        raise ModelResolutionError(f"checkpoint actor snapshot does not exist: {path}")
    return ResolvedModel(
        ref=ref,
        label=checkpoint["label"],
        kind="checkpoint",
        actor_path=str(path),
        checkpoint_id=checkpoint["id"],
    )


def _derived_seed(seed: int, pair_index: int, stream: str) -> int:
    payload = f"{seed}:{pair_index}:{stream}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") & (2**63 - 1)


def _score(result: GameResult, player_id: int) -> float:
    if result.winner is None:
        return 0.5
    return 1.0 if result.winner == player_id else 0.0


def _paired_interval(values: list[float], confidence: float) -> dict[str, float | int]:
    """Two-sided Hoeffding interval for bounded paired scores in ``[0, 1]``."""

    if not values:
        return {"estimate": 0.5, "low": 0.0, "high": 1.0, "samples": 0}
    samples = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(samples)) or np.any((samples < 0.0) | (samples > 1.0)):
        raise ValueError("paired scores must be finite and in [0, 1]")
    count = len(samples)
    estimate = float(samples.mean())
    alpha = 1.0 - confidence
    radius = math.sqrt(math.log(2.0 / alpha) / (2.0 * count))
    return {
        "estimate": estimate,
        "low": max(0.0, estimate - radius),
        "high": min(1.0, estimate + radius),
        "samples": count,
        "confidence_radius": radius,
    }


def _truncations_as_losses(
    result: dict[str, Any],
    *,
    confidence: float,
) -> dict[str, Any]:
    """Return promotion statistics with every truncated game scored as an A loss.

    Truncated games are recorded as draws (0.5 points) by the arena. Replacing
    each of those results with zero points lowers both the per-game and paired
    means by ``0.5 / games_completed``. The paired Hoeffding radius is unchanged
    because the number of completed pairs is unchanged.
    """

    pairs_completed = max(0, int(result.get("pairs_completed", 0)))
    games_completed = max(
        0,
        int(result.get("games_completed", pairs_completed * 2)),
    )
    truncated_games = max(0, int(result.get("truncated_games", 0)))
    if truncated_games > games_completed:
        raise ValueError("truncated games cannot exceed completed games")

    original_score = float(result.get("model_a_score", 0.5))
    adjusted_points = max(0.0, original_score * games_completed - 0.5 * truncated_games)
    adjusted_score = adjusted_points / games_completed if games_completed else 0.5
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    wilson = wilson_interval(adjusted_points, games_completed, z=z).as_dict()

    if pairs_completed:
        alpha = 1.0 - confidence
        radius = math.sqrt(math.log(2.0 / alpha) / (2.0 * pairs_completed))
        paired = {
            "estimate": adjusted_score,
            "low": max(0.0, adjusted_score - radius),
            "high": min(1.0, adjusted_score + radius),
            "samples": pairs_completed,
            "confidence_radius": radius,
        }
    else:
        paired = {"estimate": 0.5, "low": 0.0, "high": 1.0, "samples": 0}

    return {
        "applied": truncated_games > 0,
        "assumption": "candidate_lost_all_truncated_games",
        "truncated_games_scored_as_losses": truncated_games,
        "model_a_score": adjusted_score,
        "wilson_interval": wilson,
        "paired_interval": paired,
    }


def _early_rejection_looks(config: ArenaConfig) -> tuple[int, ...]:
    """Return fixed geometric looks strictly before the full evaluation.

    Fixing the looks before any games are observed lets a Bonferroni correction
    control the family-wise false-rejection probability without optional-peeking
    bias. The full sample remains governed by the ordinary promotion interval.
    """

    if not config.automatic_promotion or not config.early_rejection:
        return ()
    looks: list[int] = []
    pairs = config.early_rejection_min_pairs
    while pairs < config.pairs:
        looks.append(pairs)
        pairs *= 2
    return tuple(looks)


def _early_rejection_look(
    pair_scores: list[float] | np.ndarray,
    *,
    config: ArenaConfig,
    look_pairs: int,
) -> dict[str, Any]:
    """Compute a fixed-look, distribution-free upper confidence bound.

    Pair scores are bounded in ``[0, 1]``. Hoeffding's inequality therefore
    remains valid for constant outcomes (where a plug-in normal interval would
    incorrectly have zero width), and Bonferroni controls error across every
    look fixed by :func:`_early_rejection_looks`.
    """

    values = np.asarray(pair_scores, dtype=np.float64)
    looks = _early_rejection_looks(config)
    if look_pairs not in looks or len(values) != look_pairs:
        raise ValueError("early-rejection evidence must match a planned look")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("early-rejection pair scores must be finite and in [0, 1]")
    look_alpha = (1.0 - config.early_rejection_confidence) / len(looks)
    adjusted_confidence = 1.0 - look_alpha
    estimate = float(values.mean())
    confidence_radius = math.sqrt(math.log(1.0 / look_alpha) / (2.0 * len(values)))
    upper = min(1.0, estimate + confidence_radius)
    threshold = 0.5 + config.promotion_margin
    return {
        "look_index": looks.index(look_pairs) + 1,
        "look_pairs": look_pairs,
        "planned_looks": len(looks),
        "estimate": estimate,
        "upper_bound": upper,
        "confidence_radius": confidence_radius,
        "promotion_threshold": threshold,
        "configured_confidence": config.early_rejection_confidence,
        "bonferroni_look_alpha": look_alpha,
        "adjusted_one_sided_confidence": adjusted_confidence,
        "method": "bonferroni_one_sided_hoeffding",
        "reject": upper <= threshold,
    }


def _early_rejection_metadata(
    config: ArenaConfig,
    *,
    pairs_completed: int,
    latest_look: dict[str, Any] | None = None,
) -> dict[str, Any]:
    looks = _early_rejection_looks(config)
    return {
        "enabled": bool(config.automatic_promotion and config.early_rejection),
        "minimum_pairs": config.early_rejection_min_pairs,
        "configured_confidence": config.early_rejection_confidence,
        "planned_look_pairs": list(looks),
        "looks_completed": sum(look <= pairs_completed for look in looks),
        "latest_look": latest_look,
    }


def _early_acceptance_looks(config: ArenaConfig) -> tuple[int, ...]:
    """Return fixed geometric acceptance looks before the full evaluation."""

    if not config.automatic_promotion or not config.early_acceptance:
        return ()
    looks: list[int] = []
    pairs = config.early_acceptance_min_pairs
    while pairs < config.pairs:
        looks.append(pairs)
        pairs *= 2
    return tuple(looks)


def _early_acceptance_look(
    conservative_pair_scores: list[float] | np.ndarray,
    *,
    config: ArenaConfig,
    look_pairs: int,
) -> dict[str, Any]:
    """Compute a multiplicity-corrected lower bound for safe early promotion."""

    values = np.asarray(conservative_pair_scores, dtype=np.float64)
    looks = _early_acceptance_looks(config)
    if look_pairs not in looks or len(values) != look_pairs:
        raise ValueError("early-acceptance evidence must match a planned look")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("early-acceptance pair scores must be finite and in [0, 1]")
    look_alpha = (1.0 - config.early_acceptance_confidence) / len(looks)
    estimate = float(values.mean())
    confidence_radius = math.sqrt(math.log(1.0 / look_alpha) / (2.0 * len(values)))
    lower = max(0.0, estimate - confidence_radius)
    threshold = 0.5 + config.promotion_margin
    return {
        "look_index": looks.index(look_pairs) + 1,
        "look_pairs": look_pairs,
        "planned_looks": len(looks),
        "estimate": estimate,
        "lower_bound": lower,
        "confidence_radius": confidence_radius,
        "promotion_threshold": threshold,
        "configured_confidence": config.early_acceptance_confidence,
        "bonferroni_look_alpha": look_alpha,
        "adjusted_one_sided_confidence": 1.0 - look_alpha,
        "method": "bonferroni_one_sided_hoeffding",
        "accept": lower > threshold,
        "truncations_scored_as_losses": True,
    }


def _conservative_pair_scores(
    first_scores: list[float],
    second_scores: list[float],
    *,
    truncated_games: int,
) -> list[float]:
    """Return pair scores after replacing every truncated game draw with a loss."""

    values = [
        (first + second) * 0.5
        for first, second in zip(first_scores, second_scores, strict=True)
    ]
    # A truncated game is recorded as a 0.5 draw. Within a two-game pair it
    # contributes 0.25, so remove that mass. The bound depends only on the
    # mean; this also remains resumable when old per-pair records were pruned.
    remaining = min(float(sum(values)), max(0, truncated_games) * 0.25)
    for index, value in enumerate(values):
        reduction = min(value, remaining)
        values[index] = value - reduction
        remaining -= reduction
        if remaining <= 0.0:
            break
    return values


def _early_acceptance_metadata(
    config: ArenaConfig,
    *,
    pairs_completed: int,
    latest_look: dict[str, Any] | None = None,
) -> dict[str, Any]:
    looks = _early_acceptance_looks(config)
    return {
        "enabled": bool(config.automatic_promotion and config.early_acceptance),
        "minimum_pairs": config.early_acceptance_min_pairs,
        "configured_confidence": config.early_acceptance_confidence,
        "planned_look_pairs": list(looks),
        "looks_completed": sum(look <= pairs_completed for look in looks),
        "latest_look": latest_look,
    }


def _recommendation(
    config: ArenaConfig,
    *,
    pairs_completed: int,
    paired_low: float,
    paired_high: float,
    truncated_games: int = 0,
) -> str:
    threshold = 0.5 + config.promotion_margin
    if truncated_games:
        outcome = "supports promotion" if paired_low > threshold else "does not support promotion"
        return (
            f"{truncated_games:,} truncated game"
            f"{'s' if truncated_games != 1 else ''} scored as candidate losses; "
            f"the adjusted paired interval {outcome}"
        )
    if pairs_completed < config.minimum_promotion_pairs:
        return (
            f"inconclusive: {pairs_completed:,} of "
            f"{config.minimum_promotion_pairs:,} required pairs"
        )
    if pairs_completed < config.pairs:
        return f"evaluation in progress: {pairs_completed:,} of {config.pairs:,} requested pairs"
    if paired_low > threshold:
        return "model_a advantage is supported by the paired interval"
    if paired_high < threshold:
        return "model_b advantage is supported by the paired interval"
    return f"inconclusive: paired interval crosses {threshold:.1%} threshold"


def _should_extend_promotion_evaluation(
    *,
    config: ArenaConfig,
    pairs_completed: int,
    estimate: float,
    lower_bound: float,
) -> bool:
    """Return whether a completed full gate qualifies for one more 2,000-pair block."""

    return (
        config.automatic_promotion
        and config.trainer_scheduled
        and config.pairs == RECOMMENDED_PAIRS
        and pairs_completed == RECOMMENDED_PAIRS
        and estimate > 0.5
        and PROMOTION_EXTENSION_LOWER_MIN <= lower_bound <= PROMOTION_EXTENSION_LOWER_MAX
    )


def _summary(
    *,
    config: ArenaConfig,
    model_a: ResolvedModel,
    model_b: ResolvedModel,
    first_scores: list[float],
    second_scores: list[float],
    pair_records: list[dict[str, Any]],
    elapsed_seconds: float,
    total_turns: int,
    total_decisions: int,
    truncated_games: int,
    started_at: float,
    completed_at: float | None = None,
) -> dict[str, Any]:
    pairs_completed = len(first_scores)
    games_completed = pairs_completed * 2
    all_scores = [*first_scores, *second_scores]
    paired_scores = [
        (first + second) * 0.5 for first, second in zip(first_scores, second_scores, strict=True)
    ]
    model_a_wins = sum(score == 1.0 for score in all_scores)
    model_b_wins = sum(score == 0.0 for score in all_scores)
    neutral_results = sum(score == 0.5 for score in all_scores)
    draws = max(0, neutral_results - truncated_games)
    model_a_points = float(sum(all_scores))
    z = NormalDist().inv_cdf(0.5 + config.confidence / 2.0)
    wilson = wilson_interval(model_a_points, games_completed, z=z).as_dict()
    paired = _paired_interval(paired_scores, config.confidence)
    score = model_a_points / games_completed if games_completed else 0.5
    truncation_adjustment = _truncations_as_losses(
        {
            "pairs_completed": pairs_completed,
            "games_completed": games_completed,
            "model_a_score": score,
            "truncated_games": truncated_games,
        },
        confidence=config.confidence,
    )
    promotion_paired = (
        truncation_adjustment["paired_interval"] if truncated_games else paired
    )
    rate = games_completed / elapsed_seconds if elapsed_seconds > 0 else 0.0
    remaining_games = max(0, config.pairs * 2 - games_completed)
    promotion_eligible = (
        pairs_completed == config.pairs
        and pairs_completed >= config.minimum_promotion_pairs
    )
    promotion_threshold = 0.5 + config.promotion_margin
    recommendation = _recommendation(
        config,
        pairs_completed=pairs_completed,
        paired_low=float(promotion_paired["low"]),
        paired_high=float(promotion_paired["high"]),
        truncated_games=truncated_games,
    )
    return {
        "model_a": model_a.to_dict(),
        "model_b": model_b.to_dict(),
        "pairs_requested": config.pairs,
        "pairs_completed": pairs_completed,
        "games_completed": games_completed,
        "progress": pairs_completed / config.pairs,
        "model_a_wins": model_a_wins,
        "model_b_wins": model_b_wins,
        "draws": draws,
        "model_a_score": score,
        "model_b_score": 1.0 - score,
        "model_a_first_seat_score": (
            float(sum(first_scores) / pairs_completed) if pairs_completed else 0.5
        ),
        "model_a_second_seat_score": (
            float(sum(second_scores) / pairs_completed) if pairs_completed else 0.5
        ),
        "elo_difference_a_minus_b": elo_delta(score),
        "wilson_interval": wilson,
        "paired_interval": paired,
        "paired_interval_method": "two_sided_hoeffding",
        "confidence": config.confidence,
        "paired_common_seeds": True,
        "exact_seat_swap": True,
        "recent_pairs": pair_records[-20:],
        "truncated_games": truncated_games,
        "truncation_adjustment": truncation_adjustment,
        "total_turns": total_turns,
        "total_decisions": total_decisions,
        "elapsed_seconds": elapsed_seconds,
        "games_per_second": rate,
        "eta_seconds": remaining_games / rate if rate > 0 else None,
        "started_at": started_at,
        "completed_at": completed_at,
        "early_stopped": False,
        "early_stop_outcome": None,
        "early_stop_reason": None,
        "early_rejection": _early_rejection_metadata(
            config,
            pairs_completed=pairs_completed,
        ),
        "early_acceptance": _early_acceptance_metadata(
            config,
            pairs_completed=pairs_completed,
        ),
        "promotion": {
            "automatic": config.automatic_promotion,
            "eligible": promotion_eligible,
            "minimum_pairs": config.minimum_promotion_pairs,
            "tier": config.promotion_tier,
            "paired_lower_bound_required": promotion_threshold,
            "promoted": False,
            "recommendation": recommendation,
            "truncation_adjustment": truncation_adjustment,
        },
        "_first_seat_scores": first_scores,
        "_second_seat_scores": second_scores,
        "_pair_records": pair_records[-20:],
        "_total_turns": total_turns,
        "_total_decisions": total_decisions,
        "_truncated_games": truncated_games,
    }


def finalize_automatic_evaluation(
    store: Store,
    *,
    job_id: str,
    config: ArenaConfig,
    model_a: ResolvedModel,
    model_b: ResolvedModel,
    result: dict[str, Any],
) -> bool:
    """Record and possibly promote an internal candidate evaluation.

    Eligibility is recomputed from result counts and confidence bounds instead
    of trusting a caller-provided flag. Manual jobs return without touching
    checkpoint metadata.
    """

    promotion = result.setdefault("promotion", {})
    promotion.update(
        {
            "automatic": config.automatic_promotion,
            "minimum_pairs": config.minimum_promotion_pairs,
            "tier": config.promotion_tier,
            "paired_lower_bound_required": 0.5 + config.promotion_margin,
            "promoted": False,
        }
    )
    # Manual arena jobs are intentionally kept out of checkpoint metadata, but
    # trainer-scheduled diagnostics (notably Astro5 canaries) still belong to
    # the immutable candidate's evaluation history and must be visible in the
    # model registry. Promotion authority is handled separately below.
    if not config.automatic_promotion and not config.trainer_scheduled:
        return False
    if model_a.kind != "checkpoint" or model_b.kind != "checkpoint":
        raise ModelResolutionError("automatic arena jobs require two checkpoint IDs")
    if model_a.checkpoint_id is None or model_b.checkpoint_id is None:
        raise ModelResolutionError("automatic arena checkpoint metadata is incomplete")

    pairs_completed = int(result.get("pairs_completed", 0))
    paired = result.get("paired_interval") or {}
    early_stopped = bool(result.get("early_stopped", False))
    early_stop_outcome = result.get("early_stop_outcome")
    early_accepted = early_stopped and early_stop_outcome == "accepted"
    truncated_games = max(0, int(result.get("truncated_games", 0)))
    conservative = _truncations_as_losses(result, confidence=config.confidence)
    promotion["truncation_adjustment"] = conservative
    effective_paired = conservative["paired_interval"] if truncated_games else paired
    paired_low = float(effective_paired.get("low", 0.0))
    full = pairs_completed == config.pairs
    enough = pairs_completed >= config.minimum_promotion_pairs
    threshold = 0.5 + config.promotion_margin
    candidate_checkpoint = store.checkpoint(model_a.checkpoint_id)
    current_champion_id = store.get_run(candidate_checkpoint["run_id"]).get("champion_id")
    opponent_still_champion = current_champion_id == model_b.checkpoint_id
    candidate_already_champion = current_champion_id == model_a.checkpoint_id
    comparison_is_current = opponent_still_champion or candidate_already_champion
    acceptance_look = (result.get("early_acceptance") or {}).get("latest_look") or {}
    acceptance_lower = float(acceptance_look.get("lower_bound", 0.0))
    acceptance_proven = early_accepted and acceptance_lower > threshold
    promote = config.automatic_promotion and (
        ((not early_stopped and full and enough and paired_low > threshold) or acceptance_proven)
        and comparison_is_current
    )
    promotion["eligible"] = (not early_stopped and full and enough) or acceptance_proven
    promotion["promoted"] = promote
    promotion["opponent_still_champion"] = opponent_still_champion
    promotion["stale_opponent"] = not comparison_is_current
    if early_accepted and promote:
        promotion["recommendation"] = (
            f"promoted early: multiplicity-adjusted lower bound "
            f"{acceptance_lower:.3f} exceeds {threshold:.3f}"
        )
    elif early_stopped:
        promotion["recommendation"] = str(
            result.get("early_stop_reason")
            or "not promoted: early rejection upper bound did not clear the threshold"
        )
    elif not full:
        promotion["recommendation"] = "inconclusive: automatic arena job is incomplete"
    elif not enough:
        promotion["recommendation"] = (
            f"inconclusive: {pairs_completed:,} of "
            f"{config.minimum_promotion_pairs:,} required pairs"
        )
    elif not comparison_is_current:
        promotion["recommendation"] = (
            "not promoted: evaluation opponent is no longer the current champion"
        )
    elif not config.automatic_promotion:
        promotion["recommendation"] = (
            f"{config.promotion_tier} diagnostic complete: "
            f"paired score {float(result.get('model_a_score', 0.5)):.3f}"
        )
    elif promote:
        prefix = (
            f"promoted with {truncated_games:,} truncated "
            f"game{'s' if truncated_games != 1 else ''} scored as losses: "
            if truncated_games
            else "promoted model_a: "
        )
        promotion["recommendation"] = (
            f"{prefix}paired lower bound {paired_low:.3f} exceeds {threshold:.3f}"
        )
    else:
        prefix = (
            f"with {truncated_games:,} truncated "
            f"game{'s' if truncated_games != 1 else ''} scored as losses, "
            if truncated_games
            else ""
        )
        promotion["recommendation"] = (
            f"not promoted: {prefix}paired lower bound {paired_low:.3f} "
            f"does not exceed {threshold:.3f}"
        )

    evaluation = {
        "job_id": job_id,
        "opponent_checkpoint_id": model_b.checkpoint_id,
        "opponent_label": model_b.label,
        "pairs_completed": pairs_completed,
        "games_completed": int(result.get("games_completed", pairs_completed * 2)),
        "model_a_score": float(result.get("model_a_score", 0.5)),
        "wilson_interval": result.get("wilson_interval"),
        "paired_interval": paired,
        "confidence": config.confidence,
        "completed_at": result.get("completed_at"),
        "automatic": config.automatic_promotion,
        "trainer_scheduled": config.trainer_scheduled,
        "promotion_tier": config.promotion_tier,
        "promoted": promote,
        "early_stopped": early_stopped,
        "early_stop_outcome": early_stop_outcome,
        "early_stop_reason": result.get("early_stop_reason"),
        "early_rejection": result.get("early_rejection"),
        "early_acceptance": result.get("early_acceptance"),
        "truncated_games": truncated_games,
        "truncation_adjustment": conservative,
        "opponent_still_champion": opponent_still_champion,
        "stale_opponent": not comparison_is_current,
    }
    store.finalize_checkpoint_arena(
        model_a.checkpoint_id,
        evaluation,
        promote=promote,
    )
    return promote


class _ArenaCancelled(RuntimeError):
    pass


class ArenaManager:
    """Owns daemon evaluator threads; SQLite remains the source of truth."""

    def __init__(
        self,
        store: Store,
        *,
        maximum_concurrent_jobs: int = 1,
        worker_processes: int | None = None,
        recover: bool = True,
    ):
        self.store = store
        self.worker_processes = (
            _default_worker_processes()
            if worker_processes is None
            else max(1, min(16, int(worker_processes)))
        )
        self._slots = threading.Semaphore(maximum_concurrent_jobs)
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._threads: dict[str, threading.Thread] = {}
        self._job_cancellations: dict[str, threading.Event] = {}
        self._job_cancellation_hooks: dict[str, Callable[[], bool]] = {}
        if recover:
            for job in store.arena_jobs(limit=MAX_PAIRS, include_internal=True):
                if job["status"] in {"queued", "running"}:
                    self._start(job["id"])

    def create(
        self,
        model_a: str,
        model_b: str,
        config: ArenaConfig,
        *,
        cancellation_hook: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        resolved_a = resolve_model(self.store, model_a)
        resolved_b = resolve_model(self.store, model_b)
        if config.automatic_promotion and (
            resolved_a.kind != "checkpoint" or resolved_b.kind != "checkpoint"
        ):
            raise ModelResolutionError("automatic arena jobs require two checkpoint IDs")
        now = time.time()
        initial = _summary(
            config=config,
            model_a=resolved_a,
            model_b=resolved_b,
            first_scores=[],
            second_scores=[],
            pair_records=[],
            elapsed_seconds=0.0,
            total_turns=0,
            total_decisions=0,
            truncated_games=0,
            started_at=now,
        )
        job = self.store.create_arena_job(
            model_a=resolved_a.ref,
            model_b=resolved_b.ref,
            config=config.to_dict(),
            result=initial,
        )
        self._start(job["id"], cancellation_hook=cancellation_hook)
        return self.store.arena_job(job["id"])

    def create_automatic(
        self,
        candidate_checkpoint_id: str,
        opponent_checkpoint_id: str,
        *,
        pairs: int = RECOMMENDED_PAIRS,
        seed: int = 20260807,
        max_turns: int = 180,
        max_actions_per_turn: int = 160,
        confidence: float = 0.95,
        promotion_margin: float = 0.0,
        minimum_promotion_pairs: int = RECOMMENDED_PAIRS,
        promotion_tier: str = "full",
        early_rejection: bool = False,
        early_rejection_min_pairs: int = 512,
        early_rejection_confidence: float = 0.995,
        early_acceptance: bool = False,
        early_acceptance_min_pairs: int = MINIMUM_PROMOTION_PAIRS,
        early_acceptance_confidence: float = 0.995,
        cancellation_hook: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Trainer-only entry point for a promotion-eligible paired job."""

        return self.create(
            candidate_checkpoint_id,
            opponent_checkpoint_id,
            ArenaConfig(
                pairs=pairs,
                seed=seed,
                max_turns=max_turns,
                max_actions_per_turn=max_actions_per_turn,
                confidence=confidence,
                minimum_promotion_pairs=minimum_promotion_pairs,
                promotion_margin=promotion_margin,
                promotion_tier=promotion_tier,
                automatic_promotion=True,
                trainer_scheduled=True,
                early_rejection=early_rejection,
                early_rejection_min_pairs=early_rejection_min_pairs,
                early_rejection_confidence=early_rejection_confidence,
                early_acceptance=early_acceptance,
                early_acceptance_min_pairs=early_acceptance_min_pairs,
                early_acceptance_confidence=early_acceptance_confidence,
            ),
            cancellation_hook=cancellation_hook,
        )

    def get(self, job_id: str) -> dict[str, Any]:
        return self.store.arena_job(job_id)

    def list(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return self.store.arena_jobs(limit=limit)

    def _start(
        self,
        job_id: str,
        *,
        cancellation_hook: Callable[[], bool] | None = None,
    ) -> None:
        with self._lock:
            if job_id in self._threads and self._threads[job_id].is_alive():
                return
            self._job_cancellations[job_id] = threading.Event()
            if cancellation_hook is not None:
                self._job_cancellation_hooks[job_id] = cancellation_hook
            thread = threading.Thread(
                target=self._run_guarded,
                args=(job_id,),
                name=f"astro2-arena-{job_id}",
                daemon=True,
            )
            self._threads[job_id] = thread
            thread.start()

    def _job_cancelled(self, job_id: str) -> bool:
        with self._lock:
            event = self._job_cancellations[job_id]
            hook = self._job_cancellation_hooks.get(job_id)
        if hook is not None and bool(hook()):
            event.set()
        return self._stop.is_set() or event.is_set()

    def _run_guarded(self, job_id: str) -> None:
        acquired_slot = False
        try:
            while not acquired_slot:
                if self._job_cancelled(job_id):
                    raise _ArenaCancelled()
                acquired_slot = self._slots.acquire(timeout=0.1)
            self._run(job_id)
        except _ArenaCancelled:
            # A clean backend shutdown preserves resumable work. An explicit
            # per-job cancellation honors pause/stop without cancelling
            # unrelated manual arenas or permitting a cancelled promotion.
            with self._lock:
                explicitly_cancelled = self._job_cancellations.get(
                    job_id, threading.Event()
                ).is_set()
            self.store.update_arena_job(
                job_id,
                status="cancelled" if explicitly_cancelled else "queued",
                error=None,
            )
        except Exception as error:  # pragma: no cover - defensive job boundary
            self.store.update_arena_job(
                job_id,
                status="failed",
                error=f"{type(error).__name__}: {error}",
            )
        finally:
            if acquired_slot:
                self._slots.release()
            with self._lock:
                self._threads.pop(job_id, None)
                self._job_cancellations.pop(job_id, None)
                self._job_cancellation_hooks.pop(job_id, None)

    def _run(self, job_id: str) -> None:
        def cancelled() -> bool:
            return self._job_cancelled(job_id)

        job = self.store.arena_job(job_id, include_internal=True)
        config = ArenaConfig(**job["config"])
        resolved_a = resolve_model(self.store, job["model_a"])
        resolved_b = resolve_model(self.store, job["model_b"])
        previous = job["result"]
        first_scores = [float(value) for value in previous.get("_first_seat_scores", [])]
        second_scores = [float(value) for value in previous.get("_second_seat_scores", [])]
        pair_records = list(previous.get("_pair_records", []))
        total_turns = int(previous.get("_total_turns", 0))
        total_decisions = int(previous.get("_total_decisions", 0))
        truncated_games = int(previous.get("_truncated_games", 0))
        elapsed_before = float(previous.get("elapsed_seconds", 0.0))
        started_at = float(previous.get("started_at") or time.time())
        previous_early = previous.get("early_rejection") or {}
        latest_early_look = previous_early.get("latest_look")
        previous_acceptance = previous.get("early_acceptance") or {}
        latest_acceptance_look = previous_acceptance.get("latest_look")
        early_stopped = bool(previous.get("early_stopped", False))
        early_stop_outcome = previous.get("early_stop_outcome")
        early_stop_reason = previous.get("early_stop_reason")
        adaptive_extension_active = bool(previous.get("_adaptive_extension_active", False))
        base_config = config
        target_pairs = (
            RECOMMENDED_PAIRS + PROMOTION_EXTENSION_PAIRS
            if adaptive_extension_active
            else config.pairs
        )
        summary_config = replace(config, pairs=target_pairs) if adaptive_extension_active else config
        job_worker_processes = _arena_worker_processes(config, self.worker_processes)
        early_looks = frozenset(_early_rejection_looks(config))
        acceptance_looks = frozenset(_early_acceptance_looks(config))
        segment_start = time.monotonic()
        self.store.update_arena_job(job_id, status="running", error=None)
        last_update = 0.0

        def record_pair(pair: dict[str, Any]) -> None:
            nonlocal total_turns, total_decisions, truncated_games
            first_scores.append(float(pair["first_score"]))
            second_scores.append(float(pair["second_score"]))
            total_turns += int(pair["turns"])
            total_decisions += int(pair["decisions"])
            truncated_games += int(pair["truncated_games"])
            pair_records.append(pair["record"])

        statistical_looks = sorted(early_looks | acceptance_looks | {target_pairs})
        executor: ProcessPoolExecutor | None = None
        worker_cancel_event: Any | None = None
        if job_worker_processes > 1:
            context = mp.get_context("spawn")
            worker_cancel_event = context.Event()
            executor = ProcessPoolExecutor(
                max_workers=job_worker_processes,
                mp_context=context,
                initializer=_initialize_arena_worker,
                initargs=(worker_cancel_event,),
            )

        try:
            while len(first_scores) < target_pairs and not early_stopped:
                if cancelled():
                    raise _ArenaCancelled()
                pair_start = len(first_scores)
                next_look = next(look for look in statistical_looks if look > pair_start)
                wave_end = min(next_look, pair_start + job_worker_processes)
                if executor is None:
                    wave = [
                        _play_pair(
                            resolved_a,
                            resolved_b,
                            config,
                            pair_index,
                            cancel_hook=cancelled,
                        )
                        for pair_index in range(pair_start, wave_end)
                    ]
                else:
                    futures: set[Future[dict[str, Any]]] = {
                        executor.submit(
                            _play_pair,
                            resolved_a,
                            resolved_b,
                            config,
                            pair_index,
                        )
                        for pair_index in range(pair_start, wave_end)
                    }
                    wave = []
                    while futures:
                        if cancelled():
                            assert worker_cancel_event is not None
                            worker_cancel_event.set()
                            raise _ArenaCancelled()
                        done, futures = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
                        wave.extend(future.result() for future in done)
                for pair in sorted(wave, key=lambda item: int(item["pair_index"])):
                    record_pair(pair)

                pairs_completed = len(first_scores)
                if pairs_completed in early_looks:
                    pair_scores = [
                        (first_score + second_score) * 0.5
                        for first_score, second_score in zip(
                            first_scores, second_scores, strict=True
                        )
                    ]
                    latest_early_look = _early_rejection_look(
                        pair_scores,
                        config=config,
                        look_pairs=pairs_completed,
                    )
                    if bool(latest_early_look["reject"]):
                        early_stopped = True
                        early_stop_outcome = "rejected"
                        early_stop_reason = (
                            f"not promoted: early rejection at {pairs_completed:,} pairs; "
                            f"adjusted one-sided upper bound "
                            f"{float(latest_early_look['upper_bound']):.3f} is not above "
                            f"{float(latest_early_look['promotion_threshold']):.3f}"
                        )
                if not early_stopped and pairs_completed in acceptance_looks:
                    latest_acceptance_look = _early_acceptance_look(
                        _conservative_pair_scores(
                            first_scores,
                            second_scores,
                            truncated_games=truncated_games,
                        ),
                        config=config,
                        look_pairs=pairs_completed,
                    )
                    if bool(latest_acceptance_look["accept"]):
                        early_stopped = True
                        early_stop_outcome = "accepted"
                        early_stop_reason = (
                            f"promote early at {pairs_completed:,} pairs; adjusted one-sided "
                            f"lower bound {float(latest_acceptance_look['lower_bound']):.3f} "
                            f"exceeds "
                            f"{float(latest_acceptance_look['promotion_threshold']):.3f}"
                        )
                if (
                    not early_stopped
                    and not adaptive_extension_active
                    and pairs_completed == base_config.pairs
                ):
                    extension_interval = _paired_interval(
                        _conservative_pair_scores(
                            first_scores,
                            second_scores,
                            truncated_games=truncated_games,
                        ),
                        base_config.confidence,
                    )
                    if _should_extend_promotion_evaluation(
                        config=base_config,
                        pairs_completed=pairs_completed,
                        estimate=float(extension_interval["estimate"]),
                        lower_bound=float(extension_interval["low"]),
                    ):
                        adaptive_extension_active = True
                        target_pairs = RECOMMENDED_PAIRS + PROMOTION_EXTENSION_PAIRS
                        summary_config = replace(base_config, pairs=target_pairs)
                        statistical_looks = sorted(
                            early_looks | acceptance_looks | {target_pairs}
                        )
                now = time.monotonic()
                if (
                    now - last_update >= 0.25
                    or len(first_scores) == target_pairs
                    or early_stopped
                ):
                    elapsed = elapsed_before + now - segment_start
                    result = _summary(
                        config=summary_config,
                        model_a=resolved_a,
                        model_b=resolved_b,
                        first_scores=first_scores,
                        second_scores=second_scores,
                        pair_records=pair_records,
                        elapsed_seconds=elapsed,
                        total_turns=total_turns,
                        total_decisions=total_decisions,
                        truncated_games=truncated_games,
                        started_at=started_at,
                    )
                    result["early_stopped"] = early_stopped
                    result["early_stop_outcome"] = early_stop_outcome
                    result["early_stop_reason"] = early_stop_reason
                    result["early_rejection"] = _early_rejection_metadata(
                        config,
                        pairs_completed=len(first_scores),
                        latest_look=latest_early_look,
                    )
                    result["early_acceptance"] = _early_acceptance_metadata(
                        config,
                        pairs_completed=len(first_scores),
                        latest_look=latest_acceptance_look,
                    )
                    result["adaptive_extension"] = {
                        "active": adaptive_extension_active,
                        "initial_pairs": base_config.pairs,
                        "additional_pairs": (
                            PROMOTION_EXTENSION_PAIRS if adaptive_extension_active else 0
                        ),
                        "maximum_pairs": MAX_AUTOMATIC_PAIRS,
                    }
                    result["resource_policy"] = {
                        "worker_processes": job_worker_processes,
                        "trainer_isolated": config.trainer_scheduled,
                    }
                    result["_adaptive_extension_active"] = adaptive_extension_active
                    self.store.update_arena_job(job_id, status="running", result=result)
                    last_update = now
        finally:
            if executor is not None:
                if cancelled() and worker_cancel_event is not None:
                    worker_cancel_event.set()
                executor.shutdown(wait=True, cancel_futures=True)

        if cancelled():
            raise _ArenaCancelled()
        completed_at = time.time()
        elapsed = elapsed_before + time.monotonic() - segment_start
        final = _summary(
            config=summary_config,
            model_a=resolved_a,
            model_b=resolved_b,
            first_scores=first_scores,
            second_scores=second_scores,
            pair_records=pair_records,
            elapsed_seconds=elapsed,
            total_turns=total_turns,
            total_decisions=total_decisions,
            truncated_games=truncated_games,
            started_at=started_at,
            completed_at=completed_at,
        )
        final["early_stopped"] = early_stopped
        final["early_stop_outcome"] = early_stop_outcome
        final["early_stop_reason"] = early_stop_reason
        final["early_rejection"] = _early_rejection_metadata(
            config,
            pairs_completed=len(first_scores),
            latest_look=latest_early_look,
        )
        final["early_acceptance"] = _early_acceptance_metadata(
            config,
            pairs_completed=len(first_scores),
            latest_look=latest_acceptance_look,
        )
        final["adaptive_extension"] = {
            "active": adaptive_extension_active,
            "initial_pairs": base_config.pairs,
            "additional_pairs": PROMOTION_EXTENSION_PAIRS if adaptive_extension_active else 0,
            "maximum_pairs": MAX_AUTOMATIC_PAIRS,
        }
        final["resource_policy"] = {
            "worker_processes": job_worker_processes,
            "trainer_isolated": config.trainer_scheduled,
        }
        final["_adaptive_extension_active"] = adaptive_extension_active
        if early_stopped:
            final["eta_seconds"] = 0.0
        paired_interval = _paired_interval(
            [
                (first + second) * 0.5
                for first, second in zip(first_scores, second_scores, strict=True)
            ],
            config.confidence,
        )
        final["paired_interval"] = paired_interval
        final["paired_interval_method"] = "two_sided_hoeffding"
        final["promotion"]["recommendation"] = _recommendation(
            summary_config,
            pairs_completed=len(first_scores),
            paired_low=float(paired_interval["low"]),
            paired_high=float(paired_interval["high"]),
            truncated_games=truncated_games,
        )
        if cancelled():
            raise _ArenaCancelled()
        finalize_automatic_evaluation(
            self.store,
            job_id=job_id,
            config=summary_config,
            model_a=resolved_a,
            model_b=resolved_b,
            result=final,
        )
        self.store.update_arena_job(job_id, status="complete", result=final, error=None)

    def shutdown(self, timeout: float = 2.0) -> None:
        self._stop.set()
        with self._lock:
            threads = list(self._threads.values())
        deadline = time.monotonic() + timeout
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))

    def wait_for_job(self, job_id: str, timeout: float | None = None) -> bool:
        """Wait only for one manager-owned job, never unrelated arenas."""

        with self._lock:
            thread = self._threads.get(job_id)
        if thread is None:
            return self.store.arena_job(job_id)["status"] not in {"queued", "running"}
        thread.join(None if timeout is None else max(0.0, timeout))
        return not thread.is_alive()

    def cancel(self, job_id: str, timeout: float | None = None) -> bool:
        """Cancel one owned arena and wait for its terminal persisted status."""

        with self._lock:
            event = self._job_cancellations.get(job_id)
        if event is None:
            return self.store.arena_job(job_id)["status"] not in {"queued", "running"}
        event.set()
        return self.wait_for_job(job_id, timeout=timeout)

    def wait_for_idle(self, timeout: float | None = None) -> bool:
        """Wait for every job owned by this manager, including queued jobs."""

        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while True:
            with self._lock:
                threads = [thread for thread in self._threads.values() if thread.is_alive()]
            if not threads:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            threads[0].join(0.25 if remaining is None else min(0.25, remaining))


__all__ = [
    "MAX_PAIRS",
    "RECOMMENDED_PAIRS",
    "ArenaConfig",
    "ArenaManager",
    "ModelResolutionError",
    "ResolvedModel",
    "finalize_automatic_evaluation",
    "resolve_model",
]
