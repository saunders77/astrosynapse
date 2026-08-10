"""Persistent, bounded, paired arena evaluation jobs.

Every seed is played twice with exact seat reversal.  This removes most first-
player and deal noise while keeping jobs small enough to run beside training on
the 16 GB M4 target.  Jobs write progress to SQLite, so a browser can disconnect
and reconnect without owning the evaluator process.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from .baselines import BASELINE_NAMES, make_baseline
from .cards import ALL_CARDS
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
from .league import paired_bootstrap_interval
from .model import NumpyActor
from .stats import elo_delta, wilson_interval
from .storage import Store

RECOMMENDED_PAIRS = 5_000
MAX_PAIRS = 20_000


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

    def __post_init__(self) -> None:
        if not 1 <= self.pairs <= MAX_PAIRS:
            raise ValueError(f"pairs must be between 1 and {MAX_PAIRS:,}")
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
            "provisional",
            "development",
            "full",
        }:
            raise ValueError(
                "promotion_tier must be diagnostic, provisional, development, or full"
            )
        if not 8 <= self.minimum_promotion_pairs <= MAX_PAIRS:
            raise ValueError(
                f"minimum_promotion_pairs must be between 8 and {MAX_PAIRS:,}"
            )
        if self.automatic_promotion and self.pairs < self.minimum_promotion_pairs:
            raise ValueError(
                "automatic promotion jobs must run the full minimum paired evaluation"
            )

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
            self.encoder = Encoder(card_catalog=ALL_CARDS)
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
    if not values:
        return {"estimate": 0.5, "low": 0.0, "high": 1.0, "samples": 0}
    count = len(values)
    estimate = float(sum(values) / count)
    if count < 2:
        return {"estimate": estimate, "low": 0.0, "high": 1.0, "samples": count}
    variance = sum((value - estimate) ** 2 for value in values) / (count - 1)
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    radius = z * math.sqrt(variance / count)
    return {
        "estimate": estimate,
        "low": max(0.0, estimate - radius),
        "high": min(1.0, estimate + radius),
        "samples": count,
    }


def _recommendation(
    config: ArenaConfig,
    *,
    pairs_completed: int,
    paired_low: float,
    paired_high: float,
) -> str:
    threshold = 0.5 + config.promotion_margin
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
    draws = sum(score == 0.5 for score in all_scores)
    model_b_wins = games_completed - model_a_wins - draws
    model_a_points = float(sum(all_scores))
    z = NormalDist().inv_cdf(0.5 + config.confidence / 2.0)
    wilson = wilson_interval(model_a_points, games_completed, z=z).as_dict()
    paired = _paired_interval(paired_scores, config.confidence)
    score = model_a_points / games_completed if games_completed else 0.5
    rate = games_completed / elapsed_seconds if elapsed_seconds > 0 else 0.0
    remaining_games = max(0, config.pairs * 2 - games_completed)
    promotion_eligible = (
        pairs_completed == config.pairs and pairs_completed >= config.minimum_promotion_pairs
    )
    promotion_threshold = 0.5 + config.promotion_margin
    recommendation = _recommendation(
        config,
        pairs_completed=pairs_completed,
        paired_low=float(paired["low"]),
        paired_high=float(paired["high"]),
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
        "paired_interval_method": "normal_live",
        "confidence": config.confidence,
        "paired_common_seeds": True,
        "exact_seat_swap": True,
        "recent_pairs": pair_records[-20:],
        "truncated_games": truncated_games,
        "total_turns": total_turns,
        "total_decisions": total_decisions,
        "elapsed_seconds": elapsed_seconds,
        "games_per_second": rate,
        "eta_seconds": remaining_games / rate if rate > 0 else None,
        "started_at": started_at,
        "completed_at": completed_at,
        "promotion": {
            "automatic": config.automatic_promotion,
            "eligible": promotion_eligible,
            "minimum_pairs": config.minimum_promotion_pairs,
            "tier": config.promotion_tier,
            "paired_lower_bound_required": promotion_threshold,
            "promoted": False,
            "recommendation": recommendation,
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
    if not config.automatic_promotion:
        return False
    if model_a.kind != "checkpoint" or model_b.kind != "checkpoint":
        raise ModelResolutionError("automatic arena jobs require two checkpoint IDs")
    if model_a.checkpoint_id is None or model_b.checkpoint_id is None:
        raise ModelResolutionError("automatic arena checkpoint metadata is incomplete")

    pairs_completed = int(result.get("pairs_completed", 0))
    paired = result.get("paired_interval") or {}
    paired_low = float(paired.get("low", 0.0))
    full = pairs_completed == config.pairs
    enough = pairs_completed >= config.minimum_promotion_pairs
    threshold = 0.5 + config.promotion_margin
    candidate_checkpoint = store.checkpoint(model_a.checkpoint_id)
    current_champion_id = store.get_run(candidate_checkpoint["run_id"]).get("champion_id")
    opponent_still_champion = current_champion_id == model_b.checkpoint_id
    candidate_already_champion = current_champion_id == model_a.checkpoint_id
    comparison_is_current = opponent_still_champion or candidate_already_champion
    promote = full and enough and paired_low > threshold and comparison_is_current
    promotion["eligible"] = full and enough
    promotion["promoted"] = promote
    promotion["opponent_still_champion"] = opponent_still_champion
    promotion["stale_opponent"] = not comparison_is_current
    if not full:
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
    elif promote:
        promotion["recommendation"] = (
            f"promoted model_a: paired lower bound {paired_low:.3f} exceeds {threshold:.3f}"
        )
    else:
        promotion["recommendation"] = (
            f"not promoted: paired lower bound {paired_low:.3f} does not exceed {threshold:.3f}"
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
        "automatic": True,
        "promotion_tier": config.promotion_tier,
        "promoted": promote,
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

    def __init__(self, store: Store, *, maximum_concurrent_jobs: int = 1, recover: bool = True):
        self.store = store
        self._slots = threading.Semaphore(maximum_concurrent_jobs)
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._threads: dict[str, threading.Thread] = {}
        if recover:
            for job in store.arena_jobs(limit=MAX_PAIRS, include_internal=True):
                if job["status"] in {"queued", "running"}:
                    self._start(job["id"])

    def create(self, model_a: str, model_b: str, config: ArenaConfig) -> dict[str, Any]:
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
        self._start(job["id"])
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
            ),
        )

    def get(self, job_id: str) -> dict[str, Any]:
        return self.store.arena_job(job_id)

    def list(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return self.store.arena_jobs(limit=limit)

    def _start(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._threads and self._threads[job_id].is_alive():
                return
            thread = threading.Thread(
                target=self._run_guarded,
                args=(job_id,),
                name=f"astro2-arena-{job_id}",
                daemon=True,
            )
            self._threads[job_id] = thread
            thread.start()

    def _run_guarded(self, job_id: str) -> None:
        with self._slots:
            try:
                self._run(job_id)
            except _ArenaCancelled:
                # A clean backend shutdown is a pause, not evidence loss. The
                # persisted pair arrays let the next ArenaManager recover this
                # queued job from its last completed common-seed pair.
                self.store.update_arena_job(
                    job_id, status="queued", error=None
                )
            except Exception as error:  # pragma: no cover - defensive job boundary
                self.store.update_arena_job(
                    job_id,
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                )
            finally:
                with self._lock:
                    self._threads.pop(job_id, None)

    def _run(self, job_id: str) -> None:
        job = self.store.arena_job(job_id, include_internal=True)
        config = ArenaConfig(**job["config"])
        resolved_a = resolve_model(self.store, job["model_a"])
        resolved_b = resolve_model(self.store, job["model_b"])
        model_a = _LoadedModel(resolved_a)
        model_b = _LoadedModel(resolved_b)
        previous = job["result"]
        first_scores = [float(value) for value in previous.get("_first_seat_scores", [])]
        second_scores = [float(value) for value in previous.get("_second_seat_scores", [])]
        pair_records = list(previous.get("_pair_records", []))
        total_turns = int(previous.get("_total_turns", 0))
        total_decisions = int(previous.get("_total_decisions", 0))
        truncated_games = int(previous.get("_truncated_games", 0))
        elapsed_before = float(previous.get("elapsed_seconds", 0.0))
        started_at = float(previous.get("started_at") or time.time())
        segment_start = time.monotonic()
        self.store.update_arena_job(job_id, status="running", error=None)
        last_update = 0.0

        for pair_index in range(len(first_scores), config.pairs):
            if self._stop.is_set():
                raise _ArenaCancelled()
            game_seed = _derived_seed(config.seed, pair_index, "game")
            policy_a_seed = _derived_seed(config.seed, pair_index, "model_a")
            policy_b_seed = _derived_seed(config.seed, pair_index, "model_b")
            common = dict(
                config=GameConfig(
                    seed=game_seed,
                    seating=Seating.FIXED,
                    starting_player=0,
                    max_turns=config.max_turns,
                    max_actions_per_turn=config.max_actions_per_turn,
                ),
                cancel_hook=self._stop.is_set,
            )
            first = Game(
                player_names=(resolved_a.label, resolved_b.label),
                choosers=(
                    model_a.chooser(policy_a_seed),
                    model_b.chooser(policy_b_seed),
                ),
                **common,
            ).run()
            if self._stop.is_set():
                raise _ArenaCancelled()
            second = Game(
                player_names=(resolved_b.label, resolved_a.label),
                choosers=(
                    model_b.chooser(policy_b_seed),
                    model_a.chooser(policy_a_seed),
                ),
                **common,
            ).run()
            first_score = _score(first, 0)
            second_score = _score(second, 1)
            first_scores.append(first_score)
            second_scores.append(second_score)
            total_turns += first.turns + second.turns
            total_decisions += first.decisions + second.decisions
            truncated_games += int(first.truncated) + int(second.truncated)
            pair_records.append(
                {
                    "pair": pair_index + 1,
                    "seed": game_seed,
                    "first_game_seed": game_seed,
                    "second_game_seed": game_seed,
                    "model_a_first_seat_score": first_score,
                    "model_a_second_seat_score": second_score,
                    "first_game_starting_player": first.starting_player,
                    "second_game_starting_player": second.starting_player,
                }
            )
            now = time.monotonic()
            if now - last_update >= 0.25 or len(first_scores) == config.pairs:
                elapsed = elapsed_before + now - segment_start
                result = _summary(
                    config=config,
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
                self.store.update_arena_job(job_id, status="running", result=result)
                last_update = now

        completed_at = time.time()
        elapsed = elapsed_before + time.monotonic() - segment_start
        final = _summary(
            config=config,
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
        pair_scores = np.asarray(
            [
                (first + second) * 0.5
                for first, second in zip(first_scores, second_scores, strict=True)
            ],
            dtype=np.float64,
        )
        bootstrapped = paired_bootstrap_interval(
            pair_scores,
            confidence=config.confidence,
            samples=10_000,
            seed=config.seed,
        )
        final["paired_interval"] = bootstrapped.as_dict()
        final["paired_interval_method"] = "nonparametric_bootstrap"
        final["promotion"]["recommendation"] = _recommendation(
            config,
            pairs_completed=len(first_scores),
            paired_low=bootstrapped.low,
            paired_high=bootstrapped.high,
        )
        finalize_automatic_evaluation(
            self.store,
            job_id=job_id,
            config=config,
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
