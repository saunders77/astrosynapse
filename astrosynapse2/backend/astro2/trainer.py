"""M4-oriented asynchronous self-play and MLX learner loop.

CPU actor processes run the deterministic engine and lightweight NumPy model
while the parent process performs replay updates on Metal.  The overlap keeps
the M4 CPU and GPU useful at the same time without giving every worker its own
large accelerator context.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import RunConfig
from .encoding import FAMILY_COUNT, Encoder
from .hardware import RateMeter, mlx_snapshot, system_snapshot
from .league import League, Opponent
from .model import (
    ModelSpec,
    bootstrap_bce_loss,
    build_model,
    export_actor,
    load_model,
    preference_ranking_loss,
    save_model,
)
from .replay import PreferenceReplayBuffer, ReplayBuffer
from .selfplay import WorkerResult, collect_worker_batch
from .storage import Store


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


@dataclass(frozen=True, slots=True)
class _EvaluationPlan:
    tier: str
    cadence_games: int
    pairs: int
    automatic_promotion: bool


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
            cadence_games=max(
                config.checkpoint_every_games, config.evaluate_every_games // 2
            ),
            pairs=min(config.evaluation_pairs, max(1_000, config.evaluation_pairs // 5)),
            automatic_promotion=True,
        )
    return _EvaluationPlan(
        tier="full",
        cadence_games=config.evaluate_every_games,
        pairs=config.evaluation_pairs,
        automatic_promotion=True,
    )


def _epsilon(config: RunConfig, games: int) -> float:
    if games >= config.epsilon_decay_games:
        return config.epsilon_end * config.exploration_decision_scale
    progress = min(1.0, games / max(1, config.epsilon_decay_games))
    scheduled = config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)
    return scheduled * config.exploration_decision_scale


def _learning_rate(
    config: RunConfig,
    completed_updates: int,
    updates_since_optimizer_reset: int,
) -> float:
    # The decay horizon is immutable for a run.  Extending its wall-clock
    # duration therefore cannot rewind optimization to a larger learning rate.
    progress = min(1.0, max(0.0, completed_updates / config.learning_rate_decay_updates))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduled = config.min_learning_rate + (
        config.learning_rate - config.min_learning_rate
    ) * cosine
    # A brief optimizer warmup protects a fresh randomly initialized value net
    # from the unusually correlated first replay batches.
    warmup = min(1.0, (updates_since_optimizer_reset + 1) / 500.0)
    return max(config.min_learning_rate * warmup, scheduled * warmup)


def _atomic_actor_export(model: Any, spec: ModelSpec, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.stem}.partial{target.suffix}")
    export_actor(model, spec, temporary)
    temporary.replace(target)
    return target


def _latest_loadable_checkpoint(store: Store, run_id: str) -> dict[str, Any] | None:
    for checkpoint in store.checkpoints(run_id):
        path = Path(checkpoint["path"])
        if path.exists() and path.with_suffix(path.suffix + ".json").exists():
            return checkpoint
    return None


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
    checkpoint_ids = {checkpoint["id"] for checkpoint in store.checkpoints(run_id)}
    return sorted(
        (
            job
            for job in store.arena_jobs(limit=20_000, include_internal=True)
            if job["status"] == "complete"
            and job["model_a"] in checkpoint_ids
            and bool((job.get("config") or {}).get("automatic_promotion"))
            and bool((job.get("config") or {}).get("trainer_scheduled"))
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
) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = f"g{games:010d}-{int(time.time())}"
    model_path = checkpoint_dir / f"{stem}.safetensors"
    actor_path = checkpoint_dir / f"{stem}.actor.npz"
    save_model(model, spec, model_path)
    export_actor(model, spec, actor_path)
    checkpoint = store.add_checkpoint(
        run_id=run_id,
        parent_id=parent_id,
        label=f"{'Champion' if champion else 'Candidate'} · {games:,} games",
        path=str(model_path),
        actor_path=str(actor_path),
        games=games,
        champion=champion,
        evaluation={"reason": reason, "evaluated": False},
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


def _sync_league(league: League, store: Store, run_id: str) -> None:
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


def _last_scheduled_evaluation_games(
    store: Store, run_id: str, *, tier: str | None = None
) -> int:
    checkpoint_games = {item["id"]: int(item["games"]) for item in store.checkpoints(run_id)}
    return max(
        (
            checkpoint_games.get(job["model_a"], 0)
            for job in store.arena_jobs(limit=20_000)
            if job["model_a"] in checkpoint_games
            and bool(
                (job.get("config") or {}).get("trainer_scheduled")
                or (job.get("config") or {}).get("automatic_promotion")
            )
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


def _persisted_active_elapsed(store: Store, run_id: str) -> float:
    """Recover consumed training time without counting backend downtime."""

    metrics = store.metrics(run_id, after=-1, limit=1)
    if not metrics:
        return 0.0
    value = metrics[-1].get("active_elapsed_seconds", 0.0)
    try:
        elapsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return elapsed if math.isfinite(elapsed) and elapsed >= 0.0 else 0.0


def _restore_totals(store: Store, run: dict[str, Any]) -> _Totals:
    """Restore cumulative telemetry counters alongside checkpointed progress."""

    metrics = store.metrics(run["id"], after=-1, limit=1)
    latest = metrics[-1] if metrics else {}
    games = max(int(run["games"]), int(latest.get("games", 0)))
    metric_games = max(0, int(latest.get("games", games)))
    mean_turns = max(0.0, float(latest.get("mean_turns", 0.0)))
    return _Totals(
        games=games,
        decisions=max(int(run["decisions"]), int(latest.get("decisions", 0))),
        updates=max(int(run["updates"]), int(latest.get("updates", 0))),
        samples=max(0, int(latest.get("samples", 0))),
        player_wins=(
            max(0, int(latest.get("player_0_wins", 0))),
            max(0, int(latest.get("player_1_wins", 0))),
        ),
        draws=max(0, int(latest.get("draws", 0))),
        truncated=max(0, int(latest.get("truncations", 0))),
        turns=max(0, round(mean_turns * metric_games)),
        forced_choices=max(0, int(latest.get("forced_choices", 0))),
    )


def _schedule_evaluation(
    *,
    manager: Any,
    store: Store,
    run_id: str,
    checkpoint: dict[str, Any],
    config: RunConfig,
    plan: _EvaluationPlan | None = None,
) -> dict[str, Any] | None:
    champion_id = store.get_run(run_id).get("champion_id")
    if not champion_id or champion_id == checkpoint["id"]:
        return None
    plan = plan or _evaluation_plan(config, int(checkpoint["games"]))
    if plan.automatic_promotion:
        job = manager.create_automatic(
            checkpoint["id"],
            champion_id,
            pairs=plan.pairs,
            seed=config.seed + int(checkpoint["games"]),
            max_turns=config.max_turns,
            max_actions_per_turn=config.max_actions_per_turn,
            confidence=config.promotion_confidence,
            promotion_margin=config.promotion_margin,
            minimum_promotion_pairs=plan.pairs,
            promotion_tier=plan.tier,
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
                seed=config.seed + int(checkpoint["games"]),
                max_turns=config.max_turns,
                max_actions_per_turn=config.max_actions_per_turn,
                confidence=config.promotion_confidence,
                promotion_tier=plan.tier,
                automatic_promotion=False,
                trainer_scheduled=True,
            ),
        )
    store.event(
        run_id,
        "automatic_evaluation_started",
        f"Started paired evaluation for {checkpoint['label']}",
        {"job_id": job["id"], "pairs": plan.pairs, "tier": plan.tier},
    )
    return job


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
        and champion_score - float(baselines["mean_score"])
        > config.baseline_regression_tolerance
    )
    heldout_regression = (
        champion_brier is not None
        and float(heldout["game_grouped_brier"]) - champion_brier
        > config.heldout_brier_regression_tolerance
    )
    reasons: list[str] = []
    if int(tactical["raw_end_turn_violations"]) > config.max_tactical_violations:
        reasons.append("raw model logits failed the tactical dominance suite")
    if int(baselines["truncated_games"]) > 0:
        reasons.append("fixed-opponent diagnostics contained truncated games")
    if baseline_regression:
        reasons.append("fixed-opponent score regressed beyond the configured tolerance")
    if heldout_regression:
        reasons.append("held-out Brier score regressed beyond the configured tolerance")
    gate = {
        "passed": not reasons,
        "reasons": reasons,
        "champion_baseline_score": champion_score,
        "champion_heldout_brier": champion_brier,
        "baseline_regression_tolerance": config.baseline_regression_tolerance,
        "heldout_brier_regression_tolerance": config.heldout_brier_regression_tolerance,
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
        return _RolloutPlan(
            actor_paths=(current_actor, current_actor),
            baseline_names=("balanced", "balanced"),
            collect_players=(True, True),
            epsilons=(epsilon, epsilon),
            seed=seed,
            games=config.games_per_actor_batch,
            kind="self_play",
            opponent_id=None,
            current_player=None,
        )

    checkpoint_opponents = [
        item
        for item in league.opponents
        if item.kind in {"checkpoint", "champion"} and item.actor_path
    ]
    league_cutoff = config.current_selfplay_fraction + config.league_fraction
    if roll < league_cutoff and checkpoint_opponents:
        opponent = league.select(rng, mode="pfsp", kinds={"checkpoint", "champion"})
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
    )


def _train_updates(
    *,
    model: Any,
    optimizer: Any,
    replay: ReplayBuffer,
    preference_replay: PreferenceReplayBuffer,
    config: RunConfig,
    count: int,
    totals: _Totals,
    optimizer_updates_at_start: int,
    control: Any,
) -> dict[str, float]:
    if count <= 0 or len(replay) < config.replay_warmup:
        return {}

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

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
        outcome = bootstrap_bce_loss(
            model, states, actions, families, targets, masks, weights
        )[0]
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
        effective_targets = np.where(
            batch.td_valid > 0,
            config.terminal_target_weight * batch.targets
            + (1.0 - config.terminal_target_weight) * batch.td_targets,
            batch.targets,
        ).astype(np.float32)
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
                    **{
                        name: float(value.item())
                        for name, value in preference_diagnostics.items()
                    },
                    "td_target_fraction": float(batch.td_valid.mean()),
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

    encoder = Encoder()
    # Keep artifacts beside the configured SQLite store so ASTRO2_DATA_DIR and
    # the CLI's --data-dir remain self-contained.
    checkpoint_root = store.path.parent / "checkpoints" / run_id
    runtime_actor = checkpoint_root / "runtime" / "current.actor.npz"
    latest = _latest_loadable_checkpoint(store, run_id)
    if latest is not None:
        model, spec = load_model(latest["path"])
        parent_checkpoint_id = latest["id"]
    else:
        spec = ModelSpec(
            state_size=encoder.state_size,
            action_size=encoder.action_size,
            families=FAMILY_COUNT,
            hidden_size=config.hidden_size,
            action_hidden_size=max(64, config.hidden_size // 2),
            residual_blocks=config.residual_blocks,
            bootstrap_heads=config.bootstrap_heads,
        )
        model = build_model(spec)
        parent_checkpoint_id = None
    if (
        spec.state_size != encoder.state_size
        or spec.action_size != encoder.action_size
        or spec.families != FAMILY_COUNT
    ):
        raise RuntimeError("checkpoint encoder contract does not match this engine build")

    model.train()
    mx.eval(model.parameters())
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        betas=[0.9, 0.95],
        weight_decay=config.weight_decay,
        bias_correction=True,
    )
    replay = ReplayBuffer(
        capacity=config.replay_capacity,
        state_size=encoder.state_size,
        action_size=encoder.action_size,
        bootstrap_heads=config.bootstrap_heads,
        recent_sample_fraction=config.recent_sample_fraction,
        seed=config.seed + 41,
    )
    preference_replay = PreferenceReplayBuffer(
        capacity=config.preference_replay_capacity,
        state_size=encoder.state_size,
        action_size=encoder.action_size,
        seed=config.seed + 43,
    )
    totals = _restore_totals(store, run)
    optimizer_updates_at_start = totals.updates
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
        )
        parent_checkpoint_id = latest["id"]

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
    _sync_league(league, store, run_id)

    rng = np.random.default_rng(config.seed + totals.games + 73)
    rate = RateMeter.start()
    rate.last_games = totals.games
    rate.last_decisions = totals.decisions
    previous_active_elapsed = _persisted_active_elapsed(store, run_id)
    session_started_wall = time.time()
    paused_seconds = 0.0
    last_metric_at = 0.0
    last_diagnostics: dict[str, float] = {}
    metric_seq = int(time.time() * 1_000)
    seed_cursor = config.seed + totals.games * 10_007
    last_checkpoint_games = int(latest["games"] if latest else totals.games)
    evaluation_manager: Any | None = None
    processed_evaluation_jobs = {
        job["id"]
        for job in _completed_trainer_evaluations(store, run_id)
        if bool((job.get("result") or {}).get("_trainer_disposition_processed"))
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

    def maybe_schedule_evaluation(checkpoint: dict[str, Any]) -> None:
        nonlocal evaluation_manager
        champion_id = store.get_run(run_id).get("champion_id")
        if not champion_id or champion_id == checkpoint["id"]:
            return
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
            return
        plan = _evaluation_plan(config, totals.games)
        last_evaluation_games = _last_scheduled_evaluation_games(
            store, run_id, tier=plan.tier
        )
        if totals.games - last_evaluation_games < plan.cadence_games:
            return
        if evaluation_manager is None:
            from .arena import ArenaManager

            # The evaluator owns one CPU thread and persists everything. Its
            # daemon thread may outlive learning so a final comparison can
            # finish while the local backend remains open.
            evaluation_manager = ArenaManager(store, maximum_concurrent_jobs=1, recover=False)
        _schedule_evaluation(
            manager=evaluation_manager,
            store=store,
            run_id=run_id,
            checkpoint=checkpoint,
            config=config,
            plan=plan,
        )

    def set_phase(phase: str) -> None:
        status = "paused" if phase == "paused" else "running"
        store.update_run(run_id, status=status, phase=phase)

    def emit(force: bool = False, phase: str = "self_play+learning") -> None:
        nonlocal last_metric_at, metric_seq
        now = time.monotonic()
        if not force and now - last_metric_at < config.metrics_interval_seconds:
            return
        rate_values = rate.sample(totals.games, totals.decisions)
        replay_metrics = replay.metrics()
        replay_metrics["preferences"] = preference_replay.metrics()
        active_elapsed = previous_active_elapsed + max(
            0.0, time.time() - session_started_wall - paused_seconds
        )
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
            "epsilon": _epsilon(config, totals.games),
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

    # A run may resume just after a checkpoint was written but before its
    # evaluation was scheduled (for example after upgrading the backend).
    # Reconsider the latest immutable candidate without duplicating an
    # already-persisted trainer job.
    if latest is not None:
        maybe_schedule_evaluation(latest)

    # macOS uses spawn, avoiding unsafe post-Metal forks. Actors never import
    # MLX, so each process remains a small engine/NumPy worker.
    context = mp.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=config.actor_processes, mp_context=context)
    try:
        emit(force=True, phase="initializing")
        while not control.should_stop():
            pause_started = time.monotonic()
            was_paused = control.pause_requested.is_set()
            if control.wait_if_paused(set_phase):
                final_reason = "safe stop requested"
                break
            if was_paused:
                paused_seconds += time.monotonic() - pause_started

            run = store.get_run(run_id)
            config = RunConfig.model_validate(run["config"])

            # A rejected learner never becomes the behavior policy.  When its
            # asynchronous gate completes, restore the accepted champion so a
            # regression cannot keep steering future candidates indefinitely.
            for job in _completed_trainer_evaluations(store, run_id):
                if job["id"] in processed_evaluation_jobs:
                    continue
                promotion = (job.get("result") or {}).get("promotion") or {}
                if bool(promotion.get("promoted")):
                    _mark_evaluation_disposition(store, job, "promoted")
                    processed_evaluation_jobs.add(job["id"])
                    continue
                if not config.rollback_rejected_candidates:
                    _mark_evaluation_disposition(store, job, "rollback_disabled")
                    processed_evaluation_jobs.add(job["id"])
                    continue
                current_run = store.get_run(run_id)
                if current_run.get("champion_id") != job["model_b"]:
                    # A newer evaluation has already changed the accepted
                    # champion, so this stale rejection must not roll it back.
                    _mark_evaluation_disposition(store, job, "stale")
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
            active_elapsed = previous_active_elapsed + max(
                0.0, time.time() - session_started_wall - paused_seconds
            )
            duration_seconds = config.duration_minutes * 60.0
            if active_elapsed >= duration_seconds:
                break

            _sync_league(league, store, run_id)
            _atomic_actor_export(model, spec, runtime_actor)
            rollout_actor = (
                _champion_actor_path(store, run_id, runtime_actor)
                if config.behavior_policy == "champion"
                else str(runtime_actor)
            )
            epsilon = _epsilon(config, totals.games)
            futures: dict[Future[WorkerResult], _RolloutPlan] = {}
            for _ in range(config.actor_processes):
                if (
                    len(replay) < config.replay_warmup
                    or totals.updates < config.heuristic_bootstrap_updates
                ):
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
                    preference_replay.extend_compact(result.preferences)
                    totals.games += result.games
                    totals.decisions += result.decisions
                    totals.samples += len(result.samples)
                    totals.player_wins = (
                        totals.player_wins[0] + result.wins[0],
                        totals.player_wins[1] + result.wins[1],
                    )
                    totals.draws += result.draws
                    totals.truncated += result.truncated
                    totals.turns += result.turns
                    totals.forced_choices += result.forced_choices
                    if plan.opponent_id is not None and plan.current_player is not None:
                        score = result.wins[plan.current_player] + 0.5 * result.draws
                        league.record(plan.opponent_id, score, result.games)
                boundary_phase = (
                    "stopping"
                    if control.should_stop()
                    else "pausing"
                    if control.pause_requested.is_set()
                    else "self_play+learning"
                )
                emit(phase=boundary_phase)

            checkpoint_due = (
                totals.games - last_checkpoint_games >= config.checkpoint_every_games
            )
            if checkpoint_due or control.consume_checkpoint():
                checkpoint = _save_checkpoint(
                    store=store,
                    run_id=run_id,
                    model=model,
                    spec=spec,
                    checkpoint_dir=checkpoint_root,
                    games=totals.games,
                    parent_id=parent_checkpoint_id,
                    champion=False,
                    reason="scheduled" if checkpoint_due else "manual",
                )
                parent_checkpoint_id = checkpoint["id"]
                last_checkpoint_games = totals.games
                _sync_league(league, store, run_id)
                maybe_schedule_evaluation(checkpoint)

            if control.should_stop():
                final_reason = "safe stop requested"
                break

        if totals.games > last_checkpoint_games or control.checkpoint_due():
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
            )
            parent_checkpoint_id = checkpoint["id"]
            maybe_schedule_evaluation(checkpoint)
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
