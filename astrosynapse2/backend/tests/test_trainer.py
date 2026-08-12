import json
import threading
from pathlib import Path

import numpy as np
import pytest
from astro2.config import RunConfig
from astro2.encoding import FAMILY_COUNT, Encoder
from astro2.league import League
from astro2.model import ModelSpec
from astro2.storage import Store
from astro2.supervisor import InvalidTransition, RunControl, Supervisor, TrainingHandle
from astro2.trainer import (
    _ActiveElapsedClock,
    _atomic_actor_export,
    _checkpoint_quality_gate,
    _epsilon,
    _evaluation_plan,
    _evaluation_retry_state,
    _expected_model_weight_shapes,
    _finish_final_evaluations,
    _last_scheduled_evaluation_games,
    _latest_loadable_checkpoint,
    _learner_resume_checkpoint,
    _learning_rate,
    _make_bootstrap_plan,
    _make_plan,
    _next_evaluation_candidate,
    _optimizer_schedule_origin,
    _persisted_active_elapsed,
    _plateau_status,
    _restore_totals,
    _save_checkpoint,
    _schedule_evaluation,
    _sync_league,
    _Totals,
    _trainer_evaluation_outcome,
    _training_state,
)
from safetensors.numpy import save_file


def _write_test_model(path: Path, config: RunConfig) -> Path:
    """Write a CPU-only model fixture matching a run's immutable architecture."""

    encoder = Encoder(version=2 if config.training_generation >= 3 else 1)
    spec = ModelSpec(
        state_size=encoder.state_size,
        action_size=encoder.action_size,
        families=FAMILY_COUNT,
        hidden_size=config.hidden_size,
        action_hidden_size=max(64, config.hidden_size // 2),
        residual_blocks=config.residual_blocks,
        bootstrap_heads=config.bootstrap_heads,
        encoder_version=encoder.version,
    )
    arrays = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in _expected_model_weight_shapes(spec).items()
    }
    save_file(arrays, str(path))
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(spec.as_dict()),
        encoding="utf-8",
    )
    return path


class FakeArenaManager:
    def __init__(self):
        self.calls = []

    def create(self, model_a, model_b, config, **_options):
        self.calls.append(("manual", model_a, model_b, config))
        return {"id": "quick-job"}

    def create_automatic(self, model_a, model_b, **config):
        self.calls.append(("automatic", model_a, model_b, config))
        return {"id": "automatic-job"}


def test_runtime_actor_export_is_uncompressed_and_atomic(tmp_path, monkeypatch):
    target = tmp_path / "runtime" / "current.actor.npz"
    observed = {}

    def fake_export(_model, _spec, path, *, compressed):
        observed.update(path=path, compressed=compressed)
        path.write_bytes(b"complete actor")

    monkeypatch.setattr("astro2.trainer.export_actor", fake_export)
    assert _atomic_actor_export(object(), object(), target) == target

    assert observed == {
        "path": target.with_name("current.actor.partial.npz"),
        "compressed": False,
    }
    assert target.read_bytes() == b"complete actor"
    assert not observed["path"].exists()


def test_same_game_same_nanosecond_checkpoints_never_share_artifact_paths(
    tmp_path,
    monkeypatch,
):
    store = Store(tmp_path / "state.sqlite3")
    run = store.create_run(RunConfig.quick())
    checkpoint_dir = tmp_path / "checkpoints" / run["id"]

    def fake_save_model(model, _spec, path):
        Path(path).write_text(f"model:{model}")
        Path(f"{path}.json").write_text("{}")

    def fake_export_actor(model, _spec, path):
        Path(path).write_text(f"actor:{model}")

    monkeypatch.setattr("astro2.trainer.time.time_ns", lambda: 1_700_000_000_000_000_000)
    monkeypatch.setattr("astro2.trainer.save_model", fake_save_model)
    monkeypatch.setattr("astro2.trainer.export_actor", fake_export_actor)
    spec = ModelSpec(state_size=1, action_size=1, families=1)

    first = _save_checkpoint(
        store=store,
        run_id=run["id"],
        model="first",
        spec=spec,
        checkpoint_dir=checkpoint_dir,
        games=123,
        parent_id=None,
        champion=False,
        reason="first",
    )
    second = _save_checkpoint(
        store=store,
        run_id=run["id"],
        model="second",
        spec=spec,
        checkpoint_dir=checkpoint_dir,
        games=123,
        parent_id=None,
        champion=False,
        reason="second",
    )

    assert first["path"] != second["path"]
    assert first["actor_path"] != second["actor_path"]
    assert Path(first["path"]).read_text() == "model:first"
    assert Path(second["path"]).read_text() == "model:second"
    assert Path(first["actor_path"]).read_text() == "actor:first"
    assert Path(second["actor_path"]).read_text() == "actor:second"


def _checkpoints(store, run, tmp_path):
    champion = store.add_checkpoint(
        run_id=run["id"],
        label="Champion",
        path=str(tmp_path / "champion.safetensors"),
        actor_path=str(tmp_path / "champion.actor.npz"),
        games=0,
        champion=True,
    )
    candidate = store.add_checkpoint(
        run_id=run["id"],
        label="Candidate",
        path=str(tmp_path / "candidate.safetensors"),
        actor_path=str(tmp_path / "candidate.actor.npz"),
        games=2_000,
    )
    store.update_run(run["id"], champion_id=champion["id"])
    return champion, candidate


def test_exploration_schedule_reaches_configured_floor():
    config = RunConfig.quick()
    assert _epsilon(config, 0) == pytest.approx(
        config.epsilon_start * config.exploration_decision_scale
    )
    assert _epsilon(config, config.epsilon_decay_games) == pytest.approx(
        config.epsilon_end * config.exploration_decision_scale
    )
    assert _epsilon(config, config.epsilon_decay_games * 2) == pytest.approx(
        config.epsilon_end * config.exploration_decision_scale
    )


def test_astro3_recipe_repairs_policy_iteration_and_exploration_defaults():
    config = RunConfig.astro3_m4()
    assert config.training_generation == 3
    assert config.behavior_policy == "learner"
    assert config.rollback_rejected_candidates is False
    assert config.use_bootstrap_targets is False
    assert config.terminal_target_weight == 1.0
    assert config.tactical_preference_training is False
    assert config.exploration_top_k == 0
    assert config.deployment_policy_selfplay_fraction == pytest.approx(0.20)
    assert _epsilon(config, config.epsilon_decay_games) == pytest.approx(0.05)
    assert _epsilon(config, config.epsilon_decay_games, 4.0) == pytest.approx(0.20)
    assert config.persist_optimizer_state is True
    assert config.resume_replay_items > 0
    assert config.gate_baseline_regression is False
    assert config.gate_heldout_brier_regression is False


def test_heldout_brier_gate_migrates_by_training_generation():
    legacy = RunConfig.model_validate({"training_generation": 2})
    astro3 = RunConfig.model_validate({"training_generation": 3})
    explicit = RunConfig.model_validate(
        {
            "training_generation": 3,
            "gate_baseline_regression": True,
            "gate_heldout_brier_regression": True,
        }
    )

    assert legacy.gate_baseline_regression is True
    assert legacy.gate_heldout_brier_regression is True
    assert astro3.gate_baseline_regression is False
    assert astro3.gate_heldout_brier_regression is False
    assert explicit.gate_baseline_regression is True
    assert explicit.gate_heldout_brier_regression is True


def test_fresh_optimizer_repeats_learning_rate_warmup():
    config = RunConfig.quick()
    first = _learning_rate(config, completed_updates=25_000, updates_since_optimizer_reset=0)
    warmed = _learning_rate(config, completed_updates=25_000, updates_since_optimizer_reset=500)
    assert first < warmed


def test_learning_rate_does_not_rewind_when_duration_changes():
    config = RunConfig.quick()
    extended = config.model_copy(update={"duration_minutes": config.duration_minutes * 4})
    original = _learning_rate(config, completed_updates=30_000, updates_since_optimizer_reset=1_000)
    after_extension = _learning_rate(
        extended, completed_updates=30_000, updates_since_optimizer_reset=1_000
    )
    assert after_extension == original


def test_astro3_cosine_restart_escapes_a_permanent_learning_rate_floor():
    config = RunConfig.astro3_m4()
    before = _learning_rate(
        config,
        completed_updates=config.learning_rate_restart_updates - 1,
        updates_since_optimizer_reset=10_000,
    )
    restarted = _learning_rate(
        config,
        completed_updates=config.learning_rate_restart_updates,
        updates_since_optimizer_reset=10_000,
    )
    assert restarted > before * 2


def test_replay_warmup_plan_collects_both_heuristic_players():
    plan = _make_bootstrap_plan(config=RunConfig.quick(), rng=np.random.default_rng(3), seed=99)
    assert plan.actor_paths == (None, None)
    assert plan.collect_players == (True, True)
    assert plan.epsilons == (0.0, 0.0)
    assert plan.kind == "heuristic_bootstrap"


def test_astro3_collects_a_configured_share_with_exact_deployment_policy():
    config = RunConfig.astro3_m4().model_copy(
        update={
            "current_selfplay_fraction": 1.0,
            "deployment_policy_selfplay_fraction": 1.0,
            "league_fraction": 0.0,
            "baseline_fraction": 0.0,
        }
    )
    plan = _make_plan(
        config=config,
        rng=np.random.default_rng(11),
        league=League(),
        current_actor="learner.actor.npz",
        epsilon=0.17,
        seed=91,
    )

    assert plan.kind == "deployment_self_play"
    assert plan.actor_paths == ("learner.actor.npz", "learner.actor.npz")
    assert plan.collect_players == (True, True)
    assert plan.deployment_policy == (True, True)
    assert plan.epsilons == (0.0, 0.0)


def test_astro2_rollout_contract_never_uses_deployment_collection_mode():
    config = RunConfig().model_copy(
        update={
            "current_selfplay_fraction": 1.0,
            "league_fraction": 0.0,
            "baseline_fraction": 0.0,
        }
    )
    plan = _make_plan(
        config=config,
        rng=np.random.default_rng(12),
        league=League(),
        current_actor="champion.actor.npz",
        epsilon=0.02,
        seed=92,
    )

    assert plan.kind == "self_play"
    assert plan.deployment_policy == (False, False)
    assert plan.epsilons == (0.02, 0.02)


def test_deployment_rollouts_are_generation_three_only():
    with pytest.raises(ValueError, match="requires training_generation=3"):
        RunConfig(deployment_policy_selfplay_fraction=0.2)


def test_zero_game_random_checkpoint_is_not_added_to_league(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    run = store.create_run(RunConfig.quick())
    actor = tmp_path / "actor.npz"
    actor.touch()
    store.add_checkpoint(
        run_id=run["id"],
        label="Initial random model",
        path=str(tmp_path / "initial.safetensors"),
        actor_path=str(actor),
        games=0,
        champion=True,
    )
    league = League()
    _sync_league(league, store, run["id"])
    assert league.opponents == []


def test_only_accepted_nonzero_checkpoint_is_added_to_league(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    run = store.create_run(RunConfig.quick())
    actor = tmp_path / "actor.npz"
    actor.touch()
    checkpoint = store.add_checkpoint(
        run_id=run["id"],
        label="Candidate",
        path=str(tmp_path / "candidate.safetensors"),
        actor_path=str(actor),
        games=1_000,
    )
    store.update_checkpoint_evaluation(
        checkpoint["id"],
        {"latest_arena": {"promoted": True}},
    )
    league = League()
    _sync_league(league, store, run["id"])
    assert [opponent.id for opponent in league.opponents] == [checkpoint["id"]]


def test_rejected_checkpoint_is_not_added_to_league(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    run = store.create_run(RunConfig.quick())
    actor = tmp_path / "rejected.npz"
    actor.touch()
    checkpoint = store.add_checkpoint(
        run_id=run["id"],
        label="Rejected candidate",
        path=str(tmp_path / "rejected.safetensors"),
        actor_path=str(actor),
        games=1_000,
        evaluation={"latest_arena": {"promoted": False}},
    )
    league = League()
    _sync_league(league, store, run["id"])
    assert checkpoint["id"] not in {opponent.id for opponent in league.opponents}


def test_only_astro3_league_uses_compatible_cross_run_frozen_anchors(
    tmp_path,
    monkeypatch,
):
    store = Store(tmp_path / "state.sqlite3")
    legacy_run = store.create_run(RunConfig(name="Legacy run"))
    astro3_run = store.create_run(RunConfig.quick(name="Astro3 run"))

    champion_actor = tmp_path / "legacy-champion.actor.npz"
    pinned_actor = tmp_path / "pinned.actor.npz"
    ignored_actor = tmp_path / "ignored.actor.npz"
    for path in (champion_actor, pinned_actor, ignored_actor):
        path.touch()
    champion = store.add_checkpoint(
        run_id=legacy_run["id"],
        label="Legacy champion",
        path=str(tmp_path / "legacy.safetensors"),
        actor_path=str(champion_actor),
        games=4_000_000,
        champion=True,
    )
    pinned = store.add_checkpoint(
        run_id=legacy_run["id"],
        label="Pinned candidate",
        path=str(tmp_path / "pinned.safetensors"),
        actor_path=str(pinned_actor),
        games=3_500_000,
    )
    store.set_checkpoint_pinned(pinned["id"], True)
    store.add_checkpoint(
        run_id=legacy_run["id"],
        label="Ordinary rejected candidate",
        path=str(tmp_path / "ignored.safetensors"),
        actor_path=str(ignored_actor),
        games=3_000_000,
    )
    monkeypatch.setattr(
        "astro2.trainer._compatible_external_actor_path",
        lambda path: str(path) if path else None,
    )

    legacy_semantics = League()
    _sync_league(legacy_semantics, store, astro3_run["id"])
    assert legacy_semantics.opponents == []

    astro3_league = League()
    _sync_league(
        astro3_league,
        store,
        astro3_run["id"],
        include_external_anchors=True,
    )
    assert {opponent.id for opponent in astro3_league.opponents} == {
        champion["id"],
        pinned["id"],
    }
    assert {opponent.kind for opponent in astro3_league.opponents} == {"anchor"}
    assert next(item for item in astro3_league.opponents if item.id == pinned["id"]).pinned

    league_only = RunConfig.quick().model_copy(
        update={
            "current_selfplay_fraction": 0.0,
            "league_fraction": 1.0,
            "baseline_fraction": 0.0,
        }
    )
    plan = _make_plan(
        config=league_only,
        rng=np.random.default_rng(4),
        league=astro3_league,
        current_actor="current.actor.npz",
        epsilon=0.1,
        seed=55,
    )
    assert plan.kind == "league"
    assert plan.opponent_id in {champion["id"], pinned["id"]}

    store.set_checkpoint_pinned(pinned["id"], False)
    _sync_league(
        astro3_league,
        store,
        astro3_run["id"],
        include_external_anchors=True,
    )
    assert [opponent.id for opponent in astro3_league.opponents] == [champion["id"]]


def test_active_elapsed_resumes_without_counting_backend_downtime(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    run = store.create_run(RunConfig.quick())
    assert _persisted_active_elapsed(store, run["id"]) == 0.0
    store.append_metric(run["id"], 1, {"active_elapsed_seconds": 12.5})
    assert _persisted_active_elapsed(store, run["id"]) == 12.5
    checkpoint = store.add_checkpoint(
        run_id=run["id"],
        label="durable",
        path=str(tmp_path / "model.safetensors"),
        actor_path=None,
        games=5,
        evaluation={"training_state": {"active_elapsed_seconds": 7.25}},
    )
    assert _persisted_active_elapsed(store, run["id"], checkpoint) == 7.25


def test_initial_checkpoint_elapsed_is_authoritative_over_newer_metrics(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    run = store.create_run(RunConfig.quick())
    store.append_metric(run["id"], 1, {"active_elapsed_seconds": 3_600.0})
    checkpoint = store.add_checkpoint(
        run_id=run["id"],
        label="initial",
        path=str(tmp_path / "model.safetensors"),
        actor_path=None,
        games=0,
        evaluation={"training_state": {"games": 0, "active_elapsed_seconds": 0.0}},
    )
    assert _persisted_active_elapsed(store, run["id"], checkpoint) == 0.0


def test_cumulative_quality_counters_survive_run_resume(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    run = store.create_run(RunConfig.quick())
    store.update_run(run["id"], games=90, decisions=8_000, updates=70)
    store.append_metric(
        run["id"],
        1,
        {
            "games": 100,
            "decisions": 9_000,
            "updates": 80,
            "samples": 12_000,
            "player_0_wins": 46,
            "player_1_wins": 44,
            "draws": 10,
            "truncations": 3,
            "mean_turns": 22.5,
            "forced_choices": 700,
        },
    )
    totals = _restore_totals(store, store.get_run(run["id"]))
    assert (totals.games, totals.decisions, totals.updates, totals.samples) == (
        100,
        9_000,
        80,
        12_000,
    )
    assert totals.player_wins == (46, 44)
    assert (totals.draws, totals.truncated, totals.turns, totals.forced_choices) == (
        10,
        3,
        2_250,
        700,
    )


def test_checkpoint_training_state_is_authoritative_on_resume(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    run = store.create_run(RunConfig.quick())
    store.update_run(run["id"], games=500, decisions=50_000, updates=400)
    state = _training_state(
        _Totals(
            games=400,
            decisions=38_000,
            updates=320,
            samples=36_000,
            turns=8_801,
            rollout_games={"self_play": 300, "league": 100},
        ),
        seed_cursor=77,
        optimizer_updates_at_start=0,
    )
    checkpoint = store.add_checkpoint(
        run_id=run["id"],
        label="durable learner",
        path=str(tmp_path / "model.safetensors"),
        actor_path=str(tmp_path / "model.actor.npz"),
        games=400,
        evaluation={"training_state": state},
    )
    totals = _restore_totals(store, store.get_run(run["id"]), checkpoint)
    assert (totals.games, totals.decisions, totals.updates) == (400, 38_000, 320)
    assert totals.turns == 8_801
    assert totals.rollout_games == {"self_play": 300, "league": 100}


def test_partial_checkpoint_training_state_cannot_zero_mature_counters(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    run = store.create_run(RunConfig.quick())
    store.update_run(run["id"], games=500, decisions=50_000, updates=400)
    checkpoint = store.add_checkpoint(
        run_id=run["id"],
        label="legacy partial payload",
        path=str(tmp_path / "model.safetensors"),
        actor_path=None,
        games=400,
        evaluation={
            "training_state": {
                "games": 400,
                "active_elapsed_seconds": 12.0,
            }
        },
    )

    totals = _restore_totals(store, store.get_run(run["id"]), checkpoint)
    assert (totals.games, totals.decisions, totals.updates) == (500, 50_000, 400)


def test_legacy_weight_only_rollback_keeps_mature_counters(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    run = store.create_run(RunConfig.quick())
    store.update_run(run["id"], games=500, decisions=50_000, updates=400)
    champion = store.add_checkpoint(
        run_id=run["id"],
        label="old champion",
        path=str(tmp_path / "champion.safetensors"),
        actor_path=None,
        games=100,
    )
    totals = _restore_totals(store, store.get_run(run["id"]), champion)
    assert (totals.games, totals.decisions, totals.updates) == (500, 50_000, 400)


def test_extended_training_state_is_json_serializable():
    totals = _Totals(games=0, decisions=0, updates=0)
    # Use real PCG64 payloads: this is the exact shape persisted by NumPy.
    rng = np.random.default_rng(73)
    payload = _training_state(
        totals,
        seed_cursor=99,
        optimizer_updates_at_start=7,
        active_elapsed_seconds=12.5,
        rollout_rng_state=rng.bit_generator.state,
        replay_rng_state=np.random.default_rng(74).bit_generator.state,
        league_state=[{"id": "baseline:balanced", "wins": 2.5, "games": 4}],
    )
    restored = json.loads(json.dumps(payload))
    assert restored["schema_version"] == 2
    assert restored["seed_cursor"] == 99
    assert restored["optimizer_updates_at_start"] == 7


def test_optimizer_warmup_origin_round_trips_or_restarts_after_state_loss():
    state = {"optimizer_updates_at_start": 20_000}
    assert (
        _optimizer_schedule_origin(
            completed_updates=25_000,
            training_state=state,
            optimizer_restored=True,
        )
        == 20_000
    )
    assert (
        _optimizer_schedule_origin(
            completed_updates=25_000,
            training_state=state,
            optimizer_restored=False,
        )
        == 25_000
    )
    # Legacy optimizer snapshots did not record their reset boundary; retain
    # their historical assumption instead of inventing a later reset.
    assert (
        _optimizer_schedule_origin(
            completed_updates=25_000,
            training_state={},
            optimizer_restored=True,
        )
        == 0
    )


def test_resume_skips_an_artifact_that_was_already_rolled_back(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    config = RunConfig.quick().model_copy(update={"rollback_rejected_candidates": True})
    run = store.create_run(config)
    champion_path = tmp_path / "champion.safetensors"
    candidate_path = tmp_path / "candidate.safetensors"
    for path in (champion_path, candidate_path):
        _write_test_model(path, config)
    champion = store.add_checkpoint(
        run_id=run["id"],
        label="champion",
        path=str(champion_path),
        actor_path=None,
        games=100,
        champion=True,
    )
    candidate = store.add_checkpoint(
        run_id=run["id"],
        label="rejected",
        path=str(candidate_path),
        actor_path=None,
        games=200,
    )
    store.update_run(run["id"], champion_id=champion["id"])
    job = store.create_arena_job(
        model_a=candidate["id"],
        model_b=champion["id"],
        config={"automatic_promotion": True, "trainer_scheduled": True},
        result={
            "promotion": {"promoted": False},
            "_trainer_disposition_processed": True,
            "_trainer_disposition": "rolled_back",
        },
    )
    store.update_arena_job(job["id"], status="complete")
    assert _learner_resume_checkpoint(store, run["id"], config)["id"] == champion["id"]


def test_resume_falls_back_to_prior_complete_durable_checkpoint(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    config = RunConfig.quick()
    run = store.create_run(config)

    def model_files(name):
        path = tmp_path / f"{name}.safetensors"
        return _write_test_model(path, config)

    optimizer_path = tmp_path / "complete.optimizer.npz"
    np.savez(
        optimizer_path,
        __paths_json__=np.frombuffer(b"[]", dtype=np.uint8),
    )
    replay_path = tmp_path / "complete.replay.npz"
    empty = np.empty(0, dtype=np.float32)
    np.savez(
        replay_path,
        states=empty,
        actions=empty,
        families=empty,
        targets=empty,
        bootstrap_masks=empty,
        game_ids=empty,
        players=empty,
        steps=empty,
        heads=empty,
        epsilons=empty,
        td_targets=empty,
        td_valid=empty,
        sequences=empty,
        sequence_cursor=np.asarray(0, dtype=np.uint64),
    )
    complete = store.add_checkpoint(
        run_id=run["id"],
        label="complete",
        path=str(model_files("complete")),
        actor_path=None,
        games=1_000,
        evaluation={
            "artifacts": {
                "optimizer_path": str(optimizer_path),
                "replay_path": str(replay_path),
                "replay_items": 1,
            }
        },
    )
    broken_replay = tmp_path / "broken.replay.npz"
    broken_replay.write_bytes(b"not a zip archive")
    newest = store.add_checkpoint(
        run_id=run["id"],
        label="interrupted",
        path=str(model_files("interrupted")),
        actor_path=None,
        games=2_000,
        evaluation={
            "artifacts": {
                "optimizer_path": str(optimizer_path),
                "replay_path": str(broken_replay),
                "replay_items": 1,
            }
        },
    )

    selected = _latest_loadable_checkpoint(store, run["id"], config)
    assert selected["id"] == complete["id"]
    assert selected["_resume_artifacts_complete"] is True
    assert selected["_resume_skipped_checkpoint_ids"] == [newest["id"]]


@pytest.mark.parametrize("corruption", ["archive", "spec"])
def test_resume_falls_back_past_corrupt_model_files(tmp_path, corruption):
    store = Store(tmp_path / "state.sqlite3")
    config = RunConfig.quick().model_copy(
        update={
            "hidden_size": 64,
            "residual_blocks": 1,
            "bootstrap_heads": 1,
            "persist_optimizer_state": False,
            "resume_replay_items": 0,
        }
    )
    run = store.create_run(config)
    prior_path = _write_test_model(tmp_path / "prior.safetensors", config)
    prior = store.add_checkpoint(
        run_id=run["id"],
        label="prior complete",
        path=str(prior_path),
        actor_path=None,
        games=1_000,
    )
    newest_path = _write_test_model(tmp_path / "newest.safetensors", config)
    if corruption == "archive":
        newest_path.write_bytes(b"truncated safetensors archive")
    else:
        sidecar = newest_path.with_suffix(newest_path.suffix + ".json")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload["state_size"] += 1
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
    newest = store.add_checkpoint(
        run_id=run["id"],
        label="corrupt newest",
        path=str(newest_path),
        actor_path=None,
        games=2_000,
    )

    selected = _latest_loadable_checkpoint(store, run["id"], config)

    assert selected["id"] == prior["id"]
    assert selected["_resume_artifacts_complete"] is True
    assert selected["_resume_skipped_checkpoint_ids"] == [newest["id"]]


def test_stopping_thread_keeps_exclusive_trainer_slot(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    supervisor = Supervisor(store, tmp_path)
    release = threading.Event()
    thread = threading.Thread(target=release.wait)
    control = RunControl()
    thread.start()
    try:
        control.stop_requested.set()
        supervisor._handles["stopping-run"] = TrainingHandle(
            run_id="stopping-run",
            control=control,
            thread=thread,
        )
        assert supervisor.active_run_id() == "stopping-run"
        with pytest.raises(InvalidTransition, match="still stopping"):
            supervisor.start("stopping-run")
        with pytest.raises(InvalidTransition, match="still stopping"):
            supervisor.resume("stopping-run")
    finally:
        release.set()
        thread.join()


def test_paused_control_services_durable_requests_before_announcing_paused():
    control = RunControl()
    log: list[str] = []
    first_saved = threading.Event()
    second_saved = threading.Event()

    def service() -> bool:
        if not control.consume_checkpoint():
            return False
        log.append("checkpoint")
        (first_saved if log.count("checkpoint") == 1 else second_saved).set()
        return True

    def state(value: str) -> None:
        log.append(value)

    # Match Supervisor.pause ordering: request durability before exposing the
    # pause flag to the trainer boundary.
    control.checkpoint_requested.set()
    control.pause_requested.set()
    thread = threading.Thread(target=control.wait_if_paused, args=(state, service))
    thread.start()
    try:
        assert first_saved.wait(1.0)
        assert log[:2] == ["checkpoint", "paused"]

        # A checkpoint button pressed while already paused is serviced without
        # requiring a resume and returns the phase to paused afterwards.
        control.checkpoint_requested.set()
        assert second_saved.wait(1.0)
        assert log[-2:] == ["checkpoint", "paused"]
    finally:
        control.pause_requested.clear()
        thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert log[-1] == "running"


def test_manual_checkpoint_while_paused_does_not_consume_training_time():
    now = {"value": 100.0}
    clock = _ActiveElapsedClock(7.5, now=lambda: now["value"])
    control = RunControl()
    persisted_elapsed: list[float] = []
    first_saved = threading.Event()
    second_saved = threading.Event()

    def service() -> bool:
        if not control.consume_checkpoint():
            return False
        persisted_elapsed.append(clock.value())
        (first_saved if len(persisted_elapsed) == 1 else second_saved).set()
        return True

    control.checkpoint_requested.set()
    control.pause_requested.set()
    thread = threading.Thread(
        target=control.wait_if_paused,
        args=(None, service, clock.pause, clock.resume),
    )
    thread.start()
    try:
        assert first_saved.wait(1.0)
        assert persisted_elapsed == [7.5]

        # A long wall-clock pause must not leak into a later durable snapshot.
        now["value"] += 3_600.0
        control.checkpoint_requested.set()
        assert second_saved.wait(1.0)
        assert persisted_elapsed == [7.5, 7.5]
    finally:
        control.pause_requested.clear()
        thread.join(timeout=1.0)
    assert not thread.is_alive()

    now["value"] += 10.0
    assert clock.value() == 17.5


def test_quick_evaluation_is_paired_but_never_automatic(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    run = store.create_run(RunConfig.quick())
    champion, candidate = _checkpoints(store, run, tmp_path)
    manager = FakeArenaManager()
    job = _schedule_evaluation(
        manager=manager,
        store=store,
        run_id=run["id"],
        checkpoint=candidate,
        config=RunConfig.quick(),
    )
    assert job["id"] == "quick-job"
    kind, model_a, model_b, config = manager.calls[0]
    assert kind == "manual"
    assert model_a == candidate["id"]
    assert model_b == champion["id"]
    assert config.pairs == 16
    assert config.automatic_promotion is False
    assert config.trainer_scheduled is True


def test_adaptive_evaluation_grows_from_provisional_to_full():
    config = RunConfig()
    provisional = _evaluation_plan(config, 100_000)
    assert (provisional.tier, provisional.cadence_games, provisional.pairs) == (
        "provisional",
        100_000,
        200,
    )
    assert provisional.automatic_promotion is True

    development = _evaluation_plan(config, 500_000)
    assert (development.tier, development.cadence_games, development.pairs) == (
        "development",
        250_000,
        1_000,
    )

    full = _evaluation_plan(config, 1_000_000)
    assert (full.tier, full.cadence_games, full.pairs) == (
        "full",
        500_000,
        5_000,
    )


def test_provisional_evaluation_can_promote_with_its_recorded_gate(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    config = RunConfig()
    run = store.create_run(config)
    champion, candidate = _checkpoints(store, run, tmp_path)
    manager = FakeArenaManager()
    job = _schedule_evaluation(
        manager=manager,
        store=store,
        run_id=run["id"],
        checkpoint=candidate,
        config=config,
    )
    assert job["id"] == "automatic-job"
    kind, model_a, model_b, options = manager.calls[0]
    assert kind == "automatic"
    assert (model_a, model_b) == (candidate["id"], champion["id"])
    assert options["pairs"] == 200
    assert options["minimum_promotion_pairs"] == 200
    assert options["promotion_tier"] == "provisional"


def test_small_evaluation_tier_disables_rather_than_moves_configured_early_look(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    config = RunConfig().model_copy(
        update={
            "evaluation_early_rejection": True,
            "evaluation_early_rejection_min_pairs": 512,
        }
    )
    run = store.create_run(config)
    _champion, candidate = _checkpoints(store, run, tmp_path)
    manager = FakeArenaManager()

    _schedule_evaluation(
        manager=manager,
        store=store,
        run_id=run["id"],
        checkpoint=candidate,
        config=config,
    )

    options = manager.calls[0][3]
    assert options["pairs"] == 200
    assert options["early_rejection"] is False
    assert options["early_rejection_min_pairs"] == 512


def test_full_evaluation_uses_conservative_automatic_gate(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    config = RunConfig(adaptive_evaluation=False)
    run = store.create_run(config)
    champion, candidate = _checkpoints(store, run, tmp_path)
    manager = FakeArenaManager()
    job = _schedule_evaluation(
        manager=manager,
        store=store,
        run_id=run["id"],
        checkpoint=candidate,
        config=config,
    )
    assert job["id"] == "automatic-job"
    kind, model_a, model_b, options = manager.calls[0]
    assert kind == "automatic"
    assert (model_a, model_b) == (candidate["id"], champion["id"])
    assert options["pairs"] == 5_000

    persisted = store.create_arena_job(
        model_a=candidate["id"],
        model_b=champion["id"],
        config={
            "automatic_promotion": True,
            "trainer_scheduled": True,
            "promotion_tier": "full",
        },
    )
    assert _last_scheduled_evaluation_games(store, run["id"]) == 0
    store.update_arena_job(
        persisted["id"],
        status="complete",
        result={"promotion": {"eligible": True, "promoted": False}},
    )
    assert _last_scheduled_evaluation_games(store, run["id"]) == candidate["games"]
    assert _last_scheduled_evaluation_games(store, run["id"], tier="full") == candidate["games"]
    assert _last_scheduled_evaluation_games(store, run["id"], tier="provisional") == 0


def test_manual_arena_does_not_delay_trainer_evaluation_schedule(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    run = store.create_run(RunConfig.quick())
    champion, candidate = _checkpoints(store, run, tmp_path)
    store.create_arena_job(
        model_a=candidate["id"],
        model_b=champion["id"],
        config={"automatic_promotion": False, "trainer_scheduled": False},
    )
    assert _last_scheduled_evaluation_games(store, run["id"]) == 0


def test_completed_evaluation_releases_newest_due_checkpoint_once(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    config = RunConfig.quick()
    run = store.create_run(config)
    champion, candidate = _checkpoints(store, run, tmp_path)
    newest = store.add_checkpoint(
        run_id=run["id"],
        label="Newest candidate",
        path=str(tmp_path / "newest.safetensors"),
        actor_path=str(tmp_path / "newest.actor.npz"),
        games=4_000,
    )
    active = store.create_arena_job(
        model_a=candidate["id"],
        model_b=champion["id"],
        config={
            "automatic_promotion": False,
            "trainer_scheduled": True,
            "promotion_tier": "diagnostic",
        },
    )

    assert _next_evaluation_candidate(store, run["id"], config) is None
    store.update_arena_job(active["id"], status="complete", result={})

    due = _next_evaluation_candidate(store, run["id"], config)
    assert due is not None
    assert due[0]["id"] == newest["id"]
    assert due[1].tier == "diagnostic"
    assert due[1].cadence_games == 2_000

    store.create_arena_job(
        model_a=newest["id"],
        model_b=champion["id"],
        config={
            "automatic_promotion": False,
            "trainer_scheduled": True,
            "promotion_tier": "diagnostic",
        },
    )
    assert _next_evaluation_candidate(store, run["id"], config) is None


def test_invalid_evaluations_are_retryable_and_do_not_create_a_false_plateau(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    config = RunConfig.quick().model_copy(
        update={"adaptive_training": True, "plateau_patience_evaluations": 1}
    )
    run = store.create_run(config)
    champion, candidate = _checkpoints(store, run, tmp_path)
    common = {
        "model_a": candidate["id"],
        "model_b": champion["id"],
        "config": {
            "automatic_promotion": True,
            "trainer_scheduled": True,
            "promotion_tier": "diagnostic",
        },
    }

    promoted = store.create_arena_job(**common)
    store.update_arena_job(
        promoted["id"],
        status="complete",
        result={"promotion": {"eligible": True, "promoted": True}},
    )
    truncated = store.create_arena_job(**common)
    store.update_arena_job(
        truncated["id"],
        status="complete",
        result={
            "truncated_games": 1,
            "promotion": {"eligible": False, "promoted": False},
        },
    )
    stale = store.create_arena_job(**common)
    store.update_arena_job(
        stale["id"],
        status="complete",
        result={
            "promotion": {
                "eligible": True,
                "promoted": False,
                "stale_opponent": True,
            }
        },
    )
    failed = store.create_arena_job(**common)
    store.update_arena_job(failed["id"], status="failed", error="worker crashed")
    cancelled = store.create_arena_job(**common)
    store.update_arena_job(cancelled["id"], status="cancelled")
    rejected = store.create_arena_job(**common)
    store.update_arena_job(
        rejected["id"],
        status="complete",
        result={"promotion": {"eligible": True, "promoted": False}},
    )

    assert (
        _trainer_evaluation_outcome(
            truncated
            | {
                "status": "complete",
                "result": {
                    "truncated_games": 1,
                    "promotion": {"eligible": False, "promoted": False},
                },
            }
        )
        == "truncated"
    )
    plateau = _plateau_status(store, run["id"], config)
    assert plateau["consecutive_non_promotions"] == 1
    assert plateau["level"] == 1

    retry = _evaluation_retry_state(
        store,
        run["id"],
        candidate["id"],
        "diagnostic",
    )
    # The latest comparison is valid, so earlier infrastructure failures no
    # longer impose backoff on this immutable checkpoint.
    assert retry == {"ready": True, "attempts": 0, "reason": None, "retry_at": None}


def test_failed_job_does_not_consume_cadence_and_retry_uses_a_new_seed(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    config = RunConfig.quick()
    run = store.create_run(config)
    champion, candidate = _checkpoints(store, run, tmp_path)
    first_manager = FakeArenaManager()
    _schedule_evaluation(
        manager=first_manager,
        store=store,
        run_id=run["id"],
        checkpoint=candidate,
        config=config,
    )
    first_seed = first_manager.calls[0][3].seed
    assert first_seed == config.seed + candidate["games"]

    failed = store.create_arena_job(
        model_a=candidate["id"],
        model_b=champion["id"],
        config={
            "automatic_promotion": False,
            "trainer_scheduled": True,
            "promotion_tier": "diagnostic",
        },
    )
    store.update_arena_job(failed["id"], status="failed", error="inference error")
    assert _last_scheduled_evaluation_games(store, run["id"]) == 0
    assert _next_evaluation_candidate(store, run["id"], config) is None
    assert (
        _next_evaluation_candidate(
            store,
            run["id"],
            config,
            ignore_retry_backoff=True,
        )
        is not None
    )

    retry_manager = FakeArenaManager()
    _schedule_evaluation(
        manager=retry_manager,
        store=store,
        run_id=run["id"],
        checkpoint=candidate,
        config=config,
    )
    retry_seed = retry_manager.calls[0][3].seed
    assert retry_seed != first_seed


def test_natural_completion_drains_old_job_then_schedules_and_drains_newest(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    config = RunConfig.quick()
    run = store.create_run(config)
    champion, candidate = _checkpoints(store, run, tmp_path)
    newest = store.add_checkpoint(
        run_id=run["id"],
        label="Final candidate",
        path=str(tmp_path / "final.safetensors"),
        actor_path=str(tmp_path / "final.actor.npz"),
        games=4_000,
    )
    store.create_arena_job(
        model_a=candidate["id"],
        model_b=champion["id"],
        config={
            "automatic_promotion": False,
            "trainer_scheduled": True,
            "promotion_tier": "diagnostic",
        },
    )
    unrelated = store.create_arena_job(
        model_a="baseline:balanced",
        model_b="baseline:economy",
        config={"automatic_promotion": False},
    )

    class DrainingManager:
        def __init__(self):
            self.waits = 0

        def wait_for_job(self, job_id, timeout=None):
            self.waits += 1
            del timeout
            store.update_arena_job(job_id, status="complete", result={})
            return True

        def cancel(self, job_id):
            store.update_arena_job(job_id, status="cancelled")
            return True

    manager = DrainingManager()

    def schedule_latest():
        due = _next_evaluation_candidate(
            store,
            run["id"],
            config,
            ignore_retry_backoff=True,
        )
        if due is None:
            return None
        checkpoint, plan = due
        return store.create_arena_job(
            model_a=checkpoint["id"],
            model_b=champion["id"],
            config={
                "automatic_promotion": False,
                "trainer_scheduled": True,
                "promotion_tier": plan.tier,
            },
        )

    result = _finish_final_evaluations(
        store=store,
        run_id=run["id"],
        manager_getter=lambda: manager,
        schedule_latest=schedule_latest,
        process_completed=lambda: None,
    )

    assert result["status"] == "complete"
    assert manager.waits == 2
    assert len(result["scheduled_job_ids"]) == 1
    final_job = store.arena_job(result["scheduled_job_ids"][0])
    assert final_job["model_a"] == newest["id"]
    assert final_job["status"] == "complete"
    assert store.arena_job(unrelated["id"])["status"] == "queued"


def test_final_evaluation_persists_manual_checkpoint_without_cancelling_job(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    run = store.create_run(RunConfig.quick())
    champion, candidate = _checkpoints(store, run, tmp_path)
    trainer_job = store.create_arena_job(
        model_a=candidate["id"],
        model_b=champion["id"],
        config={
            "automatic_promotion": False,
            "trainer_scheduled": True,
            "promotion_tier": "diagnostic",
        },
    )
    control = RunControl()
    observed_statuses: list[str] = []

    class CheckpointingManager:
        def __init__(self):
            self.waits = 0
            self.cancelled = False

        def wait_for_job(self, job_id, timeout=None):
            assert job_id == trainer_job["id"]
            assert timeout == pytest.approx(0.25)
            self.waits += 1
            if self.waits == 1:
                store.update_arena_job(job_id, status="running")
                control.checkpoint_requested.set()
                assert control.checkpoint_due()
                return False
            store.update_arena_job(job_id, status="complete", result={})
            return True

        def cancel(self, _job_id):
            self.cancelled = True
            return True

    manager = CheckpointingManager()

    def service_checkpoint():
        if not control.consume_checkpoint():
            return False
        observed_statuses.append(store.arena_job(trainer_job["id"])["status"])
        store.add_checkpoint(
            run_id=run["id"],
            label="manual during final evaluation",
            path=str(tmp_path / "manual.safetensors"),
            actor_path=None,
            games=2_000,
        )
        return True

    result = _finish_final_evaluations(
        store=store,
        run_id=run["id"],
        manager_getter=lambda: manager,
        schedule_latest=lambda: None,
        process_completed=lambda: None,
        service_checkpoint=service_checkpoint,
    )

    assert result["status"] == "complete"
    assert result["checkpoints_serviced"] == 1
    assert observed_statuses == ["running"]
    assert control.checkpoint_due() is False
    assert manager.cancelled is False
    assert manager.waits == 2
    assert any(
        item["label"] == "manual during final evaluation" for item in store.checkpoints(run["id"])
    )


def test_final_evaluation_cancels_only_trainer_job_when_pause_is_requested(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    config = RunConfig.quick()
    run = store.create_run(config)
    champion, candidate = _checkpoints(store, run, tmp_path)
    trainer_job = store.create_arena_job(
        model_a=candidate["id"],
        model_b=champion["id"],
        config={
            "automatic_promotion": False,
            "trainer_scheduled": True,
            "promotion_tier": "diagnostic",
        },
    )
    unrelated = store.create_arena_job(
        model_a="baseline:balanced",
        model_b="baseline:economy",
        config={"automatic_promotion": False},
    )
    paused = {"value": False}

    class InterruptibleManager:
        def wait_for_job(self, job_id, timeout=None):
            assert job_id == trainer_job["id"]
            assert timeout == pytest.approx(0.25)
            paused["value"] = True
            return False

        def cancel(self, job_id):
            assert job_id == trainer_job["id"]
            store.update_arena_job(job_id, status="cancelled")
            return True

    result = _finish_final_evaluations(
        store=store,
        run_id=run["id"],
        manager_getter=InterruptibleManager,
        schedule_latest=lambda: pytest.fail("pause must prevent another evaluation"),
        process_completed=lambda: None,
        interrupt_reason=lambda: "pause_requested" if paused["value"] else None,
    )

    assert result["status"] == "interrupted"
    assert result["interrupt_reason"] == "pause_requested"
    assert store.arena_job(trainer_job["id"])["status"] == "cancelled"
    assert store.arena_job(unrelated["id"])["status"] == "queued"


def test_quality_gate_checks_raw_tactics_and_heldout_regression(tmp_path, monkeypatch):
    store = Store(tmp_path / "state.sqlite3")
    config = RunConfig.quick().model_copy(
        update={
            "heldout_brier_regression_tolerance": 0.03,
            "gate_heldout_brier_regression": True,
            "gate_raw_tactical_preferences": True,
        }
    )
    run = store.create_run(config)
    champion, candidate = _checkpoints(store, run, tmp_path)
    store.update_checkpoint_evaluation(
        champion["id"],
        {
            "quality_gate": {
                "passed": True,
                "diagnostics": {
                    "baselines": {"mean_score": 0.60},
                    "heldout": {"game_grouped_brier": 0.20},
                },
            }
        },
    )

    monkeypatch.setattr(
        "astro2.diagnostics.checkpoint_diagnostics",
        lambda *_args, **_kwargs: {
            "tactical": {
                "raw_end_turn_violations": 1,
                "masked_end_turn_violations": 0,
            },
            "heldout": {"game_grouped_brier": 0.25},
            "baselines": {"mean_score": 0.59, "truncated_games": 0},
        },
    )
    gate = _checkpoint_quality_gate(
        store=store,
        run_id=run["id"],
        checkpoint=candidate,
        config=config,
    )
    assert gate["passed"] is False
    assert any("raw model logits" in reason for reason in gate["reasons"])
    assert any("Brier" in reason for reason in gate["reasons"])


def test_astro3_quality_gate_does_not_recreate_global_end_turn_pressure(
    tmp_path,
    monkeypatch,
):
    store = Store(tmp_path / "state.sqlite3")
    config = RunConfig.astro3_m4().model_copy(update={"heldout_brier_regression_tolerance": 0.01})
    run = store.create_run(config)
    champion, candidate = _checkpoints(store, run, tmp_path)
    store.update_checkpoint_evaluation(
        champion["id"],
        {
            "quality_gate": {
                "passed": True,
                "diagnostics": {
                    "baselines": {"mean_score": 0.5},
                    "heldout": {"game_grouped_brier": 0.20},
                },
            }
        },
    )
    monkeypatch.setattr(
        "astro2.diagnostics.checkpoint_diagnostics",
        lambda *_args, **_kwargs: {
            "tactical": {
                "raw_end_turn_violations": 7,
                "masked_end_turn_violations": 0,
            },
            "strategic": {"early_high_cost_passed": True},
            "heldout": {"game_grouped_brier": 0.25},
            "baselines": {"mean_score": 0.2, "truncated_games": 0},
        },
    )
    gate = _checkpoint_quality_gate(
        store=store,
        run_id=run["id"],
        checkpoint=candidate,
        config=config,
    )
    assert gate["passed"] is True
    assert gate["tactical_gate_metric"] == "masked_end_turn_violations"
    assert gate["baseline_regression_gate_enabled"] is False
    assert gate["baseline_regression_detected"] is True
    assert gate["heldout_brier_gate_enabled"] is False
    assert gate["heldout_brier_regression_detected"] is True
    assert not any("Brier" in reason for reason in gate["reasons"])
