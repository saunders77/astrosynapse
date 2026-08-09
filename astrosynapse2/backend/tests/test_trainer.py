import threading

import numpy as np
import pytest
from astro2.config import RunConfig
from astro2.league import League
from astro2.storage import Store
from astro2.supervisor import InvalidTransition, RunControl, Supervisor, TrainingHandle
from astro2.trainer import (
    _epsilon,
    _evaluation_plan,
    _last_scheduled_evaluation_games,
    _learning_rate,
    _make_bootstrap_plan,
    _persisted_active_elapsed,
    _restore_totals,
    _schedule_evaluation,
    _sync_league,
)


class FakeArenaManager:
    def __init__(self):
        self.calls = []

    def create(self, model_a, model_b, config):
        self.calls.append(("manual", model_a, model_b, config))
        return {"id": "quick-job"}

    def create_automatic(self, model_a, model_b, **config):
        self.calls.append(("automatic", model_a, model_b, config))
        return {"id": "automatic-job"}


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
    assert _epsilon(config, 0) == config.epsilon_start
    assert _epsilon(config, config.epsilon_decay_games) == pytest.approx(config.epsilon_end)
    assert _epsilon(config, config.epsilon_decay_games * 2) == pytest.approx(
        config.epsilon_end
    )


def test_fresh_optimizer_repeats_learning_rate_warmup():
    config = RunConfig.quick()
    first = _learning_rate(config, elapsed_fraction=0.5, updates_since_optimizer_reset=0)
    warmed = _learning_rate(config, elapsed_fraction=0.5, updates_since_optimizer_reset=500)
    assert first < warmed


def test_replay_warmup_plan_collects_both_heuristic_players():
    plan = _make_bootstrap_plan(
        config=RunConfig.quick(), rng=np.random.default_rng(3), seed=99
    )
    assert plan.actor_paths == (None, None)
    assert plan.collect_players == (True, True)
    assert plan.epsilons == (0.0, 0.0)
    assert plan.kind == "heuristic_bootstrap"


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


def test_nonzero_checkpoint_is_added_to_league(tmp_path):
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
    league = League()
    _sync_league(league, store, run["id"])
    assert [opponent.id for opponent in league.opponents] == [checkpoint["id"]]


def test_active_elapsed_resumes_without_counting_backend_downtime(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    run = store.create_run(RunConfig.quick())
    assert _persisted_active_elapsed(store, run["id"]) == 0.0
    store.append_metric(run["id"], 1, {"active_elapsed_seconds": 12.5})
    assert _persisted_active_elapsed(store, run["id"]) == 12.5


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

    store.create_arena_job(
        model_a=candidate["id"],
        model_b=champion["id"],
        config={
            "automatic_promotion": True,
            "trainer_scheduled": True,
            "promotion_tier": "full",
        },
    )
    assert _last_scheduled_evaluation_games(store, run["id"]) == candidate["games"]
    assert (
        _last_scheduled_evaluation_games(store, run["id"], tier="full")
        == candidate["games"]
    )
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
