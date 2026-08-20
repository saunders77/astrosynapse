import json
import sqlite3

import numpy as np
import pytest
from astro2.config import RunConfig
from astro2.encoding import Encoder
from astro2.hardware import RateMeter
from astro2.model import ModelSpec
from astro2.stats import elo_delta, wilson_interval
from astro2.storage import Store
from astro2.supervisor import Supervisor
from astro2.trainer import _training_budget_reached


def test_wilson_interval_behaves_at_small_sample_sizes():
    interval = wilson_interval(15, 24)
    assert interval.estimate == 0.625
    assert interval.low < 0.5 < interval.high
    assert elo_delta(0.5) == 0.0


def test_rate_meter_preserves_last_completed_batch_rate(monkeypatch):
    readings = iter((10.0, 12.0, 12.5))
    monkeypatch.setattr("astro2.hardware.time.monotonic", lambda: next(readings))
    meter = RateMeter.start()
    measured = meter.sample(20, 400)
    final = meter.sample(20, 400)
    assert measured["games_per_second"] == 10.0
    assert measured["decisions_per_second"] == 200.0
    assert final["games_per_second"] == measured["games_per_second"]
    assert final["decisions_per_second"] == measured["decisions_per_second"]


def test_rate_meter_includes_idle_time_between_batched_results(monkeypatch):
    readings = iter((10.0, 11.0, 14.0))
    monkeypatch.setattr("astro2.hardware.time.monotonic", lambda: next(readings))
    meter = RateMeter.start()
    idle = meter.sample(0, 0)
    measured = meter.sample(40, 800)
    assert idle["games_per_second"] == 0.0
    assert measured["games_per_second"] == 10.0
    assert measured["decisions_per_second"] == 200.0


@pytest.mark.parametrize(
    ("config", "elapsed", "games", "evaluations", "expected", "reason"),
    [
        (
            RunConfig.quick().model_copy(update={"duration_minutes": 5}),
            300,
            0,
            0,
            True,
            "duration complete",
        ),
        (
            RunConfig.quick().model_copy(update={"budget_type": "games", "budget_games": 1_000}),
            999_999,
            1_000,
            0,
            True,
            "game budget complete",
        ),
        (
            RunConfig.quick().model_copy(
                update={"budget_type": "full_evaluations", "budget_full_evaluations": 2}
            ),
            999_999,
            999_999,
            1,
            False,
            "full-evaluation budget complete",
        ),
    ],
)
def test_training_budget_modes(config, elapsed, games, evaluations, expected, reason):
    reached, actual_reason = _training_budget_reached(
        config,
        active_elapsed=elapsed,
        games=games,
        full_evaluations=evaluations,
    )
    assert reached is expected
    assert actual_reason == reason


def test_budget_mode_requires_matching_value():
    with pytest.raises(ValueError, match="budget_games"):
        RunConfig.model_validate(
            {**RunConfig.quick().model_dump(), "budget_type": "games", "budget_games": None}
        )


def test_store_round_trip(tmp_path):
    store = Store(tmp_path / "astro2.sqlite3")
    run = store.create_run(RunConfig.quick())
    assert run["status"] == "ready"
    assert run["config"]["preset"] == "quick"

    store.append_metric(run["id"], 1, {"games": 12, "loss": 0.4})
    assert store.metrics(run["id"])[0]["games"] == 12
    store.append_metric(run["id"], 2, {"games": 24})
    store.append_metric(run["id"], 3, {"games": 36})
    assert [item["seq"] for item in store.metrics(run["id"], limit=2)] == [2, 3]
    assert [item["seq"] for item in store.metrics(run["id"], after=1)] == [2, 3]

    updated = store.update_run(run["id"], status="running", games=12)
    assert updated["games"] == 12
    assert store.events(run["id"])[0]["kind"] == "run_created"

    checkpoint = store.add_checkpoint(
        run_id=run["id"],
        label="Candidate",
        path=str(tmp_path / "candidate.safetensors"),
        actor_path=str(tmp_path / "candidate.actor.npz"),
        games=12,
    )
    assert checkpoint["is_pinned"] is False
    assert store.set_checkpoint_pinned(checkpoint["id"], True)["is_pinned"] is True


def test_store_closes_every_transaction_connection(tmp_path, monkeypatch):
    store = Store(tmp_path / "astro2.sqlite3")
    opened: list[sqlite3.Connection] = []
    connect = store._open_connection

    def tracked_connect() -> sqlite3.Connection:
        connection = connect()
        opened.append(connection)
        return connection

    monkeypatch.setattr(store, "_open_connection", tracked_connect)
    run = store.create_run(RunConfig.quick())
    store.append_metric(run["id"], 1, {"games": 16})
    store.get_run(run["id"])

    assert opened
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


def test_branch_experiment_and_controller_state_round_trip(tmp_path):
    store = Store(tmp_path / "astro2.sqlite3")
    source_run = store.create_run(RunConfig.astro4_m4())
    source = store.add_checkpoint(
        run_id=source_run["id"],
        label="Source champion",
        path=str(tmp_path / "source.safetensors"),
        actor_path=str(tmp_path / "source.actor.npz"),
        games=500_000,
        champion=True,
    )
    experiment = store.create_branch_experiment(
        name="Search fork",
        source_checkpoint_id=source["id"],
        config={"auto_advance": True},
    )
    branch_config = RunConfig.astro5_search().model_copy(
        update={
            "initial_checkpoint_id": source["id"],
            "branch_experiment_id": experiment["id"],
        }
    )
    branch_run = store.create_run(branch_config)
    member = store.add_branch_member(
        experiment_id=experiment["id"],
        run_id=branch_run["id"],
        ordinal=0,
        label="Balanced",
        overrides={"reanalysis_fraction": 0.02},
    )
    assert member["status"] == "queued"
    loaded = store.branch_experiment(experiment["id"])
    assert loaded["members"][0]["run_id"] == branch_run["id"]
    assert loaded["members"][0]["overrides"]["reanalysis_fraction"] == 0.02
    assert store.update_branch_member(branch_run["id"], status="running")["status"] == "running"

    state = store.set_controller_state(
        branch_run["id"],
        {"learning_rate_multiplier": 0.5, "branch_requested": False},
    )
    assert state["learning_rate_multiplier"] == 0.5
    assert store.controller_state(branch_run["id"])["branch_requested"] is False


def test_supervisor_copies_compatible_branch_root_artifacts(tmp_path):
    store = Store(tmp_path / "astro2.sqlite3")
    source_run = store.create_run(RunConfig.astro4_m4())
    encoder = Encoder(version=2)
    spec = ModelSpec(
        state_size=encoder.state_size,
        action_size=encoder.action_size,
        families=8,
        hidden_size=192,
        action_hidden_size=96,
        residual_blocks=3,
        bootstrap_heads=5,
        encoder_version=2,
        objective_version=2,
    )
    model_path = tmp_path / "source.safetensors"
    model_path.write_bytes(b"test model placeholder")
    model_path.with_suffix(".safetensors.json").write_text(json.dumps(spec.as_dict()))
    actor_path = tmp_path / "source.actor.npz"
    np.savez(
        actor_path,
        __spec_json__=np.frombuffer(json.dumps(spec.as_dict()).encode(), dtype=np.uint8),
    )
    source = store.add_checkpoint(
        run_id=source_run["id"],
        label="Source champion",
        path=str(model_path),
        actor_path=str(actor_path),
        games=500_000,
        champion=True,
    )
    supervisor = Supervisor(store, tmp_path)
    experiment = supervisor.create_branch_experiment(
        source_checkpoint_id=source["id"],
        name="Copied fork",
        variants=[{"label": "Balanced"}, {"label": "Search", "reanalysis_fraction": 0.02}],
        base_overrides={"duration_minutes": 5},
        start=False,
    )
    assert len(experiment["members"]) == 2
    for member in experiment["members"]:
        branch_run = store.get_run(member["run_id"])
        assert branch_run["config"]["initial_checkpoint_id"] == source["id"]
        root = store.checkpoints(member["run_id"])[0]
        assert root["parent_id"] == source["id"]
        assert root["games"] == 0
        assert root["is_champion"] is True
        assert (tmp_path / "checkpoints" / member["run_id"] / "branch-root.actor.npz").is_file()
    assert store.checkpoint(source["id"])["is_pinned"] is True
