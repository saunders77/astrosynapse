"""Promotion accounting, historical sampling, recovery, and curriculum regressions."""

import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astro2.autonomy import (
    anchor_batch,
    anchor_loss,
    commit_gate,
    gate_finished,
    historical_inventory,
    league_schedule,
    prepare_history,
)
from astro2.experiment_control import code_identity


def test_league_is_deterministic_balanced_and_mostly_incumbent():
    schedule = league_schedule("current", ["old1", "old2"], 765, 512)
    assert schedule == league_schedule("current", ["old1", "old2"], 765, 512)
    assert schedule != league_schedule("current", ["old1", "old2"], 766, 512)
    assert schedule.count("current") == 384
    for seat in (0, 1):
        assert schedule[seat::2].count("current") == 192
    assert league_schedule("current", [], 7, 8) == ["current"] * 8


def test_anchor_normalization_padding_and_gradient():
    import mlx.core as mx
    import mlx.nn as nn
    from astro2.model import ModelSpec, build_model
    from mlx.utils import tree_flatten

    spec = ModelSpec(
        state_size=4,
        action_size=3,
        families=2,
        hidden_size=8,
        action_hidden_size=4,
        residual_blocks=1,
        bootstrap_heads=2,
        objective_version=2,
    )
    model = build_model(spec)
    bank = dict(
        states=np.ones((2, 4), dtype=np.float32),
        actions=np.eye(3, dtype=np.float32),
        offsets=np.array([0, 2, 3]),
        families=np.array([0, 1], dtype=np.int32),
        logits=np.array([0.1, 0.2, 0.4], dtype=np.float32),
    )
    arrays = anchor_batch(bank, np.array([0, 1]))
    assert np.allclose((np.exp(arrays[-1]) * arrays[2]).sum(axis=1), 1)
    assert arrays[2].tolist() == [[1, 1], [1, 0]]
    value, gradients = nn.value_and_grad(model, anchor_loss)(model, *[mx.array(x) for x in arrays])
    mx.eval(value, gradients)
    assert np.isfinite(float(value)) and float(value) >= -1e-6
    grads = dict(tree_flatten(gradients))
    assert all(
        np.all(np.asarray(v) == 0) for key, v in grads.items() if key.startswith("value_output")
    )
    assert any(
        np.any(np.asarray(v) != 0) for key, v in grads.items() if key.startswith("head_outputs")
    )


def test_history_counts_only_live_compatible_rows_and_freezes_teacher(tmp_path, monkeypatch):
    from astro2 import native_actor

    spec = SimpleNamespace(state_size=4, action_size=3, families=2)
    root = tmp_path / "checkpoints" / "oldrun"
    root.mkdir(parents=True)
    shard = root / "shard"
    shard.mkdir()
    for key, value in dict(
        states=np.ones((5, 4)),
        legal_actions=np.ones((10, 3)),
        action_offsets=np.arange(0, 11, 2),
        families=np.zeros(5),
    ).items():
        np.save(shard / (key + ".npy"), value)
    (root / "last.policy-replay.json").write_text(
        json.dumps(
            dict(
                cold=dict(
                    state_size=4,
                    action_size=3,
                    shards=[
                        dict(path=str(shard), decisions=5),
                        dict(path=str(root / "deleted"), decisions=100000),
                    ],
                )
            )
        )
    )
    report, groups = historical_inventory(tmp_path, spec)
    assert report["replay_positions"] == 5 and len(groups[0]) == 1
    teacher = tmp_path / "teacher"
    teacher.write_bytes(b"fixed")
    fake = SimpleNamespace(
        spec=spec, predict_options=lambda s, a, f: np.tile([[0.1], [0.2]], (1, 2))
    )
    monkeypatch.setattr(native_actor.NativeActor, "load", lambda p: fake)
    destination = tmp_path / "history.npz"
    report = prepare_history(tmp_path, teacher, destination, seed=9, positions=4)
    assert report["sampled_positions"] == 4
    assert prepare_history(tmp_path, teacher, destination, seed=9) == report
    with np.load(destination) as bank:
        assert bank["states"].shape == (4, 4)
    destination.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        prepare_history(tmp_path, teacher, destination, seed=9)


def test_gate_budget_futility_and_single_commit():
    assert not gate_finished(dict(pairs=8192, score=0.508, passed=False), 131072)
    assert gate_finished(dict(pairs=8192, score=0.50, passed=False), 131072)
    assert gate_finished(dict(pairs=256, score=0.75, passed=True), 131072)
    state = dict(
        pending_gate=dict(model="new", actor="actor", attempt=308),
        gates=[],
        games=1,
        champion_label="champ9",
        promotions=[],
        stage=9,
        attempt=308,
    )
    commit_gate(state, dict(passed=False))
    assert state["attempt"] == 308 and state["stage"] == 9
    state["pending_gate"] = dict(model="new", actor="actor", attempt=309)
    commit_gate(state, dict(passed=True))
    assert state["attempt"] == 0 and state["stage"] == 10 and state["champion"] == "actor"
    with pytest.raises(ValueError, match="no pending"):
        commit_gate(state, dict(passed=True))


@pytest.fixture
def supervisor(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    module = importlib.import_module("autonomous_training")
    monkeypatch.setattr(module.signal, "signal", lambda *_: None)
    return module


@pytest.fixture
def campaign(tmp_path, supervisor):
    out = tmp_path / "campaign"
    out.mkdir()
    runtime = out / "runtime"
    (runtime / "astro2").mkdir(parents=True)
    (runtime / "scripts").mkdir()
    settings = dict(
        workers=2,
        hours=96,
        max_rounds=4,
        max_gate_pairs=131072,
        seed=17,
        games=512,
        round_iterations=8,
        screen_pairs=512,
        confirm_pairs=2048,
    )
    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                settings=settings,
                code_identity=code_identity(runtime / "astro2", runtime / "scripts"),
            )
        )
    )
    (out / "state.json").write_text(
        json.dumps(
            dict(
                elapsed=0,
                phase="paused",
                stage=9,
                attempt=307,
                model="model",
                champion="champ",
                champion_label="champ9",
                games=100,
                gates=[],
                promotions=[],
                pending_gate=None,
                branches=[],
                active_branch=None,
                branch_counter=0,
                events=[],
                inherited_promotions=0,
                errors=0,
            )
        )
    )
    return supervisor.Campaign(out)


def test_new_branch_has_fresh_optimizer_and_retires_after_stall(campaign):
    branch = campaign.new_branch()
    assert branch["source"] == "model" and branch["opponent"] == "champ"
    branch.update(round=2, screens=[0.49, 0.50])
    campaign.continue_or_retire(branch)
    assert campaign.state["active_branch"] is None
    next_branch = campaign.new_branch()
    assert next_branch["recipe"]["advantage_baseline"] == "constant"
    assert next_branch["seed"] != branch["seed"]
    assert next_branch["source"] == "model"


def test_confirmation_spends_inherited_attempt_before_gate(campaign, monkeypatch):
    branch = campaign.new_branch()
    branch.update(actor="actor", model="candidate", round=1, status="confirming")
    monkeypatch.setattr(
        campaign, "matches", lambda *a, **k: dict(paused=False, score=0.52, pairs=2048)
    )
    campaign.confirm(branch)
    saved = json.loads((campaign.out / "state.json").read_text())
    assert saved["attempt"] == 308 and saved["pending_gate"]["attempt"] == 308
    assert saved["pending_gate"]["seed"] != branch["seed"]
    assert saved["pending_gate"]["seed"] != branch["seed"] + 10**9 + branch["round"]


def test_promising_parent_can_accumulate_learning_without_promoting(campaign, tmp_path):
    branch = campaign.new_branch()
    model = tmp_path / "promising.safetensors"
    model.write_bytes(b"checkpoint")
    branch.update(confirmation_model=str(model), confirmation=dict(pairs=2048, score=0.508))
    assert campaign.select_parent(branch["recipe"], 1) == (str(model), branch["id"])
    assert campaign.state["champion"] == "champ" and campaign.state["attempt"] == 307
    # Periodic fresh starts prevent a noisy early winner from capturing every trial.
    assert campaign.select_parent(branch["recipe"], 3) == ("model", None)
    campaign.state["gates"].append(dict(model=str(model), score=0.499))
    assert campaign.select_parent(branch["recipe"], 1) == ("model", None)
    campaign.state["gates"].clear()
    branch["stage"] = 8
    assert campaign.select_parent(branch["recipe"], 1) == ("model", None)


def test_resume_evaluation_recovers_torn_append_without_replaying_pairs(
    campaign, supervisor, monkeypatch
):
    actor = campaign.out / "actor"
    actor.write_bytes(b"actor")
    opponent = campaign.out / "opponent"
    opponent.write_bytes(b"opponent")
    folder = campaign.out / "evaluation"
    folder.mkdir()

    def row(i):
        return dict(
            pair=i, scores=[1, 0], truncated=0, seconds=0, searches=0, changes=0, branches=0
        )

    (folder / "pairs.jsonl").write_text(json.dumps(row(0)) + '\n{"pair":')
    seen = []

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def map(self, fn, tasks, **kwargs):
            seen.extend(t[-1] for t in tasks)
            return [row(t[-1]) for t in tasks]

    monkeypatch.setattr(supervisor.concurrent.futures, "ProcessPoolExecutor", Pool)
    result = campaign.matches(str(actor), str(opponent), folder, 123, pairs=4)
    assert result["pairs"] == 4 and not result["paused"] and seen == [1, 2, 3]
    campaign.matches(str(actor), str(opponent), folder, 123, pairs=4)
    assert seen == [1, 2, 3]
    with pytest.raises(ValueError, match="identity changed"):
        campaign.matches(str(actor), str(opponent), folder, 124, pairs=4)


def test_migration_preserves_engine_and_spent_promotion_budget(tmp_path, supervisor):
    old = tmp_path / "old"
    old.mkdir()
    (old / "runtime/astro2").mkdir(parents=True)
    (old / "runtime/scripts").mkdir()
    (old / "runtime/astro2/engine.py").write_text("historical_engine = True\n")
    model = old / "champ.safetensors"
    model.write_bytes(b"checkpoint")
    Path(str(model) + ".json").write_text("{}")
    actor = old / "champ.actor.npz"
    actor.write_bytes(b"actor")
    (old / "original-champion.actor.npz").write_bytes(b"original")
    state = dict(
        phase="paused",
        model=str(model),
        champion=str(actor),
        stage=9,
        attempt=307,
        games=12000000,
        history=[],
        gates=[],
        promotions=[],
        champion_label="champ9",
        pending_gate=dict(actor="partial"),
        evaluation=dict(score=0.49, pairs=1280),
    )
    (old / "state.json").write_text(json.dumps(state))
    (old / "manifest.json").write_text(json.dumps(dict(code_identity={})))
    out = tmp_path / "new"
    out.mkdir()
    supervisor.initialize(out, old, argparse.Namespace(hours=96, workers=8))
    result = json.loads((out / "state.json").read_text())
    assert result["attempt"] == 307 and result["stage"] == 9 and result["pending_gate"] is None
    assert result["gates"][-1]["abandoned"]
    assert json.loads((old / "state.json").read_text()) == state
    assert (out / "runtime/astro2/engine.py").read_text() == "historical_engine = True\n"


def test_worker_control_applies_between_persisted_evaluation_blocks(
    campaign, supervisor, monkeypatch
):
    actor = campaign.out / "actor"
    actor.write_bytes(b"actor")
    sizes, seen = [], []

    class Pool:
        def __init__(self, max_workers, **kwargs):
            sizes.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def map(self, fn, tasks, **kwargs):
            seen.extend(t[-1] for t in tasks)
            (campaign.out / "settings.json").write_text(json.dumps(dict(workers=1)))
            return [
                dict(
                    pair=t[-1],
                    scores=[1, 0],
                    truncated=0,
                    seconds=0,
                    searches=0,
                    changes=0,
                    branches=0,
                )
                for t in tasks
            ]

    monkeypatch.setattr(supervisor.concurrent.futures, "ProcessPoolExecutor", Pool)
    result = campaign.matches(str(actor), str(actor), campaign.out / "evaluation", 123, pairs=256)
    assert sizes == [2, 1]
    assert result["pairs"] == 256 and seen == list(range(256))
