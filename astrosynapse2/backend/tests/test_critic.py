import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from astro2 import critic, server
from astro2.experiment_control import atomic_json, sha256
from fastapi.testclient import TestClient


def test_network_learns_and_resume_preserves_optimizer(tmp_path):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(128, 6)).astype(np.float32)
    y = (x[:, 0] * x[:, 1] > 0).astype(np.float32)
    model = critic.IndependentCritic(6, 32, 1)
    before = critic.calibration(model.predict(x), y)["log_loss"]
    for _ in range(200):
        model.train_batch(x, y, 0.01)
    assert critic.calibration(model.predict(x), y)["log_loss"] < before / 2
    model.save(tmp_path / "model.npz", {"epoch": 1}, model.weights)
    resumed, meta, best = critic.IndependentCritic.load(tmp_path / "model.npz")
    assert meta["epoch"] == 1
    assert best is not None
    model.train_batch(x, y, 0.001)
    resumed.train_batch(x, y, 0.001)
    for k in model.weights:
        np.testing.assert_array_equal(model.weights[k], resumed.weights[k])


def test_splits_and_calibration():
    splits = critic.game_splits(100, 8)
    assert splits == critic.game_splits(100, 8)
    assert sorted(sum(splits.values(), [])) == list(range(100))
    assert [len(splits[k]) for k in ("train", "validation", "test")] == [70, 15, 15]
    metrics = critic.calibration([0.2, 0.8], [0, 1])
    assert metrics["brier"] == pytest.approx(0.04)
    assert sum(b["positions"] for b in metrics["bins"]) == 2
    assert critic.calibration([], []) is None


def make_job(folder, config):
    folder.mkdir(exist_ok=True)
    (folder / "policy.actor.npz").write_bytes(b"frozen policy")
    atomic_json(
        folder / "job.json",
        {
            "id": folder.name,
            "config": critic.asdict(config),
            "policy_sha256": sha256(folder / "policy.actor.npz"),
            "model_label": "test policy",
            "status": "queued",
            "updated_at": time.time(),
            "created_at": time.time(),
        },
    )
    return SimpleNamespace(
        spec=SimpleNamespace(state_size=3, families=8, objective_version=2, encoder_version=2)
    )


def synthetic_collect(folder, index, config, actor, encoder, check):
    check()
    rng = np.random.default_rng(index)
    x = rng.normal(size=(16, 11)).astype(np.float32)
    critic.save_npz(
        folder / f"game-{index:05d}.npz",
        x=x,
        y=(x[:, 0] > 0).astype(np.float32),
        baseline=np.full(16, 0.5),
        forced=np.arange(16) % 2 == 0,
        family=np.arange(16) % 8,
        turn=np.arange(16),
        truncated=np.array(False),
    )


def test_end_to_end_pause_resume_and_policy_immutable(tmp_path, monkeypatch):
    config = critic.CriticConfig(games=20, epochs=2, hidden_size=32)
    continuous, paused = tmp_path / "continuous", tmp_path / "paused"
    actor = make_job(continuous, config)
    make_job(paused, config)
    monkeypatch.setattr(critic, "_load_actor_encoder", lambda _: (actor, None))
    monkeypatch.setattr(critic, "collect_game", synthetic_collect)
    critic.train_job(continuous)

    def pause_after_game(*args):
        synthetic_collect(*args)
        if args[1] == 3:
            (args[0] / "PAUSE").touch()

    monkeypatch.setattr(critic, "collect_game", pause_after_game)
    critic.train_job(paused)
    assert json.loads((paused / "job.json").read_text())["status"] == "paused"
    (paused / "PAUSE").unlink()
    monkeypatch.setattr(critic, "collect_game", synthetic_collect)
    critic.train_job(paused)
    a, _, _ = critic.IndependentCritic.load(continuous / "critic.npz")
    b, _, _ = critic.IndependentCritic.load(paused / "critic.npz")
    for k in a.weights:
        np.testing.assert_array_equal(a.weights[k], b.weights[k])
    job = json.loads((paused / "job.json").read_text())
    assert job["status"] == "complete"
    assert job["result"]["test"]["games"] == 3
    assert (paused / "policy.actor.npz").read_bytes() == b"frozen policy"


def test_collection_censors_truncation_and_includes_forced(tmp_path, monkeypatch):
    actor = SimpleNamespace(
        spec=SimpleNamespace(state_size=3, families=8),
        predict_values=lambda *args: np.zeros((1, 5)),
    )
    encoder = SimpleNamespace(
        encode_decision=lambda *args: SimpleNamespace(state=np.zeros(3), family=0)
    )

    class FakeGame:
        truncated = False

        def __init__(self, **kwargs):
            self.hook = kwargs["decision_hook"]

        def run(self):
            for i in range(20):
                self.hook(
                    i % 2, SimpleNamespace(actions=[0], observation=SimpleNamespace(turn=i)), None
                )
            return SimpleNamespace(truncated=self.truncated, winner=0)

    monkeypatch.setattr(critic, "Game", FakeGame)
    config = critic.CriticConfig(games=20, positions_per_game=16)
    critic.collect_game(tmp_path, 0, config, actor, encoder, lambda: None)
    with np.load(tmp_path / "game-00000.npz") as z:
        assert len(z["y"]) == 16
        assert z["forced"].all()
        assert set(z["y"]) == {0, 1}
    FakeGame.truncated = True
    critic.collect_game(tmp_path, 1, config, actor, encoder, lambda: None)
    with np.load(tmp_path / "game-00001.npz") as z:
        assert len(z["y"]) == 0
        assert z["truncated"]


def test_manager_and_api(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    actor = tmp_path / "policy.npz"
    actor.write_bytes(b"policy")
    monkeypatch.setattr(
        critic, "resolve_model", lambda *args: SimpleNamespace(actor_path=actor, label="Gen10")
    )
    monkeypatch.setattr(
        critic,
        "_load_actor_encoder",
        lambda _: (SimpleNamespace(spec=SimpleNamespace(objective_version=2)), None),
    )
    monkeypatch.setattr(critic.CriticManager, "launch", lambda *args: None)
    with TestClient(server.app) as client:
        assert client.get("/api/critics").json() == []
        assert (
            client.post("/api/critics", json={"model_id": "gen10", "games": 1}).status_code == 422
        )
        assert (
            client.post("/api/critics", json={"model_id": "gen10", "hidden_size": 13}).status_code
            == 422
        )
        response = client.post("/api/critics", json={"model_id": "gen10", "games": 20})
        assert response.status_code == 201
        job = response.json()
        url = f"/api/critics/{job['id']}"
        assert client.post("/api/critics", json={"model_id": "gen10"}).status_code == 422
        assert client.post(url + "/pause").json()["pause_requested"]
        assert client.post(url + "/resume").status_code == 409
        assert client.get(url + "/download/critic").status_code == 404
        path = tmp_path / "critics" / job["id"] / "job.json"
        job.update(updated_at=time.time() - 30)
        atomic_json(path, job)
        assert client.get(url).json()["status"] == "interrupted"
        assert client.post(url + "/resume").json()["status"] == "queued"
        assert client.get("/api/critics/not-an-id").status_code == 404
        assert actor.read_bytes() == b"policy"


def test_pause_mid_epoch_replays_from_durable_checkpoint(tmp_path, monkeypatch):
    config = critic.CriticConfig(games=20, epochs=2, hidden_size=32)
    folder = tmp_path / "paused-epoch"
    reference = tmp_path / "reference"
    actor = make_job(folder, config)
    make_job(reference, config)
    monkeypatch.setattr(critic, "_load_actor_encoder", lambda _: (actor, None))
    monkeypatch.setattr(critic, "collect_game", synthetic_collect)
    critic.train_job(reference)
    train = critic.IndependentCritic.train_batch

    def interrupted_train(self, *args):
        value = train(self, *args)
        if self.step == 16:  # one full epoch (14 games), then a partial epoch
            (folder / "PAUSE").touch()
        return value

    monkeypatch.setattr(critic.IndependentCritic, "train_batch", interrupted_train)
    critic.train_job(folder)
    job = json.loads((folder / "job.json").read_text())
    assert job["status"] == "paused" and job["epoch"] == 1
    (folder / "PAUSE").unlink()
    monkeypatch.setattr(critic.IndependentCritic, "train_batch", train)
    critic.train_job(folder)
    a, am, _ = critic.IndependentCritic.load(reference / "checkpoint.npz")
    b, bm, _ = critic.IndependentCritic.load(folder / "checkpoint.npz")
    assert am == bm
    for key in a.weights:
        np.testing.assert_array_equal(a.weights[key], b.weights[key])
        np.testing.assert_array_equal(a.m[key], b.m[key])
        np.testing.assert_array_equal(a.v[key], b.v[key])
