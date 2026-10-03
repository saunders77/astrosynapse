import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astro2.autopilot import CampaignConfig, worker_locked
from astro2.autopilot_data import export_trajectories, fit_candidate, game_partition
from astro2.autopilot_registry import ValuePredictor, campaign_models
from astro2.critic import IndependentCritic, save_npz
from astro2.encoding import DecisionFamily, Encoder
from astro2.experiment_control import atomic_json

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "autopilot_worker", PROJECT / "scripts/autopilot_worker.py"
)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_whole_game_export_excludes_truncations_and_splits_stably(tmp_path):
    def trajectory(target, truncated=False):
        return SimpleNamespace(
            target=target,
            truncated=truncated,
            value_states=[np.ones(4) * target] * 5,
            value_families=[0] * 5,
        )

    export_trajectories(tmp_path, 0, [trajectory(1), trajectory(0), trajectory(1, True)], 2, 7, 3)
    with np.load(tmp_path / "00000.npz") as z:
        assert z["x"].shape == (2, 32, 6)
        assert z["y"].tolist() == [1, 0]
        assert z["ids"].tolist() == [0, 1]
    assignments = {i: game_partition(f"block:{i}") for i in range(1000)}
    assert assignments == {i: game_partition(f"block:{i}") for i in reversed(range(1000))}
    assert set(assignments.values()) == set(range(10))


def test_prediction_candidate_is_not_a_promotion_and_metadata_survives(tmp_path):
    rng = np.random.default_rng(9)
    critic = IndependentCritic(5, 16, 3)
    metadata = {
        "encoder_version": 3,
        "state_size": 3,
        "families": 2,
        "config": {"rules_version": 2},
    }
    source = tmp_path / "source.npz"
    critic.save(source, metadata)
    x = rng.normal(size=(600, 32, 5)).astype(np.float32)
    y = (x[:, :, 0].mean(axis=1) > 0).astype(np.float32)
    shard = tmp_path / "shard.npz"
    save_npz(shard, x=x, y=y, ids=np.arange(600), encoder_version=np.array(3))
    result = fit_candidate(
        source, [shard], tmp_path / "fit", seed=4, seconds=10, check=lambda: None, epochs=2
    )
    assert (
        result["counts"]["train"] + result["counts"]["validation"] + result["counts"]["test"] == 600
    )
    assert "promoted" not in result
    _, meta, _ = IndependentCritic.load(tmp_path / "fit/critic.npz")
    assert meta["encoder_version"] == 3 and meta["config"]["rules_version"] == 2
    # Source critic stays immutable during candidate fitting.
    restored, _, _ = IndependentCritic.load(source)
    np.testing.assert_array_equal(restored.weights["w0"], critic.weights["w0"])


@pytest.mark.parametrize("version", [1, 2, 3])
def test_critic_predictor_uses_own_encoder(version, tmp_path):
    encoder = Encoder(version=version)
    critic = IndependentCritic(encoder.state_size + len(DecisionFamily), 16)
    path = tmp_path / "critic.npz"
    critic.save(path, {"encoder_version": version, "config": {"rules_version": 2}})
    predictor = ValuePredictor({"kind": "independent", "path": str(path)})
    assert predictor.encoder.version == version


def test_campaign_registry_preserves_associated_critic_and_champion(tmp_path):
    folder = tmp_path / "autopilot" / ("a" * 32)
    folder.mkdir(parents=True)
    actor = folder / "policy.actor.npz"
    actor.touch()
    atomic_json(
        folder / "state.json",
        {
            "champion_id": "p1",
            "models": [
                {
                    "id": "p1",
                    "actor_path": str(actor),
                    "critic_id": "c2",
                    "label": "Policy",
                    "created_at": 0,
                }
            ],
        },
    )
    model = campaign_models(tmp_path)[0]
    assert model["is_champion"] and model["critic_id"] == "c2"


def test_paired_contrast_respects_matching_and_does_not_promote_ties():
    rows = [{"pair": i, "scores": [1, 0]} for i in range(2000)]
    report = worker.paired_difference(rows, rows, 0.025)
    assert report["difference"] == 0 and report["lower"] <= 0
    with pytest.raises(ValueError, match="Matched"):
        worker.paired_difference(rows, rows[::-1], 0.025)


def test_promotion_reentry_does_not_duplicate():
    campaign = object.__new__(worker.Campaign)
    campaign.policy_gate({"finished": True}, {})
    campaign.critic_stage({"finished": True})


def test_invalid_budgets_and_training_seed_duplicates():
    base = dict(policy_id="p", critic_id="c", opponent_ids=["p"], panel_ids=["p"])
    with pytest.raises(ValueError, match="multiples"):
        CampaignConfig(**base, block_games=21111)
    with pytest.raises(ValueError, match="distinct"):
        CampaignConfig(**base, seeds=[1, 1])


def test_worker_lock_detects_other_owner(tmp_path):
    import fcntl

    assert not worker_locked(tmp_path)
    with (tmp_path / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert worker_locked(tmp_path)


def test_api_discovery_and_invalid_controls(tmp_path, monkeypatch):
    from astro2 import server
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    with TestClient(server.app) as client:
        assert client.get("/api/autopilot").json() == []
        assert client.get("/api/autopilot/critics").json() == []
        assert client.get("/api/autopilot/matches").json() == []
        assert client.post("/api/autopilot/not-a-path/resume").status_code == 404
        assert (
            client.post(
                "/api/autopilot/matches", json={"model_a": "bad", "model_b": "bad"}
            ).status_code
            == 422
        )


def test_better_prediction_without_better_policy_does_not_promote_critic(tmp_path):
    campaign = object.__new__(worker.Campaign)
    campaign.config = SimpleNamespace(seeds=[1, 2], probe_games=20, probe_pairs=2000, seed=5)
    pending = dict(
        key="critic-1", champion_model="model", champion_actor="actor", critic_path="old", attempt=1
    )
    campaign.state = dict(pending=pending, events=[], critic_id="old", critics=[], promotions=[])
    campaign.save = lambda: None
    out = tmp_path / "critic-1"
    out.mkdir()
    atomic_json(
        out / "result.json",
        {"qualified": True, "test_before": {"log_loss": 0.5}, "test_after": {"log_loss": 0.4}},
    )
    campaign.begin_job = lambda *_: out
    campaign.train = lambda *_, **__: {"model": "candidate.safetensors"}
    rows = [{"pair": i, "scores": [1, 0]} for i in range(2000)]
    campaign.evaluate = lambda *_, **__: (rows, {})
    campaign.critic_stage(pending)
    assert campaign.state["critic_id"] == "old"
    assert campaign.state["promotions"] == []
    assert campaign.state["events"][-1]["kind"] == "critic_inconclusive"
    assert pending["finished"]


def test_resources_and_restore_require_stopped_campaign(tmp_path, monkeypatch):
    from astro2 import server
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    ident = "a" * 32
    folder = tmp_path / "autopilot" / ident
    folder.mkdir(parents=True)
    config = CampaignConfig(policy_id="p", critic_id="c", opponent_ids=["p"], panel_ids=["p"])
    atomic_json(folder / "config.json", config.model_dump())
    atomic_json(
        folder / "state.json",
        dict(
            id=ident,
            status="paused",
            pending=None,
            champion_id="p1",
            critic_id="c1",
            models=[
                {"id": "p0", "was_champion": True},
                {"id": "p1", "was_champion": True},
                {"id": "candidate", "was_champion": False},
            ],
            critics=[{"id": "c0"}, {"id": "c1"}],
            events=[],
        ),
    )
    with TestClient(server.app) as client:
        response = client.patch(
            f"/api/autopilot/{ident}/settings",
            json=dict(max_hours=48, storage_gb=40, workers=4, policy_fraction=0.8),
        )
        assert response.status_code == 200 and response.json()["config"]["max_hours"] == 48
        assert (
            client.post(
                f"/api/autopilot/{ident}/champions/restore",
                json={"policy_id": "candidate", "critic_id": "c0"},
            ).status_code
            == 409
        )
        restored = client.post(
            f"/api/autopilot/{ident}/champions/restore", json={"policy_id": "p0", "critic_id": "c0"}
        )
        assert restored.status_code == 200 and restored.json()["champion_id"] == "p0"
