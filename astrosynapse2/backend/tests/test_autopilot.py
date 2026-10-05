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


@pytest.mark.parametrize("existing_attempt", [None, 27])
def test_better_prediction_without_better_policy_does_not_promote_critic(
    tmp_path, existing_attempt
):
    campaign = object.__new__(worker.Campaign)
    campaign.config = SimpleNamespace(
        seeds=[1, 2], probe_games=20, probe_pairs=2000, seed=5, exploration_recipes=False
    )
    pending = dict(
        key="critic-1", champion_model="model", champion_actor="actor", critic_path="old"
    )
    if existing_attempt is not None:
        pending["attempt"] = existing_attempt
    campaign.state = dict(
        pending=pending, events=[], critic_id="old", critics=[], promotions=[], critic_attempt=27
    )
    campaign.save = lambda: None
    out = tmp_path / "critic-1"
    out.mkdir()
    atomic_json(
        out / "result.json",
        {"qualified": True, "test_before": {"log_loss": 0.5}, "test_after": {"log_loss": 0.4}},
    )
    campaign.begin_job = lambda *_: out
    calls = []

    def train(*args, **kwargs):
        calls.append(kwargs)
        return {"model": "candidate.safetensors"}

    campaign.train = train
    rows = [{"pair": i, "scores": [1, 0]} for i in range(2000)]
    campaign.evaluate = lambda *_, **__: (rows, {})
    campaign.critic_stage(pending)
    assert campaign.state["critic_id"] == "old"
    assert campaign.state["promotions"] == []
    assert campaign.state["events"][-1]["kind"] == "critic_inconclusive"
    assert pending["finished"]
    assert campaign.state["critic_attempt"] == (28 if existing_attempt is None else 27)
    assert calls[0]["seed"] == calls[1]["seed"]
    assert calls[0]["recipe"] == calls[1]["recipe"]


def gate_extension_fixture(tmp_path):
    from astro2.experiment_control import code_identity

    for name in ["runtime/astro2", "runtime/scripts", "tasks/policy-00055-gate"]:
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "runtime/astro2/sequential.py").write_text("UNCHANGED = True\n")
    for name in [
        "candidate.actor.npz",
        "candidate.safetensors",
        "champion.actor.npz",
        "critic.npz",
    ]:
        (tmp_path / name).write_text(name)
    atomic_json(tmp_path / "config.json", recovery_config(gate_pairs=50000).model_dump())
    atomic_json(
        tmp_path / "code.json",
        code_identity(tmp_path / "runtime/astro2", tmp_path / "runtime/scripts"),
    )
    atomic_json(tmp_path / "jobs.json", {})
    atomic_json(tmp_path / "inputs.json", {})
    actor = str(tmp_path / "candidate.actor.npz")
    state = dict(
        status="paused",
        active_job="policy-00055-gate",
        attempt=25,
        critic_attempt=31,
        champion_id="champion",
        critic_id="critic",
        models=[
            dict(id="candidate", actor_path=actor, path=str(tmp_path / "candidate.safetensors"))
        ],
        pending=dict(
            kind="policy",
            key="policy-00055",
            attempt=25,
            selection=dict(actor=actor),
            champion_actor=str(tmp_path / "champion.actor.npz"),
            critic_path=str(tmp_path / "critic.npz"),
        ),
    )
    atomic_json(tmp_path / "state.json", state)
    atomic_json(
        tmp_path / "tasks/policy-00055-gate/pairs.json",
        [dict(pair=i, scores=[1, 0], truncated=0) for i in range(8)],
    )
    return state


def test_gate_budget_extension_keeps_the_same_attempt_models_and_prefix(tmp_path):
    import json

    from extend_autopilot_gate import extend

    state = gate_extension_fixture(tmp_path)
    before = {
        p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file() and p.name != "config.json"
    }
    report = extend(tmp_path, pairs=100000, policy_fraction=0.95)
    assert report["attempt"] == 25 and report["alpha"] == 0.025 / (25 * 26)
    assert report["completed_pairs"] == 8
    assert before == {p: p.read_bytes() for p in before}
    assert json.loads((tmp_path / "state.json").read_text()) == state
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["gate_pairs"] == 100000 and config["policy_fraction"] == 0.95
    assert config["max_hours"] == 24  # No unrequested extension of active-hour budget.


@pytest.mark.parametrize("invalid", ["running", "finished", "result", "prefix", "runtime", "limit"])
def test_gate_budget_extension_refuses_invalid_or_decided_tests(tmp_path, invalid):
    from extend_autopilot_gate import extend

    state = gate_extension_fixture(tmp_path)
    if invalid == "running":
        state["status"] = "running"
    if invalid == "finished":
        state["pending"]["finished"] = True
    atomic_json(tmp_path / "state.json", state)
    if invalid == "result":
        atomic_json(tmp_path / "tasks/policy-00055-gate/result.json", {"passed": False})
    if invalid == "prefix":
        atomic_json(tmp_path / "tasks/policy-00055-gate/pairs.json", [dict(pair=1, scores=[1, 0])])
    if invalid == "runtime":
        (tmp_path / "runtime/astro2/sequential.py").write_text("CHANGED = True\n")
    before = (tmp_path / "config.json").read_bytes()
    with pytest.raises(ValueError):
        extend(tmp_path, pairs=50000 if invalid == "limit" else 100000, policy_fraction=0.95)
    assert (tmp_path / "config.json").read_bytes() == before


def recovery_config(**kwargs):
    return CampaignConfig(
        policy_id="p", critic_id="c", opponent_ids=["p"], panel_ids=["p"], **kwargs
    )


def test_stale_lanes_restart_and_optimizer_continuation_is_bounded():
    config = recovery_config(lane_max_blocks=3, exploration_recipes=True)
    champion = {"id": "champion"}
    tip = dict(champion_id="champion", model="candidate", optimizer="moments", blocks=2)
    assert worker.policy_source(tip, champion, config) == tip
    assert worker.policy_source({**tip, "blocks": 3}, champion, config) == {}
    assert worker.policy_source({**tip, "champion_id": "old"}, champion, config) == {}
    # Legacy tips have no bounded continuation count and must be rebased on migration.
    assert worker.policy_source({"champion_id": "champion"}, champion, config) == {}
    assert worker.training_recipe(config, 0) != worker.training_recipe(config, 1)


def test_new_credit_recipe_rebases_legacy_lanes_and_uses_frozen_critic():
    config = recovery_config(temporal_credit=True, exploration_recipes=True, lane_max_blocks=3)
    tip = dict(champion_id="p", model="old-estimator", blocks=1)
    assert worker.policy_source(tip, {"id": "p"}, config) == {}
    recipe = worker.training_recipe(config)
    assert recipe == dict(temperature=0.1, entropy_weight=0, gae_lambda=0.95)
    tip["recipe"] = recipe
    assert worker.policy_source(tip, {"id": "p"}, config) == tip


def test_intermediate_selection_freezes_winner_and_matching_optimizer(tmp_path):
    campaign = object.__new__(worker.Campaign)
    campaign.config = recovery_config(policy_checkpoint_games=[5000, 10000, 20000])
    campaign.save = lambda: None
    for games in [5000, 10000, 20000]:
        for suffix in ["safetensors", "actor.npz", "optimizer.npz"]:
            (tmp_path / f"g{games:08d}.{suffix}").touch()
    calls = []

    def evaluate(key, actor, opponent, pairs, seed):
        calls.append((key, seed))
        score = {5000: 0.52, 10000: 0.54, 20000: 0.49}[int(Path(actor).name[1:9])]
        return [], dict(score=score, truncated=0)

    campaign.evaluate = evaluate
    pending = dict(
        key="policy-00051", source="champion", champion_model="champion", champion_actor="actor"
    )
    trained = dict(model=str(tmp_path / "g00020000.safetensors"))
    selected = campaign.select_checkpoint(pending, trained)
    assert selected["games"] == 10000
    assert selected["optimizer"] == str(tmp_path / "g00010000.optimizer.npz")
    assert selected["source_score"] == 0.5
    assert len(calls) == 3 and len({seed for _, seed in calls}) == 1
    assert worker.stream_seed(campaign.config.seed, pending["key"] + "-gate") != calls[0][1]
    assert campaign.select_checkpoint(pending, trained) == selected
    assert len(calls) == 3  # Resume never reselects on new games.

    # A favorable selection result cannot bypass the independent gate.
    campaign.state = dict(pending=pending, attempt=23, events=[])
    campaign.evaluate = lambda *_, **__: (
        [],
        dict(score=0.5, lower=0.49, pairs=50000, passed=False),
    )
    candidate = dict(id="candidate", actor_path=selected["actor"])
    campaign.policy_gate(pending, candidate)
    assert campaign.state["attempt"] == 24
    assert campaign.state["events"][-1]["kind"] == "policy_inconclusive"


def test_selection_checkpoint_requires_matching_optimizer_before_evaluation(tmp_path):
    campaign = object.__new__(worker.Campaign)
    campaign.config = recovery_config(policy_checkpoint_games=[5000])
    with pytest.raises(ValueError, match="optimizer"):
        campaign.select_checkpoint(
            dict(key="p"), dict(model=str(tmp_path / "g00020000.safetensors"))
        )


def test_prediction_rejection_does_not_spend_strength_attempt(tmp_path):
    campaign = object.__new__(worker.Campaign)
    pending = dict(key="critic-0028")
    campaign.state = dict(pending=pending, events=[], critic_attempt=27)
    campaign.save = lambda: None
    campaign.begin_job = lambda *_: tmp_path
    atomic_json(tmp_path / "result.json", {"qualified": False})
    campaign.critic_stage(pending)
    assert campaign.state["critic_attempt"] == 27
    assert "attempt" not in pending
    assert pending["finished"]


def test_stronger_screen_does_not_spend_attempt_on_weak_nominee():
    campaign = object.__new__(worker.Campaign)
    campaign.config = recovery_config(screen_pairs=2048, screen_min_score=0.505)
    pending = dict(key="policy-00031", champion_actor="champion")
    campaign.state = dict(pending=pending, events=[], attempt=20)
    campaign.save = lambda: None
    campaign.evaluate = lambda *_, **__: ([], {"score": 0.503, "truncated": 0})
    campaign.policy_gate(pending, {"id": "candidate", "actor_path": "candidate"})
    assert campaign.state["attempt"] == 20
    assert pending["finished"]


def test_gate_futility_preserves_existing_prefix_and_never_promotes(tmp_path):
    campaign = object.__new__(worker.Campaign)
    campaign.config = recovery_config(gate_futility_pairs=2000)
    campaign.jobs = {"gate": {"seconds": 0}}
    campaign.save = lambda: None
    campaign.begin_job = lambda *_: tmp_path
    rows = [
        dict(pair=i, scores=[1, 0], truncated=0, seconds=0, searches=0, changes=0, branches=0)
        for i in range(2048)
    ]
    atomic_json(tmp_path / "pairs.json", rows)
    prefix = (tmp_path / "pairs.json").read_bytes()
    returned, report = campaign.evaluate("gate", "a", "b", 50000, 1234, 0.00005)
    assert returned == rows and not report["passed"]
    assert report["lower"] <= 0.5
    assert prefix == (tmp_path / "pairs.json").read_bytes()


def test_runtime_maintenance_preserves_champions_attempts_and_evidence(tmp_path):
    from astro2.experiment_control import code_identity
    from maintain_autopilot_runtime import PATCHABLE, maintain

    project, folder = tmp_path / "project", tmp_path / "campaign"
    folder.mkdir()
    for name in PATCHABLE:
        old = folder / "runtime" / name
        new = project / ("backend" if name.startswith("astro2/") else "") / name
        old.parent.mkdir(parents=True, exist_ok=True)
        new.parent.mkdir(parents=True, exist_ok=True)
        old.write_text("VERSION = 1\n")
        new.write_text("VERSION = 2\n")
    engine = folder / "runtime/astro2/engine.py"
    engine.write_text("RULES = 2\n")
    for name in ("champion", "actor", "critic"):
        (folder / name).write_bytes(name.encode())
    state = dict(
        status="paused",
        pending=None,
        champion_id="p",
        critic_id="c",
        attempt=20,
        critic_attempt=27,
        models=[dict(id="p", path=str(folder / "champion"), actor_path=str(folder / "actor"))],
        critics=[dict(id="c", path=str(folder / "critic"))],
    )
    atomic_json(folder / "state.json", state)
    atomic_json(folder / "config.json", recovery_config().model_dump())
    atomic_json(folder / "jobs.json", {})
    atomic_json(folder / "inputs.json", {})
    atomic_json(
        folder / "code.json", code_identity(folder / "runtime/astro2", folder / "runtime/scripts")
    )
    report = maintain(folder, project, {"lane_max_blocks": 3})
    assert report["attempt"] == 20 and report["critic_attempt"] == 27
    assert (folder / "state.json").read_bytes() == (
        Path(report["backup"]) / "state.json"
    ).read_bytes()
    assert engine.read_text() == "RULES = 2\n"
    state["pending"] = {"kind": "policy"}
    atomic_json(folder / "state.json", state)
    with pytest.raises(ValueError, match="boundary"):
        maintain(folder, project, {})


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
