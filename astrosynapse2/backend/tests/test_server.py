from pathlib import Path

from astro2 import server
from astro2.retention import prune_checkpoint_artifacts
from fastapi.testclient import TestClient


def test_run_lifecycle_api(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    with TestClient(server.app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["ok"] is True

        presets = client.get("/api/presets").json()
        astro5 = presets["astro5_search"]
        assert astro5["games_per_actor_batch"] == 4
        assert astro5["rollout_tasks_per_actor"] == 4
        assert astro5["reanalysis_fraction"] == 0.0025
        assert astro5["reanalysis_max_per_game"] == 1
        assert astro5["reanalysis_max_actions"] == 4
        assert astro5["reanalysis_rollouts_per_action"] == 1
        assert astro5["reanalysis_horizon_turns"] == 2
        assert astro5["checkpoint_every_games"] == 10_000
        assert astro5["canary_every_games"] == 10_000
        assert astro5["canary_pairs"] == 64
        assert astro5["evaluate_every_games"] == 50_000
        assert astro5["governor_interval_games"] == 500
        assert astro5["evaluation_early_acceptance"] is True
        assert astro5["evaluation_early_acceptance_min_pairs"] == 1_000
        assert astro5["policy_replay_capacity"] == 250_000
        assert astro5["policy_replay_disk_capacity"] == 5_000_000
        assert astro5["policy_replay_disk_sample_fraction"] == 0.30
        assert astro5["policy_replay_disk_shard_items"] == 8_192
        assert astro5["resume_replay_items"] == 250_000
        assert presets["astro4_m4"]["training_generation"] == 4
        assert presets["astro4_m4"]["seed"] == 20260813
        assert presets["astro4_m4"]["batch_size"] == 256
        assert presets["astro4_m4"]["counterfactual_fraction"] > 0
        assert presets["astro4_m4"]["counterfactual_loss_weight"] == 0.05
        assert presets["astro4_m4"]["rollback_rejected_candidates"] is True
        assert presets["astro4_m4"]["randomized_prior_scale"] > 0
        assert presets["astro4_m4"]["checkpoint_every_games"] == 50_000
        assert presets["astro4_m4"]["evaluate_every_games"] == 250_000
        assert presets["astro4_m4"]["policy_replay_disk_capacity"] == 0
        assert presets["astro3_m4"]["training_generation"] == 3
        assert presets["astro3_m4"]["seed"] == 20260807
        assert presets["astro3_m4"]["deployment_policy_selfplay_fraction"] == 0.2
        assert presets["quick"]["training_generation"] == 3
        assert presets["quick"]["seed"] == 20260807
        assert presets["m4_24h"]["training_generation"] == 2
        assert presets["m4_24h"]["seed"] == 20260807
        assert presets["m4_24h"]["deployment_policy_selfplay_fraction"] == 0.0

        created = client.post(
            "/api/runs",
            json={
                "preset": "quick",
                "name": "API smoke",
                "overrides": {"seed": 271828},
                "start": False,
            },
        )
        assert created.status_code == 201
        run = created.json()
        assert run["status"] == "ready"
        assert run["config"]["seed"] == 271828

        listing = client.get("/api/runs").json()
        assert listing[0]["id"] == run["id"]
        detail = client.get(f"/api/runs/{run['id']}").json()
        assert detail["run"]["name"] == "API smoke"

        events = client.get(f"/api/runs/{run['id']}/events")
        assert events.status_code == 200
        assert events.json()[0]["kind"] == "run_created"

        actor = tmp_path / "candidate.actor.npz"
        weights = tmp_path / "candidate.safetensors"
        actor.write_bytes(b"actor")
        weights.write_bytes(b"weights")
        checkpoint = client.app.state.store.add_checkpoint(
            run_id=run["id"],
            label="Candidate",
            path=str(weights),
            actor_path=str(actor),
            games=100,
        )
        model = client.get(f"/api/models?run_id={run['id']}").json()[0]
        assert model["id"] == checkpoint["id"]
        assert model["size_bytes"] == len(b"actorweights")
        assert model["evaluated"] is False
        enriched_run = next(
            item for item in client.get("/api/runs").json() if item["id"] == run["id"]
        )
        assert enriched_run["latest_model"]["label"] == "Candidate"
        assert enriched_run["deployment_model"] is None
        assert (
            client.patch(f"/api/models/{checkpoint['id']}", json={"pinned": True}).json()[
                "is_pinned"
            ]
            is True
        )
        assert client.get(f"/api/models/{checkpoint['id']}/actor").content == b"actor"

        other = client.post(
            "/api/runs",
            json={"preset": "quick", "name": "Other run", "start": False},
        ).json()
        other_checkpoint = client.app.state.store.add_checkpoint(
            run_id=other["id"],
            label="Other candidate",
            path=str(weights),
            actor_path=str(actor),
            games=100,
        )
        own_job = client.app.state.store.create_arena_job(
            model_a=checkpoint["id"],
            model_b="baseline:balanced",
            config={"pairs": 8},
        )
        client.app.state.store.create_arena_job(
            model_a=other_checkpoint["id"],
            model_b="baseline:balanced",
            config={"pairs": 8},
        )
        filtered_jobs = client.get(f"/api/arena?run_id={run['id']}").json()
        assert [job["id"] for job in filtered_jobs] == [own_job["id"]]
        assert client.get("/api/arena?run_id=missing").status_code == 404

        patched = client.patch(
            f"/api/runs/{run['id']}/config",
            json={"changes": {"epsilon_end": 0.04}},
        )
        assert patched.status_code == 200
        assert patched.json()["config"]["epsilon_end"] == 0.04


def test_models_hide_tainted_random_restart_lineage(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    with TestClient(server.app) as client:
        run = client.post(
            "/api/runs",
            json={"preset": "quick", "name": "Recovery", "start": False},
        ).json()
        store = client.app.state.store
        verified = store.add_checkpoint(
            run_id=run["id"], label="Candidate 98k", path=str(tmp_path / "verified"),
            actor_path=None, games=98_000, evaluation={"reason": "pause"},
        )
        bad_anchor = store.add_checkpoint(
            run_id=run["id"], label="Champion 98k", path=str(tmp_path / "bad"),
            actor_path=None, games=98_000, champion=True,
            evaluation={"reason": "initial random model"},
        )
        descendant = store.add_checkpoint(
            run_id=run["id"], label="Candidate 127k", path=str(tmp_path / "child"),
            actor_path=None, games=127_000, parent_id=bad_anchor["id"],
            evaluation={"reason": "final"},
        )

        visible = client.get(f"/api/models?run_id={run['id']}").json()
        audited = client.get(
            f"/api/models?run_id={run['id']}&include_tainted=true"
        ).json()

        assert [item["id"] for item in visible] == [verified["id"]]
        assert {item["id"] for item in audited} == {
            verified["id"], bad_anchor["id"], descendant["id"]
        }
        tainted = {item["id"] for item in audited if item["integrity_status"] != "verified_lineage"}
        assert tainted == {bad_anchor["id"], descendant["id"]}

def test_run_seed_rejects_values_that_cannot_round_trip_through_the_dashboard(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    with TestClient(server.app) as client:
        for invalid_seed in (-1, 9_007_199_254_740_992):
            response = client.post(
                "/api/runs",
                json={
                    "preset": "astro3_m4",
                    "overrides": {"seed": invalid_seed},
                    "start": False,
                },
            )
            assert response.status_code == 409


def test_card_analysis_api_queues_and_polls_a_fixed_thousand_game_candidate_job(
    tmp_path, monkeypatch
):
    calls = []

    class FakeCardAnalysisManager:
        def __init__(self, store, output_dir):
            self.store = store
            self.output_dir = output_dir

        def create(self, model_id, kind, config):
            calls.append((model_id, str(kind), config))
            return {
                "id": "analysis-1",
                "status": "queued",
                "model_id": model_id,
                "kind": str(kind),
                "config": config.to_dict(),
                "result": {
                    "games_requested": config.games,
                    "games_completed": 0,
                    "progress": 0.0,
                },
            }

        def get(self, job_id):
            if job_id != "analysis-1":
                raise KeyError(job_id)
            return {"id": job_id, "status": "running", "result": {"games_completed": 7}}

        def list(self, *, limit, model_id=None):
            return [{"id": "analysis-1", "model_id": model_id, "limit": limit}]

        def shutdown(self):
            pass

    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "CardAnalysisManager", FakeCardAnalysisManager)
    with TestClient(server.app) as client:
        response = client.post(
            "/api/card-analysis",
            json={"model_id": "candidate-42", "kind": "scrap", "games": 1_000},
        )

        assert response.status_code == 201
        assert response.json()["config"]["games"] == 1_000
        assert calls[0][0:2] == ("candidate-42", "scrap")
        assert client.get("/api/card-analysis/analysis-1").json()["status"] == "running"
        assert client.get("/api/card-analysis/missing").status_code == 404
        listing = client.get(
            "/api/card-analysis?limit=3&model_id=candidate-42"
        ).json()
        assert listing == [{"id": "analysis-1", "model_id": "candidate-42", "limit": 3}]
        assert client.get("/api/card-analysis?run_id=missing").status_code == 404


def test_generation_two_custom_run_can_be_loaded_and_cloned(tmp_path, monkeypatch):
    """Exercise the request shape emitted by the GUI for a loaded custom run."""

    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    with TestClient(server.app) as client:
        source = client.post(
            "/api/runs",
            json={
                "preset": "m4_24h",
                "name": "Legacy custom source",
                "overrides": {
                    "preset": "custom",
                    "training_generation": 2,
                    "seed": 8675309,
                    "deployment_policy_selfplay_fraction": 0.0,
                },
                "start": False,
            },
        )
        assert source.status_code == 201

        source_config = client.get(f"/api/runs/{source.json()['id']}").json()["run"]["config"]
        assert source_config["preset"] == "custom"
        assert source_config["training_generation"] == 2
        assert source_config["seed"] == 8675309
        assert source_config["deployment_policy_selfplay_fraction"] == 0.0

        # The GUI selects the generation-compatible launch base, sends all
        # represented fields as overrides, and sends the deployment fraction
        # explicitly instead of inheriting Astro3's 0.20 default.
        clone_overrides = dict(source_config)
        clone_overrides.pop("name")
        clone = client.post(
            "/api/runs",
            json={
                "preset": "m4_24h",
                "name": "Legacy custom clone",
                "overrides": clone_overrides,
                "start": False,
            },
        )
        assert clone.status_code == 201
        assert clone.json()["name"] == "Legacy custom clone"
        assert clone.json()["config"]["training_generation"] == 2
        assert clone.json()["config"]["seed"] == 8675309
        assert clone.json()["config"]["deployment_policy_selfplay_fraction"] == 0.0


def test_pruned_checkpoint_history_is_explicitly_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    with TestClient(server.app) as client:
        run = client.post(
            "/api/runs",
            json={"preset": "quick", "name": "Retention API", "start": False},
        ).json()
        root = tmp_path / "checkpoints" / run["id"]
        root.mkdir(parents=True)
        checkpoints = []
        for index in range(4):
            stem = root / f"g{index:010d}-api"
            model = Path(f"{stem}.safetensors")
            actor = Path(f"{stem}.actor.npz")
            optimizer = Path(f"{stem}.optimizer.npz")
            replay = Path(f"{stem}.replay.npz")
            for path in (model, actor, optimizer, replay, Path(f"{model}.json")):
                path.write_bytes(path.name.encode())
            checkpoints.append(
                client.app.state.store.add_checkpoint(
                    run_id=run["id"],
                    label=f"API checkpoint {index}",
                    path=str(model),
                    actor_path=str(actor),
                    games=index * 100,
                    champion=index == 1,
                    evaluation={
                        "artifacts": {
                            "optimizer_path": str(optimizer),
                            "replay_path": str(replay),
                            "replay_items": 10,
                        }
                    },
                )
            )

        prune_checkpoint_artifacts(
            client.app.state.store,
            run["id"],
            keep_checkpoints=2,
        )
        models = client.get(f"/api/models?run_id={run['id']}").json()
        stale = next(item for item in models if item["id"] == checkpoints[0]["id"])
        assert stale["artifact_state"] == "pruned"
        assert stale["actor_available"] is False
        assert stale["model_available"] is False
        assert stale["playable"] is False
        assert stale["actor_downloadable"] is False
        assert stale["evaluation"]["artifact_retention"]["pruned"] is True
        assert len(models) == 4

        actor_response = client.get(f"/api/models/{stale['id']}/actor")
        assert actor_response.status_code == 409
        assert "pruned by checkpoint retention" in actor_response.json()["detail"]
        game_response = client.post(
            "/api/games",
            json={"model_id": stale["id"], "seed": 7, "human_starts": True},
        )
        assert game_response.status_code == 409
        assert "pruned by checkpoint retention" in game_response.json()["detail"]
