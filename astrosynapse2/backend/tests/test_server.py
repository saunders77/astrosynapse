from astro2 import server
from fastapi.testclient import TestClient


def test_run_lifecycle_api(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    with TestClient(server.app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["ok"] is True

        created = client.post(
            "/api/runs",
            json={"preset": "quick", "name": "API smoke", "start": False},
        )
        assert created.status_code == 201
        run = created.json()
        assert run["status"] == "ready"

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
        assert client.patch(
            f"/api/models/{checkpoint['id']}", json={"pinned": True}
        ).json()["is_pinned"] is True
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
