import json
import os

import pytest
from astro2 import server
from fastapi import HTTPException


def test_progress_is_real_saved_state_and_partial_metric_append_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    root = tmp_path / "progressive"
    run = root / "test"
    stage = run / "stage-000"
    stage.mkdir(parents=True)
    (root / "current.json").write_text(json.dumps({"path": str(run)}))
    (run / "state.json").write_text(
        json.dumps(
            dict(
                stage=0,
                games_before_stage=100,
                games=100,
                phase="training",
                pid=os.getpid(),
                heartbeat=0,
            )
        )
    )
    (run / "manifest.json").write_text(json.dumps(dict(settings={}, promotion_contract="verified")))
    (stage / "metrics.jsonl").write_text('{"games": 32}\n{"unfinished":')
    result = server.progressive_progress()
    assert result["games"] == 132
    assert result["running"]
    assert result["latest"] == {"games": 32}
    assert result["run_name"] == "Astro6 · test"
    assert result["checkpoint_name"] == "—"
    assert server.progressive_command("pause")["ok"]
    assert (run / "STOP").exists()
    with pytest.raises(HTTPException):
        server.progressive_command("delete")


def test_progress_pointer_cannot_escape_managed_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    root = tmp_path / "progressive"
    root.mkdir()
    (root / "current.json").write_text(json.dumps({"path": "/private/tmp"}))
    with pytest.raises(HTTPException):
        server.progressive_progress()


def test_astro6_discovery_resolution_and_manual_rules(tmp_path, monkeypatch):
    from astro2.arena import resolve_model
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    run = tmp_path / "progressive" / "campaign"
    stage = run / "stage-000"
    stage.mkdir(parents=True)
    actor = stage / "g00008192.actor.npz"
    actor.write_bytes(b"retained actor")
    model = stage / "g00008192.safetensors"
    (run / "state.json").write_text(
        json.dumps(
            {
                "name": "Astro6 run",
                "champion": str(actor),
                "history": [{"checkpoint": str(model), "games": 8192}],
                "promotions": [{"actor": str(actor)}],
            }
        )
    )
    with TestClient(server.app) as client:
        models = client.get("/api/models").json()
        assert len(models) == 1
        checkpoint = models[0]
        assert "Generation 1" in checkpoint["label"]
        assert "campaign" in checkpoint["run_name"]
        assert checkpoint["games"] == 8192
        assert checkpoint["actor_available"]
        assert not checkpoint["branch_compatible"]
        assert checkpoint["evaluation"] == {}
        assert client.get("/api/models", params={"run_id": checkpoint["run_id"]}).json() == models
        assert resolve_model(server.app.state.store, checkpoint["id"]).actor_path == str(actor)
        assert server.app.state.store.checkpoints() == []
        assert client.get(f"/api/models/{checkpoint['id']}/actor").content == b"retained actor"
        assert (
            client.patch(f"/api/models/{checkpoint['id']}", json={"pinned": True}).status_code
            == 409
        )
        monkeypatch.setattr(server.app.state.arena, "_start", lambda *a, **k: None)
        arena = client.post(
            "/api/arena",
            json={
                "model_a": checkpoint["id"],
                "model_b": "baseline:balanced",
                "pairs": 1,
            },
        )
        assert arena.status_code == 201, arena.text
        assert arena.json()["config"]["rules_version"] == 2
        captured = []

        def create_analysis(model_id, kind, config):
            captured.append(config)
            return {"ok": True}

        monkeypatch.setattr(server.app.state.card_analysis, "create", create_analysis)
        for kind in ("scrap", "acquire", "acquire_bucketed"):
            response = client.post(
                "/api/card-analysis", json={"model_id": checkpoint["id"], "kind": kind}
            )
            assert response.status_code == 201
            assert captured[-1].rules_version == 2
        (stage / "g00016384.actor.npz").write_bytes(b"new actor")
        refreshed = client.get("/api/models").json()
        assert len(refreshed) == 2
        assert checkpoint["id"] in {item["id"] for item in refreshed}


def test_discovery_rejects_actor_symlinks_outside_campaign(tmp_path):
    from astro2.progressive_models import progressive_models

    run = tmp_path / "progressive" / "test"
    stage = run / "stage-000"
    stage.mkdir(parents=True)
    (run / "state.json").write_text("{}")
    outside = tmp_path / "outside.actor.npz"
    outside.write_bytes(b"outside")
    (stage / "escape.actor.npz").symlink_to(outside)
    assert progressive_models(tmp_path) == []
