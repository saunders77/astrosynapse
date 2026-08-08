from __future__ import annotations

import time

import pytest
from astro2 import server
from astro2.arena import (
    MAX_PAIRS,
    RECOMMENDED_PAIRS,
    ArenaConfig,
    ArenaManager,
    ModelResolutionError,
    ResolvedModel,
    finalize_automatic_evaluation,
    resolve_model,
)
from astro2.config import RunConfig
from astro2.storage import Store
from fastapi.testclient import TestClient


def _wait_for_job(manager: ArenaManager, job_id: str, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if job["status"] in {"complete", "failed", "cancelled"}:
            return job
        time.sleep(0.01)
    raise AssertionError("arena job did not finish")


def test_arena_config_is_bounded_and_conservative():
    assert ArenaConfig().pairs == RECOMMENDED_PAIRS
    assert ArenaConfig().minimum_promotion_pairs == RECOMMENDED_PAIRS
    with pytest.raises(ValueError):
        ArenaConfig(pairs=MAX_PAIRS + 1)
    with pytest.raises(ValueError):
        ArenaConfig(pairs=2, minimum_promotion_pairs=2)
    with pytest.raises(ValueError):
        ArenaConfig(pairs=10, automatic_promotion=True)


def test_model_refs_accept_named_baselines_and_reject_unknown_refs(tmp_path):
    store = Store(tmp_path / "arena.sqlite3")
    model = resolve_model(store, "balanced")
    assert model.ref == "baseline:balanced"
    assert model.kind == "baseline"
    with pytest.raises(ModelResolutionError):
        resolve_model(store, "does-not-exist")


def test_paired_arena_job_is_persistent_live_and_exactly_seat_swapped(tmp_path):
    path = tmp_path / "arena.sqlite3"
    store = Store(path)
    manager = ArenaManager(store, recover=False)
    created = manager.create(
        "baseline:balanced",
        "baseline:aggressive",
        ArenaConfig(pairs=3, seed=77, max_turns=80, max_actions_per_turn=100),
    )
    complete = _wait_for_job(manager, created["id"])
    manager.shutdown()

    assert complete["status"] == "complete", complete.get("error")
    result = complete["result"]
    assert result["pairs_completed"] == 3
    assert result["games_completed"] == 6
    assert result["progress"] == 1.0
    assert result["paired_common_seeds"] is True
    assert result["exact_seat_swap"] is True
    assert result["wilson_interval"]["samples"] == 6
    assert result["paired_interval"]["samples"] == 3
    assert result["paired_interval_method"] == "nonparametric_bootstrap"
    assert result["promotion"]["automatic"] is False
    assert result["promotion"]["eligible"] is False
    assert len({pair["seed"] for pair in result["recent_pairs"]}) == 3
    assert all(
        pair["first_game_seed"] == pair["second_game_seed"] for pair in result["recent_pairs"]
    )
    assert all(pair["first_game_starting_player"] == 0 for pair in result["recent_pairs"])
    assert all(pair["second_game_starting_player"] == 0 for pair in result["recent_pairs"])
    assert not any(key.startswith("_") for key in result)

    # A new Store instance sees the finished result: browser/process ownership
    # is not required to retrieve arena progress.
    reconnected = Store(path).arena_job(created["id"])
    assert reconnected["status"] == "complete"
    assert reconnected["result"]["games_completed"] == 6


def test_arena_store_crud_round_trip(tmp_path):
    store = Store(tmp_path / "arena.sqlite3")
    job = store.create_arena_job(
        model_a="baseline:first",
        model_b="baseline:random",
        config=ArenaConfig(pairs=1).to_dict(),
        result={"pairs_completed": 0, "_first_seat_scores": []},
    )
    assert store.arena_job(job["id"])["result"] == {"pairs_completed": 0}
    updated = store.update_arena_job(job["id"], status="running", result={"pairs_completed": 1})
    assert updated["status"] == "running"
    assert store.arena_jobs()[0]["id"] == job["id"]
    store.delete_arena_job(job["id"])
    with pytest.raises(KeyError):
        store.get_arena_job(job["id"])


def test_arena_api_create_list_detail_and_reconnect(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    with TestClient(server.app) as client:
        response = client.post(
            "/api/arena",
            json={
                "model_a": "balanced",
                "model_b": "economy",
                "pairs": 2,
                "seed": 19,
                "max_turns": 80,
                "max_actions_per_turn": 100,
            },
        )
        assert response.status_code == 201
        job_id = response.json()["id"]
        deadline = time.monotonic() + 10
        detail = None
        while time.monotonic() < deadline:
            detail = client.get(f"/api/arena/{job_id}").json()
            if detail["status"] == "complete":
                break
            time.sleep(0.01)
        assert detail is not None and detail["status"] == "complete", detail
        assert detail["result"]["games_completed"] == 4
        assert client.get("/api/arena").json()[0]["id"] == job_id
        assert client.get("/api/arena/missing").status_code == 404
        assert (
            client.post(
                "/api/arena",
                json={"model_a": "missing", "model_b": "balanced", "pairs": 1},
            ).status_code
            == 404
        )

    with TestClient(server.app) as reconnected:
        persisted = reconnected.get(f"/api/arena/{job_id}")
        assert persisted.status_code == 200
        assert persisted.json()["result"]["games_completed"] == 4


def test_automatic_finalization_never_promotes_manual_or_tiny_jobs_but_promotes_full(
    tmp_path,
):
    store = Store(tmp_path / "arena.sqlite3")
    run = store.create_run(RunConfig.quick())
    candidate = store.add_checkpoint(
        run_id=run["id"],
        label="candidate",
        path=str(tmp_path / "candidate.safetensors"),
        actor_path=str(tmp_path / "candidate.npz"),
        games=10_000,
    )
    champion = store.add_checkpoint(
        run_id=run["id"],
        label="champion",
        path=str(tmp_path / "champion.safetensors"),
        actor_path=str(tmp_path / "champion.npz"),
        games=5_000,
        champion=True,
    )
    model_a = ResolvedModel(
        ref=candidate["id"],
        label=candidate["label"],
        kind="checkpoint",
        checkpoint_id=candidate["id"],
    )
    model_b = ResolvedModel(
        ref=champion["id"],
        label=champion["label"],
        kind="checkpoint",
        checkpoint_id=champion["id"],
    )

    manual_result = {
        "pairs_completed": 5_000,
        "games_completed": 10_000,
        "model_a_score": 0.55,
        "paired_interval": {"estimate": 0.55, "low": 0.52, "high": 0.58, "samples": 5_000},
        "promotion": {},
    }
    assert not finalize_automatic_evaluation(
        store,
        job_id="manual",
        config=ArenaConfig(pairs=5_000, automatic_promotion=False),
        model_a=model_a,
        model_b=model_b,
        result=manual_result,
    )
    assert store.get_run(run["id"])["champion_id"] == champion["id"]
    assert "latest_arena" not in store.checkpoint(candidate["id"])["evaluation"]

    automatic = ArenaConfig(pairs=5_000, automatic_promotion=True)
    tiny_result = {
        "pairs_completed": 8,
        "games_completed": 16,
        "model_a_score": 0.75,
        "paired_interval": {"estimate": 0.75, "low": 0.70, "high": 0.80, "samples": 8},
        "promotion": {},
    }
    assert not finalize_automatic_evaluation(
        store,
        job_id="tiny",
        config=automatic,
        model_a=model_a,
        model_b=model_b,
        result=tiny_result,
    )
    assert tiny_result["promotion"]["eligible"] is False
    assert tiny_result["promotion"]["promoted"] is False
    assert store.get_run(run["id"])["champion_id"] == champion["id"]

    qualifying = {
        "pairs_completed": 5_000,
        "games_completed": 10_000,
        "model_a_score": 0.54,
        "wilson_interval": {"estimate": 0.54, "low": 0.53, "high": 0.55, "samples": 10_000},
        "paired_interval": {"estimate": 0.54, "low": 0.51, "high": 0.57, "samples": 5_000},
        "completed_at": time.time(),
        "promotion": {},
    }
    assert finalize_automatic_evaluation(
        store,
        job_id="qualifying",
        config=automatic,
        model_a=model_a,
        model_b=model_b,
        result=qualifying,
    )
    assert qualifying["promotion"]["eligible"] is True
    assert qualifying["promotion"]["promoted"] is True
    assert store.get_run(run["id"])["champion_id"] == candidate["id"]
    assert store.checkpoint(candidate["id"])["is_champion"] is True
    assert store.checkpoint(champion["id"])["is_champion"] is False
    latest = store.checkpoint(candidate["id"])["evaluation"]["latest_arena"]
    assert latest["job_id"] == "qualifying"
    assert latest["promoted"] is True
