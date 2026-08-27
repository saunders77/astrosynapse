from __future__ import annotations

import threading
import time

import astro2.arena as arena_module
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
from astro2.engine import GameResult
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
    assert ArenaConfig().early_rejection is False
    assert ArenaConfig().early_rejection_min_pairs == 512
    assert ArenaConfig().early_rejection_confidence == 0.995
    with pytest.raises(ValueError):
        ArenaConfig(pairs=MAX_PAIRS + 1)
    with pytest.raises(ValueError):
        ArenaConfig(pairs=2, minimum_promotion_pairs=2)
    with pytest.raises(ValueError):
        ArenaConfig(pairs=10, automatic_promotion=True)
    with pytest.raises(ValueError):
        ArenaConfig(early_rejection=True)
    with pytest.raises(ValueError):
        ArenaConfig(
            pairs=512,
            minimum_promotion_pairs=512,
            automatic_promotion=True,
            early_rejection=True,
        )
    with pytest.raises(ValueError):
        ArenaConfig(early_rejection_confidence=1.0)


def test_arena_summary_separates_true_draws_from_truncations():
    config = ArenaConfig(pairs=1)
    model_a = ResolvedModel("baseline:balanced", "A", "baseline")
    model_b = ResolvedModel("baseline:economy", "B", "baseline")
    common = dict(
        config=config,
        model_a=model_a,
        model_b=model_b,
        first_scores=[0.5],
        second_scores=[0.5],
        pair_records=[],
        elapsed_seconds=1.0,
        total_turns=2,
        total_decisions=2,
        started_at=1.0,
    )

    clean = arena_module._summary(**common, truncated_games=0)
    truncated = arena_module._summary(**common, truncated_games=1)

    assert clean["draws"] == 2
    assert clean["truncated_games"] == 0
    assert truncated["draws"] == 1
    assert truncated["truncated_games"] == 1
    assert truncated["promotion"]["eligible"] is False  # Below the 2,000-pair minimum.
    assert truncated["truncation_adjustment"]["model_a_score"] == 0.25
    assert "scored as candidate losses" in truncated["promotion"]["recommendation"]


def test_early_rejection_hoeffding_bound_stays_wide_for_zero_variance():
    config = ArenaConfig(
        pairs=1_000,
        minimum_promotion_pairs=1_000,
        automatic_promotion=True,
        early_rejection=True,
        early_rejection_min_pairs=8,
        early_rejection_confidence=0.995,
    )

    first = arena_module._early_rejection_look(
        [0.0] * 8,
        config=config,
        look_pairs=8,
    )
    assert first["method"] == "bonferroni_one_sided_hoeffding"
    assert first["confidence_radius"] > 0.0
    assert first["upper_bound"] > 0.5
    assert first["reject"] is False

    second = arena_module._early_rejection_look(
        [0.0] * 16,
        config=config,
        look_pairs=16,
    )
    assert 0.0 < second["upper_bound"] < 0.5
    assert second["reject"] is True

    tied = arena_module._early_rejection_look(
        [0.5] * 16,
        config=config,
        look_pairs=16,
    )
    assert tied["upper_bound"] > 0.5
    assert tied["reject"] is False


def test_final_paired_interval_is_valid_for_constant_samples():
    losing = arena_module._paired_interval([0.0] * 2_000, 0.95)
    winning = arena_module._paired_interval([1.0] * 2_000, 0.95)

    assert losing["estimate"] == 0.0
    assert 0.0 < losing["high"] < 0.1
    assert winning["estimate"] == 1.0
    assert 0.9 < winning["low"] < 1.0
    assert losing["confidence_radius"] == pytest.approx(winning["confidence_radius"])


def test_only_trainer_owned_automatic_arenas_may_use_the_4000_pair_cap():
    with pytest.raises(ValueError):
        ArenaConfig(pairs=4_000)
    with pytest.raises(ValueError):
        ArenaConfig(pairs=4_000, automatic_promotion=True)

    config = ArenaConfig(
        pairs=4_000,
        automatic_promotion=True,
        trainer_scheduled=True,
    )
    assert config.pairs == 4_000


def test_full_promotion_arenas_use_all_available_workers():
    full = ArenaConfig(automatic_promotion=True, trainer_scheduled=True)
    canary = ArenaConfig(
        pairs=64,
        promotion_tier="canary",
        trainer_scheduled=True,
    )

    assert arena_module._arena_worker_processes(full, 10) == 10
    assert arena_module._arena_worker_processes(canary, 10) == 2


def test_default_arena_pool_uses_all_detected_cpus(monkeypatch):
    monkeypatch.delenv("ASTRO2_ARENA_WORKERS", raising=False)
    monkeypatch.setattr(arena_module.os, "cpu_count", lambda: 10)

    assert arena_module._default_worker_processes() == 10


@pytest.mark.parametrize(
    ("estimate", "lower_bound", "expected"),
    [
        (0.51, 0.48, True),
        (0.51, 0.50, True),
        (0.50, 0.49, False),
        (0.51, 0.479, False),
        (0.51, 0.501, False),
    ],
)
def test_promotion_extension_uses_the_requested_mean_and_interval_window(
    estimate,
    lower_bound,
    expected,
):
    config = ArenaConfig(automatic_promotion=True, trainer_scheduled=True)
    assert (
        arena_module._should_extend_promotion_evaluation(
            config=config,
            pairs_completed=2_000,
            estimate=estimate,
            lower_bound=lower_bound,
        )
        is expected
    )


def test_marginal_positive_full_evaluation_runs_one_more_2000_pair_block(
    tmp_path,
    monkeypatch,
):
    store = Store(tmp_path / "arena.sqlite3")
    run = store.create_run(RunConfig.quick())
    champion_actor = tmp_path / "champion.npz"
    candidate_actor = tmp_path / "candidate.npz"
    champion_actor.touch()
    candidate_actor.touch()
    champion = store.add_checkpoint(
        run_id=run["id"],
        label="champion",
        path=str(tmp_path / "champion.safetensors"),
        actor_path=str(champion_actor),
        games=2_000,
        champion=True,
    )
    candidate = store.add_checkpoint(
        run_id=run["id"],
        label="candidate",
        path=str(tmp_path / "candidate.safetensors"),
        actor_path=str(candidate_actor),
        games=10_000,
    )

    def fake_pair(_model_a, _model_b, _config, pair_index, **_kwargs):
        score = 1.0 if pair_index % 100 < 52 else 0.0
        return {
            "pair_index": pair_index,
            "first_score": score,
            "second_score": score,
            "turns": 2,
            "decisions": 2,
            "truncated_games": 0,
            "record": {"pair_index": pair_index},
        }

    monkeypatch.setattr(arena_module, "_play_pair", fake_pair)
    manager = ArenaManager(store, worker_processes=1, recover=False)
    created = manager.create_automatic(candidate["id"], champion["id"])
    complete = _wait_for_job(manager, created["id"])
    manager.shutdown()

    assert complete["status"] == "complete", complete.get("error")
    result = complete["result"]
    assert result["pairs_completed"] == 4_000
    assert result["pairs_requested"] == 4_000
    assert result["adaptive_extension"] == {
        "active": True,
        "initial_pairs": 2_000,
        "additional_pairs": 2_000,
        "block_pairs": 2_000,
        "maximum_pairs": 4_000,
        "look_adjusted_confidence": pytest.approx(0.975),
    }
    assert result["paired_interval"]["estimate"] == pytest.approx(0.52)
    assert 0.48 <= result["paired_interval"]["low"] <= 0.50
    assert result["promotion"]["promoted"] is False


def test_mature_evaluation_can_extend_across_multiple_blocks(tmp_path, monkeypatch):
    store = Store(tmp_path / "arena.sqlite3")
    run = store.create_run(RunConfig.quick())
    champion_actor = tmp_path / "champion.npz"
    candidate_actor = tmp_path / "candidate.npz"
    champion_actor.touch()
    candidate_actor.touch()
    champion = store.add_checkpoint(
        run_id=run["id"],
        label="champion",
        path=str(tmp_path / "champion.safetensors"),
        actor_path=str(champion_actor),
        games=2_000,
        champion=True,
    )
    candidate = store.add_checkpoint(
        run_id=run["id"],
        label="candidate",
        path=str(tmp_path / "candidate.safetensors"),
        actor_path=str(candidate_actor),
        games=10_000,
    )

    def fake_pair(_model_a, _model_b, _config, pair_index, **_kwargs):
        score = 1.0 if pair_index % 100 < 51 else 0.0
        return {
            "pair_index": pair_index,
            "first_score": score,
            "second_score": score,
            "turns": 2,
            "decisions": 2,
            "truncated_games": 0,
            "record": {"pair_index": pair_index},
        }

    monkeypatch.setattr(arena_module, "_play_pair", fake_pair)
    manager = ArenaManager(store, worker_processes=1, recover=False)
    created = manager.create_automatic(
        candidate["id"],
        champion["id"],
        extension_max_pairs=8_000,
        extension_min_lower_bound=0.0,
    )
    complete = _wait_for_job(manager, created["id"])
    manager.shutdown()

    assert complete["status"] == "complete", complete.get("error")
    result = complete["result"]
    assert result["pairs_completed"] == 8_000
    assert result["adaptive_extension"]["additional_pairs"] == 6_000
    assert result["adaptive_extension"]["maximum_pairs"] == 8_000
    assert result["adaptive_extension"]["look_adjusted_confidence"] == pytest.approx(
        0.9875
    )
    assert result["promotion"]["promoted"] is False


def test_automatic_arena_early_rejects_only_at_adjusted_safe_look(
    tmp_path,
    monkeypatch,
):
    store = Store(tmp_path / "arena.sqlite3")
    run = store.create_run(RunConfig.quick())
    champion_actor = tmp_path / "champion.npz"
    weak_actor = tmp_path / "weak.npz"
    strong_actor = tmp_path / "strong.npz"
    champion_actor.touch()
    weak_actor.touch()
    strong_actor.touch()
    champion = store.add_checkpoint(
        run_id=run["id"],
        label="champion",
        path=str(tmp_path / "champion.safetensors"),
        actor_path=str(champion_actor),
        games=2_000,
        champion=True,
    )
    weak = store.add_checkpoint(
        run_id=run["id"],
        label="weak",
        path=str(tmp_path / "weak.safetensors"),
        actor_path=str(weak_actor),
        games=10_000,
    )
    strong = store.add_checkpoint(
        run_id=run["id"],
        label="strong",
        path=str(tmp_path / "strong.safetensors"),
        actor_path=str(strong_actor),
        games=12_000,
    )
    candidate_loses = {"value": True}

    class FakeLoadedModel:
        def __init__(self, resolved):
            self.resolved = resolved

        def chooser(self, _seed):
            return object()

    class FakeGame:
        def __init__(self, *, player_names, config, **_kwargs):
            self.player_names = player_names
            self.config = config

        def run(self):
            champion_player = self.player_names.index("champion")
            candidate_player = 1 - champion_player
            winner = champion_player if candidate_loses["value"] else candidate_player
            return GameResult(
                winner=winner,
                turns=1,
                decisions=1,
                forced_choices=0,
                truncated=False,
                truncation_reason=None,
                seed=self.config.seed,
                starting_player=0,
            )

    monkeypatch.setattr(arena_module, "_LoadedModel", FakeLoadedModel)
    monkeypatch.setattr(arena_module, "Game", FakeGame)
    manager = ArenaManager(store, worker_processes=1, recover=False)
    options = {
        "pairs": 1_000,
        "minimum_promotion_pairs": 1_000,
        "early_rejection": True,
        "early_rejection_min_pairs": 8,
        "early_rejection_confidence": 0.995,
    }

    weak_job = manager.create_automatic(weak["id"], champion["id"], **options)
    weak_complete = _wait_for_job(manager, weak_job["id"])
    weak_result = weak_complete["result"]
    assert weak_complete["status"] == "complete"
    assert weak_complete["error"] is None
    assert weak_result["pairs_completed"] == 16
    assert weak_result["games_completed"] == 32
    assert weak_result["early_stopped"] is True
    assert "upper bound" in weak_result["early_stop_reason"]
    assert weak_result["early_rejection"]["planned_look_pairs"] == [
        8,
        16,
        32,
        64,
        128,
        256,
        512,
    ]
    assert weak_result["early_rejection"]["looks_completed"] == 2
    look = weak_result["early_rejection"]["latest_look"]
    assert look["method"] == "bonferroni_one_sided_hoeffding"
    assert look["look_index"] == 2
    assert look["configured_confidence"] == pytest.approx(0.995)
    assert look["bonferroni_look_alpha"] == pytest.approx((1.0 - 0.995) / 7)
    assert 0.0 < look["upper_bound"] < 0.5
    assert look["reject"] is True
    assert weak_result["promotion"]["eligible"] is False
    assert weak_result["promotion"]["promoted"] is False
    assert store.get_run(run["id"])["champion_id"] == champion["id"]
    weak_evaluation = store.checkpoint(weak["id"])["evaluation"]["latest_arena"]
    assert weak_evaluation["early_stopped"] is True

    # A strong candidate can promote at a separately planned, more stringent
    # acceptance look without waiting for all 2,000 requested pairs.
    candidate_loses["value"] = False
    strong_job = manager.create_automatic(
        strong["id"],
        champion["id"],
        **{
            **options,
            "pairs": 2_000,
            "minimum_promotion_pairs": 2_000,
            "early_acceptance": True,
            "early_acceptance_min_pairs": 1_000,
            "early_acceptance_confidence": 0.995,
        },
    )
    strong_complete = _wait_for_job(manager, strong_job["id"])
    manager.shutdown()
    strong_result = strong_complete["result"]
    assert strong_result["pairs_completed"] == 1_000
    assert strong_result["early_stopped"] is True
    assert strong_result["early_stop_outcome"] == "accepted"
    assert strong_result["early_acceptance"]["latest_look"]["accept"] is True
    assert strong_result["early_rejection"]["looks_completed"] == 7
    assert strong_result["early_rejection"]["latest_look"]["reject"] is False
    assert strong_result["promotion"]["promoted"] is True


def test_cancelling_running_or_queued_automatic_arena_prevents_promotion(tmp_path, monkeypatch):
    store = Store(tmp_path / "arena.sqlite3")
    run = store.create_run(RunConfig.quick())
    champion_actor = tmp_path / "champion.npz"
    candidate_actor = tmp_path / "candidate.npz"
    champion_actor.touch()
    candidate_actor.touch()
    champion = store.add_checkpoint(
        run_id=run["id"],
        label="champion",
        path=str(tmp_path / "champion.safetensors"),
        actor_path=str(champion_actor),
        games=1_000,
        champion=True,
    )
    candidate = store.add_checkpoint(
        run_id=run["id"],
        label="candidate",
        path=str(tmp_path / "candidate.safetensors"),
        actor_path=str(candidate_actor),
        games=2_000,
    )
    started = threading.Event()
    stop_requested = threading.Event()

    class FakeLoadedModel:
        def __init__(self, resolved):
            self.resolved = resolved

        def chooser(self, _seed):
            return object()

    class BlockingGame:
        def __init__(self, *, config, cancel_hook, **_kwargs):
            self.config = config
            self.cancel_hook = cancel_hook

        def run(self):
            started.set()
            deadline = time.monotonic() + 2.0
            while not self.cancel_hook() and time.monotonic() < deadline:
                time.sleep(0.001)
            return GameResult(
                winner=None,
                turns=0,
                decisions=0,
                forced_choices=0,
                truncated=True,
                truncation_reason="cancelled",
                seed=self.config.seed,
                starting_player=0,
            )

    monkeypatch.setattr(arena_module, "_LoadedModel", FakeLoadedModel)
    monkeypatch.setattr(arena_module, "Game", BlockingGame)
    manager = ArenaManager(store, worker_processes=1, recover=False)
    job = manager.create_automatic(
        candidate["id"],
        champion["id"],
        pairs=1_000,
        minimum_promotion_pairs=1_000,
        cancellation_hook=stop_requested.is_set,
    )
    assert started.wait(1.0)
    queued = manager.create_automatic(
        candidate["id"],
        champion["id"],
        pairs=1_000,
        minimum_promotion_pairs=1_000,
    )
    assert manager.cancel(queued["id"], timeout=1.0) is True
    assert store.arena_job(queued["id"])["status"] == "cancelled"
    assert store.arena_job(job["id"])["status"] == "running"
    stop_requested.set()
    assert manager.wait_for_job(job["id"], timeout=2.0) is True
    manager.shutdown()

    assert store.arena_job(job["id"])["status"] == "cancelled"
    assert store.get_run(run["id"])["champion_id"] == champion["id"]
    assert "latest_arena" not in store.checkpoint(candidate["id"])["evaluation"]


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
    manager = ArenaManager(store, worker_processes=1, recover=False)
    created = manager.create(
        "baseline:balanced",
        "baseline:aggressive",
        ArenaConfig(pairs=3, seed=77, max_turns=80, max_actions_per_turn=100),
    )
    assert manager.wait_for_idle(timeout=10.0) is True
    complete = manager.get(created["id"])
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
    assert result["paired_interval_method"] == "two_sided_hoeffding"
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


def test_arena_store_uses_indexed_trainer_filters(tmp_path):
    store = Store(tmp_path / "arena-filter.sqlite3")
    run = store.create_run(RunConfig.quick())
    checkpoint = store.add_checkpoint(
        run_id=run["id"],
        label="candidate",
        path=str(tmp_path / "model.safetensors"),
        actor_path=str(tmp_path / "actor.npz"),
        games=100,
    )
    trainer = store.create_arena_job(
        model_a=checkpoint["id"],
        model_b="baseline:balanced",
        config=ArenaConfig(
            pairs=8,
            promotion_tier="canary",
            trainer_scheduled=True,
        ).to_dict(),
    )
    store.create_arena_job(
        model_a="baseline:first",
        model_b="baseline:random",
        config=ArenaConfig(pairs=1).to_dict(),
    )

    selected = store.arena_jobs(
        run_id=run["id"],
        statuses=("queued",),
        promotion_tier="canary",
        trainer_scheduled=True,
    )
    assert [job["id"] for job in selected] == [trainer["id"]]


def test_arena_api_create_list_detail_and_reconnect(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setenv("ASTRO2_ARENA_WORKERS", "1")
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
        games=2_000,
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
        "pairs_completed": 2_000,
        "games_completed": 10_000,
        "model_a_score": 0.55,
        "paired_interval": {"estimate": 0.55, "low": 0.52, "high": 0.58, "samples": 2_000},
        "promotion": {},
    }
    assert not finalize_automatic_evaluation(
        store,
        job_id="manual",
        config=ArenaConfig(pairs=2_000, automatic_promotion=False),
        model_a=model_a,
        model_b=model_b,
        result=manual_result,
    )
    assert store.get_run(run["id"])["champion_id"] == champion["id"]
    assert "latest_arena" not in store.checkpoint(candidate["id"])["evaluation"]

    canary_result = {
        "pairs_completed": 64,
        "games_completed": 128,
        "model_a_score": 0.625,
        "paired_interval": {
            "estimate": 0.625,
            "low": 0.50,
            "high": 0.75,
            "samples": 64,
        },
        "promotion": {},
    }
    canary = ArenaConfig(
        pairs=64,
        minimum_promotion_pairs=64,
        promotion_tier="canary",
        automatic_promotion=False,
        trainer_scheduled=True,
    )
    assert not finalize_automatic_evaluation(
        store,
        job_id="canary",
        config=canary,
        model_a=model_a,
        model_b=model_b,
        result=canary_result,
    )
    assert store.get_run(run["id"])["champion_id"] == champion["id"]
    latest_canary = store.checkpoint(candidate["id"])["evaluation"]["latest_arena"]
    assert latest_canary["job_id"] == "canary"
    assert latest_canary["promotion_tier"] == "canary"
    assert latest_canary["promoted"] is False
    assert latest_canary["automatic"] is False
    assert latest_canary["trainer_scheduled"] is True
    assert canary_result["promotion"]["eligible"] is True
    assert canary_result["promotion"]["promoted"] is False

    automatic = ArenaConfig(pairs=2_000, automatic_promotion=True)
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

    marginal_truncated = {
        "pairs_completed": 2_000,
        "games_completed": 10_000,
        "model_a_score": 0.5272,
        "paired_interval": {
            "estimate": 0.5272,
            "low": 0.508,
            "high": 0.5464,
            "samples": 2_000,
        },
        "truncated_games": 200,
        "promotion": {},
    }
    assert not finalize_automatic_evaluation(
        store,
        job_id="marginal-truncated",
        config=automatic,
        model_a=model_a,
        model_b=model_b,
        result=marginal_truncated,
    )
    assert marginal_truncated["promotion"]["eligible"] is True
    assert marginal_truncated["promotion"]["promoted"] is False
    assert marginal_truncated["promotion"]["truncation_adjustment"]["model_a_score"] == pytest.approx(
        0.5172
    )
    assert marginal_truncated["promotion"]["truncation_adjustment"]["paired_interval"]["low"] < 0.5
    assert store.get_run(run["id"])["champion_id"] == champion["id"]

    truncated_result = {
        "pairs_completed": 2_000,
        "games_completed": 10_000,
        "model_a_score": 0.55,
        "paired_interval": {
            "estimate": 0.55,
            "low": 0.52,
            "high": 0.58,
            "samples": 2_000,
        },
        "truncated_games": 1,
        "promotion": {},
    }
    assert finalize_automatic_evaluation(
        store,
        job_id="truncated",
        config=automatic,
        model_a=model_a,
        model_b=model_b,
        result=truncated_result,
    )
    assert truncated_result["promotion"]["eligible"] is True
    assert truncated_result["promotion"]["promoted"] is True
    adjustment = truncated_result["promotion"]["truncation_adjustment"]
    assert adjustment["model_a_score"] == pytest.approx(0.54995)
    assert adjustment["paired_interval"]["low"] > 0.5
    assert "scored as losses" in truncated_result["promotion"]["recommendation"]
    assert store.get_run(run["id"])["champion_id"] == candidate["id"]

    provisional = ArenaConfig(
        pairs=1_000,
        minimum_promotion_pairs=1_000,
        promotion_tier="provisional",
        automatic_promotion=True,
    )
    provisional_result = {
        "pairs_completed": 1_000,
        "games_completed": 2_000,
        "model_a_score": 0.65,
        "paired_interval": {
            "estimate": 0.65,
            "low": 0.58,
            "high": 0.72,
            "samples": 1_000,
        },
        "promotion": {},
    }
    assert finalize_automatic_evaluation(
        store,
        job_id="provisional",
        config=provisional,
        model_a=model_a,
        model_b=model_b,
        result=provisional_result,
    )
    assert provisional_result["promotion"]["tier"] == "provisional"
    latest = store.checkpoint(candidate["id"])["evaluation"]["latest_arena"]
    assert latest["promotion_tier"] == "provisional"

    qualifying = {
        "pairs_completed": 2_000,
        "games_completed": 10_000,
        "model_a_score": 0.54,
        "wilson_interval": {"estimate": 0.54, "low": 0.53, "high": 0.55, "samples": 10_000},
        "paired_interval": {"estimate": 0.54, "low": 0.51, "high": 0.57, "samples": 2_000},
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


def test_stale_automatic_evaluation_cannot_replace_a_newer_champion(tmp_path):
    store = Store(tmp_path / "arena.sqlite3")
    run = store.create_run(RunConfig.quick())
    old_champion = store.add_checkpoint(
        run_id=run["id"],
        label="old champion",
        path=str(tmp_path / "old.safetensors"),
        actor_path=str(tmp_path / "old.npz"),
        games=2_000,
        champion=True,
    )
    stale_candidate = store.add_checkpoint(
        run_id=run["id"],
        label="stale candidate",
        path=str(tmp_path / "stale.safetensors"),
        actor_path=str(tmp_path / "stale.npz"),
        games=10_000,
    )
    newer_champion = store.add_checkpoint(
        run_id=run["id"],
        label="newer champion",
        path=str(tmp_path / "new.safetensors"),
        actor_path=str(tmp_path / "new.npz"),
        games=12_000,
        champion=True,
    )
    model_a = ResolvedModel(
        ref=stale_candidate["id"],
        label=stale_candidate["label"],
        kind="checkpoint",
        checkpoint_id=stale_candidate["id"],
    )
    model_b = ResolvedModel(
        ref=old_champion["id"],
        label=old_champion["label"],
        kind="checkpoint",
        checkpoint_id=old_champion["id"],
    )
    result = {
        "pairs_completed": 1_000,
        "games_completed": 2_000,
        "model_a_score": 0.65,
        "paired_interval": {
            "estimate": 0.65,
            "low": 0.58,
            "high": 0.72,
            "samples": 1_000,
        },
        "promotion": {},
    }
    promoted = finalize_automatic_evaluation(
        store,
        job_id="stale",
        config=ArenaConfig(
            pairs=1_000,
            minimum_promotion_pairs=1_000,
            promotion_tier="provisional",
            automatic_promotion=True,
        ),
        model_a=model_a,
        model_b=model_b,
        result=result,
    )
    assert promoted is False
    assert result["promotion"]["stale_opponent"] is True
    assert store.get_run(run["id"])["champion_id"] == newer_champion["id"]
