import gzip
import json
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest
from astro2 import acquire_student as student
from astro2 import server
from astro2.baselines import HeuristicChooser
from astro2.cards import CARD_BY_ID
from astro2.engine import Action, ActionKind, Decision, DecisionFamily, Game, GameConfig
from astro2.experiment_control import atomic_json
from fastapi.testclient import TestClient


def observation(**changes):
    base = Game(config=GameConfig(seed=19, starting_player=0)).observation(0)
    return replace(
        base, turn=1, trade_row=tuple(CARD_BY_ID[i] for i in (4, 11, 15, 30, 40)), **changes
    )


def event(sampler, o, action):
    # Single-action decisions must count just as much as any other moment.
    sampler.observe(o.player_id, Decision(DecisionFamily.MAIN, o, (action,)), action)


def sample_turn(seed):
    sampler = student.TurnSampler(seed)
    actions = [
        Action(ActionKind.PLAY_CARD, card_id=0),
        Action(ActionKind.ACQUIRE, card_id=2, amount=2),
        Action(ActionKind.PLAY_CARD, card_id=0),
        Action(ActionKind.ACQUIRE, card_id=4, amount=2),
        Action(ActionKind.END_TURN),
    ]
    for trade, action in zip((0, 2, 0, 3, 1), actions, strict=True):
        event(sampler, observation(trade=trade), action)
    return sampler.rows(0)[0]


def test_uniform_moments_next_acquisition_and_hindsight_trade():
    histogram = Counter()
    for seed in range(1500):
        row = sample_turn(seed)
        histogram[row["moment"]] += 1
        assert row["moments"] == 5
        assert row["target"] == [2, 2, 4, 4, -1][row["moment"]]
        assert row["total_trade"] == 5  # Includes trade generated after first purchase.
        assert row["spent"] == [0, 0, 2, 2, 4][row["moment"]]
    assert all(240 < n < 360 for n in histogram.values())


def test_free_acquisition_uses_target_not_source_card():
    sampler = student.TurnSampler(1)
    event(
        sampler,
        observation(hand=(CARD_BY_ID[5],)),
        Action(ActionKind.FREE_ACQUIRE, card_id=5, target_card_id=40),
    )
    row = sampler.rows(0)[0]
    assert row["target"] == 40
    assert row["spent"] == 0
    assert row["excluded_reason"] is None


def test_unseen_future_market_target_is_not_mislabeled_none():
    sampler = student.TurnSampler(1)
    event(sampler, observation(), Action(ActionKind.ACQUIRE, card_id=3, amount=6))
    row = sampler.rows(0)[0]
    assert row["target"] == 3
    assert row["excluded_reason"] == "future_market_card"
    assert student.metrics({"id": 1, "score": 0}, [row])["all_turn_accuracy"] == 0


def test_terminal_turn_and_truncation_do_not_cross_turn_boundaries():
    sampler = student.TurnSampler(12)
    event(sampler, observation(), Action(ActionKind.END_TURN))
    event(
        sampler,
        replace(observation(), turn=2, player_id=1),
        Action(ActionKind.ACQUIRE, card_id=2, amount=2),
    )
    rows = sampler.rows(0, truncated=True)
    assert rows[0]["target"] == -1
    assert rows[0]["excluded_reason"] is None
    assert rows[1]["excluded_reason"] == "unfinished_turn"


def test_real_games_capture_both_players_and_forced_moments():
    sampler = student.TurnSampler(11)
    chooser = HeuristicChooser("balanced")
    game = Game(
        config=GameConfig(seed=42, rules_version=2),
        choosers=(chooser, chooser),
        decision_hook=sampler.observe,
    )
    result = game.run()
    rows = sampler.rows(0, truncated=result.truncated)
    assert len(rows) == result.turns
    assert sum(row["moments"] for row in rows) == result.decisions
    assert {row["player_id"] for row in rows} == {0, 1}
    for row in rows:
        assert row["total_trade"] >= row["spent"] + row["observation"]["trade"]
        assert row["excluded_reason"] or row["target"] in row["candidates"]


def test_tree_learns_context_dependent_preferences_with_bounded_size():
    rows = []
    for i in range(200):
        # A small deck prefers A; a large deck prefers B, despite identical options.
        small = i % 2 == 0
        rows.append(
            {
                "x": [[0, 0], [1, int(small)], [1, int(not small)]],
                "candidates": [-1, 4, 11],
                "target": 4 if small else 11,
                "excluded_reason": None,
            }
        )
    tree = student.fit_tree(rows, ["Candidate is a card", "Candidate fits deck"], 11)
    assert student.tree_size(tree) <= 11
    assert student.metrics(tree, rows)["accuracy"] == 1
    assert "Candidate fits deck" in student.english_tree(tree)
    assert "Yes →" in student.english_tree(tree)
    assert tree == student.fit_tree(rows, ["Candidate is a card", "Candidate fits deck"], 11)


def test_candidate_set_and_inference_share_feature_schema():
    o = replace(
        observation(trade=2),
        explorers_remaining=0,
        trade_row=(CARD_BY_ID[4], CARD_BY_ID[4], None, None, None),
    )
    assert student.candidate_ids(o) == [-1, 4]
    names = list(student.features(o, -1, 5, 2))
    none_index = names.index("Candidate is None")
    tree = {
        "id": 1,
        "feature": none_index,
        "name": names[none_index],
        "threshold": 0.5,
        "left": {"id": 2, "score": 1},
        "right": {"id": 3, "score": 0},
    }
    artifact = {
        "schema_version": 1,
        "feature_names": names,
        "tree": tree,
        "config": {"rules_version": 2},
    }
    advice = student.recommend(artifact, o, 5, 2)
    assert advice["recommendation"]["card_id"] == 4
    assert advice["recommendation"]["path"][0]["answer"] == "yes"
    with pytest.raises(ValueError, match="must cover"):
        student.recommend(artifact, o, 3, 2)
    with pytest.raises(ValueError, match="finite"):
        student.recommend(artifact, o, float("nan"), 0)
    artifact["tree"] = {"id": 1, "score": 0.4}
    assert student.recommend(artifact, o, 5, 2)["recommendation"]["card_id"] == -1


def test_whole_game_split_is_disjoint_and_reproducible():
    splits = student.split_games(100, 11)
    assert splits == student.split_games(100, 11)
    assert Counter(splits.values()) == {"train": 70, "validation": 15, "test": 15}


def test_full_training_persists_tree_and_auditable_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(student, "_load_actor_encoder", lambda _: (None, None))
    monkeypatch.setattr(student, "_GreedyChooser", lambda *_: HeuristicChooser("balanced"))
    job_id = "a" * 32
    folder = tmp_path / job_id
    folder.mkdir()
    job = {
        "id": job_id,
        "model_id": "teacher",
        "model_label": "Teacher",
        "teacher_sha256": "test",
        "created_at": 0,
        "config": asdict_config(20),
    }
    atomic_json(folder / "job.json", job)
    student.train_job(folder)
    manager = student.StudentManager(None, tmp_path)
    saved = manager.get(job_id)
    assert saved["status"] == "complete"
    artifact = manager.artifact(job_id)
    assert artifact["node_count"] <= 11
    assert "always_none_accuracy" in artifact["metrics"]["test"]
    with gzip.open(folder / "samples.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == artifact["counts"]["turns"]
    assert len({(r["game"], r["turn"], r["player_id"]) for r in rows}) == len(rows)
    assert all("observation" in r for r in rows)
    for split, ids in artifact["split_games"].items():
        assert all(r["game"] in ids for r in rows if r["split"] == split)
    assert manager.list()[0]["summary"]["node_count"] == artifact["node_count"]


def asdict_config(games):
    return {"games": games, "seed": 45, "max_nodes": 11, "rules_version": 2}


def test_student_api_create_recommend_download_and_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    with TestClient(server.app) as client:
        assert client.get("/api/acquire-students").json() == []
        assert client.get("/api/acquire-students/missing").status_code == 404
        assert (
            client.post(
                "/api/acquire-students", json={"model_id": "missing", "max_nodes": 12}
            ).status_code
            == 422
        )
        assert client.post("/api/acquire-students", json={"model_id": "missing"}).status_code == 404
        o = Game(config=GameConfig(seed=11, starting_player=0)).observation(0)
        artifact = {
            "schema_version": 1,
            "config": {"rules_version": 2},
            "feature_names": list(student.features(o, -1, 5, 0)),
            "tree": {"id": 1, "score": 0.5},
            "node_count": 1,
            "metrics": {},
        }
        job_id = "b" * 32
        folder = tmp_path / "acquire_students" / job_id
        folder.mkdir()
        atomic_json(
            folder / "job.json",
            {"id": job_id, "status": "complete", "created_at": 0, "result": artifact},
        )
        (folder / "tree.txt").write_text("English tree")
        response = client.post(
            f"/api/acquire-students/{job_id}/recommend",
            json={"observation": o.to_dict(), "total_trade": 5, "spent": 0},
        )
        assert response.status_code == 200
        assert response.json()["recommendation"]["name"] == "None"
        assert client.get(f"/api/acquire-students/{job_id}/download/tree").text == "English tree"
        assert client.get(f"/api/acquire-students/{job_id}/download/teacher").status_code == 404


def test_manager_copies_teacher_and_rejects_concurrent_jobs(tmp_path, monkeypatch):
    teacher = tmp_path / "original.npz"
    teacher.write_bytes(b"immutable teacher")
    monkeypatch.setattr(
        student,
        "resolve_model",
        lambda *_: SimpleNamespace(kind="checkpoint", actor_path=teacher, label="Teacher"),
    )
    monkeypatch.setattr(
        student.subprocess, "Popen", lambda *_, **__: SimpleNamespace(poll=lambda: None)
    )
    manager = student.StudentManager(None, tmp_path / "students")
    job = manager.create("teacher", student.StudentConfig(games=20))
    assert (manager.folder(job["id"]) / "teacher.actor.npz").read_bytes() == teacher.read_bytes()
    with pytest.raises(ValueError, match="already training"):
        manager.create("teacher", student.StudentConfig(games=20))
    manager.cancel(job["id"])
    assert (manager.folder(job["id"]) / "STOP").exists()
    with pytest.raises(KeyError):
        manager.folder("../outside")
