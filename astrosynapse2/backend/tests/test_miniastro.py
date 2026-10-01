import gzip
import json
from dataclasses import replace

import numpy as np
import pytest
from astro2 import acquire_student as students
from astro2 import miniastro
from astro2.baselines import HeuristicChooser
from astro2.engine import Game, GameConfig
from astro2.experiment_control import atomic_json


def vector_names():
    return list(miniastro.feature_vector(Game().observation(0), -1, 5, 0))


@pytest.mark.parametrize("budget", [1000, 10000, 100000])
def test_real_parameter_budget_and_exact_contributions(budget):
    names = vector_names()
    spec = miniastro.architecture(names, budget)
    params = miniastro.initialize(spec, 9)
    assert budget * 0.9 < sum(p.size for p in params.values()) <= budget
    assert sum(p.size for p in params.values()) == spec["parameter_count"]
    x = np.random.default_rng(1).normal(size=(3, 7, len(names))).astype(np.float32)
    logits, terms, units = miniastro.forward(params, spec, x, detailed=True)
    np.testing.assert_allclose(logits, terms.sum(axis=-1), atol=1e-6)
    for i, unit in enumerate(units):
        np.testing.assert_allclose(unit.sum(axis=-1), terms[..., i + 1], atol=1e-6)
    permutation = [4, 2, 1, 0, 6, 3, 5]
    permuted = miniastro.forward(params, spec, x[:, permutation])[0]
    np.testing.assert_allclose(permuted, logits[:, permutation], atol=1e-6)


def test_numpy_and_mlx_scores_and_nonzero_training_gradients_agree():
    import mlx.core as mx

    names = vector_names()
    spec = miniastro.architecture(names, 10000)
    params = miniastro.initialize(spec, 5)
    x = np.random.default_rng(3).normal(size=(4, 7, len(names))).astype(np.float32)
    expected = miniastro.forward(params, spec, x)[0]
    mx_params = {k: mx.array(v) for k, v in params.items()}
    actual = miniastro.forward_mlx(mx_params, spec, mx.array(x))
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-6)

    def loss(weights):
        scores = miniastro.forward_mlx(weights, spec, mx.array(x))
        return mx.mean(mx.logsumexp(scores, axis=1) - scores[:, 0])

    value, gradients = mx.value_and_grad(loss)(mx_params)
    mx.eval(value, gradients)
    assert np.isfinite(float(value))
    assert all(np.any(np.asarray(g) != 0) for g in gradients.values())


def test_normalization_never_uses_validation_or_test_inputs():
    data = {
        "x": np.array([[1, 2], [2, 3], [100, 500]], dtype=np.float32),
        "pointers": np.array([[0, 1], [2, -1]]),
        "splits": np.array([0, 2]),
        "targets": np.array([0, 0]),
    }
    mean, scale = miniastro.normalization(data)
    data["x"][2] = [1e9, 1e10]
    again_mean, again_scale = miniastro.normalization(data)
    np.testing.assert_array_equal(mean, again_mean)
    np.testing.assert_array_equal(scale, again_scale)


def test_miniastro_full_training_cache_export_and_recommendation(tmp_path, monkeypatch):
    monkeypatch.setattr(students, "_load_actor_encoder", lambda _: (None, None))
    monkeypatch.setattr(students, "_GreedyChooser", lambda *_: HeuristicChooser("balanced"))
    folder = tmp_path / ("f" * 32)
    folder.mkdir()
    config = students.StudentConfig(
        games=20, student_type="miniastro", parameter_budget=1000, epochs=2
    )
    atomic_json(
        folder / "job.json",
        {
            "id": folder.name,
            "model_id": "teacher",
            "model_label": "Teacher",
            "teacher_sha256": "test",
            "config": students.asdict(config),
            "created_at": 0,
        },
    )
    students.train_job(folder)
    job = json.loads((folder / "job.json").read_text())
    assert job["status"] == "complete"
    artifact = job["result"]
    assert artifact["student_type"] == "miniastro"
    assert artifact["parameter_count"] <= 1000
    assert (folder / "miniastro.npz").exists()
    assert (folder / "errors.jsonl.gz").exists()
    assert len(artifact["history"]) == 2
    o = replace(Game(config=GameConfig(starting_player=0)).observation(0), trade=3)
    advice = students.recommend(artifact, o, 5, 0)
    assert advice["student_type"] == "miniastro"
    assert sum(c["policy_share"] for c in advice["candidates"]) == pytest.approx(1)
    for c in advice["candidates"]:
        assert sum(g["value"] for g in c["contributions"]) == pytest.approx(c["score"], abs=1e-6)
    with gzip.open(folder / "errors.jsonl.gz", "rt") as stream:
        for line in stream:
            error = json.loads(line)
            assert error["game"] in artifact["split_games"]["test"]
    manager = students.StudentManager(None, tmp_path)
    assert manager.list()[0]["summary"]["parameter_count"] == artifact["parameter_count"]
    from astro2 import server
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server, "DATA_DIR", tmp_path / "api")
    with TestClient(server.app) as client:
        server.app.state.acquire_students = manager
        prefix = f"/api/acquire-students/{folder.name}"
        detail = client.get(f"{prefix}/unit?group=0&unit=0")
        assert detail.status_code == 200
        weights = miniastro.load_weights(artifact["weights_path"])
        assert detail.json()["bias"] == float(weights["b0"][0])
        assert len(detail.json()["inputs"]) == len(artifact["architecture"]["groups"][0]["indices"])
        assert client.get(f"{prefix}/unit?group=0&unit=1999").status_code == 422
        assert client.get(f"{prefix}/download/weights").status_code == 200
        assert client.get(f"{prefix}/download/tree").status_code == 404
        response = client.post(
            f"{prefix}/recommend", json={"observation": o.to_dict(), "total_trade": 5, "spent": 0}
        )
        assert response.status_code == 200
        assert response.json()["recommendation"]["card_id"] == advice["recommendation"]["card_id"]


def test_reusing_dataset_checks_teacher_and_rules(tmp_path, monkeypatch):
    manager = students.StudentManager(None, tmp_path)
    source_id = "c" * 32
    folder = tmp_path / source_id
    folder.mkdir()
    atomic_json(
        folder / "job.json",
        {
            "id": source_id,
            "created_at": 0,
            "status": "complete",
            "result": {
                "model_id": "teacher",
                "node_count": 3,
                "metrics": {},
                "config": {"games": 20, "rules_version": 1},
            },
        },
    )
    from types import SimpleNamespace

    monkeypatch.setattr(students, "resolve_model", lambda *_: SimpleNamespace(kind="checkpoint"))
    with pytest.raises(ValueError, match="teacher checkpoint"):
        manager.create("other", students.StudentConfig(games=20, source_student_id=source_id))
    with pytest.raises(ValueError, match="rules must match"):
        manager.create("teacher", students.StudentConfig(games=20, source_student_id=source_id))
