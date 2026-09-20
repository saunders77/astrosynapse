"""Resume regressions without starting a learner or requiring Metal."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def supervisor(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "progressive_supervisor_test", scripts / "progressive_training.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_saved_budgets_survive_ui_resume_and_explicit_overrides(supervisor):
    settings = dict(hours=144, workers=8, games=512, output="old", resume=False)
    args = argparse.Namespace(hours=None, workers=None, games=2, output="new", resume=True)
    supervisor.restore_settings(args, settings)
    assert vars(args) == dict(hours=144, workers=8, games=512, output="new", resume=True)
    args.hours, args.workers = 180, 4
    supervisor.restore_settings(args, settings)
    assert (args.hours, args.workers) == (180, 4)


@pytest.mark.parametrize(
    "report,attempt,finished",
    [
        ({"pairs": 256, "score": 0.8, "passed": True}, 1, True),
        ({"pairs": 2048, "score": 0.49}, 2, True),
        ({"pairs": 8192, "score": 0.505}, 3, True),
        ({"pairs": 2048, "score": 0.51}, 2, False),
        ({"pairs": 256, "score": 0.3}, 1, False),
        ({"pairs": 2048, "score": 0.3}, None, False),
        ({"pairs": 32768, "score": 0.51}, None, True),
        ({}, None, False),
    ],
)
def test_completed_gate_is_not_extended_after_resume(supervisor, report, attempt, finished):
    assert supervisor.gate_finished(report, 32768, attempt) is finished


def test_resume_completes_interrupted_anchor_before_training(supervisor, tmp_path, monkeypatch):
    out = tmp_path / "run"
    frozen = out / "runtime/scripts/progressive_training.py"
    frozen.parent.mkdir(parents=True)
    frozen.touch()
    monkeypatch.setattr(supervisor, "__file__", str(frozen))
    monkeypatch.setattr(supervisor, "code_identity", lambda *_: {})
    settings = dict(
        hours=144,
        workers=8,
        games=512,
        round_iterations=16,
        screen_pairs=512,
        max_gate_pairs=32768,
        seed=17,
    )
    (out / "manifest.json").write_text(json.dumps(dict(settings=settings, code_identity={})))
    champion = out / "champion.npz"
    champion.write_bytes(b"test champion")
    (out / "original-champion.actor.npz").write_bytes(b"original")
    state = dict(
        elapsed=108 * 3600,
        phase="paused",
        stage=1,
        round=0,
        attempt=0,
        promotions=[{}],
        pending_gate=None,
        champion=str(champion),
    )
    (out / "state.json").write_text(json.dumps(state))
    anchor = out / "anchor-001"
    anchor.mkdir()

    def row(i):
        return dict(
            pair=i, scores=[1.0, 0.0], seconds=0.0, searches=0, changes=0, branches=0, truncated=0
        )

    (anchor / "pairs.jsonl").write_text("".join(json.dumps(row(i)) + "\n" for i in range(1792)))
    seen = []

    class Pool:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 8

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def map(self, fn, tasks, **kwargs):
            seen.extend(t[-1] for t in tasks)
            (out / "STOP").touch()
            return [row(t[-1]) for t in tasks]

    monkeypatch.setattr(supervisor.concurrent.futures, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *_a, **_k: pytest.fail("learner started before safe pause"),
    )
    monkeypatch.setattr(sys, "argv", ["progressive_training.py", "--resume", "--output", str(out)])
    supervisor.main()
    result = json.loads((out / "state.json").read_text())
    assert seen == list(range(1792, 2048))
    assert result["promotions"][0]["original_champion_benchmark"]["pairs"] == 2048
    assert result["promotions"][0]["original_champion_benchmark"]["paused"] is False
    assert result["phase"] == "paused"
    assert json.loads((out / "manifest.json").read_text())["settings"]["hours"] == 144


def test_runtime_maintenance_preserves_provenance_and_rejects_unknown_changes(
    tmp_path, monkeypatch
):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    from astro2.experiment_control import code_identity
    from maintain_progressive_runtime import maintain

    out, project = tmp_path / "run", tmp_path / "project"
    runtime = out / "runtime"
    (runtime / "astro2").mkdir(parents=True)
    (runtime / "scripts").mkdir()
    (project / "scripts").mkdir(parents=True)
    for name in ["progressive_training.py", "onpolicy_experiment.py"]:
        (runtime / "scripts" / name).write_text("version = 1\n")
        (project / "scripts" / name).write_text("version = 2\n")
    (runtime / "astro2/engine.py").write_text("rules = 1\n")
    before = code_identity(runtime / "astro2", runtime / "scripts")
    manifest = dict(code_identity=before, settings={"hours": 144})
    (out / "manifest.json").write_text(json.dumps(manifest))
    state = dict(phase="paused", stage=1, games=123)
    (out / "state.json").write_text(json.dumps(state))
    (out / "STOP").touch()
    for stage in ["stage-000", "stage-001"]:
        (out / stage).mkdir()
        (out / stage / "manifest.json").write_text(json.dumps(manifest))
    revision = maintain(out, project, "operational fix")
    after = code_identity(runtime / "astro2", runtime / "scripts")
    assert after["astro2/engine.py"] == before["astro2/engine.py"]
    assert json.loads((out / "manifest.json").read_text())["code_identity"] == after
    assert json.loads((out / "stage-001/manifest.json").read_text())["code_identity"] == after
    assert json.loads((out / "stage-000/manifest.json").read_text())["code_identity"] == before
    assert json.loads((out / "state.json").read_text()) == state
    assert (out / "STOP").exists()
    assert json.loads((Path(revision["backup"]) / "manifest.json").read_text()) == manifest
    assert maintain(out, project, "repeat")["changed_files"] == []
    (project / "backend/astro2").mkdir(parents=True)
    (project / "backend/astro2/onpolicy.py").write_text("critic = 2\n")
    (runtime / "astro2/onpolicy.py").write_text("critic = 1\n")
    # Add the old learner module to the known identity before migration.
    old_identity = code_identity(runtime / "astro2", runtime / "scripts")
    for path in [out / "manifest.json", out / "stage-001/manifest.json"]:
        payload = json.loads(path.read_text())
        payload["code_identity"] = old_identity
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="explicit --separate-critic"):
        maintain(out, project, "must opt in")
    revision = maintain(out, project, "critic correction", separate_critic=True)
    migrated = json.loads((out / "stage-001/manifest.json").read_text())
    assert migrated["separate_critic"] is True
    assert migrated["learner_version"] == 4
    assert migrated["critic_learning_rate"] == 0.0003
    assert json.loads((out / "state.json").read_text()) == state
    assert (runtime / "astro2/engine.py").read_text() == "rules = 1\n"
    assert (Path(revision["backup"]) / "runtime/astro2/onpolicy.py").read_text() == "critic = 1\n"
    (runtime / "astro2/engine.py").write_text("rules = 2\n")
    with pytest.raises(ValueError, match="unknown modification"):
        maintain(out, project, "bad patch")
