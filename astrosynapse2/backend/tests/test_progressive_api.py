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
