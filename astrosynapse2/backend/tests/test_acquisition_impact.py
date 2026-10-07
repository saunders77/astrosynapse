from types import SimpleNamespace

import pytest
from astro2.acquisition_impact import build_report, paired_interval, safe_folder
from astro2.acquisition_impact_api import router
from astro2.cards import EXPLORER
from astro2.experiment_control import atomic_json
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_paired_intervals_are_signed_and_nonzero_when_no_discordance():
    lo, hi = paired_interval(0, 0, 64)
    assert lo < 0 < hi
    assert hi == pytest.approx(-lo)
    forward = paired_interval(20, 3, 64)
    reverse = paired_interval(3, 20, 64)
    assert forward[0] > 0
    assert forward[0] == pytest.approx(-reverse[1])
    assert paired_interval(0, 0, 0) == [None, None]


def make_report(tmp_path):
    folder = tmp_path / "miniastro_regret" / "test-run"
    folder.mkdir(parents=True)
    observation = {
        "own_authority": 20,
        "opponent_authority": 30,
        "trade": 2,
        "trade_row": [EXPLORER.to_dict()],
        "hand": [],
        "own_deck": [],
        "own_known_top": [],
        "own_discard": [],
        "own_in_play": [],
        "opponent_hidden": [],
        "opponent_known_hand": [],
        "opponent_known_top": [],
        "opponent_discard": [],
        "opponent_in_play": [],
    }
    base = {
        "turn": 5,
        "player_id": 0,
        "teacher_card": 2,
        "student_card": -1,
        "disagreement": True,
        "total_trade": 2,
        "spent": 0,
        "candidates": [
            {
                "card_id": 2,
                "score": 1,
                "features": {
                    "Your total deck size (all zones)": 10,
                    "Candidate faction cards in your deck": 0,
                },
            }
        ],
    }
    for gi, (count, outcome) in enumerate([(3, [1, 0]), (1, [0, 1])]):
        atomic_json(
            folder / f"game-{gi:05d}.json",
            {
                "game": gi,
                "source_truncated": False,
                "counts": {"eligible_positions": count + 1},
                "events": [{**base, "game": gi} for _ in range(count)],
                "roots": [
                    {
                        **base,
                        "game": gi,
                        "root": 0,
                        "observation": observation,
                        "inclusion_probability": 1 / count,
                        "outcomes": [outcome] * 64,
                    }
                ],
            },
        )
    atomic_json(
        folder / "progress.json",
        {
            "status": "complete",
            "games_completed": 2,
            "updated_at": 1,
            "reference_model": {"label": "Reference", "ref": "r"},
            "settings": {"student": "s", "rollouts": 64},
            "counts": {"eligible_positions": 6, "disagreements": 4},
            "reference_sha256": "rhash",
            "weights_sha256": "shash",
        },
    )
    return folder, build_report(folder, bootstraps=200)


def test_inverse_sampling_weights_and_frequency_use_full_source_games(tmp_path):
    folder, report = make_report(tmp_path)
    # Naive unweighted mean would be zero: roots represent 3 and 1 disagreements.
    assert report["overall"]["effect"] == pytest.approx(0.5)
    assert report["overall"]["weighted_positions"] == 4
    assert report["overall"]["burden"] == pytest.approx(2 / 6)
    assert report["disagreement_rate"] == pytest.approx(4 / 6)
    assert report["overall"]["sampled_games"] == 2
    assert report["valid_pairs"] == 128
    assert len(report["positions"]) == 2
    assert report["positions"][1]["effect"] == -1
    assert sum(g["burden"] for g in report["categories"]) == pytest.approx(
        report["overall"]["burden"]
    )
    assert sum(h["weighted_count"] for h in report["histogram"]) == 4
    assert safe_folder(folder.parent, "test-run") == folder
    with pytest.raises(ValueError):
        safe_folder(folder.parent, "../test-run")


def test_report_and_position_api(tmp_path):
    folder, report = make_report(tmp_path)
    app = FastAPI()
    app.include_router(router)
    app.state.acquire_students = SimpleNamespace(output_dir=tmp_path / "acquire_students")
    with TestClient(app) as client:
        assert client.get("/api/acquisition-impact").json()[0]["id"] == "test-run"
        prefix = "/api/acquisition-impact/test-run"
        assert client.get(prefix).json()["overall"]["effect"] == report["overall"]["effect"]
        detail = client.get(prefix + "/positions/0/0")
        assert detail.status_code == 200
        assert detail.json()["trade_row"] == [{"card": "Explorer", "count": 1}]
        assert client.get(prefix + "/positions/100/0").status_code == 404
        assert client.get(prefix + "/positions/0/100").status_code == 404
        assert client.get(prefix + "/positions/-1/0").status_code == 404
        assert client.get("/api/acquisition-impact/missing").status_code == 404
        assert "attachment" in client.get(prefix + "/download").headers["content-disposition"]
