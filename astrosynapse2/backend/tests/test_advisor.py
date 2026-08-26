from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astro2 import server
from astro2.advisor import (
    AdvisorEvaluateRequest,
    AdvisorInputError,
    CheckpointAdvisor,
    decision_from_request,
    main_phase_actions,
)
from astro2.cards import CARD_BY_NAME, SCOUT, VIPER
from astro2.engine import Game, GameConfig, _InPlay
from fastapi.testclient import TestClient


def _request_payload(*, model_id: str = "checkpoint-1") -> dict:
    game = Game(config=GameConfig(seed=17, starting_player=0))
    return {
        "model_id": model_id,
        "observation": game.observation(0).to_dict(),
    }


def test_advisor_hydrates_engine_observation_and_generates_main_actions():
    game = Game(config=GameConfig(seed=11, starting_player=0))
    own = game.players[0]
    opponent = game.players[1]
    own.hand = [SCOUT, VIPER, SCOUT]
    own.trade = 4
    own.combat = 6
    own.in_play = [
        _InPlay(101, CARD_BY_NAME["Trading Post"], CARD_BY_NAME["Trading Post"], False),
        _InPlay(102, CARD_BY_NAME["Battle Station"], CARD_BY_NAME["Battle Station"], True),
    ]
    opponent.in_play = [
        _InPlay(201, CARD_BY_NAME["War World"], CARD_BY_NAME["War World"], True)
    ]
    game._uid = 201

    engine_observation = game.observation(0)
    request = AdvisorEvaluateRequest.model_validate(
        {"model_id": "checkpoint-1", "observation": engine_observation.to_dict()}
    )
    hydrated = request.observation.observation()

    assert hydrated == engine_observation
    expected = game._main_actions(own)
    generated = main_phase_actions(hydrated)
    assert [action.semantic_key for action in generated] == [
        action.semantic_key for action in expected
    ]
    assert [action.kind for action in generated].count(generated[0].kind) == 2
    assert decision_from_request(request).prompt == "Main phase"


def test_advisor_scores_generated_and_supplied_nested_decisions_with_actor_chooser(
    tmp_path,
):
    loads: list[Path] = []

    class FakeActorChooser:
        def __init__(self, actor_path):
            loads.append(Path(actor_path))
            self.actor = SimpleNamespace(spec=SimpleNamespace(objective_version=2))

        def score(self, decision):
            values = np.arange(1, len(decision.actions) + 1, dtype=np.float64)
            values /= values.sum()
            return len(decision.actions) - 1, values, 0.63

    actor_path = tmp_path / "advisor.actor.npz"
    actor_path.write_bytes(b"cached actor signature")
    advisor = CheckpointAdvisor(chooser_factory=FakeActorChooser)
    main_payload = _request_payload()
    main_payload["decision"] = {
        "family": "main",
        "prompt": "Choose Astro5's next action",
        # Main actions are generated on the server; a client list is ignored.
        "actions": [{"kind": "end_turn", "label": "Untrusted client action"}],
    }
    main_request = AdvisorEvaluateRequest.model_validate(main_payload)

    main = advisor.evaluate(actor_path, main_request)
    assert main.family == "main"
    assert main.prompt == "Choose Astro5's next action"
    assert len(main.actions) > 1
    assert main.score_semantics == "policy_probability"
    assert main.expected_win_rate == pytest.approx(0.63)
    assert sum(action.model_value for action in main.actions) == pytest.approx(1.0)
    assert sum(action.model_recommended for action in main.actions) == 1
    assert main.actions[-1].model_recommended is True

    nested_payload = _request_payload()
    nested_payload["observation"]["own_discard"].append(VIPER.to_dict())
    nested_payload["observation"]["own_in_play"].append(
        {
            "card": CARD_BY_NAME["Missile Bot"].to_dict(),
            "activated": True,
            "ally_triggered": False,
            "copied_from_stealth_needle": False,
        }
    )
    nested_payload["decision"] = {
        "family": "scrap",
        "prompt": "Missile Bot: scrap from hand or discard",
        "actions": [
            {
                "kind": "scrap_card",
                "card_id": 0,
                "source_zone": "hand",
            },
            # Physical duplicate Scouts collapse to the same semantic choice,
            # matching Game._choose before checkpoint scoring.
            {
                "kind": "scrap_card",
                "card_id": 0,
                "source_zone": "hand",
            },
            {
                "kind": "scrap_card",
                "card_id": 1,
                "source_zone": "discard",
            },
            {"kind": "decline", "card_id": 20, "ability": "scrap_any"},
        ],
    }
    nested = advisor.evaluate(
        actor_path,
        AdvisorEvaluateRequest.model_validate(nested_payload),
    )
    assert nested.family == "scrap"
    assert nested.prompt == "Missile Bot: scrap from hand or discard"
    assert [action.kind for action in nested.actions] == [
        "scrap_card",
        "scrap_card",
        "decline",
    ]
    assert nested.actions[1].source_zone == "discard"
    assert len(loads) == 1

    undefined_action = _request_payload()
    undefined_action["decision"] = {
        "family": "scrap",
        "actions": [{"kind": "scrap_card", "source_zone": "hand"}],
    }
    with pytest.raises(AdvisorInputError, match="defined card_id"):
        decision_from_request(AdvisorEvaluateRequest.model_validate(undefined_action))


def test_triggered_decision_remains_scoreable_after_its_source_was_scrapped():
    payload = _request_payload()
    payload["observation"]["opponent_in_play"].append(
        {
            "card": CARD_BY_NAME["Trading Post"].to_dict(),
            "activated": True,
            "ally_triggered": False,
            "copied_from_stealth_needle": False,
        }
    )
    payload["decision"] = {
        "family": "destroy_base",
        "prompt": "Battlecruiser: optionally destroy a base",
        "actions": [
            {
                "kind": "destroy_base",
                "card_id": CARD_BY_NAME["Battlecruiser"].card_id,
                "target_card_id": CARD_BY_NAME["Trading Post"].card_id,
                "ability": "destroy_base",
                "source_zone": "opponent_in_play",
            },
            {
                "kind": "decline",
                "card_id": CARD_BY_NAME["Battlecruiser"].card_id,
                "ability": "destroy_base",
            },
        ],
    }

    decision = decision_from_request(AdvisorEvaluateRequest.model_validate(payload))

    assert decision.family.value == "destroy_base"
    assert [action.kind.value for action in decision.actions] == ["destroy_base", "decline"]


def test_advisor_accepts_empty_trade_row_slots_after_trade_deck_exhaustion():
    payload = _request_payload()
    payload["observation"]["trade_row"][0] = None

    request = AdvisorEvaluateRequest.model_validate(payload)
    decision = decision_from_request(request)

    assert request.observation.observation().trade_row[0] is None
    assert decision.actions[-1].kind.value == "end_turn"


def test_card_catalog_and_advisor_http_contract(tmp_path, monkeypatch):
    class FakeActorChooser:
        def __init__(self, _actor_path):
            self.actor = SimpleNamespace(spec=SimpleNamespace(objective_version=2))

        def score(self, decision):
            values = np.full(len(decision.actions), 1.0 / len(decision.actions))
            return 0, values, 0.57

    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    with TestClient(server.app) as client:
        client.app.state.advisor = CheckpointAdvisor(chooser_factory=FakeActorChooser)
        cards = client.get("/api/cards")
        assert cards.status_code == 200
        catalog = cards.json()
        assert len(catalog) == 49
        assert [card["card_id"] for card in catalog] == list(range(49))
        assert next(card for card in catalog if card["name"] == "Explorer")["cost"] == 2

        run = client.post(
            "/api/runs",
            json={"preset": "quick", "name": "Advisor API", "start": False},
        ).json()
        actor_path = tmp_path / "api-advisor.actor.npz"
        actor_path.write_bytes(b"actor")
        checkpoint = client.app.state.store.add_checkpoint(
            run_id=run["id"],
            label="Advisor checkpoint",
            path=str(tmp_path / "model.safetensors"),
            actor_path=str(actor_path),
            games=42,
        )
        payload = _request_payload(model_id=checkpoint["id"])
        response = client.post("/api/advisor/evaluate", json=payload)

        assert response.status_code == 200
        result = response.json()
        assert set(result) == {
            "family",
            "prompt",
            "score_semantics",
            "expected_win_rate",
            "actions",
        }
        assert result["family"] == "main"
        assert result["prompt"] == "Main phase"
        assert result["score_semantics"] == "policy_probability"
        assert result["expected_win_rate"] == pytest.approx(0.57)
        assert result["actions"]
        assert set(result["actions"][0]) == {
            "id",
            "label",
            "kind",
            "card_id",
            "target_card_id",
            "ability",
            "source_zone",
            "amount",
            "amount2",
            "model_value",
            "model_recommended",
        }

        undefined = _request_payload(model_id=checkpoint["id"])
        undefined["observation"]["hand"][0] = {
            "card_id": -1,
            "name": "Undefined",
        }
        invalid = client.post("/api/advisor/evaluate", json=undefined)
        assert invalid.status_code == 422
        assert "card_id" in str(invalid.json()["detail"])

        missing = _request_payload(model_id="missing-checkpoint")
        assert client.post("/api/advisor/evaluate", json=missing).status_code == 404

        unavailable = client.app.state.store.add_checkpoint(
            run_id=run["id"],
            label="Pruned advisor checkpoint",
            path=str(tmp_path / "pruned.safetensors"),
            actor_path=None,
            games=43,
        )
        unavailable_payload = _request_payload(model_id=unavailable["id"])
        unavailable_response = client.post(
            "/api/advisor/evaluate", json=unavailable_payload
        )
        assert unavailable_response.status_code == 409
        assert "actor snapshot is unavailable" in unavailable_response.json()["detail"]
