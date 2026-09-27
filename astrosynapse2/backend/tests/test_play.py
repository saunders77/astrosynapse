from types import SimpleNamespace

import numpy as np
import pytest
from astro2.encoding import FAMILY_COUNT, Encoder
from astro2.model import ModelSpec, build_model, export_actor
from astro2.play import ActorChooser, PlayManager


def test_human_game_advances_after_legal_choice():
    manager = PlayManager()
    initial = manager.create(seed=17, human_starts=True)
    assert initial["rules_version"] == 2
    assert initial["status"] == "your_turn"
    assert initial["model_score_semantics"] is None
    assert initial["expected_win_rate"] is None
    assert initial["decision"]["actions"]
    play_actions = [
        action for action in initial["decision"]["actions"] if action["kind"] == "play_card"
    ]
    assert play_actions
    assert all(action["card_id"] >= 0 for action in play_actions)
    assert len(initial["card_zones"]["own"]["hand"]) == 3
    assert len(initial["card_zones"]["own"]["deck"]) == 7
    assert set(initial["card_zones"]["opponent"]) == {"hidden"}
    assert len(initial["card_zones"]["opponent"]["hidden"]) == 10
    assert len(initial["observation"]["opponent_hidden"]) == 10
    assert initial["observation"]["opponent_known_hand"] == []
    assert [card["card_id"] for card in initial["card_zones"]["own"]["deck"]] == sorted(
        card["card_id"] for card in initial["card_zones"]["own"]["deck"]
    )

    session = manager.get(initial["id"])
    next_state = session.choose(0)
    assert next_state["id"] == initial["id"]
    assert next_state["action_log"]
    manager.shutdown()
    assert manager.list() == []


def test_checkpoint_opponent_can_move_first_and_expose_the_model_lens(monkeypatch):
    def fake_init(chooser, _actor_path):
        chooser.actor = SimpleNamespace(spec=SimpleNamespace(objective_version=2))

    def fake_score(_chooser, decision):
        probabilities = np.full(len(decision.actions), 1.0 / len(decision.actions))
        return 0, probabilities, 0.61

    monkeypatch.setattr(ActorChooser, "__init__", fake_init)
    monkeypatch.setattr(ActorChooser, "score", fake_score)

    manager = PlayManager()
    initial = manager.create(
        seed=23,
        human_starts=False,
        actor_path="test.actor.npz",
        model_label="Test checkpoint",
    )

    assert initial["status"] == "your_turn"
    assert initial["error"] is None
    assert initial["decision"]["actions"]
    assert initial["model_score_semantics"] == "policy_probability"
    assert initial["expected_win_rate"] == pytest.approx(0.61)
    assert sum(action["model_value"] for action in initial["decision"]["actions"]) == pytest.approx(1.0)
    manager.shutdown()


def test_astro4_play_snapshot_exposes_separate_state_win_rate(tmp_path):
    encoder = Encoder(version=2)
    spec = ModelSpec(
        state_size=encoder.state_size,
        action_size=encoder.action_size,
        families=FAMILY_COUNT,
        encoder_version=2,
        hidden_size=32,
        action_hidden_size=16,
        residual_blocks=1,
        bootstrap_heads=3,
        objective_version=2,
    )
    actor_path = export_actor(build_model(spec), spec, tmp_path / "astro4.actor.npz")
    manager = PlayManager()
    initial = manager.create(seed=23, human_starts=True, actor_path=actor_path)

    assert initial["model_score_semantics"] == "policy_probability"
    assert 0 <= initial["expected_win_rate"] <= 1
    policy_shares = [action["model_value"] for action in initial["decision"]["actions"]]
    assert sum(policy_shares) == pytest.approx(1.0)
    manager.shutdown()


def test_play_all_batches_original_hand_and_logs_each_card():
    manager = PlayManager()
    try:
        initial = manager.create(seed=17, human_starts=True)
        session = manager.get(initial["id"])
        assert initial["can_play_all"]
        assert not any(a["kind"] == "play_all" for a in initial["decision"]["actions"])
        state = session.play_all()
        assert state["status"] == "your_turn"
        assert state["observation"]["hand"] == []
        assert len(state["action_log"]) == 3
        assert all(entry["action"]["kind"] == "play_card" for entry in state["action_log"])
        assert not state["can_play_all"]
        with pytest.raises(ValueError):
            session.play_all()
    finally:
        manager.shutdown()


@pytest.mark.parametrize("card_id, available", [(10, True), (22, False)])
def test_play_all_probe_preserves_state_and_stops_for_choices(card_id, available):
    from astro2.cards import CARD_BY_ID

    manager = PlayManager()
    try:
        initial = manager.create(seed=17, human_starts=True)
        session = manager.get(initial["id"])
        with session._condition:
            player = session.game.players[0]
            player.hand = [CARD_BY_ID[card_id], CARD_BY_ID[0]]
            player.deck = [CARD_BY_ID[1], CARD_BY_ID[1]]
            pending = session._pending
            # Keep the decision object used by the waiting engine, updating its
            # legal options to match this deliberately constructed position.
            object.__setattr__(pending, "actions", session.game._main_actions(player))
            before = session.game.state_dict(include_hidden=True)
            assert bool(session._play_all_plan()) is available
            assert session.game.state_dict(include_hidden=True) == before
            assert not session._action_log
        if available:
            state = session.play_all()
            assert len(state["observation"]["hand"]) == 1
            assert len(state["action_log"]) == 2
        else:
            with pytest.raises(ValueError):
                session.play_all()
    finally:
        manager.shutdown()
