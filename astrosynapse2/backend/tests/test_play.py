from astro2.play import PlayManager


def test_human_game_advances_after_legal_choice():
    manager = PlayManager()
    initial = manager.create(seed=17, human_starts=True)
    assert initial["status"] == "your_turn"
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
