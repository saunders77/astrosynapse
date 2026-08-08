from astro2.play import PlayManager


def test_human_game_advances_after_legal_choice():
    manager = PlayManager()
    initial = manager.create(seed=17, human_starts=True)
    assert initial["status"] == "your_turn"
    assert initial["decision"]["actions"]
    assert len(initial["card_zones"]["own"]["hand"]) == 3
    assert len(initial["card_zones"]["own"]["deck"]) == 7
    assert len(initial["card_zones"]["opponent"]["hand"]) == 5
    assert len(initial["card_zones"]["opponent"]["deck"]) == 5
    assert [card["card_id"] for card in initial["card_zones"]["own"]["deck"]] == sorted(
        card["card_id"] for card in initial["card_zones"]["own"]["deck"]
    )

    session = manager.get(initial["id"])
    next_state = session.choose(0)
    assert next_state["id"] == initial["id"]
    assert next_state["action_log"]
    manager.shutdown()
    assert manager.list() == []
