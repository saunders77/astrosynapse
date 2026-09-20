from dataclasses import replace

import pytest
from astro2.cards import CARD_BY_NAME as CARDS
from astro2.engine import ActionKind as K
from astro2.engine import Decision, Game, _InPlay, model_action_indices
from astro2.engine import DecisionFamily as F
from astro2.lethal import find_lethal_plan


def position(hand=(), bases=(), enemy=(), deck=(), discard=(), authority=10):
    game = Game()
    own, opponent = game.players
    own.hand = [CARDS[name] for name in hand]
    own.deck = [CARDS[name] for name in deck]
    own.discard = [CARDS[name] for name in discard]
    own.known_top.clear()
    own.revealed_hand.clear()
    own.in_play = [_InPlay(i + 1, CARDS[name], CARDS[name]) for i, name in enumerate(bases)]
    opponent.in_play = [_InPlay(i + 101, CARDS[name], CARDS[name]) for i, name in enumerate(enemy)]
    opponent.authority = authority
    game._uid = 200
    game.choosers[0] = lambda _, d: model_action_indices(d)[0]
    return game


def decision(game):
    return Decision(F.MAIN, game.observation(0), game._main_actions(game.players[0]))


def finish(game):
    before = game.card_conservation()
    history = []
    game.decision_hook = lambda _, d, a: history.append((d.family, a))
    own = game.players[0]
    for _ in range(100):
        if game._winner is not None:
            break
        selected = game._choose(own, F.MAIN, game._main_actions(own), "Main phase")
        assert selected.kind != K.END_TURN
        game._apply_main_action(own, selected)
    assert game._winner == 0
    assert game.card_conservation() == before
    return history


@pytest.mark.parametrize(
    "hand,enemy,authority",
    [
        (["Viper", "Viper"], [], 2),
        (["Explorer"], [], 2),
        (["Dreadnaught"], ["Battle Station"], 7),
        (["Missile Mech", "Viper"], ["Battle Station"], 7),
        (["Command Ship", "Cutter"], ["Battle Station"], 9),
        (["Blob Destroyer", "Ram"], ["Battle Station", "Space Station"], 9),
        (["Battlecruiser"], ["Battle Station"], 5),
        (["Patrol Mech"], [], 5),
        (["Patrol Mech", "Stealth Needle"], [], 10),
        (["Cutter", "Mech World"], [], 4),
        (["Viper"], ["Fleet HQ"], 1),
    ],
)
def test_verified_lethal_uses_legal_actions(hand, enemy, authority):
    game = position(hand=hand, enemy=enemy, authority=authority)
    assert find_lethal_plan(game, decision(game))
    finish(game)


def test_turn_start_includes_base_resources_and_scrap():
    game = position(hand=["Viper"], bases=["Battle Station", "War World"], authority=9)
    game._take_turn(game.players[0])
    assert game._winner == 0


def test_draw_whole_deck_and_recheck_on_draw():
    game = position(hand=["Command Ship"], deck=["Dreadnaught", "Battle Blob"], authority=29)
    assert find_lethal_plan(game, decision(game))
    finish(game)
    # A partial draw isn't forecast, but the newly drawn hand is reconsidered.
    game = position(hand=["Corvette"], deck=["Scout", "Dreadnaught"], authority=13)
    assert not find_lethal_plan(game, decision(game))
    game._play_card(game.players[0], 0)
    assert find_lethal_plan(game, decision(game))
    finish(game)


def test_partial_deck_draw_and_discard_are_not_speculative_combat():
    game = position(hand=["Corvette"], deck=["Dreadnaught", "Scout"], authority=13)
    assert not find_lethal_plan(game, decision(game))
    game = position(hand=["Corvette"], discard=["Dreadnaught"], authority=13)
    assert not find_lethal_plan(game, decision(game))


def test_draw_cannot_bootstrap_itself_from_hidden_deck():
    game = position(hand=["Corvette"], deck=["Corvette", "Dreadnaught"], authority=15)
    assert not find_lethal_plan(game, decision(game))


def test_deck_order_does_not_change_plan():
    game = position(
        hand=["Command Ship", "Survey Ship"],
        deck=["Battle Blob", "Dreadnaught", "Viper"],
        authority=30,
    )
    first = find_lethal_plan(game, decision(game))
    game.players[0].deck.reverse()
    assert find_lethal_plan(game, decision(game)) == first
    assert first
    finish(game)


def test_counting_combat_does_not_double_count_activated_cards():
    game = position(bases=["War World"], authority=4)
    game.players[0].in_play[0].activated = True
    game.players[0].combat = 3
    assert not find_lethal_plan(game, decision(game))


def test_outposts_can_make_apparent_lethal_insufficient():
    game = position(hand=["Dreadnaught"], enemy=["Battle Station", "Battle Station"], authority=3)
    assert not find_lethal_plan(game, decision(game))


def test_required_scrap_cost_is_paid():
    game = position(bases=["Machine Base"], deck=["Dreadnaught"], authority=12)
    assert not find_lethal_plan(game, decision(game))
    game.players[0].hand = [CARDS["Scout"]]
    assert find_lethal_plan(game, decision(game))
    finish(game)


def test_recycle_and_brain_world_pay_for_their_draws():
    for base in ["Recycling Station", "Brain World"]:
        game = position(
            hand=["Scout", "Scout"], bases=[base], deck=["Battle Blob", "Dreadnaught"], authority=24
        )
        assert find_lethal_plan(game, decision(game))
        finish(game)


def test_human_choices_and_serialized_decisions_are_unmodified():
    game = position(hand=["Dreadnaught"], authority=12)
    game.choosers[0] = lambda _, d: next(i for i, a in enumerate(d.actions) if a.kind == K.END_TURN)
    game._take_turn(game.players[0])
    assert game._winner is None
    d = replace(decision(game), opaque=game._model_lethal_action)
    assert "opaque" not in d.to_json()


def test_solver_does_not_mutate_live_state_or_rng():
    game = position(hand=["Command Ship"], deck=["Dreadnaught", "Battle Blob"], authority=29)
    before = game.to_json(include_hidden=True)
    rng = game.players[0].rng.getstate()
    assert find_lethal_plan(game, decision(game))
    assert game.to_json(include_hidden=True) == before
    assert game.players[0].rng.getstate() == rng


def test_draw_fleet_hq_before_spending_ships():
    game = position(hand=["Viper", "Survey Ship"], deck=["Fleet HQ"], authority=2)
    assert find_lethal_plan(game, decision(game))
    history = finish(game)
    plays = [a.card_id for _, a in history if a.kind == K.PLAY_CARD]
    assert plays.index(CARDS["Fleet HQ"].card_id) < plays.index(CARDS["Viper"].card_id)


def test_blob_world_draw_mode_does_not_also_award_its_combat():
    game = position(hand=["Battle Pod"], bases=["Blob World"], deck=["Dreadnaught"], authority=23)
    assert not find_lethal_plan(game, decision(game))
    game.players[1].authority = 18
    assert find_lethal_plan(game, decision(game))
    finish(game)


def test_search_budget_falls_back_without_changing_the_game():
    game = position(hand=["Viper"], authority=1)
    before = game.to_json(include_hidden=True)
    assert not find_lethal_plan(game, decision(game), budget=0)
    assert game.to_json(include_hidden=True) == before
