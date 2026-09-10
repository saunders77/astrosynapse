from collections import Counter

import numpy as np
import pytest
from astro2.cards import CARD_BY_NAME
from astro2.engine import Action, ActionKind, Decision, DecisionFamily, Game, GameConfig
from astro2.engine_encoding import EngineEncoder
from astro2.planning import PlanningConfig, residual_features, sample_belief


def ids(cards):
    return tuple(card.card_id for card in cards)


def hidden_signature(game):
    return (
        tuple((ids(p.hand), ids(p.deck), p.rng.getstate()) for p in game.players),
        ids(game.trade_deck),
        game.rng_streams.market.getstate(),
    )


def test_belief_sampling_cannot_depend_on_actual_hidden_order_assignment_or_rng():
    original = Game(config=GameConfig(seed=19))
    original.players[1].hand = [CARD_BY_NAME["Scout"], CARD_BY_NAME["Viper"]]
    original.players[1].deck = [CARD_BY_NAME["Explorer"], CARD_BY_NAME["Trade Bot"]]
    equivalent = original.fork()
    equivalent.players[0].deck.reverse()
    equivalent.players[1].hand = original.players[1].deck[:]
    equivalent.players[1].deck = original.players[1].hand[:]
    equivalent.trade_deck.reverse()
    equivalent.players[1].rng.seed(991)
    encoder = EngineEncoder(version=2)
    np.testing.assert_array_equal(
        encoder.encode_state(original.observation(0)),
        encoder.encode_state(equivalent.observation(0)),
    )
    assert hidden_signature(sample_belief(original, 0, 123)) == hidden_signature(
        sample_belief(equivalent, 0, 123)
    )


def test_sampling_preserves_public_observation_known_cards_and_live_state():
    game = Game(config=GameConfig(seed=41))
    game.players[0].known_top = game.players[0].deck[-2:]
    game.players[1].known_top = game.players[1].deck[-1:]
    game.players[1].revealed_hand = game.players[1].hand[:1]
    before = hidden_signature(game)
    encoder = EngineEncoder(version=2)
    encoded = encoder.encode_state(game.observation(0))
    for seed in range(8):
        sampled = sample_belief(game, 0, seed)
        np.testing.assert_array_equal(encoded, encoder.encode_state(sampled.observation(0)))
        assert sampled.players[0].deck[-2:] == game.players[0].deck[-2:]
        assert sampled.players[1].deck[-1:] == game.players[1].deck[-1:]
        assert Counter(sampled.players[1].hand)[game.players[1].revealed_hand[0]] > 0
    assert hidden_signature(game) == before


def test_direct_adapter_uses_only_public_state_and_zero_coefficients_are_identity():
    game = Game(config=GameConfig(seed=5))
    card = CARD_BY_NAME["Trade Bot"]
    action = Action(ActionKind.ACQUIRE, card_id=card.card_id, amount=card.cost)
    decision = Decision(DecisionFamily.MAIN, game.observation(0), (action,))
    features = residual_features(decision, action)
    assert features.shape == (104,)
    assert features[card.card_id] == 1
    assert features @ np.zeros(104) == 0
    assert np.isfinite(features).all()


@pytest.mark.parametrize(
    "options", [{"rollouts": -1}, {"max_actions": 1}, {"residual": [1]}, {"horizon_turns": -1}]
)
def test_invalid_search_budgets_are_rejected(options):
    with pytest.raises(ValueError):
        PlanningConfig(**options)
