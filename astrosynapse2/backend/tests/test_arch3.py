from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
from astro2.cards import CARD_BY_ID as C
from astro2.encoding import Encoder, Zone
from astro2.engine import Game, GameConfig, make_random_chooser
from astro2.engine_encoding import EngineEncoder


def counts(cards):
    return Counter(c.card_id for c in cards)


@pytest.mark.parametrize("remaining", [1, 2, 3, 4, 5])
def test_cleanup_remembers_exhausted_deck_through_shuffle_and_fork(remaining):
    game = Game(config=GameConfig(seed=42))
    player = game.players[1]
    player.deck = [C[3 + i] for i in range(remaining)]
    player.hand = [C[0]] * 5
    player.discard = [C[1]] * 8
    player.in_play.clear()
    expected = counts(player.deck)
    game._cleanup_and_draw(player)
    observed = game.observation(0)
    assert counts(observed.opponent_inferred_hand) == expected
    assert observed.opponent_known_hand == ()
    assert len(player.hand) == 5
    clone = game.fork()
    clone.resample_public_belief(0, 72)
    assert counts(clone.players[1].hand) >= expected
    assert clone.observation(0) == observed
    index = next(i for i, card in enumerate(player.hand) if card.card_id in expected)
    card = player.hand[index]
    game._discard_from_hand(player, index)
    expected.subtract([card.card_id])
    assert counts(game.observation(0).opponent_inferred_hand) == +expected


def test_small_deck_and_ship_to_top_overlap_do_not_double_count():
    game = Game()
    player = game.players[1]
    player.deck, player.known_top = [C[3], C[4]], [C[4]]
    player.hand, player.discard, player.in_play = [], [C[0]] * 8, []
    game._cleanup_and_draw(player)
    obs = game.observation(0)
    assert counts(obs.opponent_inferred_hand) == {3: 1, 4: 1}
    assert counts(obs.opponent_known_hand) == {4: 1}
    player.deck.clear()
    assert counts(game.observation(0).opponent_inferred_hand) == counts(player.hand)


def test_six_unknown_cards_do_not_reveal_the_actual_five_drawn():
    game = Game()
    player = game.players[1]
    player.deck = [C[i] for i in range(3, 9)]
    player.hand, player.discard, player.in_play = [], [], []
    game._cleanup_and_draw(player)
    assert game.observation(0).opponent_inferred_hand == ()


def test_arch3_preserves_old_prefix_and_counts_market_without_order():
    game = Game()
    old, new, fast = Encoder(version=2), Encoder(version=3), EngineEncoder(version=3)
    obs = replace(game.observation(0), opponent_inferred_hand=(C[3], C[3], C[4]))
    a, b = old.encode_state(obs), new.encode_state(obs)
    np.testing.assert_array_equal(b[: len(a)], a)
    np.testing.assert_array_equal(fast.encode_state(obs), b)
    assert b[-49 + 3] == 2 and b[-49 + 4] == 1
    trade = new.zone_counts(b, Zone.TRADE_DECK)
    assert trade.sum() == len(game.trade_deck)
    np.testing.assert_array_equal(
        new.encode_state(replace(obs, trade_deck=obs.trade_deck[::-1])), b
    )
    game._refill_market_slot(0)
    after = new.zone_counts(new.encode_state(game.observation(0)), Zone.TRADE_DECK)
    assert trade.sum() - after.sum() == 1


def test_known_hand_is_always_public_subset_in_random_games():
    def observe(_pid, decision, _selected):
        obs = game.observation(decision.observation.player_id)
        player = game.players[1 - obs.player_id]
        assert counts(player.hand) >= counts(obs.opponent_inferred_hand)
        clone = game.fork()
        clone.resample_public_belief(obs.player_id, obs.action_number + obs.turn * 100)
        assert clone.observation(obs.player_id) == obs

    for seed in range(3):
        game = Game(
            choosers=(make_random_chooser(seed), make_random_chooser(seed + 7)),
            config=GameConfig(seed=seed, max_turns=40),
            decision_hook=observe,
        )
        game.run()


def test_port_preserves_weights_and_adds_zero_columns(tmp_path):
    import json

    from astro2.arch3 import port_actor
    from astro2.experiment_control import sha256
    from astro2.model import ModelSpec, NumpyActor

    old = Encoder(version=2)
    spec = ModelSpec(old.state_size, old.action_size, 8, encoder_version=2)
    rng = np.random.default_rng(78)
    weight = rng.normal(size=(16, old.state_size)).astype(np.float32)
    source = tmp_path / "original.npz"
    np.savez(
        source,
        **{
            "state_in.weight": weight,
            "state_in.bias": np.ones(16),
            "__spec_json__": np.frombuffer(json.dumps(spec.as_dict()).encode(), np.uint8),
        },
    )
    digest = sha256(source)
    lineage = port_actor(source, tmp_path / "new")
    actor = NumpyActor.load(lineage["actor"])
    assert actor.spec.encoder_version == 3
    np.testing.assert_array_equal(actor.weights["state_in.weight"][:, : old.state_size], weight)
    assert not actor.weights["state_in.weight"][:, old.state_size :].any()
    assert sha256(source) == digest == lineage["source_sha256"]
    with pytest.raises(FileExistsError):
        port_actor(source, tmp_path / "new")


def test_inferred_and_later_top_draw_same_type_are_distinct_copies():
    game = Game()
    player = game.players[1]
    player.deck, player.hand, player.discard = [C[3]], [], [C[0]] * 10
    player.in_play.clear()
    game._cleanup_and_draw(player)
    game._place_acquired(player, C[3], force_top=True)
    game._draw(player, 1)
    assert counts(game.observation(0).opponent_inferred_hand)[3] == 2
    index = next(i for i, c in enumerate(player.hand) if c.card_id == 3)
    game._discard_from_hand(player, index)
    assert counts(game.observation(0).opponent_inferred_hand)[3] == 1
