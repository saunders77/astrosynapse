from types import SimpleNamespace

import numpy as np
import pytest
from astro2.baselines import BASELINE_NAMES, HeuristicChooser, RandomChooser, make_baseline
from astro2.encoding import DecisionFamily as EncodedDecisionFamily
from astro2.encoding import Encoder
from astro2.engine import Decision, DecisionFamily, Game, GameConfig
from astro2.selfplay import (
    ActorPolicy,
    CompactSamples,
    PlayerExploration,
    collect_game,
    collect_worker_batch,
)


def test_every_baseline_returns_only_legal_actions_and_seeded_random_repeats():
    for offset, name in enumerate(BASELINE_NAMES):
        result = Game(
            choosers=(make_baseline(name, 99), HeuristicChooser()),
            config=GameConfig(seed=400 + offset, max_turns=30),
        ).run()
        assert result.turns > 0

    first = Game(
        choosers=(RandomChooser(7), RandomChooser(8)),
        config=GameConfig(seed=123, max_turns=50),
    ).run()
    second = Game(
        choosers=(RandomChooser(7), RandomChooser(8)),
        config=GameConfig(seed=123, max_turns=50),
    ).run()
    assert first == second


class CountingPolicy:
    def __init__(self, choice: int):
        self.choice = choice
        self.calls = 0

    def __call__(self, _player_id, decision):
        self.calls += 1
        return min(self.choice, len(decision.actions) - 1)


class RecordingActor:
    def __init__(self, encoder: Encoder, bootstrap_heads: int = 3):
        self.spec = SimpleNamespace(
            state_size=encoder.state_size,
            action_size=encoder.action_size,
            families=len(EncodedDecisionFamily),
            bootstrap_heads=bootstrap_heads,
        )
        self.calls = []
        self.value_calls = 0

    def choose(self, _state, actions, _family, **kwargs):
        self.calls.append(kwargs)
        return 0, np.full(len(actions), 0.5, dtype=np.float32)

    def predict_options(self, _state, actions, _family):
        values = np.linspace(0.2, 0.8, len(actions), dtype=np.float32)
        return np.repeat(values[:, None], self.spec.bootstrap_heads, axis=1)

    def predict_values(self, states, _families):
        self.value_calls += 1
        count = 1 if np.asarray(states).ndim == 1 else len(states)
        return np.zeros((count, self.spec.bootstrap_heads), dtype=np.float32)


def test_deployment_policy_uses_mean_heads_without_epsilon_or_random_prior():
    encoder = Encoder()
    actor = RecordingActor(encoder)
    policy = ActorPolicy(actor, encoder)
    game = Game(config=GameConfig(seed=14))
    player = game.players[0]
    decision = Decision(
        DecisionFamily.MAIN,
        game.observation(player.player_id),
        game._main_actions(player),
    )
    exploration = PlayerExploration(
        head=2,
        epsilon=0.75,
        bootstrap_mask=np.asarray([0, 0, 1], dtype=np.uint8),
        deployment_policy=True,
    )

    policy.score(
        decision,
        exploration,
        np.random.default_rng(7),
        exploration_top_k=0,
        randomized_prior_scale=0.9,
    )

    assert len(actor.calls) == 1
    call = actor.calls[0]
    assert call["head"] is None
    assert call["epsilon"] == 0.0
    assert call["exploration_top_k"] == 0
    assert call["randomized_prior_scale"] == 0.0
    assert isinstance(call["rng"], np.random.Generator)


def test_deployment_trajectory_keeps_valid_bootstrap_head_metadata():
    collected = collect_game(
        (CountingPolicy(0), CountingPolicy(1)),
        seed=43,
        encoder=Encoder(),
        bootstrap_heads=4,
        epsilons=(0.8, 0.9),
        heads=(1, 3),
        deployment_policy=True,
        max_turns=80,
    )

    assert collected.epsilons == (0.0, 0.0)
    for player in (0, 1):
        assert collected.bootstrap_masks[player][collected.heads[player]] == 1
        samples = [sample for sample in collected.samples if sample.player == player]
        assert samples
        assert {sample.head for sample in samples} == {collected.heads[player]}
        assert {sample.epsilon for sample in samples} == {0.0}
        assert all(sample.bootstrap_mask[collected.heads[player]] == 1 for sample in samples)


@pytest.mark.parametrize("current_player", [0, 1])
def test_collector_supports_uncollected_legacy_actor_with_fewer_heads(current_player):
    astro3_encoder = Encoder(version=2)
    astro2_encoder = Encoder(version=1)
    current_actor = RecordingActor(astro3_encoder, bootstrap_heads=5)
    legacy_actor = RecordingActor(astro2_encoder, bootstrap_heads=3)
    policies = [
        ActorPolicy(legacy_actor, astro2_encoder),
        ActorPolicy(legacy_actor, astro2_encoder),
    ]
    policies[current_player] = ActorPolicy(current_actor, astro3_encoder)
    collected = collect_game(
        policies,
        seed=430 + current_player,
        encoder=astro3_encoder,
        bootstrap_heads=5,
        collect_players=(current_player == 0, current_player == 1),
        collect_preferences=False,
        max_turns=80,
    )

    assert collected.samples
    assert {sample.player for sample in collected.samples} == {current_player}
    assert {sample.bootstrap_mask.shape for sample in collected.samples} == {(5,)}
    assert {sample.head for sample in collected.samples} == {collected.heads[current_player]}
    assert 0 <= collected.heads[current_player] < 5
    assert 0 <= collected.heads[1 - current_player] < 3
    assert all(sample.bootstrap_mask[sample.head] == 1 for sample in collected.samples)


def test_collector_rejects_collected_actor_with_wrong_replay_head_count():
    encoder = Encoder(version=2)
    mismatched = ActorPolicy(RecordingActor(encoder, bootstrap_heads=3), encoder)

    with pytest.raises(ValueError, match="collected ActorPolicy"):
        collect_game(
            (mismatched, CountingPolicy(0)),
            seed=432,
            encoder=encoder,
            bootstrap_heads=5,
            collect_players=(True, False),
            max_turns=20,
        )


def test_collector_skips_forced_choices_and_uses_exact_player_terminal_targets():
    policies = (CountingPolicy(0), CountingPolicy(1))
    collected = collect_game(
        policies,
        seed=42,
        encoder=Encoder(),
        bootstrap_heads=4,
        epsilons=(0.12, 0.34),
        max_turns=80,
    )
    result = collected.result
    assert len(collected.samples) == result.decisions - result.forced_choices
    assert len(collected.samples) == policies[0].calls + policies[1].calls

    expected = (0.5, 0.5)
    if result.winner is not None:
        expected = (1.0, 0.0) if result.winner == 0 else (0.0, 1.0)
    for player in (0, 1):
        samples = [sample for sample in collected.samples if sample.player == player]
        assert samples
        assert {sample.target for sample in samples} == {expected[player]}
        assert {sample.head for sample in samples} == {collected.heads[player]}
        assert {sample.epsilon for sample in samples} == {collected.epsilons[player]}
        assert all(
            np.array_equal(sample.bootstrap_mask, collected.bootstrap_masks[player])
            for sample in samples
        )
        assert collected.bootstrap_masks[player][collected.heads[player]] == 1
    assert collected.preferences


def test_worker_helper_returns_compact_arrays_and_aggregate_stats_without_mlx():
    encoder = Encoder()
    result = collect_worker_batch(
        (None, None),
        games=3,
        seed=700,
        baseline_names=("balanced", "random"),
        bootstrap_heads=3,
        collect_players=(True, True),
        max_turns=20,
    )
    assert result.games == 3
    assert sum(result.wins) + result.draws + result.truncated == 3
    assert result.samples.states.shape[1] == encoder.state_size
    assert result.samples.actions.shape[1] == encoder.action_size
    assert result.samples.states.dtype == np.float16
    assert result.preferences.states.shape[1] == encoder.state_size
    items = result.samples.replay_items()
    assert len(items) == len(result.samples)
    round_trip = CompactSamples.from_items(
        items,
        state_size=encoder.state_size,
        action_size=encoder.action_size,
        bootstrap_heads=3,
    )
    np.testing.assert_array_equal(round_trip.families, result.samples.families)


def test_truncated_game_is_counted_separately_and_never_enters_replay():
    collected = collect_game(
        (HeuristicChooser(), HeuristicChooser()),
        seed=811,
        encoder=Encoder(),
        bootstrap_heads=3,
        collect_players=(True, True),
        max_turns=1,
    )

    assert collected.result.truncated is True
    assert collected.result.winner is None
    assert collected.samples == ()
    assert collected.preferences == ()

    worker = collect_worker_batch(
        (None, None),
        games=2,
        seed=812,
        baseline_names=("balanced", "balanced"),
        bootstrap_heads=3,
        collect_players=(True, True),
        max_turns=1,
    )
    assert worker.truncated == 2
    assert worker.draws == 0
    assert sum(worker.wins) == 0
    assert len(worker.samples) == 0


def test_generation_four_collection_keeps_complete_legal_sets_and_counterfactual_pairs():
    collected = collect_game(
        (HeuristicChooser(), HeuristicChooser()),
        seed=919,
        encoder=Encoder(version=2),
        bootstrap_heads=5,
        collect_policy_decisions=True,
        collect_preferences=False,
        counterfactual_fraction=1.0,
        counterfactual_max_per_game=1,
        max_turns=80,
    )
    assert collected.policy_samples
    assert all(len(item.legal_actions) >= 2 for item in collected.policy_samples)
    assert all(
        0 <= item.selected_index < len(item.legal_actions) for item in collected.policy_samples
    )
    assert all(0 < item.behavior_probability <= 1 for item in collected.policy_samples)
    assert len(collected.preferences) <= 1


def test_generation_five_reanalysis_attaches_action_distribution_and_value():
    encoder = Encoder(version=2)
    actor = RecordingActor(encoder, bootstrap_heads=5)
    policy = ActorPolicy(actor, encoder)
    collected = collect_game(
        (policy, policy),
        seed=923,
        encoder=encoder,
        bootstrap_heads=5,
        collect_policy_decisions=True,
        collect_preferences=False,
        collect_outcome_decisions=False,
        reanalysis_fraction=1.0,
        reanalysis_max_per_game=1,
        reanalysis_max_actions=2,
        reanalysis_rollouts_per_action=2,
        reanalysis_horizon_turns=2,
        reanalysis_policy_temperature=0.35,
        max_turns=100,
    )
    searched = [item for item in collected.policy_samples if item.search_valid]
    assert len(searched) == 1
    assert np.isclose(np.sum(searched[0].search_policy), 1.0)
    assert np.sum(searched[0].search_mask) == 2
    assert 0 <= searched[0].search_value <= 1
    assert collected.search_repeatability_positions == 1
    assert collected.search_top_action_agreements in {0, 1}
    assert collected.search_policy_js_sum >= 0
    assert collected.search_value_abs_delta_sum >= 0
    assert actor.value_calls > 0


def test_turn_gae_assigns_one_bootstrapped_advantage_to_every_decision_in_a_turn():
    encoder = Encoder(version=2)
    actor = RecordingActor(encoder, bootstrap_heads=5)
    policy = ActorPolicy(actor, encoder)
    collected = collect_game(
        (policy, policy),
        seed=927,
        encoder=encoder,
        bootstrap_heads=5,
        collect_policy_decisions=True,
        collect_preferences=False,
        collect_outcome_decisions=False,
        policy_actor_advantage="turn_gae",
        policy_actor_gae_lambda=0.95,
        max_turns=180,
    )

    assert collected.result.truncated is False
    assert collected.policy_samples
    assert actor.value_calls == 2
    assert all(item.actor_advantage_valid for item in collected.policy_samples)
    assert all(item.collection_value == pytest.approx(0.5) for item in collected.policy_samples)
    by_player_turn: dict[tuple[int, int], set[float]] = {}
    for item in collected.policy_samples:
        by_player_turn.setdefault((item.player, item.turn), set()).add(item.actor_advantage)
    assert all(len(advantages) == 1 for advantages in by_player_turn.values())
    assert any(abs(item.actor_advantage) > 0 for item in collected.policy_samples)
