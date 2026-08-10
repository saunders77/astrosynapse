import numpy as np
from astro2.baselines import BASELINE_NAMES, HeuristicChooser, RandomChooser, make_baseline
from astro2.encoding import Encoder
from astro2.engine import Game, GameConfig
from astro2.selfplay import CompactSamples, collect_game, collect_worker_batch


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
    assert sum(result.wins) + result.draws == 3
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
