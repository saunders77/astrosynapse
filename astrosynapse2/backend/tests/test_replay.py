import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from astro2.encoding import DecisionFamily
from astro2.replay import (
    GameBalancedPolicyReplayBuffer,
    PolicyItem,
    PreferenceItem,
    PreferenceReplayBuffer,
    ReplayItem,
    StratifiedReplayBuffer,
    _FamilyRing,
    make_bootstrap_mask,
)
from astro2.selfplay import CompactPolicySamples, CompactPreferences, CompactSamples


def item(step: int, family: DecisionFamily, *, state_size=5, action_size=4, heads=3):
    return ReplayItem(
        state=np.full(state_size, step, dtype=np.float32),
        action=np.full(action_size, step + 0.5, dtype=np.float32),
        family=family,
        target=float(step % 2),
        bootstrap_mask=np.ones(heads, dtype=np.uint8),
        game_id=step // 3,
        player=step % 2,
        step=step,
        head=step % heads,
        epsilon=0.1,
        td_target=0.25,
        td_valid=True,
    )


def test_independent_family_rings_retain_rare_decisions_and_are_bounded():
    replay = StratifiedReplayBuffer(
        capacity=32,
        state_size=5,
        action_size=4,
        bootstrap_heads=3,
        seed=7,
    )
    for step in range(100):
        replay.add(item(step, DecisionFamily.MAIN))
    replay.add(item(101, DecisionFamily.COPY_SHIP))
    replay.add(item(102, DecisionFamily.SCRAP))

    metrics = replay.metrics()
    assert len(replay) <= 32
    assert metrics["families"]["main"]["overwrites"] > 0
    assert metrics["families"]["copy_ship"]["size"] == 1
    assert metrics["families"]["scrap"]["size"] == 1

    batch = replay.sample(24, recent_fraction=0.0)
    assert batch.states.shape == (24, 5)
    assert batch.actions.shape == (24, 4)
    assert batch.bootstrap_mask.shape == (24, 3)
    assert batch.states.dtype == np.float32
    assert set(batch.families.tolist()) == {
        int(DecisionFamily.MAIN),
        int(DecisionFamily.COPY_SHIP),
        int(DecisionFamily.SCRAP),
    }
    assert set(batch.learner_inputs()) == {
        "states",
        "actions",
        "families",
        "targets",
        "bootstrap_mask",
        "sample_weights",
    }


def test_recent_sampling_draws_from_the_newest_window():
    replay = StratifiedReplayBuffer(
        capacity=64,
        state_size=5,
        action_size=4,
        bootstrap_heads=3,
        family_capacity_weights={DecisionFamily.MAIN: 1.0},
        recent_sample_fraction=1.0,
        recent_window_fraction=0.2,
        seed=11,
    )
    for step in range(50):
        replay.add(item(step, DecisionFamily.MAIN))
    batch = replay.sample(20)
    # The pool expands to the requested unique batch size rather than creating
    # duplicates from the nominal ten-item (20%) recent window.
    assert batch.steps.min() >= 30


def test_fast_ring_sampling_matches_reference_when_physical_order_is_sorted():
    ring = _FamilyRing(37, state_size=1, action_size=1, bootstrap_heads=1, storage_dtype=np.float16)
    ring.size = ring.capacity
    rng = np.random.default_rng(419)
    reference_rng = np.random.default_rng(419)

    actual = ring.sample_indices(10, 3, 0.25, rng)
    ordered = ring.chronological_indices()
    window_size = max(3, int(np.ceil(ring.size * 0.25)))
    recent = reference_rng.choice(ordered[-window_size:], size=3, replace=False)
    pool = np.setdiff1d(ordered, recent, assume_unique=False)
    general = reference_rng.choice(pool, size=7, replace=False)
    expected = np.concatenate((recent, general))
    reference_rng.shuffle(expected)

    np.testing.assert_array_equal(actual, expected)


def test_wrapped_fast_ring_sampling_preserves_expected_inclusion_distribution():
    ring = _FamilyRing(37, state_size=1, action_size=1, bootstrap_heads=1, storage_dtype=np.float16)
    ring.size = ring.capacity
    ring.write_index = 13
    ordered = ring.chronological_indices()
    recent_pool = ordered[-10:]
    counts = np.zeros(ring.size, dtype=np.int64)
    rng = np.random.default_rng(421)
    trials = 6_000

    for _ in range(trials):
        selected = ring.sample_indices(10, 3, 0.25, rng)
        assert len(np.unique(selected)) == len(selected)
        counts[selected] += 1

    outside_pool = np.setdiff1d(ordered, recent_pool, assume_unique=True)
    # Three of ten recent rows are selected first. A recent row not selected
    # there can still be one of seven uniformly sampled general rows.
    expected_recent = 3 / 10 + (1 - 3 / 10) * (7 / (37 - 3))
    expected_outside = 7 / (37 - 3)
    np.testing.assert_allclose(counts[recent_pool] / trials, expected_recent, atol=0.025)
    np.testing.assert_allclose(counts[outside_pool] / trials, expected_outside, atol=0.025)


def test_stratified_sampling_can_be_importance_corrected_to_write_distribution():
    replay = StratifiedReplayBuffer(
        capacity=200,
        state_size=5,
        action_size=4,
        bootstrap_heads=3,
        family_capacity_weights={
            DecisionFamily.MAIN: 0.9,
            DecisionFamily.SCRAP: 0.1,
        },
        family_sampling_weights={
            DecisionFamily.MAIN: 0.5,
            DecisionFamily.SCRAP: 0.5,
        },
        importance_correct_sampling=True,
        recent_sample_fraction=0.0,
        seed=37,
    )
    for step in range(90):
        replay.add(item(step, DecisionFamily.MAIN))
    for step in range(90, 100):
        replay.add(item(step, DecisionFamily.SCRAP))

    batch = replay.sample(100)
    main = batch.sample_weights[batch.families == int(DecisionFamily.MAIN)]
    scrap = batch.sample_weights[batch.families == int(DecisionFamily.SCRAP)]
    assert len(main) == len(scrap) == 50
    np.testing.assert_allclose(main.mean(), 1.8, rtol=1e-6)
    np.testing.assert_allclose(scrap.mean(), 0.2, rtol=1e-6)
    np.testing.assert_allclose(batch.sample_weights.mean(), 1.0, rtol=1e-6)
    metrics = replay.metrics()
    assert metrics["importance_correct_sampling"] is True
    assert metrics["recent_sample_fraction_realized"] == 0.0
    assert metrics["importance_weights"]["minimum"] == pytest.approx(0.2)
    assert metrics["importance_weights"]["maximum"] == pytest.approx(1.8)
    assert metrics["importance_weights"]["effective_sample_size"] == pytest.approx(
        100**2 / (50 * 1.8**2 + 50 * 0.2**2)
    )


def test_bootstrap_mask_is_nonempty_and_contains_generating_head():
    rng = np.random.default_rng(123)
    for _ in range(100):
        mask = make_bootstrap_mask(5, rng, inclusion_probability=0.05, required_head=3)
        assert mask.shape == (5,)
        assert mask[3] == 1
        assert mask.any()


def test_extend_validates_atomically():
    replay = StratifiedReplayBuffer(
        capacity=16,
        state_size=5,
        action_size=4,
        bootstrap_heads=3,
    )
    malformed = replace(item(2, DecisionFamily.MAIN), state=np.zeros(99, dtype=np.float32))
    try:
        replay.extend([item(1, DecisionFamily.MAIN), malformed])
    except ValueError:
        pass
    else:
        raise AssertionError("malformed replay item was accepted")
    assert len(replay) == 0


def test_vectorized_compact_extend_matches_scalar_rings_through_wraparound():
    kwargs = dict(
        capacity=32,
        state_size=5,
        action_size=4,
        bootstrap_heads=3,
        seed=19,
    )
    scalar = StratifiedReplayBuffer(**kwargs)
    vectorized = StratifiedReplayBuffer(**kwargs)
    items = [
        item(
            step,
            DecisionFamily.MAIN if step % 4 else DecisionFamily.COPY_SHIP,
        )
        for step in range(80)
    ]
    for chunk in (items[:11], items[11:]):
        scalar.extend(chunk)
        compact = CompactSamples.from_items(
            chunk,
            state_size=5,
            action_size=4,
            bootstrap_heads=3,
        )
        assert vectorized.extend_compact(compact) == len(chunk)

    assert scalar.metrics() == vectorized.metrics()
    assert scalar._sequence == vectorized._sequence
    for family in DecisionFamily:
        expected_ring = scalar._rings[family]
        actual_ring = vectorized._rings[family]
        expected = expected_ring.chronological_indices()
        actual = actual_ring.chronological_indices()
        for name in (
            "states",
            "actions",
            "targets",
            "bootstrap_masks",
            "game_ids",
            "players",
            "steps",
            "heads",
            "epsilons",
            "td_targets",
            "td_valid",
            "sequences",
        ):
            np.testing.assert_array_equal(
                getattr(actual_ring, name)[actual], getattr(expected_ring, name)[expected]
            )


def test_compact_extend_rejects_negative_family_without_partial_write():
    replay = StratifiedReplayBuffer(
        capacity=16,
        state_size=5,
        action_size=4,
        bootstrap_heads=3,
    )
    compact = CompactSamples.from_items(
        [item(1, DecisionFamily.MAIN)],
        state_size=5,
        action_size=4,
        bootstrap_heads=3,
    )
    malformed = replace(compact, families=np.asarray([-1], dtype=np.int16))
    try:
        replay.extend_compact(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("negative family was accepted")
    assert len(replay) == 0


def test_preference_replay_is_bounded_and_round_trips_compact_rows():
    entries = [
        PreferenceItem(
            state=np.full(5, step, dtype=np.float32),
            preferred_action=np.full(4, step + 1, dtype=np.float32),
            disfavored_action=np.full(4, step - 1, dtype=np.float32),
            family=DecisionFamily.MAIN,
            bootstrap_mask=np.asarray([1, step % 2, 0], dtype=np.uint8),
        )
        for step in range(12)
    ]
    compact = CompactPreferences.from_items(entries, state_size=5, action_size=4, bootstrap_heads=3)
    replay = PreferenceReplayBuffer(
        capacity=8, state_size=5, action_size=4, bootstrap_heads=3, seed=3
    )
    assert replay.extend_compact(compact) == 12
    assert len(replay) == 8
    assert replay.metrics()["overwrites"] == 4
    batch = replay.sample(6)
    assert batch.states.shape == (6, 5)
    assert batch.preferred_actions.shape == (6, 4)
    assert batch.bootstrap_mask.shape == (6, 3)


def test_policy_replay_samples_player_games_before_decisions():
    replay = GameBalancedPolicyReplayBuffer(
        capacity=64, state_size=5, action_size=4, bootstrap_heads=3, max_actions=5, seed=17
    )
    entries = []
    for game_id, decisions in ((10, 30), (20, 2)):
        for step in range(decisions):
            entries.append(
                PolicyItem(
                    state=np.full(5, step, dtype=np.float32),
                    legal_actions=np.stack(
                        (np.full(4, step, dtype=np.float32), np.full(4, step + 1, dtype=np.float32))
                    ),
                    selected_index=step % 2,
                    family=DecisionFamily.MAIN,
                    target=float(game_id == 10),
                    behavior_probability=0.5,
                    bootstrap_mask=np.ones(3, dtype=np.uint8),
                    game_id=game_id,
                    player=0,
                    step=step,
                )
            )
    compact = CompactPolicySamples.from_items(
        entries, state_size=5, action_size=4, bootstrap_heads=3
    )
    assert replay.extend_compact(compact) == 32
    batch = replay.sample(2_000)
    assert 0.45 < np.mean(batch.game_ids == 10) < 0.55
    assert batch.legal_actions.shape == (2_000, 2, 4)
    assert batch.legal_mask.shape == (2_000, 2)
    assert replay.metrics()["sampling"] == (
        "uniform_player_game_then_mixed_natural_and_stratified_decision"
    )


def test_policy_replay_pads_only_to_the_largest_sampled_legal_set():
    replay = GameBalancedPolicyReplayBuffer(
        capacity=16, state_size=5, action_size=4, bootstrap_heads=3, max_actions=8, seed=23
    )
    entries = []
    for game_id, action_count in ((10, 2), (20, 5)):
        entries.append(
            PolicyItem(
                state=np.full(5, game_id, dtype=np.float32),
                legal_actions=np.arange(action_count * 4, dtype=np.float32).reshape(
                    action_count, 4
                ),
                selected_index=action_count - 1,
                family=DecisionFamily.MAIN,
                target=1.0,
                behavior_probability=0.5,
                bootstrap_mask=np.ones(3, dtype=np.uint8),
                game_id=game_id,
                player=0,
                step=0,
            )
        )
    assert replay.extend(entries) == 2

    batch = replay.sample(2)

    assert set(batch.game_ids.tolist()) == {10, 20}
    assert batch.legal_actions.shape == (2, 5, 4)
    np.testing.assert_array_equal(np.sort(batch.legal_mask.sum(axis=1)), np.asarray([2.0, 5.0]))


def test_astro5_policy_reservoir_preserves_search_and_round_trips(tmp_path):
    replay = GameBalancedPolicyReplayBuffer(
        capacity=64,
        state_size=12,
        action_size=4,
        bootstrap_heads=3,
        max_actions=5,
        max_decisions_per_player_game=3,
        family_balanced=True,
        seed=29,
    )
    entries = []
    for step in range(10):
        searched = step == 7
        entries.append(
            PolicyItem(
                state=np.full(12, step / 50, dtype=np.float32),
                legal_actions=np.stack(
                    (np.full(4, step, dtype=np.float32), np.full(4, step + 1, dtype=np.float32))
                ),
                selected_index=0,
                family=DecisionFamily.MAIN if step % 2 else DecisionFamily.DISCARD,
                target=1.0,
                behavior_probability=0.5,
                bootstrap_mask=np.ones(3, dtype=np.uint8),
                game_id=51,
                player=0,
                step=step,
                search_policy=np.asarray([0.8, 0.2] if searched else [0, 0]),
                search_mask=np.asarray([1, 1] if searched else [0, 0]),
                search_value=0.7,
                search_valid=searched,
                rollout_source=3,
                opponent_key=987654321,
                collected_at_game=123_456,
                collection_policy_probability=0.42,
                behavior_head=2,
                behavior_epsilon=0.03,
                deployment_policy=False,
            )
        )
    assert replay.extend(entries) == 10
    assert len(replay) == 3
    assert replay.metrics()["searched_decisions"] == 1

    path = tmp_path / "policy-replay.npz"
    assert replay.snapshot(path) == 3
    restored = GameBalancedPolicyReplayBuffer(
        capacity=64,
        state_size=12,
        action_size=4,
        bootstrap_heads=3,
        max_actions=5,
        max_decisions_per_player_game=3,
        family_balanced=True,
        seed=31,
    )
    assert restored.restore(path) == 3
    assert restored.metrics()["searched_decisions"] == 1
    assert restored.metrics()["player_games"] == 1
    restored_batch = restored.sample(3)
    assert set(restored_batch.rollout_sources.tolist()) == {3}
    assert set(restored_batch.opponent_keys.tolist()) == {987654321}
    assert set(restored_batch.collected_at_games.tolist()) == {123_456}
    np.testing.assert_allclose(restored_batch.collection_policy_probabilities, 0.42, atol=1e-3)
    assert set(restored_batch.behavior_heads.tolist()) == {2}
    np.testing.assert_allclose(restored_batch.behavior_epsilons, 0.03, atol=1e-3)
    assert set(restored_batch.sample_tiers.tolist()) == {0}


def test_policy_replay_mixes_natural_and_family_balanced_sampling():
    replay = GameBalancedPolicyReplayBuffer(
        capacity=128,
        state_size=12,
        action_size=4,
        bootstrap_heads=3,
        max_decisions_per_player_game=100,
        family_balanced=True,
        family_balanced_fraction=0.2,
        seed=53,
    )
    entries = []
    for step in range(100):
        entries.append(
            PolicyItem(
                state=np.full(12, step / 50, dtype=np.float32),
                legal_actions=np.stack((np.zeros(4), np.ones(4))).astype(np.float32),
                selected_index=0,
                family=(DecisionFamily.DISCARD if step < 10 else DecisionFamily.MAIN),
                target=1.0,
                behavior_probability=0.5,
                bootstrap_mask=np.ones(3, dtype=np.uint8),
                game_id=77,
                player=0,
                step=step,
            )
        )
    replay.extend(entries)
    batch = replay.sample(4_000)
    discard_fraction = np.mean(batch.families == int(DecisionFamily.DISCARD))
    assert 0.12 < discard_fraction < 0.22
    metrics = replay.metrics()
    assert metrics["family_balanced_fraction"] == 0.2
    assert sum(metrics["decision_distribution"]["sampled_by_family"]) == 4_000


def test_policy_replay_incremental_manifest_restores_latest_window(tmp_path):
    replay = GameBalancedPolicyReplayBuffer(
        capacity=6, state_size=5, action_size=4, bootstrap_heads=3, max_actions=5, seed=37
    )

    def episode(game_id: int) -> list[PolicyItem]:
        return [
            PolicyItem(
                state=np.full(5, game_id + step, dtype=np.float32),
                legal_actions=np.stack((np.zeros(4), np.ones(4))).astype(np.float32),
                selected_index=step,
                family=DecisionFamily.MAIN,
                target=float(step),
                behavior_probability=0.5,
                bootstrap_mask=np.ones(3, dtype=np.uint8),
                game_id=game_id,
                player=0,
                step=step,
            )
            for step in range(2)
        ]

    replay.extend(episode(1))
    replay.enable_incremental_snapshots(tmp_path / "journal", max_items=4)
    replay.extend(episode(2))
    first_manifest = tmp_path / "first.json"
    assert replay.snapshot_incremental(first_manifest, max_items=4) == 4
    assert len(json.loads(first_manifest.read_text())["segments"]) == 2

    replay.extend(episode(3))
    second_manifest = tmp_path / "second.json"
    assert replay.snapshot_incremental(second_manifest, max_items=4) == 4
    assert len(json.loads(second_manifest.read_text())["segments"]) == 3

    restored = GameBalancedPolicyReplayBuffer(
        capacity=6, state_size=5, action_size=4, bootstrap_heads=3, max_actions=5, seed=39
    )
    assert restored.restore(second_manifest) == 4
    assert set(restored._episodes) == {(2, 0), (3, 0)}


def test_policy_replay_moves_old_episodes_to_bounded_mmap_disk_tier(tmp_path):
    def episode(game_id: int) -> list[PolicyItem]:
        return [
            PolicyItem(
                state=np.full(12, (game_id + step) / 50, dtype=np.float32),
                legal_actions=np.stack((np.zeros(4), np.ones(4))).astype(np.float32),
                selected_index=step,
                family=DecisionFamily.MAIN,
                target=float(step),
                behavior_probability=0.5,
                bootstrap_mask=np.ones(3, dtype=np.uint8),
                game_id=game_id,
                player=0,
                step=step,
            )
            for step in range(2)
        ]

    disk = tmp_path / "cold"
    replay = GameBalancedPolicyReplayBuffer(
        capacity=4,
        state_size=12,
        action_size=4,
        bootstrap_heads=3,
        max_actions=5,
        disk_directory=disk,
        disk_capacity=4,
        disk_sample_fraction=1.0,
        disk_shard_items=2,
        seed=43,
    )
    for game_id in range(1, 6):
        replay.extend(episode(game_id))

    metrics = replay.metrics()
    assert metrics["hot"]["size"] == 4
    assert metrics["cold"]["size"] == 4
    assert metrics["size"] == 8
    assert metrics["capacity"] == 8
    assert metrics["cold"]["shards"] == 2
    assert metrics["cold"]["pending_decisions"] == 0

    batch = replay.sample(100)
    assert set(batch.game_ids.tolist()) == {2, 3}
    assert len(replay._cold._mapped) <= 4
    assert all(isinstance(arrays["states"], np.memmap) for arrays in replay._cold._mapped.values())


def test_hybrid_policy_replay_checkpoint_restores_hot_and_cold_tiers(tmp_path):
    def episode(game_id: int) -> list[PolicyItem]:
        return [
            PolicyItem(
                state=np.full(12, (game_id + step) / 50, dtype=np.float32),
                legal_actions=np.stack((np.zeros(4), np.ones(4))).astype(np.float32),
                selected_index=step,
                family=DecisionFamily.MAIN,
                target=float(step),
                behavior_probability=0.5,
                bootstrap_mask=np.ones(3, dtype=np.uint8),
                game_id=game_id,
                player=0,
                step=step,
            )
            for step in range(2)
        ]

    replay = GameBalancedPolicyReplayBuffer(
        capacity=4,
        state_size=12,
        action_size=4,
        bootstrap_heads=3,
        disk_directory=tmp_path / "source-cold",
        disk_capacity=8,
        disk_sample_fraction=0.5,
        disk_shard_items=2,
        seed=47,
    )
    replay.enable_incremental_snapshots(tmp_path / "source-journal", max_items=4)
    for game_id in range(1, 6):
        replay.extend(episode(game_id))
    manifest = tmp_path / "checkpoint.policy-replay.json"
    assert replay.snapshot_incremental(manifest, max_items=4) == 10
    payload = json.loads(manifest.read_text())
    assert payload["format"] == "hybrid_game_reservoir_v3"
    assert payload["hot_items"] == 4
    assert payload["cold_items"] == 6

    restored = GameBalancedPolicyReplayBuffer(
        capacity=4,
        state_size=12,
        action_size=4,
        bootstrap_heads=3,
        disk_directory=tmp_path / "restored-cold",
        disk_capacity=8,
        disk_sample_fraction=0.5,
        disk_shard_items=2,
        seed=53,
    )
    assert restored.restore(manifest) == 10
    restored_metrics = restored.metrics()
    assert restored_metrics["hot"]["size"] == 4
    assert restored_metrics["cold"]["size"] == 6
    sampled_game_ids = {
        int(game_id) for _ in range(8) for game_id in restored.sample(50).game_ids.tolist()
    }
    assert sampled_game_ids == {1, 2, 3, 4, 5}
    assert all(
        shard.path.parent == (tmp_path / "restored-cold").resolve()
        for shard in restored._cold._shards
    )

    # Clearing and restoring from the same live store must not mark restored
    # shards obsolete when the next durable checkpoint is committed.
    restored_snapshot = restored._cold.snapshot_payload()
    restored.clear()
    restored._cold.import_payload(restored_snapshot)
    restored._cold.commit_snapshot()
    assert all(shard.path.is_dir() for shard in restored._cold._shards)

    first_cold_paths = {Path(value["path"]) for value in payload["cold"]["shards"]}
    replay.extend(episode(6))
    replay.extend(episode(7))
    second_manifest = tmp_path / "next.policy-replay.json"
    replay.snapshot_incremental(second_manifest, max_items=4)
    second_payload = json.loads(second_manifest.read_text())
    second_cold_paths = {Path(value["path"]) for value in second_payload["cold"]["shards"]}
    retired_paths = first_cold_paths - second_cold_paths
    assert retired_paths
    replay.commit_incremental_snapshot()
    assert all(path.is_dir() for path in retired_paths)

    manifest.unlink()
    replay.commit_incremental_snapshot()
    assert all(not path.exists() for path in retired_paths)


def test_recent_replay_snapshot_round_trips_across_families(tmp_path):
    replay = StratifiedReplayBuffer(
        capacity=64,
        state_size=5,
        action_size=4,
        bootstrap_heads=3,
        seed=29,
    )
    for step in range(40):
        family = DecisionFamily.COPY_SHIP if step % 7 == 0 else DecisionFamily.MAIN
        replay.add(item(step, family))
    replay.sample(8)
    original_metrics = replay.metrics()

    path = tmp_path / "resume.replay.npz"
    assert replay.snapshot(path, max_items=12) == 12
    restored = StratifiedReplayBuffer(
        capacity=64,
        state_size=5,
        action_size=4,
        bootstrap_heads=3,
        seed=31,
    )
    assert restored.restore(path) == 12
    assert len(restored) == 12
    restored_steps = sorted(
        int(value)
        for ring in restored._rings.values()
        for value in ring.steps[ring.chronological_indices()]
    )
    assert restored_steps == list(range(28, 40))
    assert restored._sequence == replay._sequence
    restored_metrics = restored.metrics()
    assert restored_metrics["writes"] == original_metrics["writes"]
    assert restored_metrics["overwrites"] == original_metrics["overwrites"]
    assert restored_metrics["sample_calls"] == original_metrics["sample_calls"]
    assert restored_metrics["samples_drawn"] == original_metrics["samples_drawn"]
    for family in DecisionFamily:
        name = family.name.lower()
        assert (
            restored_metrics["families"][name]["writes"]
            == original_metrics["families"][name]["writes"]
        )
        assert (
            restored_metrics["families"][name]["samples_drawn"]
            == original_metrics["families"][name]["samples_drawn"]
        )


def test_full_replay_snapshot_round_trips_every_ring_without_gathering(tmp_path):
    replay = StratifiedReplayBuffer(
        capacity=64,
        state_size=5,
        action_size=4,
        bootstrap_heads=3,
        seed=43,
    )
    for step in range(160):
        family = (
            DecisionFamily.COPY_SHIP
            if step % 11 == 0
            else DecisionFamily.SCRAP
            if step % 5 == 0
            else DecisionFamily.MAIN
        )
        replay.add(item(step, family))
    replay.sample(24)

    path = tmp_path / "full.replay.npz"
    assert replay.snapshot_full(path) == len(replay)
    restored = StratifiedReplayBuffer(
        capacity=64,
        state_size=5,
        action_size=4,
        bootstrap_heads=3,
        seed=47,
    )
    assert restored.restore(path) == len(replay)
    assert restored.metrics() == replay.metrics()
    assert restored._sequence == replay._sequence
    for family in DecisionFamily:
        expected_ring = replay._rings[family]
        actual_ring = restored._rings[family]
        assert actual_ring.size == expected_ring.size
        assert actual_ring.write_index == expected_ring.write_index
        for name in (
            "states",
            "actions",
            "targets",
            "bootstrap_masks",
            "game_ids",
            "players",
            "steps",
            "heads",
            "epsilons",
            "td_targets",
            "td_valid",
            "sequences",
        ):
            np.testing.assert_array_equal(
                getattr(actual_ring, name)[: actual_ring.size],
                getattr(expected_ring, name)[: expected_ring.size],
            )


def test_replay_sampling_rng_resumes_at_the_same_batch():
    kwargs = dict(
        capacity=64,
        state_size=5,
        action_size=4,
        bootstrap_heads=3,
        recent_sample_fraction=0.25,
    )
    first = StratifiedReplayBuffer(**kwargs, seed=101)
    restored = StratifiedReplayBuffer(**kwargs, seed=999)
    entries = [
        item(step, DecisionFamily.SCRAP if step % 9 == 0 else DecisionFamily.MAIN)
        for step in range(50)
    ]
    first.extend(entries)
    restored.extend(entries)
    state = first.rng_state()

    expected = first.sample(24)
    restored.restore_rng_state(state)
    actual = restored.sample(24)
    np.testing.assert_array_equal(actual.sequences, expected.sequences)
    np.testing.assert_array_equal(actual.families, expected.families)
    np.testing.assert_array_equal(actual.sample_weights, expected.sample_weights)
