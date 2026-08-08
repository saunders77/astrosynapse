from dataclasses import replace

import numpy as np
from astro2.encoding import DecisionFamily
from astro2.replay import ReplayItem, StratifiedReplayBuffer, make_bootstrap_mask
from astro2.selfplay import CompactSamples


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
