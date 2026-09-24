from types import SimpleNamespace

import numpy as np
import pytest

from astro2.counterfactual import (
    ComparisonConfig,
    comparison_loss,
    continuation_choosers,
    economic_indices,
    load_positions,
    paired_counts,
    save_positions,
    training_batch,
)
from astro2.engine import Action, ActionKind, DecisionFamily


def test_paired_truncation_discards_all_candidates_for_that_chance_sample():
    wins, losses, n = paired_counts([[0, 1, 0, 1], [1, 0, np.nan, 1], [0, 1, 1, 0]])
    assert n == 3
    np.testing.assert_array_equal(wins, [0, 1, 0])
    np.testing.assert_array_equal(losses, [0, 1, 1])
    wins, losses, n = paired_counts([[0, 1], [0, 1]])
    assert n == 2 and not wins.any() and not losses.any()
    with pytest.raises(ValueError):
        paired_counts([[0.5], [1]])


def test_alternatives_are_not_restricted_by_policy_probability(monkeypatch):
    import astro2.counterfactual as module

    chosen = Action(ActionKind.ACQUIRE, card_id=3)
    other = Action(ActionKind.ACQUIRE, card_id=4)
    duplicate = Action(ActionKind.ACQUIRE, card_id=4, opaque=(10,))
    end = Action(ActionKind.END_TURN)
    decision = SimpleNamespace(family=DecisionFamily.MAIN, actions=[chosen, other, duplicate, end])
    monkeypatch.setattr(module, "model_action_indices", lambda d: range(len(d.actions)))
    assert economic_indices(decision, chosen) == [chosen, other, end]
    play = Action(ActionKind.PLAY_CARD, card_id=0)
    decision.actions.append(play)
    assert economic_indices(decision, chosen) == []


def test_continuations_keep_actual_opponent_and_reset_each_seat_rng(monkeypatch):
    import astro2.counterfactual as module

    actor = SimpleNamespace(spec=SimpleNamespace(encoder_version=1))
    opponent = SimpleNamespace(spec=SimpleNamespace(encoder_version=1))
    monkeypatch.setattr(module, "_ActorChooser", lambda a, e, s: SimpleNamespace(actor=a, seed=s))
    a = continuation_choosers(actor, opponent, 1, 123)
    b = continuation_choosers(actor, opponent, 1, 123)
    assert a[1].actor is actor and a[0].actor is opponent
    assert a[0].seed != a[1].seed
    assert a[0].seed == b[0].seed and a[1].seed == b[1].seed
    assert a[0] is not b[0] and a[1] is not b[1]


def _example():
    return dict(
        state=np.array([1.0], dtype=np.float32),
        actions=np.eye(2, dtype=np.float32),
        family=0,
        teacher_logits=np.array([10.0, -10.0], dtype=np.float32),
        wins=np.array([0.0, 4.0], dtype=np.float32),
        losses=np.zeros(2, dtype=np.float32),
        reference=0,
        samples=8,
        alternatives=1,
        game=17,
    )


def test_ragged_dataset_roundtrip_preserves_game_split_and_labels(tmp_path):
    row = _example()
    second = {
        **row,
        "game": 20,
        "samples": 0,
        "alternatives": 0,
        "wins": np.zeros(2, dtype=np.float32),
    }
    path = tmp_path / "rows.npz"
    save_positions(path, [row, second])
    restored = load_positions(path)
    for original, loaded in zip([row, second], restored, strict=True):
        for name in original:
            np.testing.assert_array_equal(original[name], loaded[name])
    assert restored[0]["game"] % 5 != 0 and restored[1]["game"] % 5 == 0


def test_preferred_never_sampled_action_receives_gradient_and_ties_do_not():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(2, 1, bias=False)
            self.head.update({"weight": mx.array([[10.0, -10.0]])})

        def state_features(self, states):
            return states

        def action_logits_from_features(self, states, actions, families):
            return self.head(actions)

    model = Model()
    row = _example()
    arrays = tuple(mx.array(a) for a in training_batch([row]))
    value, gradients = nn.value_and_grad(model, lambda: comparison_loss(model, *arrays)[0])()
    assert np.isfinite(float(value.item()))
    gradient = np.asarray(gradients["head"]["weight"])[0]
    assert gradient[1] < -1 and gradient[0] > 1
    # Swap the discordant outcomes and the direction must reverse.
    row["wins"], row["losses"] = row["losses"], row["wins"]
    arrays = tuple(mx.array(a) for a in training_batch([row]))
    # Use neutral current logits to keep the reverse logistic gradient away
    # from saturation, and match the reference for zero regularizer gradient.
    model.head.update({"weight": mx.zeros((1, 2))})
    row["teacher_logits"] = np.zeros(2, dtype=np.float32)
    arrays = tuple(mx.array(a) for a in training_batch([row]))
    _, gradients = nn.value_and_grad(model, lambda: comparison_loss(model, *arrays)[0])()
    assert np.asarray(gradients["head"]["weight"])[0, 1] > 0
    row["losses"] = np.zeros(2, dtype=np.float32)
    arrays = tuple(mx.array(a) for a in training_batch([row]))
    _, gradients = nn.value_and_grad(model, lambda: comparison_loss(model, *arrays)[0])()
    assert all(np.allclose(np.asarray(g), 0) for _, g in tree_flatten(gradients))


def test_padding_has_no_effect_on_loss():
    import mlx.core as mx

    from astro2.model import ModelSpec, build_model

    spec = ModelSpec(
        1,
        2,
        1,
        hidden_size=8,
        action_hidden_size=4,
        residual_blocks=0,
        bootstrap_heads=1,
        objective_version=2,
    )
    model = build_model(spec)
    batch = training_batch([_example()])
    initial = float(comparison_loss(model, *(mx.array(a) for a in batch))[0].item())
    padded = list(batch)
    for i in [1, 2, 4, 5, 6]:
        widths = [(0, 0), (0, 3)] + ([(0, 0)] if i == 1 else [])
        padded[i] = np.pad(padded[i], widths)
    actual = float(comparison_loss(model, *(mx.array(a) for a in padded))[0].item())
    assert actual == pytest.approx(initial, abs=1e-5)


@pytest.mark.parametrize(
    "kwargs", [{"rollouts": 0}, {"roots": 0}, {"max_actions": 1}, {"rules_version": 3}]
)
def test_invalid_budget_rejected(kwargs):
    with pytest.raises(ValueError):
        ComparisonConfig(**kwargs)
