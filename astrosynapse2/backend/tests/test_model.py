import json
import zipfile
from pathlib import Path

import numpy as np
from astro2.model import (
    ModelSpec,
    NumpyActor,
    actor_critic_policy_loss,
    build_model,
    export_actor,
    load_optimizer_state,
    preference_ranking_loss,
    regenerate_actor_snapshot,
    save_optimizer_state,
)
from safetensors.numpy import save_file


def test_model_spec_preserves_the_legacy_positional_argument_order():
    spec = ModelSpec(17, 11, 4, 32, 16, 2, 5, 1e-4)
    assert spec.hidden_size == 32
    assert spec.action_hidden_size == 16
    assert spec.residual_blocks == 2
    assert spec.bootstrap_heads == 5
    assert spec.layer_norm_eps == 1e-4
    assert spec.encoder_version == 1


def test_epsilon_exploration_is_restricted_to_top_k_actions():
    actor = object.__new__(NumpyActor)
    actor.predict_options = lambda _state, _actions, _family: np.asarray(
        [[-5.0], [-2.0], [0.1], [0.2], [0.3]], dtype=np.float32
    )
    rng = np.random.default_rng(17)
    choices = {
        actor.choose(
            np.zeros(1),
            np.zeros((5, 1)),
            0,
            head=0,
            epsilon=1.0,
            exploration_top_k=2,
            rng=rng,
        )[0]
        for _ in range(100)
    }
    assert choices == {3, 4}


def test_zero_top_k_gives_every_action_exploration_support():
    actor = object.__new__(NumpyActor)
    actor.predict_options = lambda _state, _actions, _family: np.asarray(
        [[-20.0], [-10.0], [0.0], [10.0], [20.0]], dtype=np.float32
    )
    rng = np.random.default_rng(19)
    choices = {
        actor.choose(
            np.zeros(1),
            np.zeros((5, 1)),
            0,
            head=0,
            epsilon=1.0,
            exploration_top_k=0,
            rng=rng,
        )[0]
        for _ in range(300)
    }
    assert choices == set(range(5))


def test_randomized_priors_are_fixed_per_head_and_distinct_between_heads():
    actor = object.__new__(NumpyActor)
    actor.spec = ModelSpec(state_size=7, action_size=5, families=2, bootstrap_heads=3)
    actor._prior_cache = {}
    state = np.arange(7, dtype=np.float32) / 7
    actions = np.arange(20, dtype=np.float32).reshape(4, 5) / 20
    first = actor._randomized_prior(state, actions, family=1, head=0)
    repeated = actor._randomized_prior(state, actions, family=1, head=0)
    other = actor._randomized_prior(state, actions, family=1, head=1)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other)


def test_single_eligible_action_keeps_its_learned_value():
    actor = object.__new__(NumpyActor)
    actor.predict_options = lambda _state, _actions, _family: np.asarray([[-2.0]], dtype=np.float32)
    index, probabilities = actor.choose(
        np.zeros(1),
        np.zeros((1, 1)),
        0,
        head=0,
    )
    assert index == 0
    assert probabilities[0] == np.float32(1.0 / (1.0 + np.exp(2.0)))


def test_generation_four_actor_returns_normalized_legal_action_policy():
    actor = object.__new__(NumpyActor)
    actor.spec = ModelSpec(1, 1, 1, objective_version=2)
    actor.predict_options = lambda _state, _actions, _family: np.asarray(
        [[-1.0], [0.0], [1.0]], dtype=np.float32
    )
    index, probabilities = actor.choose(np.zeros(1), np.zeros((3, 1)), 0, head=0)
    assert index == 2
    np.testing.assert_allclose(probabilities.sum(), 1.0)


def test_numpy_actor_matches_mlx(tmp_path):
    import mlx.core as mx

    spec = ModelSpec(
        state_size=17,
        action_size=11,
        families=4,
        hidden_size=32,
        action_hidden_size=16,
        residual_blocks=2,
        bootstrap_heads=3,
    )
    model = build_model(spec)
    rng = np.random.default_rng(7)
    states = rng.normal(size=(9, spec.state_size)).astype(np.float32)
    actions = rng.normal(size=(9, spec.action_size)).astype(np.float32)
    families = rng.integers(0, spec.families, size=9, dtype=np.int32)

    expected = np.asarray(model(mx.array(states), mx.array(actions), mx.array(families)))
    actor_path = export_actor(model, spec, tmp_path / "actor.npz")
    actual = NumpyActor.load(actor_path).predict(states, actions, families)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)

    same_family = np.full(len(actions), 2, dtype=np.int32)
    batch_expected = np.asarray(model(mx.array(states), mx.array(actions), mx.array(same_family)))
    option_actual = NumpyActor.load(actor_path).predict_options(states[0], actions, 2)
    repeated_expected = np.asarray(
        model(
            mx.array(np.repeat(states[0:1], len(actions), axis=0)),
            mx.array(actions),
            mx.array(same_family),
        )
    )
    np.testing.assert_allclose(option_actual, repeated_expected, rtol=2e-4, atol=2e-4)
    assert batch_expected.shape == option_actual.shape

    actor = NumpyActor.load(actor_path)
    for head in range(spec.bootstrap_heads):
        np.testing.assert_allclose(
            actor.predict_option_head(states[0], actions, 2, head),
            option_actual[:, head],
            rtol=2e-4,
            atol=2e-4,
        )
    batched = actor.predict_option_value_batches(
        (states[0], states[1], states[2]),
        (actions[:2], actions[:5], actions[:3]),
        (2, 1, 3),
        (0, None, 2),
    )
    np.testing.assert_allclose(batched[0], actor.predict_options(states[0], actions[:2], 2)[:, 0])
    np.testing.assert_allclose(
        batched[1], actor.predict_options(states[1], actions[:5], 1).mean(axis=1)
    )
    np.testing.assert_allclose(batched[2], actor.predict_options(states[2], actions[:3], 3)[:, 2])


def test_actor_exports_support_fast_uncompressed_runtime_archives(tmp_path):
    import mlx.core as mx

    spec = ModelSpec(
        state_size=8,
        action_size=6,
        families=2,
        hidden_size=32,
        action_hidden_size=16,
        residual_blocks=1,
        bootstrap_heads=3,
    )
    model = build_model(spec)
    state = np.arange(spec.state_size, dtype=np.float32).reshape(1, -1) / 10
    action = np.arange(spec.action_size, dtype=np.float32).reshape(1, -1) / 10
    family = np.asarray([1], dtype=np.int32)
    expected = np.asarray(model(mx.array(state), mx.array(action), mx.array(family)))

    checkpoint_path = export_actor(model, spec, tmp_path / "checkpoint.npz")
    runtime_path = export_actor(
        model,
        spec,
        tmp_path / "runtime.npz",
        compressed=False,
    )

    with zipfile.ZipFile(checkpoint_path) as archive:
        assert {item.compress_type for item in archive.infolist()} == {zipfile.ZIP_DEFLATED}
    with zipfile.ZipFile(runtime_path) as archive:
        assert {item.compress_type for item in archive.infolist()} == {zipfile.ZIP_STORED}

    for path in (checkpoint_path, runtime_path):
        actual = NumpyActor.load(path).predict(state, action, family)
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


def test_regenerate_actor_snapshot_converts_portable_weights_without_mlx(tmp_path):
    spec = ModelSpec(
        state_size=8,
        action_size=6,
        families=2,
        hidden_size=32,
        action_hidden_size=16,
        residual_blocks=1,
        bootstrap_heads=3,
        encoder_version=2,
        objective_version=2,
    )
    model_path = tmp_path / "checkpoint.safetensors"
    actor_path = tmp_path / "checkpoint.actor.npz"
    weights = {"state_in.weight": np.arange(24, dtype=np.float32).reshape(3, 8)}
    save_file(weights, model_path)
    Path(f"{model_path}.json").write_text(json.dumps(spec.as_dict()))

    result = regenerate_actor_snapshot(model_path, actor_path)
    actor = NumpyActor.load(result)

    assert result == actor_path
    assert actor.spec == spec
    np.testing.assert_array_equal(actor.weights["state_in.weight"], weights["state_in.weight"])
    assert not list(tmp_path.glob("*.partial.npz"))


def test_generation_four_numpy_actor_and_actor_critic_loss(tmp_path):
    import mlx.core as mx

    spec = ModelSpec(
        state_size=8,
        action_size=6,
        families=2,
        hidden_size=32,
        action_hidden_size=16,
        residual_blocks=1,
        bootstrap_heads=3,
        objective_version=2,
    )
    model = build_model(spec)
    rng = np.random.default_rng(29)
    states = rng.normal(size=(4, spec.state_size)).astype(np.float32)
    legal_actions = rng.normal(size=(4, 5, spec.action_size)).astype(np.float32)
    legal_mask = np.ones((4, 5), dtype=np.float32)
    families = np.asarray([0, 1, 0, 1], dtype=np.int32)
    loss, diagnostics = actor_critic_policy_loss(
        model,
        mx.array(states),
        mx.array(legal_actions),
        mx.array(legal_mask),
        mx.array(np.asarray([0, 1, 2, 3], dtype=np.int32)),
        mx.array(families),
        mx.array(np.asarray([1, 0, 1, 0], dtype=np.float32)),
        mx.array(np.full(4, 0.2, dtype=np.float32)),
        mx.array(np.ones((4, 3), dtype=np.float32)),
        mx.array(np.ones(4, dtype=np.float32)),
        search_policy_targets=mx.array(np.full((4, 5), 0.2, dtype=np.float32)),
        search_mask=mx.array(np.ones((4, 5), dtype=np.float32)),
        search_values=mx.array(np.full(4, 0.5, dtype=np.float32)),
        search_valid=mx.array(np.asarray([1, 0, 0, 0], dtype=np.float32)),
        search_policy_loss_weight=1.0,
        search_value_loss_weight=1.0,
        search_loss_reference_positions=4,
        collection_policy_probabilities=mx.array(np.full(4, 0.2, dtype=np.float32)),
        behavior_heads=mx.array(np.asarray([0, 1, 2, -1], dtype=np.int32)),
        importance_groups={"first_half": mx.array(np.asarray([1, 1, 0, 0], dtype=np.float32))},
    )
    mx.eval(loss, *diagnostics.values())
    assert np.isfinite(float(loss.item()))
    assert float(diagnostics["importance_samples_first_half"].item()) == 2.0
    assert float(diagnostics["search_weight_scale"].item()) == 0.25
    assert float(diagnostics["collection_policy_samples"].item()) == 4.0
    assert all(f"importance_ratio_head_{head}" in diagnostics for head in range(3))

    value_only_loss, value_only_diagnostics = actor_critic_policy_loss(
        model,
        mx.array(states),
        mx.array(legal_actions),
        mx.array(legal_mask),
        mx.array(np.asarray([0, 1, 2, 3], dtype=np.int32)),
        mx.array(families),
        mx.array(np.asarray([1, 0, 1, 0], dtype=np.float32)),
        mx.array(np.full(4, 0.2, dtype=np.float32)),
        mx.array(np.ones((4, 3), dtype=np.float32)),
        mx.array(np.ones(4, dtype=np.float32)),
        actor_sample_weights=mx.array(np.zeros(4, dtype=np.float32)),
        reference_model=model,
        reference_policy_kl_weight=1.0,
    )
    mx.eval(value_only_loss, *value_only_diagnostics.values())
    assert float(value_only_diagnostics["actor_sample_fraction"].item()) == 0.0
    assert float(value_only_diagnostics["policy_loss"].item()) == 0.0
    assert float(value_only_diagnostics["policy_entropy"].item()) == 0.0
    assert float(value_only_diagnostics["reference_policy_kl"].item()) == 0.0
    assert float(value_only_diagnostics["value_loss"].item()) > 0.0

    path = export_actor(model, spec, tmp_path / "generation4.actor.npz")
    actor = NumpyActor.load(path)
    expected_values = np.asarray(model.state_values(mx.array(states), mx.array(families)))
    np.testing.assert_allclose(actor.predict_values(states, families), expected_values, atol=2e-4)
    flat_states = np.repeat(states[:1], 5, axis=0)
    expected_policy = np.asarray(
        model(
            mx.array(flat_states),
            mx.array(legal_actions[0]),
            mx.array(np.zeros(5, dtype=np.int32)),
        )
    )
    np.testing.assert_allclose(
        actor.predict_options(states[0], legal_actions[0], 0), expected_policy, atol=2e-4
    )


def test_actor_critic_loss_is_invariant_to_masked_padding_width():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    spec = ModelSpec(
        state_size=8,
        action_size=6,
        families=2,
        hidden_size=32,
        action_hidden_size=16,
        residual_blocks=1,
        bootstrap_heads=3,
        objective_version=2,
    )
    model = build_model(spec)
    rng = np.random.default_rng(31)
    batch_size = 4
    counts = np.asarray([2, 3, 4, 5])
    states = rng.normal(size=(batch_size, spec.state_size)).astype(np.float32)
    actions = rng.normal(size=(batch_size, 64, spec.action_size)).astype(np.float32)
    mask = np.zeros((batch_size, 64), dtype=np.float32)
    for index, count in enumerate(counts):
        mask[index, :count] = 1.0
        actions[index, count:] = 0.0
    arguments = (
        mx.array(states),
        mx.array(np.asarray([1, 2, 3, 4], dtype=np.int32)),
        mx.array(np.asarray([0, 1, 0, 1], dtype=np.int32)),
        mx.array(np.asarray([1, 0, 1, 0], dtype=np.float32)),
        mx.array(np.full(batch_size, 0.25, dtype=np.float32)),
        mx.array(np.ones((batch_size, spec.bootstrap_heads), dtype=np.float32)),
        mx.array(np.ones(batch_size, dtype=np.float32)),
    )

    padded_loss, padded_diagnostics = actor_critic_policy_loss(
        model,
        arguments[0],
        mx.array(actions),
        mx.array(mask),
        *arguments[1:],
    )
    compact_loss, compact_diagnostics = actor_critic_policy_loss(
        model,
        arguments[0],
        mx.array(actions[:, :5]),
        mx.array(mask[:, :5]),
        *arguments[1:],
    )
    mx.eval(
        padded_loss,
        compact_loss,
        *padded_diagnostics.values(),
        *compact_diagnostics.values(),
    )

    np.testing.assert_allclose(float(compact_loss.item()), float(padded_loss.item()), rtol=1e-5)
    for name in padded_diagnostics:
        np.testing.assert_allclose(
            float(compact_diagnostics[name].item()),
            float(padded_diagnostics[name].item()),
            rtol=1e-5,
            atol=1e-6,
            err_msg=name,
        )

    def loss_for_width(legal_actions, legal_mask):
        return actor_critic_policy_loss(
            model,
            arguments[0],
            legal_actions,
            legal_mask,
            *arguments[1:],
        )[0]

    loss_and_grad = nn.value_and_grad(model, loss_for_width)
    _padded_loss, padded_gradients = loss_and_grad(mx.array(actions), mx.array(mask))
    _compact_loss, compact_gradients = loss_and_grad(
        mx.array(actions[:, :5]), mx.array(mask[:, :5])
    )
    padded_leaves = dict(tree_flatten(padded_gradients))
    compact_leaves = dict(tree_flatten(compact_gradients))
    mx.eval(*padded_leaves.values(), *compact_leaves.values())
    assert padded_leaves.keys() == compact_leaves.keys()
    for name in padded_leaves:
        np.testing.assert_allclose(
            np.asarray(compact_leaves[name]),
            np.asarray(padded_leaves[name]),
            rtol=1e-5,
            atol=1e-6,
            err_msg=name,
        )


def test_tactical_preference_loss_is_finite_and_reports_ordering_metrics():
    import mlx.core as mx

    spec = ModelSpec(
        state_size=8,
        action_size=6,
        families=2,
        hidden_size=32,
        action_hidden_size=16,
        residual_blocks=1,
        bootstrap_heads=3,
    )
    model = build_model(spec)
    rng = np.random.default_rng(23)
    loss, diagnostics = preference_ranking_loss(
        model,
        mx.array(rng.normal(size=(5, spec.state_size)).astype(np.float32)),
        mx.array(rng.normal(size=(5, spec.action_size)).astype(np.float32)),
        mx.array(rng.normal(size=(5, spec.action_size)).astype(np.float32)),
        mx.array(np.zeros(5, dtype=np.int32)),
        margin=1.0,
        bootstrap_mask=mx.array(np.tile(np.asarray([[1, 0, 0]], dtype=np.float32), (5, 1))),
    )
    mx.eval(loss, *diagnostics.values())
    assert np.isfinite(float(loss.item()))
    assert set(diagnostics) == {"preference_accuracy", "preference_margin_mean"}


def test_optimizer_state_round_trips_without_losing_tree_structure(tmp_path):
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    spec = ModelSpec(
        state_size=8,
        action_size=6,
        families=2,
        hidden_size=32,
        action_hidden_size=16,
        residual_blocks=1,
        bootstrap_heads=3,
    )
    model = build_model(spec)
    optimizer = optim.AdamW(learning_rate=2e-4, weight_decay=1e-4)
    optimizer.init(model.trainable_parameters())
    optimizer.state["step"] = mx.array(37, mx.uint64)
    mx.eval(optimizer.state)

    path = save_optimizer_state(optimizer, tmp_path / "optimizer.npz")
    restored = optim.AdamW(learning_rate=9e-4, weight_decay=1e-4)
    restored.init(model.trainable_parameters())
    assert load_optimizer_state(restored, path) is True

    expected = list(tree_flatten(optimizer.state))
    actual = list(tree_flatten(restored.state))
    assert [name for name, _value in actual] == [name for name, _value in expected]
    for (_name, expected_value), (_other_name, actual_value) in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(np.asarray(actual_value), np.asarray(expected_value))
