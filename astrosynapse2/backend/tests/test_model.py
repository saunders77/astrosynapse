import numpy as np
from astro2.model import (
    ModelSpec,
    NumpyActor,
    build_model,
    export_actor,
    preference_ranking_loss,
)


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


def test_single_eligible_action_keeps_its_learned_value():
    actor = object.__new__(NumpyActor)
    actor.predict_options = lambda _state, _actions, _family: np.asarray(
        [[-2.0]], dtype=np.float32
    )
    index, probabilities = actor.choose(
        np.zeros(1),
        np.zeros((1, 1)),
        0,
        head=0,
    )
    assert index == 0
    assert probabilities[0] == np.float32(1.0 / (1.0 + np.exp(2.0)))


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
    )
    mx.eval(loss, *diagnostics.values())
    assert np.isfinite(float(loss.item()))
    assert set(diagnostics) == {"preference_accuracy", "preference_margin_mean"}
