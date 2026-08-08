import numpy as np
from astro2.model import ModelSpec, NumpyActor, build_model, export_actor


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
