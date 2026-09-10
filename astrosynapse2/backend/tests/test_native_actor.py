import platform

import numpy as np
import pytest
from astro2.model import ModelSpec, NumpyActor, build_model, export_actor
from astro2.native_actor import NativeActor

pytestmark = pytest.mark.skipif(platform.system() != "Darwin", reason="Accelerate runtime")


@pytest.fixture
def actors(tmp_path):
    import mlx.core as mx

    mx.random.seed(531)
    spec = ModelSpec(
        12,
        7,
        3,
        hidden_size=16,
        action_hidden_size=8,
        residual_blocks=2,
        bootstrap_heads=2,
        objective_version=2,
    )
    path = tmp_path / "small.actor.npz"
    export_actor(build_model(spec), spec, path)
    return NumpyActor.load(path), NativeActor.load(path)


def test_native_matches_numpy_for_all_heads_families_and_values(actors):
    reference, native = actors
    rng = np.random.default_rng(700)
    states = rng.normal(size=(5, 12)).astype(np.float32)
    actions = rng.normal(size=(9, 7)).astype(np.float32)
    for state in states:
        for family in range(3):
            np.testing.assert_allclose(
                native.predict_options(state, actions, family),
                reference.predict_options(state, actions, family),
                atol=2e-5,
            )
            for head in range(2):
                np.testing.assert_allclose(
                    native.predict_option_head(state, actions, family, head),
                    reference.predict_option_head(state, actions, family, head),
                    atol=2e-5,
                )
    families = np.array([0, 1, 2, 1, 0])
    np.testing.assert_allclose(
        native.predict_values(states, families),
        reference.predict_values(states, families),
        atol=2e-5,
    )


def test_native_rejects_invalid_bounds_before_entering_c(actors):
    _, native = actors
    with pytest.raises(ValueError):
        native.predict_options(np.zeros(11), np.zeros((1, 7)), 0)
    with pytest.raises(ValueError):
        native.predict_options(np.zeros(12), np.zeros((1, 7)), 3)
    with pytest.raises(ValueError):
        native.predict_option_head(np.zeros(12), np.zeros((1, 7)), 0, 2)
    with pytest.raises(ValueError):
        native.predict_values(np.zeros((2, 12)), np.array([0]))
    weights = dict(native.weights)
    weights["state_in.weight"] = np.full_like(weights["state_in.weight"], np.nan)
    with pytest.raises(ValueError, match="nonfinite"):
        NativeActor(native.spec, weights)
