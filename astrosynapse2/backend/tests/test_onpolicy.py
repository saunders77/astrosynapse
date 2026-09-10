import numpy as np

from astro2.model import ModelSpec, build_model
from astro2.onpolicy import masked_log_policy, ppo_loss


def fixture():
    import mlx.core as mx

    mx.random.seed(17)
    model = build_model(
        ModelSpec(
            12,
            5,
            3,
            hidden_size=16,
            action_hidden_size=8,
            residual_blocks=1,
            bootstrap_heads=2,
            objective_version=2,
        )
    )
    states = mx.random.normal((4, 12))
    actions = mx.random.normal((4, 5, 5))
    mask = mx.array([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
    families = mx.array([0, 1, 2, 0])
    selected = mx.array([0, 1, 2, 1])
    logp, _ = masked_log_policy(model, states, actions, mask, families, 0.1)
    old = mx.stop_gradient(mx.take_along_axis(logp, selected[:, None], axis=1).squeeze(1))
    return model, (states, actions, mask, families, selected, old)


def test_fresh_behavior_and_update_probabilities_agree_and_padding_has_no_mass():
    import mlx.core as mx

    model, arrays = fixture()
    logp, _ = masked_log_policy(model, *arrays[:4], 0.1)
    probabilities = np.asarray(mx.exp(logp))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    assert not probabilities[np.asarray(arrays[2]) == 0].any()
    _, metrics = ppo_loss(
        model,
        *arrays,
        mx.array([1.0, -1.0, 1.0, -1.0]),
        mx.array([1.0, 0.0, 1.0, 0.0]),
        mx.stop_gradient(logp),
    )
    assert abs(metrics["ratio_mean"].item() - 1) < 1e-6
    assert metrics["approx_kl"].item() < 1e-7
    assert metrics["clip_fraction"].item() == 0
    assert abs(metrics["categorical_kl"].item()) < 1e-7


def test_critic_cannot_move_policy_parameters_but_actor_can():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    model, arrays = fixture()
    targets = mx.array([1.0, 0.0, 1.0, 0.0])

    def critic_loss(*inputs):
        return ppo_loss(model, *inputs, mx.zeros(4), targets)[0]

    _, gradients = nn.value_and_grad(model, critic_loss)(*arrays)
    norms = {name: float(mx.sum(mx.abs(value)).item()) for name, value in tree_flatten(gradients)}
    assert all(norm == 0 for name, norm in norms.items() if not name.startswith("value_output."))
    assert sum(norm for name, norm in norms.items() if name.startswith("value_output.")) > 0

    def actor_loss(*inputs):
        return ppo_loss(model, *inputs, mx.array([1.0, -1.0, 1.0, -1.0]), targets, value_weight=0)[
            0
        ]

    _, gradients = nn.value_and_grad(model, actor_loss)(*arrays)
    norms = {name: float(mx.sum(mx.abs(value)).item()) for name, value in tree_flatten(gradients)}
    assert sum(norm for name, norm in norms.items() if name.startswith("state_in.")) > 0
    assert all(norm == 0 for name, norm in norms.items() if name.startswith("value_output."))
