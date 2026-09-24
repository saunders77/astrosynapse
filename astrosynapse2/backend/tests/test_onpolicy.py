import numpy as np

from astro2.model import ModelSpec, build_model
from astro2.onpolicy import critic_calibration, critic_logits, masked_log_policy, ppo_loss


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


def test_separate_critic_fits_outcomes_without_changing_policy(tmp_path):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    from astro2.model import load_optimizer_state, save_optimizer_state

    model, arrays = fixture()
    before = {k: np.asarray(v).copy() for k, v in tree_flatten(model.parameters())}
    head = nn.Linear(16, 6)
    head.update(model.value_output.parameters())
    model.value_output.freeze()
    features = mx.stop_gradient(model.state_features(arrays[0]))
    targets = mx.array([1.0, 0.0, 1.0, 0.0])

    def loss():
        logits = critic_logits(head, features, arrays[3], 3, 2)
        return nn.losses.binary_cross_entropy(
            logits,
            mx.broadcast_to(targets[:, None], logits.shape),
            with_logits=True,
            reduction="mean",
        )

    initial = float(loss().item())
    optimizer = optim.Adam(learning_rate=0.003)
    gradient = nn.value_and_grad(head, loss)
    for _ in range(30):
        _, grads = gradient()
        optimizer.update(head, grads)
        mx.eval(head.parameters(), optimizer.state)
    assert float(loss().item()) < initial
    model.value_output.update(head.parameters())
    after = {k: np.asarray(v) for k, v in tree_flatten(model.parameters())}
    for name in before:
        if not name.startswith("value_output."):
            np.testing.assert_array_equal(before[name], after[name])
    assert any(
        not np.array_equal(before[k], after[k]) for k in before if k.startswith("value_output.")
    )

    # A policy update excludes the value head even when the actor optimizer has
    # inherited nonzero critic momentum from the old combined objective.
    model.value_output.unfreeze()
    actor_optimizer = optim.Adam(learning_rate=0.001)
    _, grads = nn.value_and_grad(
        model,
        lambda: ppo_loss(
            model,
            *arrays,
            mx.zeros(4),
            targets,
        )[0],
    )()
    actor_optimizer.update(model, grads)
    mx.eval(model.parameters(), actor_optimizer.state)
    model.value_output.freeze()
    frozen = {k: np.asarray(v).copy() for k, v in tree_flatten(model.value_output.parameters())}
    _, grads = nn.value_and_grad(
        model,
        lambda: ppo_loss(
            model,
            *arrays,
            mx.ones(4),
            targets,
            value_weight=0,
        )[0],
    )()
    actor_optimizer.update(model, grads)
    mx.eval(model.parameters(), actor_optimizer.state)
    for k, v in tree_flatten(model.value_output.parameters()):
        np.testing.assert_array_equal(frozen[k], np.asarray(v))

    path = tmp_path / "critic.npz"
    save_optimizer_state(optimizer, path)
    restored = optim.Adam(learning_rate=0.003)
    assert load_optimizer_state(restored, path)
    for (name, value), (other_name, other) in zip(
        tree_flatten(optimizer.state), tree_flatten(restored.state), strict=True
    ):
        assert name == other_name
        np.testing.assert_array_equal(np.asarray(value), np.asarray(other))


def test_critic_calibration_separates_forced_and_family_errors():
    report = critic_calibration([0.9, 0.5, 0.2], [0.0, 1.0, 0.0], [0, 0, 1], [True, False, False])
    assert report["forced"]["positions"] == 1
    assert abs(report["forced"]["brier"] - 0.81) < 1e-7
    assert report["choices"]["positions"] == 2
    assert report["families"]["1"]["positions"] == 1


def test_forced_positions_train_critic_but_not_policy(monkeypatch):
    from types import SimpleNamespace

    import astro2.onpolicy as module

    decision = SimpleNamespace(actions=[object(), object()], observation=None)
    forced = SimpleNamespace(actions=[object()], observation=None)
    masked = SimpleNamespace(actions=[object(), object()], observation=None, masked=True)
    actor = SimpleNamespace(
        spec=SimpleNamespace(encoder_version=1),
        predict_options=lambda *_: np.array([[0.0], [0.0]]),
    )
    monkeypatch.setattr(module, "cached_actor", lambda _: actor)
    monkeypatch.setattr(
        module,
        "EngineEncoder",
        lambda **_: SimpleNamespace(
            encode_decision=lambda *_: SimpleNamespace(
                state=np.ones(2), actions=np.ones((2, 3)), family=0
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "model_action_indices",
        lambda d: [0] if getattr(d, "masked", False) else list(range(len(d.actions))),
    )

    class Game:
        def __init__(self, choosers, decision_hook, **_):
            self.choose = choosers[0]
            self.hook = decision_hook

        def run(self):
            for current in [forced, masked, decision]:
                # Match the engine: a single rules-legal action bypasses the
                # chooser, while a dominance-masked choice still calls it.
                action = (
                    current.actions[0] if len(current.actions) == 1 else self.choose(0, current)
                )
                self.hook(0, current, action)
            self.hook(1, forced, forced.actions[0])
            return SimpleNamespace(truncated=False, winner=0)

    monkeypatch.setattr(module, "Game", Game)
    trajectory = module.collect_trajectory(("actor", "opponent", 0.03, 17, 0, None, 1, True))
    assert len(trajectory.states) == 1
    assert len(trajectory.value_states) == 3
    assert trajectory.value_forced == [True, True, False]
    assert trajectory.target == 1.0


def test_real_engine_value_hook_covers_bypassed_decisions_without_changing_play(monkeypatch):
    from types import SimpleNamespace

    import astro2.onpolicy as module
    from astro2.baselines import make_baseline
    from astro2.engine import Game as RealGame

    actor = SimpleNamespace(
        spec=SimpleNamespace(encoder_version=1),
        predict_options=lambda state, actions, family: np.zeros(
            (len(actions), 1), dtype=np.float32
        ),
    )
    monkeypatch.setattr(module, "cached_actor", lambda _: actor)
    monkeypatch.setattr(
        module, "_ActorChooser", lambda a, e, seed: make_baseline("balanced", seed=seed)
    )
    observed = []

    def game_factory(**kwargs):
        recorder = kwargs.pop("decision_hook")

        def hook(pid, decision, action):
            if pid == 0:
                observed.append(len(decision.actions))
            if recorder is not None:
                recorder(pid, decision, action)

        return RealGame(**kwargs, decision_hook=hook)

    monkeypatch.setattr(module, "Game", game_factory)
    enabled = module.collect_trajectory(("a", "b", 0.03, 1729, 0, None, 1, True))
    assert 1 in observed  # These bypassed the chooser in the actual engine.
    assert len(enabled.value_states) == len(observed)
    disabled = module.collect_trajectory(("a", "b", 0.03, 1729, 0, None, 1, False))
    assert not disabled.value_states
    assert enabled.target == disabled.target and enabled.truncated == disabled.truncated
    assert enabled.selected == disabled.selected
    np.testing.assert_array_equal(enabled.states, disabled.states)
