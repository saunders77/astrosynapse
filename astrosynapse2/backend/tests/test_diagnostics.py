import astro2.diagnostics as diagnostics
import numpy as np
from astro2.diagnostics import all_family_decision_suite, ensemble_metrics
from astro2.engine import DecisionFamily


class SplitHeadActor:
    def predict_options(self, _state, actions, _family):
        order = np.arange(len(actions), dtype=np.float32)
        return np.stack((order, -order, order * 0.25), axis=1)


def test_all_family_ensemble_suite_reports_independent_head_choices():
    decisions = all_family_decision_suite(seed=419)
    assert {decision.family for decision in decisions} == set(DecisionFamily)

    metrics = ensemble_metrics(SplitHeadActor(), decisions)

    assert metrics["positions"] == len(DecisionFamily)
    assert metrics["families"] == len(DecisionFamily)
    assert metrics["head_argmax_disagreements"] == len(decisions)
    assert metrics["head_argmax_disagreement_rate"] == 1.0
    assert metrics["mean_probability_std"] > 0.0


def test_checkpoint_diagnostics_contains_only_general_quality_groups(monkeypatch):
    actor = SplitHeadActor()

    class Loader:
        @staticmethod
        def load(_path):
            return actor

    monkeypatch.setattr(diagnostics, "NumpyActor", Loader)
    monkeypatch.setattr(
        diagnostics,
        "heldout_outcome_metrics",
        lambda _actor, **_kwargs: {"games": 0, "samples": 0, "game_grouped_brier": 0.2},
    )
    monkeypatch.setattr(
        diagnostics,
        "baseline_metrics",
        lambda _path, **_kwargs: {"mean_score": 0.5, "truncated_games": 0},
    )

    result = diagnostics.checkpoint_diagnostics(
        "unused.actor.npz",
        seed=521,
        games=2,
        baseline_pairs=1,
    )

    assert set(result) == {"ensemble", "heldout", "baselines"}
    assert result["ensemble"]["families"] == len(DecisionFamily)
    assert result["heldout"]["game_grouped_brier"] == 0.2
    assert result["baselines"]["mean_score"] == 0.5
