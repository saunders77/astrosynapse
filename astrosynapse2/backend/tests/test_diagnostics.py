import astro2.diagnostics as diagnostics
import numpy as np
from astro2.baselines import HeuristicChooser
from astro2.cards import CARD_BY_ID
from astro2.diagnostics import (
    all_family_decision_suite,
    ensemble_metrics,
    strategic_decision_suite,
    strategic_metrics,
    tactical_metrics,
)
from astro2.encoding import ActionKind as EncodedActionKind
from astro2.engine import ActionKind, Decision, DecisionFamily, Game, GameConfig


class EndFavoringActor:
    def predict_options(self, _state, actions, _family):
        logits = np.zeros((len(actions), 1), dtype=np.float32)
        logits[-1, 0] = 10.0
        return logits


class ScrapFavoringActor:
    def predict_options(self, _state, actions, _family):
        logits = np.zeros((len(actions), 3), dtype=np.float32)
        scrap = actions[:, int(EncodedActionKind.SCRAP_FROM_PLAY)] > 0
        keep = actions[:, int(EncodedActionKind.END_TURN)] > 0
        logits[scrap] = 12.0
        logits[keep] = -12.0
        return logits


class SplitHeadActor:
    def predict_options(self, _state, actions, _family):
        order = np.arange(len(actions), dtype=np.float32)
        return np.stack((order, -order, order * 0.25), axis=1)


def test_tactical_diagnostic_reports_raw_error_but_masked_policy_is_safe():
    game = Game(config=GameConfig(seed=91))
    player = game.players[0]
    decision = Decision(
        DecisionFamily.MAIN,
        game.observation(player.player_id),
        game._main_actions(player),
    )
    assert decision.actions[-1].kind == ActionKind.END_TURN
    metrics = tactical_metrics(EndFavoringActor(), (decision,))
    assert metrics["raw_end_turn_violations"] == 1
    assert metrics["masked_end_turn_violations"] == 0
    assert metrics["positions"] == 1


def test_strategic_suite_rejects_actor_that_scraps_expensive_cards_early():
    decisions = strategic_decision_suite(seed=313)
    metrics = strategic_metrics(ScrapFavoringActor(), decisions)

    assert metrics["optional_scrap_positions"] == len(decisions)
    assert metrics["early_high_cost_positions"] >= 4
    assert metrics["early_high_cost_scrap_over_keep_count"] == metrics["early_high_cost_positions"]
    assert metrics["early_high_cost_scrap_over_keep_rate"] == 1.0
    assert metrics["early_high_cost_mean_scrap_over_keep_logit_margin"] == 24.0
    assert metrics["early_high_cost_passed"] is False


def test_bootstrap_baseline_keeps_every_high_cost_card_in_strategic_suite():
    chooser = HeuristicChooser("balanced")
    checked = 0
    for decision in strategic_decision_suite(seed=317):
        scrap = next(
            action for action in decision.actions if action.kind == ActionKind.SCRAP_FOR_ABILITY
        )
        if CARD_BY_ID[scrap.card_id].cost < diagnostics.HIGH_COST_MIN:
            continue
        checked += 1
        assert chooser(0, decision).kind == ActionKind.END_TURN
    assert checked >= 4


def test_all_family_ensemble_suite_reports_independent_head_choices():
    decisions = all_family_decision_suite(seed=419)
    assert {decision.family for decision in decisions} == set(DecisionFamily)

    metrics = ensemble_metrics(SplitHeadActor(), decisions)

    assert metrics["positions"] == len(DecisionFamily)
    assert metrics["families"] == len(DecisionFamily)
    assert metrics["head_argmax_disagreements"] == len(decisions)
    assert metrics["head_argmax_disagreement_rate"] == 1.0
    assert metrics["mean_probability_std"] > 0.0


def test_checkpoint_diagnostics_keeps_existing_groups_and_adds_regressions(monkeypatch):
    actor = ScrapFavoringActor()

    class Loader:
        @staticmethod
        def load(_path):
            return actor

    monkeypatch.setattr(diagnostics, "NumpyActor", Loader)
    monkeypatch.setattr(diagnostics, "_behavioral_suite", lambda **_kwargs: ())
    monkeypatch.setattr(
        diagnostics,
        "tactical_metrics",
        lambda _actor, _decisions: {"positions": 0},
    )
    monkeypatch.setattr(
        diagnostics,
        "heldout_outcome_metrics",
        lambda _actor, **_kwargs: {"games": 0, "samples": 0},
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

    assert set(result) == {"tactical", "strategic", "ensemble", "heldout", "baselines"}
    assert result["strategic"]["early_high_cost_passed"] is False
    assert result["ensemble"]["families"] == len(DecisionFamily)
