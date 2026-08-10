import numpy as np
from astro2.diagnostics import tactical_metrics
from astro2.engine import ActionKind, Decision, DecisionFamily, Game, GameConfig


class EndFavoringActor:
    def predict_options(self, _state, actions, _family):
        logits = np.zeros((len(actions), 1), dtype=np.float32)
        logits[-1, 0] = 10.0
        return logits


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
