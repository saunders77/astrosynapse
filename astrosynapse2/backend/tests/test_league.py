import numpy as np
from astro2.league import League, Opponent, decide_promotion


def test_pfsp_prefers_harder_opponents():
    league = League(
        [
            Opponent("easy", None, "baseline", "easy", wins=90, games=100),
            Opponent("hard", None, "checkpoint", "hard", wins=20, games=100),
        ]
    )
    rng = np.random.default_rng(2)
    choices = [league.select(rng, mode="pfsp").id for _ in range(500)]
    assert choices.count("hard") > choices.count("easy") * 10


def test_promotion_requires_enough_paired_evidence():
    first = np.ones(100)
    second = np.ones(100)
    early = decide_promotion(first, second, minimum_pairs=5_000, bootstrap_samples=100)
    assert early.promote is False

    decisive = decide_promotion(
        np.ones(5_000), np.ones(5_000), minimum_pairs=5_000, bootstrap_samples=100
    )
    assert decisive.promote is True


def test_league_matchup_statistics_round_trip_without_stale_paths():
    original = League(
        [
            Opponent("old", "/old/path", "checkpoint", "old", wins=7.5, games=12),
            Opponent("baseline:balanced", None, "baseline", "balanced", wins=2, games=4),
        ]
    )
    restored = League(
        [
            Opponent("new", "/new/model", "checkpoint", "new"),
            Opponent("old", "/new/path", "champion", "renamed"),
        ]
    )
    assert restored.restore(original.snapshot()) == 1
    assert [item.id for item in restored.opponents] == ["old", "new"]
    assert restored.opponents[0].actor_path == "/new/path"
    assert (restored.opponents[0].wins, restored.opponents[0].games) == (7.5, 12)
    assert (restored.opponents[1].wins, restored.opponents[1].games) == (0.0, 0)
