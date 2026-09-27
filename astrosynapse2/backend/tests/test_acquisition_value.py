from dataclasses import replace

import numpy as np
import pytest
from astro2.acquisition_value import (
    FREE_IDS,
    INDEX,
    AcquisitionSample,
    _Likelihood,
    bundle_counts,
    calibrate_acquisition_report,
    extract_acquisition_samples,
    rate_acquisition_samples,
)
from astro2.card_analysis import AcquisitionContext, rate_bucketed_acquire_decisions
from astro2.cards import CARD_BY_ID
from astro2.engine import Action, ActionKind, Decision, DecisionFamily, Game, GameConfig


def decision(turn, trade, row, actions, family=DecisionFamily.MAIN):
    observation = replace(
        Game(config=GameConfig(seed=1)).observation(0),
        turn=turn,
        trade=trade,
        trade_row=tuple(CARD_BY_ID[cid] if cid is not None else None for cid in row),
        explorers_remaining=3,
    )
    return Decision(family, observation, tuple(actions))


def test_records_multi_purchase_stop_and_free_target_without_refill_hindsight():
    explorer = Action(ActionKind.ACQUIRE, card_id=2, source_zone="explorer_supply")
    pod = Action(ActionKind.ACQUIRE, card_id=4, source_zone="trade_row")
    ram = Action(ActionKind.ACQUIRE, card_id=11, source_zone="trade_row")
    end = Action(ActionKind.END_TURN)
    free = Action(ActionKind.FREE_ACQUIRE, card_id=99, target_card_id=11, source_zone="trade_row")
    decline = Action(ActionKind.DECLINE)
    first = decision(2, 4, [4, None, None, None, None], [explorer, pod, end])
    refill = decision(2, 2, [11, None, None, None, None], [explorer, end])
    free_choice = decision(
        2, 0, [11, None, None, None, None], [free, decline], DecisionFamily.FREE_ACQUIRE
    )
    last = decision(2, 2, [4, None, None, None, None], [explorer, pod, end])
    result = extract_acquisition_samples(
        [(0, first, pod), (0, refill, explorer), (0, free_choice, free), (0, last, end)],
        7,
    )
    samples = result["decisions"]
    assert result["acquisitions_recorded"] == 3
    assert result["single_card_turns"] == 1
    assert [sample.chosen for sample in samples] == [4, 2, 11, -1]
    assert sum(sample.weight for sample in samples) == pytest.approx(1)
    assert all(sample.context.acquired_cards == 0 for sample in samples)
    assert 11 not in [cid for cid, _, _ in samples[0].market]
    assert 11 not in [cid for cid, _, _ in samples[1].market]  # unaffordable
    assert samples[2].single
    assert samples[2].market == ((11, 1, 0),)
    assert ram.card_id == 11


def test_bundles_respect_budget_supply_multiplicity_and_free_constraints():
    market = ((2, 2, 2), (4, 2, 2), (11, 1, 3))
    bundles = bundle_counts(market, 4, False, True)
    assert all(row @ np.array([2, 2, 3]) <= 4 for row in bundles)
    assert any(np.array_equal(row, [1, 1, 0]) for row in bundles)
    assert any(np.array_equal(row, [0, 2, 0]) for row in bundles)
    assert len({tuple(row) for row in bundles}) == len(bundles)
    free = bundle_counts(((4, 2, 0), (11, 1, 0)), 0, True, True)
    assert all(sum(row) <= 1 for row in free)


def sample(chosen, game=0, turn=3):
    return AcquisitionSample(
        game,
        0,
        AcquisitionContext(turn, 50, 0, 50, None),
        ((2, 2, 2), (4, 1, 2), (11, 1, 3)),
        4,
        False,
        True,
        chosen,
    )


def test_bundle_likelihood_gradient_and_action_probabilities():
    values = np.zeros(len(FREE_IDS))
    values[INDEX[2]], values[INDEX[4]], values[INDEX[11]] = 2, 1.1, 3.7
    likelihood = _Likelihood([sample(4)])
    _, gradient = likelihood.evaluate(values)
    for cid in (2, 4, 11):
        delta = np.zeros(len(FREE_IDS))
        delta[INDEX[cid]] = 1e-5
        numerical = (
            likelihood.evaluate(values + delta)[0] - likelihood.evaluate(values - delta)[0]
        ) / 2e-5
        assert numerical == pytest.approx(gradient[INDEX[cid]], abs=1e-7)
    # Undo the common regularizer to check that marginalized actions sum to one.
    probabilities = [
        np.exp(-_Likelihood([sample(cid)]).evaluate(values)[0] + 0.25 / 2 * (values @ values))
        for cid in (-1, 2, 4, 11)
    ]
    assert sum(probabilities) == pytest.approx(1)


def test_recovers_known_values_and_returns_clustered_intervals():
    rng = np.random.default_rng(42)
    prototype = sample(2)
    bundles = bundle_counts(prototype.market, 4, False, True)
    true_values = np.array([2, 1.1, 3.7])
    probabilities = np.exp(bundles @ true_values)
    probabilities /= probabilities.sum()
    samples = []
    for game in range(1800):
        bundle = bundles[rng.choice(len(bundles), p=probabilities)]
        chosen = -1 if bundle.sum() == 0 else int(rng.choice([2, 4, 11], p=bundle / bundle.sum()))
        samples.append(sample(chosen, game))
    result = rate_acquisition_samples(samples)
    assert result["fit_converged"]
    entries = {entry["key"]: entry for entry in result["leaderboard"]}
    for cid, truth in ((4, 1.1), (11, 3.7)):
        entry = entries[f"card:{cid}"]
        assert entry["value"] == pytest.approx(truth, abs=0.4)
        assert entry["ci_lower"] < truth < entry["ci_upper"]
    assert entries["card:2"]["ci_lower"] < 2 < entries["card:2"]["ci_upper"]
    assert not entries["card:2"]["fixed_anchor"]
    assert entries["no_card"]["value"] == 0
    assert entries["card:12"]["value"] is None
    assert entries["card:12"]["ci_lower"] is None
    # Repeating observations inside the same game is not new independent evidence.
    duplicated = rate_acquisition_samples(
        [replace(s, weight=0.5) for s in samples for _ in range(2)]
    )
    duplicate_entries = {entry["key"]: entry for entry in duplicated["leaderboard"]}
    assert duplicate_entries["card:4"]["uncertainty"] == pytest.approx(
        entries["card:4"]["uncertainty"]
    )
    reversed_result = rate_acquisition_samples(list(reversed(samples)))
    assert reversed_result["leaderboard"] == result["leaderboard"]
    smaller = rate_acquisition_samples(samples[:450])
    smaller_entries = {entry["key"]: entry for entry in smaller["leaderboard"]}
    assert entries["card:4"]["uncertainty"] < smaller_entries["card:4"]["uncertainty"]


def test_bucket_values_and_empty_evidence_remain_distinct_from_legacy_elo():
    samples = [sample(4, game=game, turn=3) for game in range(4)]
    charts = rate_bucketed_acquire_decisions(samples, acquisition_values=True)
    turn_chart = charts[0]
    assert turn_chart["rating_model"] == "visible_bundle_acquisition_value_v2"
    assert turn_chart["buckets"][2]["captured_decisions"] == 4
    empty = turn_chart["buckets"][0]
    assert empty["captured_decisions"] == 0
    assert all(
        entry["value"] is None for entry in empty["leaderboard"] if not entry["fixed_anchor"]
    )


def test_value_run_persists_overall_fit_and_reloads_new_format(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from astro2 import card_analysis
    from astro2.storage import Store

    samples = [
        sample(cid, game=game, turn=turn)
        for game in range(6)
        for turn, cid in ((1, 4), (2, 11), (3, 2))
    ]
    monkeypatch.setattr(
        card_analysis,
        "resolve_model",
        lambda *args: SimpleNamespace(
            kind="checkpoint",
            actor_path="unused",
            checkpoint_id="test",
            ref="test",
            label="Test",
        ),
    )
    monkeypatch.setattr(
        card_analysis,
        "_simulate_games",
        lambda *args, **kwargs: {
            "decisions": samples,
            "games_completed": 6,
            "truncated_games": 0,
            "turns_observed": 18,
            "single_card_turns": 18,
        },
    )
    result = card_analysis.run_card_analysis(
        None,
        "test",
        "acquire_bucketed",
        card_analysis.CardAnalysisConfig(games=6),
        output_dir=tmp_path,
    )
    saved = json.loads(Path(result["json_path"]).read_text())
    assert "_value_" in result["json_path"]
    assert saved["acquisitions_recorded"] == 18
    assert saved["calibration"]["status"] == "calibrated"
    assert saved["eligible_turns"] == 18
    raw_entries = {
        entry["key"]: entry for entry in rate_acquisition_samples(samples)["leaderboard"]
    }
    for entry in saved["leaderboard"]:
        assert entry["raw_elo"] == raw_entries[entry["key"]]["raw_elo"]
        assert entry["elo"] == pytest.approx(entry["raw_elo"] * saved["calibration"]["factor"])
    assert "Whole-game acquisition values (95% CI)" in saved["report_text"]
    manager = card_analysis.CardAnalysisManager(Store(tmp_path / "store.sqlite3"), tmp_path)
    try:
        restored = manager.get(manager.list()[0]["id"])["result"]
        assert restored["leaderboard"] == saved["leaderboard"]
        assert restored["bucketed_charts"] == saved["bucketed_charts"]
    finally:
        manager.shutdown()


def test_turn_three_scale_is_shared_and_explorer_can_be_negative_later():
    samples = [
        AcquisitionSample(
            game,
            0,
            AcquisitionContext(turn, authority, 0, 50, None),
            ((2, 1, 2),),
            2,
            False,
            True,
            2 if chosen else -1,
        )
        for game in range(200)
        for turn, authority, chosen in ((3, 50, game % 5 != 0), (8, 20, game % 5 == 0))
    ]
    result = rate_acquisition_samples(samples)
    result["bucketed_charts"] = rate_bucketed_acquire_decisions(samples, acquisition_values=True)
    calibrate_acquisition_report(result)
    assert result["calibration"]["status"] == "calibrated"
    factor = result["calibration"]["factor"]
    assert factor > 0
    turns = result["bucketed_charts"][0]["buckets"]
    third = next(entry for entry in turns[2]["leaderboard"] if entry["key"] == "card:2")
    later = next(entry for entry in turns[7]["leaderboard"] if entry["key"] == "card:2")
    assert third["value"] == 2.0
    assert third["calibration_reference"]
    assert third["ci_lower"] < 2 < third["ci_upper"]
    assert later["value"] < 0
    assert later["ci_upper"] < 0
    assert not later["fixed_anchor"]
    assert not later["calibration_reference"]
    overall = next(entry for entry in result["leaderboard"] if entry["key"] == "card:2")
    assert overall["value"] == pytest.approx(0, abs=1e-6)
    for fit in [result, *(b for chart in result["bucketed_charts"] for b in chart["buckets"])]:
        assert fit["normalization_factor"] == factor
        for entry in fit["leaderboard"]:
            if entry["supported"]:
                assert entry["value"] == pytest.approx(entry["raw_elo"] * factor)
                assert entry["uncertainty"] == pytest.approx(entry["raw_uncertainty"] * factor)
                assert entry["ci_lower"] == pytest.approx(entry["raw_ci_lower"] * factor)
                assert entry["ci_upper"] == pytest.approx(entry["raw_ci_upper"] * factor)
            if entry["key"] == "no_card":
                assert entry["value"] == entry["ci_lower"] == entry["ci_upper"] == 0
    import copy

    previous = copy.deepcopy(result)
    calibrate_acquisition_report(result)
    assert result == previous


@pytest.mark.parametrize("case", ["missing", "unsupported", "zero", "negative"])
def test_invalid_turn_three_reference_does_not_reverse_or_invent_a_scale(case):
    result = rate_acquisition_samples([sample(2, game=i) for i in range(10)])
    reference_fit = rate_acquisition_samples([sample(2, game=i) for i in range(10)])
    reference = next(entry for entry in reference_fit["leaderboard"] if entry["key"] == "card:2")
    if case == "unsupported":
        reference["supported"] = False
        reference["value"] = None
    elif case in {"zero", "negative"}:
        reference["raw_elo"] = reference["value"] = 0.0 if case == "zero" else -2.0
    result["bucketed_charts"] = [
        {
            "key": "turn",
            "buckets": []
            if case == "missing"
            else [
                {"key": "3", **reference_fit},
            ],
        }
    ]
    calibrate_acquisition_report(result)
    assert result["calibration"]["status"] == "unavailable"
    assert result["normalization_factor"] == 1
    assert all(entry["elo"] == entry["raw_elo"] for entry in result["leaderboard"])
