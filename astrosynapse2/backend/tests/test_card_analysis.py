from dataclasses import replace

import pytest
from astro2 import card_analysis
from astro2.card_analysis import (
    AcquisitionContext,
    AnalysisKind,
    ChoiceDecision,
    ChoiceOption,
    extract_single_card_turn_decisions,
    rate_bucketed_acquire_decisions,
    rate_choice_decisions,
)
from astro2.cards import CARD_BY_ID
from astro2.engine import Action, ActionKind, Decision, DecisionFamily, Game, GameConfig


def _decision(turn, family, actions):
    observation = replace(Game(config=GameConfig(seed=turn)).observation(0), turn=turn)
    return Decision(family, observation, tuple(actions))


def test_acquire_probe_excludes_entire_turn_when_two_cards_are_acquired():
    explorer = Action(ActionKind.ACQUIRE, card_id=2, source_zone="explorer_supply")
    battle_pod = Action(ActionKind.ACQUIRE, card_id=4, source_zone="trade_row")
    ram = Action(ActionKind.ACQUIRE, card_id=11, source_zone="trade_row")
    end = Action(ActionKind.END_TURN)
    first = _decision(1, DecisionFamily.MAIN, (explorer, battle_pod, end))
    second_a = _decision(2, DecisionFamily.MAIN, (explorer, battle_pod, ram, end))
    second_b = _decision(2, DecisionFamily.MAIN, (explorer, ram, end))

    extracted = extract_single_card_turn_decisions(
        [(0, first, explorer), (0, second_a, battle_pod), (0, second_b, explorer)],
        AnalysisKind.ACQUIRE,
    )

    assert extracted["single_card_turns"] == 1
    assert len(extracted["decisions"]) == 1
    decision = extracted["decisions"][0]
    assert decision.winner.card_name == "Explorer"
    assert [option.card_name for option in decision.alternatives] == ["Battle Pod"]


def test_scrap_probe_counts_in_play_scrap_for_single_card_turn_filter():
    scout = Action(ActionKind.SCRAP_CARD, card_id=0, source_zone="hand")
    viper = Action(ActionKind.SCRAP_CARD, card_id=1, source_zone="discard")
    decline = Action(ActionKind.DECLINE, card_id=20, ability="scrap_any")
    scrap_choice = _decision(3, DecisionFamily.SCRAP, (scout, viper, decline))
    in_play_scrap = _decision(
        3,
        DecisionFamily.MAIN,
        (Action(ActionKind.SCRAP_FOR_ABILITY, card_id=2), Action(ActionKind.END_TURN)),
    )
    clean_choice = _decision(4, DecisionFamily.SCRAP, (scout, viper, decline))

    extracted = extract_single_card_turn_decisions(
        [
            (0, scrap_choice, scout),
            (0, in_play_scrap, in_play_scrap.actions[0]),
            (0, clean_choice, scout),
        ],
        AnalysisKind.SCRAP,
    )

    assert extracted["single_card_turns"] == 1
    assert len(extracted["decisions"]) == 1
    decision = extracted["decisions"][0]
    assert decision.winner.label == "Scout"
    assert [option.label for option in decision.alternatives] == ["Viper"]


def test_scrap_elo_uses_every_unchosen_legal_card():
    decision = ChoiceDecision(
        ChoiceOption("card:0", "Scout", "", "Scout"),
        (
            ChoiceOption("card:1", "Viper", "", "Viper"),
            ChoiceOption("card:2", "Explorer", "", "Explorer"),
        ),
    )

    rated = rate_choice_decisions([decision], AnalysisKind.SCRAP)
    entries = {entry["key"]: entry for entry in rated["leaderboard"]}

    assert rated["scored_decisions"] == 1
    assert rated["pairwise_comparisons"] == 2
    assert entries["card:0"]["elo"] > 1_000
    assert entries["card:1"]["elo"] < 1_000
    assert entries["card:2"]["elo"] < 1_000
    assert entries["card:0"]["pairwise_comparisons"] == 2


def test_acquire_elo_is_normalized_to_explorer_and_rates_all_alternatives():
    decision = ChoiceDecision(
        ChoiceOption("card:2", CARD_BY_ID[2].name, "", CARD_BY_ID[2].name),
        (
            ChoiceOption("card:4", CARD_BY_ID[4].name, "", CARD_BY_ID[4].name),
            ChoiceOption("card:11", CARD_BY_ID[11].name, "", CARD_BY_ID[11].name),
        ),
    )

    rated = rate_choice_decisions([decision], AnalysisKind.ACQUIRE)
    entries = {entry["key"]: entry for entry in rated["leaderboard"]}

    assert rated["pairwise_comparisons"] == 2
    assert entries["card:2"]["elo"] == pytest.approx(2.0)
    assert entries["card:2"]["elo"] > entries["card:4"]["elo"]
    assert entries["card:2"]["elo"] > entries["card:11"]["elo"]


def test_acquire_normalization_preserves_order_when_explorer_raw_elo_is_negative():
    state = card_analysis._initial_rating_state(AnalysisKind.ACQUIRE)
    state["ratings"]["card:2"] = -1.5
    state["ratings"]["card:4"] = 1_965.0
    state["information"]["card:2"] = 4.0
    state["information"]["card:4"] = 4.0

    entries = {
        entry["key"]: entry
        for entry in card_analysis._leaderboard(state, AnalysisKind.ACQUIRE, 24.0)
    }

    assert entries["card:2"]["elo"] == pytest.approx(2.0)
    assert entries["card:4"]["elo"] > entries["card:2"]["elo"]
    assert entries["card:4"]["elo"] == pytest.approx(1_968.5)
    assert entries["card:4"]["card_cost"] == 2
    assert entries["card:4"]["uncertainty"] == entries["card:4"]["raw_uncertainty"]


def test_acquire_context_is_captured_before_the_purchase_and_tracks_opponent_color():
    battle_pod = Action(ActionKind.ACQUIRE, card_id=4, source_zone="trade_row")
    explorer = Action(ActionKind.ACQUIRE, card_id=2, source_zone="explorer_supply")
    ram = Action(ActionKind.ACQUIRE, card_id=11, source_zone="trade_row")
    player_one = _decision(1, DecisionFamily.MAIN, (battle_pod, explorer))
    player_zero = replace(
        _decision(2, DecisionFamily.MAIN, (explorer, ram)),
        observation=replace(
            _decision(2, DecisionFamily.MAIN, (explorer, ram)).observation,
            own_authority=37,
            opponent_authority=24,
        ),
    )

    extracted = extract_single_card_turn_decisions(
        [(1, player_one, battle_pod), (0, player_zero, explorer)],
        AnalysisKind.ACQUIRE_BUCKETED,
    )

    contexts = [decision.context for decision in extracted["decisions"]]
    assert contexts[0] == AcquisitionContext(1, 50, 0, 50, None)
    assert contexts[1] == AcquisitionContext(2, 37, 0, 24, "green")


def test_bucketed_acquire_rates_the_same_decisions_in_all_five_post_hoc_views():
    explorer = ChoiceOption("card:2", CARD_BY_ID[2].name, "", CARD_BY_ID[2].name)
    battle_pod = ChoiceOption("card:4", CARD_BY_ID[4].name, "", CARD_BY_ID[4].name)
    decisions = [
        ChoiceDecision(
            explorer,
            (battle_pod,),
            AcquisitionContext(1, 9, 0, 27, None),
        ),
        ChoiceDecision(
            battle_pod,
            (explorer,),
            AcquisitionContext(33, 17, 5, 31, "green"),
        ),
    ]

    charts = rate_bucketed_acquire_decisions(decisions)
    chart_by_key = {chart["key"]: chart for chart in charts}

    assert list(chart_by_key) == [
        "turn",
        "own_authority",
        "acquired_cards",
        "opponent_authority",
        "opponent_top_color",
    ]
    assert len(chart_by_key["turn"]["buckets"]) == 30
    assert chart_by_key["turn"]["buckets"][-1]["label"] == "30+"
    assert chart_by_key["turn"]["buckets"][-1]["captured_decisions"] == 1
    assert chart_by_key["own_authority"]["buckets"][1]["label"] == "10–19"
    assert [bucket["label"] for bucket in chart_by_key["acquired_cards"]["buckets"]] == [
        "0", "1", "2", "3", "4–5", "6–8", "9–13", "14–21", "22+"
    ]
    assert chart_by_key["opponent_top_color"]["unbucketed_decisions"] == 1
    green = chart_by_key["opponent_top_color"]["buckets"][1]
    assert green["captured_decisions"] == 1
    scored_entries = [entry for entry in green["leaderboard"] if entry["decision_count"]]
    assert {entry["card_color"] for entry in scored_entries} == {"green", "neutral"}
    assert all(entry["uncertainty"] is not None for entry in scored_entries)
