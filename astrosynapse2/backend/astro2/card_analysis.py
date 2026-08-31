"""Candidate card-choice Elo probes for acquisition and deck-thinning decisions.

The probes intentionally measure the policy that is present in a checkpoint,
not game outcomes. A selected card wins against every other card that was
legal in the same decision. Whole turns are retained only when zero or one
card was acquired or scrapped, preventing a multi-card turn from being treated
as several independent preferences.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import threading
import time
import uuid
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from .arena import ModelResolutionError, resolve_model
from .cards import ALL_CARDS, CARD_BY_ID, Faction
from .encoding import DecisionFamily as EncodedDecisionFamily
from .encoding import Encoder
from .engine import (
    Action,
    ActionKind,
    Decision,
    DecisionFamily,
    Game,
    GameConfig,
    model_action_indices,
)
from .engine_encoding import EngineEncoder
from .model import NumpyActor
from .storage import Store

DEFAULT_ANALYSIS_GAMES = 1_000
BUCKETED_ANALYSIS_GAMES = 10_000
MAX_ANALYSIS_GAMES = 10_000
DEFAULT_K_FACTOR = 24.0
INITIAL_ELO = 1_000.0
ACQUIRE_EXPLORER_TARGET = 2.0
ELO_LOGISTIC_SCALE = math.log(10.0) / 400.0
ADAPTIVE_K_REFERENCE_DECISIONS = 16.0
ADAPTIVE_K_PRIOR_DECISIONS = 1.0
ADAPTIVE_K_MIN_MULTIPLIER = 0.25
ADAPTIVE_K_MAX_MULTIPLIER = 4.0


class AnalysisKind(StrEnum):
    SCRAP = "scrap"
    ACQUIRE = "acquire"
    ACQUIRE_BUCKETED = "acquire_bucketed"


def _is_acquire_kind(kind: AnalysisKind) -> bool:
    return kind in {AnalysisKind.ACQUIRE, AnalysisKind.ACQUIRE_BUCKETED}


def default_games_for_kind(kind: AnalysisKind | str) -> int:
    return (
        BUCKETED_ANALYSIS_GAMES
        if AnalysisKind(kind) == AnalysisKind.ACQUIRE_BUCKETED
        else DEFAULT_ANALYSIS_GAMES
    )


@dataclass(frozen=True, slots=True)
class CardAnalysisConfig:
    games: int = DEFAULT_ANALYSIS_GAMES
    seed: int = 20260813
    max_turns: int = 400
    max_actions_per_turn: int = 200
    k_factor: float = DEFAULT_K_FACTOR
    workers: int = max(1, min(4, (os.cpu_count() or 4) - 1))

    def __post_init__(self) -> None:
        if not 1 <= self.games <= MAX_ANALYSIS_GAMES:
            raise ValueError(f"games must be between 1 and {MAX_ANALYSIS_GAMES:,}")
        if not 20 <= self.max_turns <= 500:
            raise ValueError("max_turns must be between 20 and 500")
        if not 20 <= self.max_actions_per_turn <= 500:
            raise ValueError("max_actions_per_turn must be between 20 and 500")
        if not math.isfinite(self.k_factor) or self.k_factor <= 0:
            raise ValueError("k_factor must be greater than zero")
        if not 1 <= self.workers <= 16:
            raise ValueError("workers must be between 1 and 16")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ChoiceOption:
    key: str
    card_name: str
    source: str
    label: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


NO_CARD_OPTION = ChoiceOption("no_card", "No Card", "", "No Card")
NO_DISCARD_OPTION = ChoiceOption("no_discard", "No Discard", "", "No Discard")


@dataclass(frozen=True, slots=True)
class ChoiceDecision:
    winner: ChoiceOption
    alternatives: tuple[ChoiceOption, ...]
    context: AcquisitionContext | None = None


@dataclass(frozen=True, slots=True)
class AcquisitionContext:
    """Choice-time state attached to an acquisition result for post-hoc grouping."""

    turn: int
    own_authority: int
    acquired_cards: int
    opponent_authority: int
    opponent_top_color: str | None


@dataclass(slots=True)
class _TurnChoices:
    acquired_cards: int = 0
    scrapped_cards: int = 0
    acquire_decisions: list[ChoiceDecision] | None = None
    scrap_decisions: list[ChoiceDecision] | None = None
    end_turn_decision: ChoiceDecision | None = None

    def __post_init__(self) -> None:
        self.acquire_decisions = [] if self.acquire_decisions is None else self.acquire_decisions
        self.scrap_decisions = [] if self.scrap_decisions is None else self.scrap_decisions


class _GreedyChooser:
    """The same mean-head, epsilon-free policy used for deployment."""

    def __init__(self, actor: NumpyActor, encoder: Encoder, seed: int):
        self.actor = actor
        self.encoder = encoder
        self.rng = np.random.default_rng(seed)

    def __call__(self, _player_id: int, decision: Decision) -> Action:
        encoded = self.encoder.encode_decision(decision.observation, decision)
        eligible = np.asarray(model_action_indices(decision), dtype=np.int64)
        local_index, _values = self.actor.choose(
            encoded.state,
            encoded.actions[eligible],
            int(encoded.family),
            epsilon=0.0,
            head=None,
            rng=self.rng,
        )
        return decision.actions[int(eligible[local_index])]


_POLICY_CACHE: OrderedDict[str, tuple[int, int, NumpyActor, Encoder]] = OrderedDict()


def _load_actor_encoder(actor_path: str) -> tuple[NumpyActor, Encoder]:
    path = str(Path(actor_path).expanduser().resolve())
    stat = Path(path).stat()
    cached = _POLICY_CACHE.get(path)
    if cached is None or cached[:2] != (stat.st_mtime_ns, stat.st_size):
        actor = NumpyActor.load(path)
        encoder = EngineEncoder(version=actor.spec.encoder_version)
        _POLICY_CACHE[path] = (stat.st_mtime_ns, stat.st_size, actor, encoder)
    else:
        actor, encoder = cached[2], cached[3]
    _POLICY_CACHE.move_to_end(path)
    while len(_POLICY_CACHE) > 2:
        _POLICY_CACHE.popitem(last=False)
    return actor, encoder


def _derived_seed(seed: int, game_index: int, stream: int) -> int:
    sequence = np.random.SeedSequence([int(seed) % (1 << 64), game_index, stream])
    return int(sequence.generate_state(1, dtype=np.uint64)[0] & np.uint64((1 << 63) - 1))


def _acquire_option(action: Action) -> ChoiceOption | None:
    if action.kind not in {ActionKind.ACQUIRE, ActionKind.FREE_ACQUIRE}:
        return None
    card = CARD_BY_ID.get(action.card_id)
    if card is None:
        return None
    return ChoiceOption(f"card:{card.card_id}", card.name, "", card.name)


def _scrap_option(action: Action) -> ChoiceOption | None:
    if action.kind == ActionKind.DECLINE:
        return NO_DISCARD_OPTION
    if action.kind != ActionKind.SCRAP_CARD:
        return None
    card = CARD_BY_ID.get(action.card_id)
    if card is None:
        return None
    # The requested score is per card, so hand and discard observations feed
    # one shared option rather than producing two source-specific ratings.
    return ChoiceOption(f"card:{card.card_id}", card.name, "", card.name)


def _choice_decision(
    decision: Decision,
    selected: Action,
    option_from_action: Callable[[Action], ChoiceOption | None],
    context: AcquisitionContext | None = None,
) -> ChoiceDecision | None:
    winner = option_from_action(selected)
    if winner is None:
        return None
    alternatives: list[ChoiceOption] = []
    seen = {winner.key}
    for action in decision.actions:
        option = option_from_action(action)
        if option is None or option.key in seen:
            continue
        seen.add(option.key)
        alternatives.append(option)
    return ChoiceDecision(winner, tuple(alternatives), context)


def _no_card_decision(
    decision: Decision,
    context: AcquisitionContext,
) -> ChoiceDecision | None:
    """Build the end-of-turn choice among declining and affordable cards."""

    alternatives: list[ChoiceOption] = []
    seen: set[str] = set()
    for action in decision.actions:
        option = _acquire_option(action)
        card = CARD_BY_ID.get(action.card_id)
        if (
            option is None
            or card is None
            or card.cost > decision.observation.trade
            or option.key in seen
        ):
            continue
        seen.add(option.key)
        alternatives.append(option)
    if not alternatives:
        return None
    return ChoiceDecision(NO_CARD_OPTION, tuple(alternatives), context)


_FACTION_COLORS = {
    Faction.MACHINE_CULT: "red",
    Faction.BLOB: "green",
    Faction.TRADE_FEDERATION: "blue",
    Faction.STAR_EMPIRE: "yellow",
}
_COLOR_ORDER = ("red", "green", "blue", "yellow")


def _opponent_top_acquired_color(
    counts: Counter[str],
    most_recent: dict[str, int],
) -> str | None:
    maximum = max((counts.get(color, 0) for color in _COLOR_ORDER), default=0)
    if maximum <= 0:
        return None
    leaders = [color for color in _COLOR_ORDER if counts.get(color, 0) == maximum]
    return max(leaders, key=lambda color: most_recent.get(color, -1))


def extract_single_card_turn_decisions(
    events: Iterable[tuple[int, Decision, Action]],
    kind: AnalysisKind | str,
) -> dict[str, Any]:
    """Extract eligible choices from one game of decision-hook events.

    All actual acquisition mechanisms count toward the acquire-per-turn filter.
    Both hand/discard scraps and in-play scrap-for-ability actions count toward
    the scrap-per-turn filter, while only hand/discard choice sets are rated.
    """

    resolved_kind = AnalysisKind(kind)
    turns: dict[tuple[int, int], _TurnChoices] = {}
    acquired_counts = {0: 0, 1: 0}
    acquired_colors = {0: Counter(), 1: Counter()}
    most_recent_colors: dict[int, dict[str, int]] = {0: {}, 1: {}}
    acquisition_sequence = 0
    for player_id, decision, selected in events:
        key = (int(player_id), int(decision.observation.turn))
        turn = turns.setdefault(key, _TurnChoices())
        if selected.kind in {ActionKind.ACQUIRE, ActionKind.FREE_ACQUIRE}:
            turn.acquired_cards += 1
            opponent_id = 1 - int(player_id)
            context = AcquisitionContext(
                turn=max(1, int(decision.observation.turn)),
                own_authority=max(0, int(decision.observation.own_authority)),
                acquired_cards=acquired_counts[int(player_id)],
                opponent_authority=max(0, int(decision.observation.opponent_authority)),
                opponent_top_color=_opponent_top_acquired_color(
                    acquired_colors[opponent_id], most_recent_colors[opponent_id]
                ),
            )
            choice = _choice_decision(decision, selected, _acquire_option, context)
            if choice is not None:
                assert turn.acquire_decisions is not None
                turn.acquire_decisions.append(choice)
            acquired_counts[int(player_id)] += 1
            card = CARD_BY_ID.get(selected.card_id)
            color = _FACTION_COLORS.get(card.faction) if card is not None else None
            if color is not None:
                acquisition_sequence += 1
                acquired_colors[int(player_id)][color] += 1
                most_recent_colors[int(player_id)][color] = acquisition_sequence
        if selected.kind == ActionKind.END_TURN:
            opponent_id = 1 - int(player_id)
            context = AcquisitionContext(
                turn=max(1, int(decision.observation.turn)),
                own_authority=max(0, int(decision.observation.own_authority)),
                acquired_cards=acquired_counts[int(player_id)],
                opponent_authority=max(0, int(decision.observation.opponent_authority)),
                opponent_top_color=_opponent_top_acquired_color(
                    acquired_colors[opponent_id], most_recent_colors[opponent_id]
                ),
            )
            turn.end_turn_decision = _no_card_decision(decision, context)
        if selected.kind in {ActionKind.SCRAP_CARD, ActionKind.SCRAP_FOR_ABILITY}:
            turn.scrapped_cards += 1
        if decision.family == DecisionFamily.SCRAP:
            choice = _choice_decision(decision, selected, _scrap_option)
            if choice is not None:
                assert turn.scrap_decisions is not None
                turn.scrap_decisions.append(choice)

    single_card_turns = 0
    decisions: list[ChoiceDecision] = []
    for turn in turns.values():
        if _is_acquire_kind(resolved_kind):
            if turn.acquired_cards > 1:
                continue
            if turn.acquired_cards == 1:
                single_card_turns += 1
                # Rate the acquired card against the alternatives at purchase,
                # then give it a separate head-to-head win over No Card. This
                # keeps No Card from indirectly competing with cards that were
                # merely present earlier in the turn.
                acquired = (turn.acquire_decisions or [])[:1]
                decisions.extend(acquired)
                if acquired:
                    choice = acquired[0]
                    decisions.append(
                        ChoiceDecision(choice.winner, (NO_CARD_OPTION,), choice.context)
                    )
            elif turn.end_turn_decision is not None:
                single_card_turns += 1
                decisions.append(turn.end_turn_decision)
            continue

        if turn.scrapped_cards > 1:
            continue
        turn_decisions = turn.scrap_decisions or []
        if turn_decisions:
            single_card_turns += 1
            decisions.extend(turn_decisions)
    return {
        "turns_observed": len(turns),
        "single_card_turns": single_card_turns,
        "decisions": decisions,
    }


def _adaptive_k_multiplier(decisions: int) -> float:
    multiplier = math.sqrt(
        (ADAPTIVE_K_REFERENCE_DECISIONS + ADAPTIVE_K_PRIOR_DECISIONS)
        / (max(0, decisions) + ADAPTIVE_K_PRIOR_DECISIONS)
    )
    return min(ADAPTIVE_K_MAX_MULTIPLIER, max(ADAPTIVE_K_MIN_MULTIPLIER, multiplier))


def _adaptive_k(base: float, decisions: int) -> float:
    return base * _adaptive_k_multiplier(decisions)


def _initial_rating_state(kind: AnalysisKind) -> dict[str, Any]:
    if _is_acquire_kind(kind):
        options = [
            ChoiceOption(f"card:{card.card_id}", card.name, "", card.name)
            for card in ALL_CARDS
            if card.card_id not in {0, 1}  # starters never enter an acquire choice
        ]
        options.append(NO_CARD_OPTION)
    else:
        options = [
            ChoiceOption(f"card:{card.card_id}", card.name, "", card.name)
            for card in ALL_CARDS
        ]
        options.append(NO_DISCARD_OPTION)
    return {
        "ratings": {option.key: INITIAL_ELO for option in options},
        "decisions": {option.key: 0 for option in options},
        "comparisons": {option.key: 0 for option in options},
        "wins": {option.key: 0 for option in options},
        "losses": {option.key: 0 for option in options},
        "information": {option.key: 0.0 for option in options},
        "options": {option.key: option for option in options},
        "order": [option.key for option in options],
    }


def _ensure_option(state: dict[str, Any], option: ChoiceOption) -> None:
    if option.key in state["options"]:
        return
    state["options"][option.key] = option
    state["order"].append(option.key)
    for field, value in (
        ("ratings", INITIAL_ELO),
        ("decisions", 0),
        ("comparisons", 0),
        ("wins", 0),
        ("losses", 0),
        ("information", 0.0),
    ):
        state[field][option.key] = value


def _unique_alternatives(decision: ChoiceDecision) -> list[ChoiceOption]:
    result: list[ChoiceOption] = []
    seen = {decision.winner.key}
    for option in decision.alternatives:
        if option.key in seen:
            continue
        seen.add(option.key)
        result.append(option)
    return result


def _apply_acquire_result(state: dict[str, Any], decision: ChoiceDecision, base_k: float) -> int:
    alternatives = _unique_alternatives(decision)
    if not alternatives:
        return 0
    participants = [decision.winner, *alternatives]
    for option in participants:
        _ensure_option(state, option)
    ratings = state["ratings"]
    maximum = max(ratings[option.key] for option in participants)
    weights = {
        option.key: math.exp((ratings[option.key] - maximum) * ELO_LOGISTIC_SCALE)
        for option in participants
    }
    total = sum(weights.values())
    probabilities = {key: weight / total for key, weight in weights.items()}
    prior = {option.key: state["decisions"][option.key] for option in participants}
    for option in participants:
        key = option.key
        probability = probabilities[key]
        actual = 1.0 if key == decision.winner.key else 0.0
        ratings[key] += _adaptive_k(base_k, prior[key]) * (actual - probability)
        state["decisions"][key] += 1
        state["comparisons"][key] += len(participants) - 1
        state["information"][key] += probability * (1.0 - probability)
        if actual:
            state["wins"][key] += 1
        else:
            state["losses"][key] += 1
    return len(alternatives)


def _apply_scrap_result(state: dict[str, Any], decision: ChoiceDecision, base_k: float) -> int:
    alternatives = _unique_alternatives(decision)
    if not alternatives:
        return 0
    winner = decision.winner
    _ensure_option(state, winner)
    for option in alternatives:
        _ensure_option(state, option)
    ratings = state["ratings"]
    prior = {option.key: state["decisions"][option.key] for option in [winner, *alternatives]}
    base_winner = ratings[winner.key]
    deltas = {winner.key: 0.0}
    for loser in alternatives:
        expected = 1.0 / (1.0 + 10.0 ** ((ratings[loser.key] - base_winner) / 400.0))
        remainder = 1.0 - expected
        deltas[winner.key] += _adaptive_k(base_k, prior[winner.key]) * remainder
        deltas[loser.key] = deltas.get(loser.key, 0.0) - _adaptive_k(
            base_k, prior[loser.key]
        ) * remainder
        state["comparisons"][winner.key] += 1
        state["comparisons"][loser.key] += 1
        state["wins"][winner.key] += 1
        state["losses"][loser.key] += 1
        information = expected * (1.0 - expected)
        state["information"][winner.key] += information
        state["information"][loser.key] += information
    for option in [winner, *alternatives]:
        state["decisions"][option.key] += 1
    for key, delta in deltas.items():
        ratings[key] += delta
    return len(alternatives)


def _leaderboard(state: dict[str, Any], kind: AnalysisKind, base_k: float) -> list[dict[str, Any]]:
    normalization_offset = 0.0
    if _is_acquire_kind(kind):
        explorer_key = "card:2"
        explorer_rating = state["ratings"].get(explorer_key, INITIAL_ELO)
        normalization_offset = ACQUIRE_EXPLORER_TARGET - explorer_rating
    entries: list[dict[str, Any]] = []
    for key in state["order"]:
        option: ChoiceOption = state["options"][key]
        information = float(state["information"][key])
        raw_uncertainty = (
            1.0 / (ELO_LOGISTIC_SCALE * math.sqrt(information)) if information > 0 else None
        )
        raw_elo = float(state["ratings"][key])
        decisions = int(state["decisions"][key])
        entries.append(
            {
                **option.to_dict(),
                "card_color": _card_color_for_option(option),
                "card_cost": _card_cost_for_option(option),
                "elo": round(raw_elo + normalization_offset, 4),
                "raw_elo": round(raw_elo, 4),
                "uncertainty": None if raw_uncertainty is None else round(raw_uncertainty, 4),
                "raw_uncertainty": None if raw_uncertainty is None else round(raw_uncertainty, 4),
                "decision_count": decisions,
                "pairwise_comparisons": int(state["comparisons"][key]),
                "wins": int(state["wins"][key]),
                "losses": int(state["losses"][key]),
                "next_k_factor": round(_adaptive_k(base_k, decisions), 4),
            }
        )
    entries.sort(key=lambda entry: (-float(entry["elo"]), str(entry["label"])))
    return entries


def _card_color_for_option(option: ChoiceOption) -> str:
    try:
        card_id = int(option.key.split(":", 1)[1])
    except (IndexError, ValueError):
        return "neutral"
    card = CARD_BY_ID.get(card_id)
    if card is None:
        return "neutral"
    return _FACTION_COLORS.get(card.faction, "neutral")


def _card_cost_for_option(option: ChoiceOption) -> int:
    try:
        card_id = int(option.key.split(":", 1)[1])
    except (IndexError, ValueError):
        return 0
    card = CARD_BY_ID.get(card_id)
    return int(card.cost) if card is not None else 0


def rate_choice_decisions(
    decisions: Sequence[ChoiceDecision],
    kind: AnalysisKind | str,
    *,
    k_factor: float = DEFAULT_K_FACTOR,
) -> dict[str, Any]:
    resolved_kind = AnalysisKind(kind)
    state = _initial_rating_state(resolved_kind)
    comparisons = 0
    scored = 0
    apply = _apply_acquire_result if _is_acquire_kind(resolved_kind) else _apply_scrap_result
    for decision in decisions:
        added = apply(state, decision, k_factor)
        comparisons += added
        scored += int(added > 0)
    explorer_raw = state["ratings"].get("card:2", INITIAL_ELO)
    normalization_offset = (
        ACQUIRE_EXPLORER_TARGET - explorer_raw if _is_acquire_kind(resolved_kind) else 0.0
    )
    return {
        "scored_decisions": scored,
        "pairwise_comparisons": comparisons,
        "normalization_factor": 1.0,
        "normalization_offset": normalization_offset,
        "explorer_raw_elo": explorer_raw if _is_acquire_kind(resolved_kind) else None,
        "leaderboard": _leaderboard(state, resolved_kind, k_factor),
    }


_ACQUIRED_CARD_BUCKETS = (
    ("0", "0", 0, 0),
    ("1", "1", 1, 1),
    ("2", "2", 2, 2),
    ("3", "3", 3, 3),
    ("4_5", "4–5", 4, 5),
    ("6_8", "6–8", 6, 8),
    ("9_13", "9–13", 9, 13),
    ("14_21", "14–21", 14, 21),
    ("22_plus", "22+", 22, None),
)


def _ten_authority_bucket(value: int) -> tuple[str, str]:
    start = max(0, int(value)) // 10 * 10
    return f"{start}_{start + 9}", f"{start}–{start + 9}"


def _acquired_card_bucket(value: int) -> tuple[str, str]:
    for key, label, minimum, maximum in _ACQUIRED_CARD_BUCKETS:
        if value >= minimum and (maximum is None or value <= maximum):
            return key, label
    return "22_plus", "22+"


def rate_bucketed_acquire_decisions(
    decisions: Sequence[ChoiceDecision],
    *,
    k_factor: float = DEFAULT_K_FACTOR,
) -> list[dict[str, Any]]:
    """Rate the same captured choices independently in five context groupings."""

    contextual = [decision for decision in decisions if decision.context is not None]
    maximum_own_authority = max(
        (max(0, decision.context.own_authority) for decision in contextual), default=0
    )
    maximum_opponent_authority = max(
        (max(0, decision.context.opponent_authority) for decision in contextual), default=0
    )
    own_authority_starts = list(range(0, maximum_own_authority // 10 * 10 + 1, 10))
    opponent_authority_starts = list(
        range(0, maximum_opponent_authority // 10 * 10 + 1, 10)
    )
    definitions: list[tuple[str, str, list[tuple[str, str]]]] = [
        (
            "turn",
            "Turn number",
            [(str(turn), "30+" if turn == 30 else str(turn)) for turn in range(1, 31)],
        ),
        (
            "own_authority",
            "Your authority",
            [(f"{start}_{start + 9}", f"{start}–{start + 9}") for start in own_authority_starts],
        ),
        (
            "acquired_cards",
            "Cards already acquired",
            [(key, label) for key, label, _minimum, _maximum in _ACQUIRED_CARD_BUCKETS],
        ),
        (
            "opponent_authority",
            "Opponent authority",
            [
                (f"{start}_{start + 9}", f"{start}–{start + 9}")
                for start in opponent_authority_starts
            ],
        ),
        (
            "opponent_top_color",
            "Opponent top-acquired color",
            [(color, color.title()) for color in _COLOR_ORDER],
        ),
    ]

    grouped: dict[str, dict[str, list[ChoiceDecision]]] = {
        key: {bucket_key: [] for bucket_key, _label in buckets}
        for key, _label, buckets in definitions
    }
    unbucketed_colors = 0
    for decision in contextual:
        context = decision.context
        assert context is not None
        grouped["turn"][str(min(30, max(1, context.turn)))].append(decision)
        grouped["own_authority"][_ten_authority_bucket(context.own_authority)[0]].append(
            decision
        )
        grouped["acquired_cards"][_acquired_card_bucket(context.acquired_cards)[0]].append(
            decision
        )
        grouped["opponent_authority"][
            _ten_authority_bucket(context.opponent_authority)[0]
        ].append(decision)
        if context.opponent_top_color is None:
            unbucketed_colors += 1
        else:
            grouped["opponent_top_color"][context.opponent_top_color].append(decision)

    charts: list[dict[str, Any]] = []
    for key, label, buckets in definitions:
        bucket_results = []
        for bucket_key, bucket_label in buckets:
            bucket_decisions = grouped[key].get(bucket_key, [])
            rated = rate_choice_decisions(
                bucket_decisions,
                AnalysisKind.ACQUIRE,
                k_factor=k_factor,
            )
            bucket_results.append(
                {
                    "key": bucket_key,
                    "label": bucket_label,
                    "captured_decisions": len(bucket_decisions),
                    **rated,
                }
            )
        charts.append(
            {
                "key": key,
                "label": label,
                "buckets": bucket_results,
                "unbucketed_decisions": unbucketed_colors if key == "opponent_top_color" else 0,
            }
        )
    return charts


def _simulate_game_batch(
    actor_path: str,
    kind: AnalysisKind,
    config: CardAnalysisConfig,
    game_indices: Sequence[int],
    *,
    progress: Callable[[int, int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    actor, encoder = _load_actor_encoder(actor_path)
    if actor.spec.state_size != encoder.state_size or actor.spec.action_size != encoder.action_size:
        raise ModelResolutionError("checkpoint actor encoder is incompatible")
    if actor.spec.families != len(EncodedDecisionFamily):
        raise ModelResolutionError("checkpoint actor decision families are incompatible")

    decisions: list[ChoiceDecision] = []
    games_completed = 0
    truncated_games = 0
    turns_observed = 0
    single_card_turns = 0
    for game_index in game_indices:
        if cancelled is not None and cancelled():
            break
        events: list[tuple[int, Decision, Action]] = []

        def capture(
            player_id: int,
            decision: Decision,
            selected: Action,
            game_events: list[tuple[int, Decision, Action]] = events,
        ) -> None:
            game_events.append((player_id, decision, selected))

        game_seed = _derived_seed(config.seed, game_index, 0)
        game = Game(
            player_names=("candidate_a", "candidate_b"),
            choosers=(
                _GreedyChooser(actor, encoder, _derived_seed(config.seed, game_index, 1)),
                _GreedyChooser(actor, encoder, _derived_seed(config.seed, game_index, 2)),
            ),
            config=GameConfig(
                seed=game_seed,
                max_turns=config.max_turns,
                max_actions_per_turn=config.max_actions_per_turn,
            ),
            decision_hook=capture,
        )
        result = game.run()
        games_completed += 1
        truncated_games += int(result.truncated)
        extracted = extract_single_card_turn_decisions(events, kind)
        turns_observed += int(extracted["turns_observed"])
        single_card_turns += int(extracted["single_card_turns"])
        decisions.extend(extracted["decisions"])
        if progress is not None:
            progress(games_completed, single_card_turns, len(decisions))
    return {
        "games_completed": games_completed,
        "truncated_games": truncated_games,
        "turns_observed": turns_observed,
        "single_card_turns": single_card_turns,
        "decisions": decisions,
    }


def _simulate_game_batch_worker(task: dict[str, Any]) -> dict[str, Any]:
    config = CardAnalysisConfig(**task["config"])
    return _simulate_game_batch(
        task["actor_path"],
        AnalysisKind(task["kind"]),
        config,
        task["game_indices"],
    )


def _simulate_games(
    actor_path: str,
    kind: AnalysisKind,
    config: CardAnalysisConfig,
    *,
    progress: Callable[[int, int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    worker_count = min(config.workers, config.games)
    if worker_count <= 1:
        return _simulate_game_batch(
            actor_path,
            kind,
            config,
            range(config.games),
            progress=progress,
            cancelled=cancelled,
        )

    # Short batches keep GUI progress and shutdown latency responsive. Results
    # are merged in original game-index order because adaptive Elo depends on
    # observation order even though game simulation itself is independent.
    batch_size = max(1, min(8, math.ceil(config.games / (worker_count * 8))))
    batches = [
        list(range(start, min(config.games, start + batch_size)))
        for start in range(0, config.games, batch_size)
    ]
    context = mp.get_context("spawn")
    try:
        executor = ProcessPoolExecutor(max_workers=worker_count, mp_context=context)
    except OSError:
        # Restricted shells can deny POSIX semaphore discovery. The GUI still
        # gets a correct job in that environment, just without process speedup.
        return _simulate_game_batch(
            actor_path,
            kind,
            config,
            range(config.games),
            progress=progress,
            cancelled=cancelled,
        )
    futures: dict[Future[dict[str, Any]], int] = {}
    for index, game_indices in enumerate(batches):
        futures[
            executor.submit(
                _simulate_game_batch_worker,
                {
                    "actor_path": actor_path,
                    "kind": kind.value,
                    "config": config.to_dict(),
                    "game_indices": game_indices,
                },
            )
        ] = index

    completed_games = 0
    completed_single_card_turns = 0
    completed_decisions = 0
    results: dict[int, dict[str, Any]] = {}
    try:
        for future in as_completed(futures):
            batch = future.result()
            results[futures[future]] = batch
            completed_games += int(batch["games_completed"])
            completed_single_card_turns += int(batch["single_card_turns"])
            completed_decisions += len(batch["decisions"])
            if progress is not None:
                progress(completed_games, completed_single_card_turns, completed_decisions)
            if cancelled is not None and cancelled():
                break
    finally:
        should_cancel = cancelled is not None and cancelled()
        executor.shutdown(wait=True, cancel_futures=should_cancel)

    combined: dict[str, Any] = {
        "games_completed": 0,
        "truncated_games": 0,
        "turns_observed": 0,
        "single_card_turns": 0,
        "decisions": [],
    }
    for index in sorted(results):
        batch = results[index]
        for field in (
            "games_completed",
            "truncated_games",
            "turns_observed",
            "single_card_turns",
        ):
            combined[field] += int(batch[field])
        combined["decisions"].extend(batch["decisions"])
    return combined


def format_analysis_report(result: dict[str, Any]) -> str:
    kind = AnalysisKind(result["kind"])
    display_kind = "Bucketed Acquire" if kind == AnalysisKind.ACQUIRE_BUCKETED else kind.value.title()
    lines = [
        f"{display_kind} Elo Test for {result['model']['label']}",
        f"Checkpoint: {result['model']['id']}",
        f"Games: {result['games_completed']} / {result['games_requested']}",
        "Policy: greedy mean-head deployment policy",
        f"Turns observed: {result['turns_observed']}",
        f"Eligible zero-or-one-{'acquire' if _is_acquire_kind(kind) else 'scrap'} turns: {result['single_card_turns']}",
        f"Scored decisions: {result['scored_decisions']}",
        f"Pairwise comparisons: {result['pairwise_comparisons']}",
        f"Truncated games: {result['truncated_games']}",
        "The chosen card beats every unchosen card that was legal in the same decision.",
        (
            "No Card faces only the card actually acquired, or all cards affordable when a zero-acquire turn ended."
            if _is_acquire_kind(kind)
            else "No Discard is rated alongside the cards in each hand/discard scrap choice."
        ),
        "Turns containing more than one acquired or scrapped card are excluded in full.",
    ]
    if kind == AnalysisKind.ACQUIRE_BUCKETED:
        lines.extend(
            [
                "Ratings were grouped after simulation from the purchase-time or turn-end choice state.",
                "Turn 30 includes turn 30 and later; opponent color uses the most recently acquired tied leader.",
                "Opponent-color states before any colored opponent acquisition are reported as unbucketed.",
                "",
                "Charts: turn, own authority, acquired-card count, opponent authority, opponent top-acquired color.",
            ]
        )
    lines.extend(["", "Rankings"])
    for index, entry in enumerate(result["leaderboard"], 1):
        uncertainty = entry.get("uncertainty")
        uncertainty_text = "-" if uncertainty is None else f"{float(uncertainty):.2f}"
        lines.append(
            f"{index:>3}. {entry['label']:<32} Elo {float(entry['elo']):>8.2f}  "
            f"+/- {uncertainty_text:>8}  dec {int(entry['decision_count']):>5}  "
            f"cmp {int(entry['pairwise_comparisons']):>6}"
        )
    return "\n".join(lines)


def run_card_analysis(
    store: Store,
    model_id: str,
    kind: AnalysisKind | str,
    config: CardAnalysisConfig | None = None,
    *,
    progress: Callable[[int, int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    resolved_kind = AnalysisKind(kind)
    resolved_config = config or CardAnalysisConfig(games=default_games_for_kind(resolved_kind))
    resolved = resolve_model(store, model_id)
    if resolved.kind != "checkpoint" or resolved.actor_path is None:
        raise ModelResolutionError("card-choice analysis requires a checkpoint candidate")
    started = time.monotonic()
    simulated = _simulate_games(
        resolved.actor_path,
        resolved_kind,
        resolved_config,
        progress=progress,
        cancelled=cancelled,
    )
    decisions = simulated.pop("decisions")
    rated = rate_choice_decisions(decisions, resolved_kind, k_factor=resolved_config.k_factor)
    result: dict[str, Any] = {
        "kind": resolved_kind.value,
        "model": {
            "id": resolved.checkpoint_id or resolved.ref,
            "label": resolved.label,
            "kind": resolved.kind,
        },
        "games_requested": resolved_config.games,
        **simulated,
        **rated,
        "config": resolved_config.to_dict(),
        "rating_model": (
            "multinomial_elo_plackett_luce_adaptive_k"
            if _is_acquire_kind(resolved_kind)
            else "pairwise_elo_adaptive_k"
        ),
        "duration_seconds": time.monotonic() - started,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    if resolved_kind == AnalysisKind.ACQUIRE_BUCKETED:
        result["bucketed_charts"] = rate_bucketed_acquire_decisions(
            decisions, k_factor=resolved_config.k_factor
        )
    result["report_text"] = format_analysis_report(result)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        stem = f"card_{resolved_kind.value}_elo_{model_id}_{stamp}"
        report_path = output_dir / f"{stem}.txt"
        json_path = output_dir / f"{stem}.json"
        report_path.write_text(result["report_text"], encoding="utf-8")
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["report_path"] = str(report_path.resolve())
        result["json_path"] = str(json_path.resolve())
    return result


class CardAnalysisManager:
    """Run one bounded analysis at a time without tying it to an HTTP request."""

    def __init__(self, store: Store, output_dir: Path):
        self.store = store
        self.output_dir = output_dir
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="card-analysis")
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = self._load_completed_reports()
        self._futures: dict[str, Future[Any]] = {}
        self._cancel = threading.Event()

    def _load_completed_reports(self) -> dict[str, dict[str, Any]]:
        """Restore completed analyses from their durable JSON report files."""

        jobs: dict[str, dict[str, Any]] = {}
        if not self.output_dir.is_dir():
            return jobs
        for path in sorted(self.output_dir.glob("card_*_elo_*.json")):
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
                kind = AnalysisKind(result["kind"])
                model = result["model"]
                model_id = str(model["id"])
                completed_at = str(
                    result.get("completed_at")
                    or datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
                )
                config = result.get("config", {})
                if not isinstance(config, dict):
                    config = {}
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            job_id = f"saved-{path.stem}"
            jobs[job_id] = {
                "id": job_id,
                "status": "complete",
                "model_id": model_id,
                "model_label": str(model.get("label", model_id)),
                "kind": kind.value,
                "config": config,
                "created_at": completed_at,
                "updated_at": completed_at,
                "result": result,
                "error": None,
                "saved_report": True,
            }
        return jobs

    def create(
        self,
        model_id: str,
        kind: AnalysisKind | str,
        config: CardAnalysisConfig | None = None,
    ) -> dict[str, Any]:
        resolved_kind = AnalysisKind(kind)
        resolved_config = config or CardAnalysisConfig(games=default_games_for_kind(resolved_kind))
        resolved = resolve_model(self.store, model_id)
        if resolved.kind != "checkpoint":
            raise ModelResolutionError("card-choice analysis requires a checkpoint candidate")
        job_id = uuid.uuid4().hex
        now = datetime.now(UTC).isoformat()
        job = {
            "id": job_id,
            "status": "queued",
            "model_id": model_id,
            "model_label": resolved.label,
            "kind": resolved_kind.value,
            "config": resolved_config.to_dict(),
            "created_at": now,
            "updated_at": now,
            "result": {
                "games_requested": resolved_config.games,
                "games_completed": 0,
                "single_card_turns": 0,
                "decisions_captured": 0,
                "progress": 0.0,
            },
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._futures[job_id] = self._executor.submit(
                self._run, job_id, model_id, resolved_kind, resolved_config
            )
            return json.loads(json.dumps(job))

    def _run(
        self,
        job_id: str,
        model_id: str,
        kind: AnalysisKind,
        config: CardAnalysisConfig,
    ) -> None:
        with self._lock:
            self._jobs[job_id]["status"] = "running"
            self._jobs[job_id]["updated_at"] = datetime.now(UTC).isoformat()

        def progress(games: int, single_card_turns: int, decisions: int) -> None:
            with self._lock:
                result = self._jobs[job_id]["result"]
                result.update(
                    games_completed=games,
                    single_card_turns=single_card_turns,
                    decisions_captured=decisions,
                    progress=games / config.games,
                )
                self._jobs[job_id]["updated_at"] = datetime.now(UTC).isoformat()

        try:
            result = run_card_analysis(
                self.store,
                model_id,
                kind,
                config,
                progress=progress,
                cancelled=self._cancel.is_set,
                output_dir=self.output_dir,
            )
            status = "cancelled" if result["games_completed"] < config.games else "complete"
            with self._lock:
                self._jobs[job_id].update(
                    status=status,
                    result=result,
                    updated_at=datetime.now(UTC).isoformat(),
                )
        except Exception as error:  # background errors must remain inspectable by the GUI
            with self._lock:
                self._jobs[job_id].update(
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                    updated_at=datetime.now(UTC).isoformat(),
                )

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return json.loads(json.dumps(self._jobs[job_id]))

    def list(self, *, limit: int = 50, model_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda job: str(job.get("updated_at", "")),
                reverse=True,
            )
            if model_id is not None:
                jobs = [job for job in jobs if job["model_id"] == model_id]
            return json.loads(json.dumps(jobs[:limit]))

    def shutdown(self) -> None:
        self._cancel.set()
        self._executor.shutdown(wait=True, cancel_futures=True)


__all__ = [
    "ACQUIRE_EXPLORER_TARGET",
    "AcquisitionContext",
    "AnalysisKind",
    "BUCKETED_ANALYSIS_GAMES",
    "CardAnalysisConfig",
    "CardAnalysisManager",
    "ChoiceDecision",
    "ChoiceOption",
    "DEFAULT_ANALYSIS_GAMES",
    "MAX_ANALYSIS_GAMES",
    "default_games_for_kind",
    "extract_single_card_turn_decisions",
    "format_analysis_report",
    "rate_choice_decisions",
    "rate_bucketed_acquire_decisions",
    "run_card_analysis",
]
