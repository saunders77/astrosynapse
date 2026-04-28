"""Resumable self-play training for the Star Realms simulator.

This module trains a lightweight PPO-style actor-critic without external ML
dependencies. The policy learns through the existing `choose(player, options,
state) -> int` interface used by `sim.py`.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import threading
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from chooser import choose as human_choose
from sim import Game, cardDetails, factions

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

try:
    import torch_directml  # type: ignore
except ImportError as exc:
    torch_directml = None
    TORCH_DIRECTML_IMPORT_ERROR = exc
else:
    TORCH_DIRECTML_IMPORT_ERROR = None


ROOT_DIR = Path(__file__).resolve().parent
RUNS_DIR = ROOT_DIR / "starrealms_policies"
LATEST_RUN_NAME = "default"
INITIAL_ELO = 1000.0
CARD_ACQUIRE_ELO_TEST_GAMES = 200
CARD_ACQUIRE_ELO_K_FACTOR = 24.0
CARD_ZONE_SCALE = 15.0
NUMERIC_OPTION_SCALE = 10.0
EPSILON = 1e-12
LEGACY_MODEL_ARCHITECTURE = "legacy_v1"
CURRENT_MODEL_ARCHITECTURE = "deep_v2"
MODEL_TYPE_DEEP = "deep"
MODEL_TYPE_DEFAULT = "default"
DEFAULT_MODEL_HIDDEN_SIZE = 24
DEEP_MODEL_HIDDEN_SIZE = 96
DEVICE_AUTO = "auto"
DEVICE_CPU = "cpu"
DEVICE_DIRECTML = "directml"
DEVICE_CUDA = "cuda"
DEVICE_MPS = "mps"
NOOP_ACTIONS = {"endTurn", "nodiscard", "nokill", "noRowScrap", "noScrapFromHand", "nocopy"}
TRAINING_DECISIONS_PER_GAME_ALL = "ALL"
TRAINING_DECISIONS_PER_GAME_1 = "1"
TRAINING_DECISIONS_PER_GAME_5 = "5"
TRAINING_DECISIONS_PER_GAME_OPTIONS = (
    TRAINING_DECISIONS_PER_GAME_1,
    TRAINING_DECISIONS_PER_GAME_5,
    TRAINING_DECISIONS_PER_GAME_ALL,
)
DEFAULT_MIN_TRAIN_TEMPERATURE_RATIO = 0.5 / 0.9
DEFAULT_PLATEAU_MAX_TRAIN_TEMPERATURE_RATIO = 1.15 / 0.9


def _normalize_ability(raw_ability: Any) -> str:
    if not isinstance(raw_ability, str) or raw_ability == "":
        return "none"
    return _normalize_ability_cached(raw_ability)


@lru_cache(maxsize=None)
def _normalize_ability_cached(raw_ability: str) -> str:
    if raw_ability == "none":
        return "none"
    if raw_ability[0] == "-":
        return raw_ability[1:]
    return raw_ability


ABILITY_FAMILIES = sorted(
    {
        _normalize_ability(card[7])
        for card in cardDetails
    }
    | {
        _normalize_ability(card[8])
        for card in cardDetails
    }
    | {
        _normalize_ability(card[10])
        for card in cardDetails
    }
)
ABILITY_TO_INDEX = {name: idx for idx, name in enumerate(ABILITY_FAMILIES)}
FACTION_TO_INDEX = {name: idx for idx, name in enumerate(factions)}
CARD_TYPE_TO_INDEX = {"ship": 0, "base": 1, "outp": 2}
ACTION_TYPES = [
    "play",
    "abilityOption",
    "scrapFromPlay",
    "attack",
    "attackOpponent",
    "acquire",
    "endTurn",
    "discardNormal",
    "discardDraw",
    "nodiscard",
    "gainattack",
    "draw",
    "trade",
    "authority",
    "nocopy",
    "copyship",
    "freeAcquire",
    "killbase",
    "nokill",
    "rowscrap",
    "noRowScrap",
    "scrapFromHandNormal",
    "scrapFromDiscardNormal",
    "scrapFromHandDraw",
    "scrapFromDiscardDraw",
    "noScrapFromHand",
    "other",
]
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTION_TYPES)}
CARD_NAME_ORDER = [str(card[0]) for card in cardDetails]
CARD_BY_NAME = {str(card[0]): card for card in cardDetails}
CARD_COST_BY_NAME = {str(card[0]): int(card[1]) for card in cardDetails}

CARD_FEATURE_SIZE = 8 + len(factions) + len(CARD_TYPE_TO_INDEX) + len(ABILITY_FAMILIES) + 2
ZONE_FEATURE_SIZE = CARD_FEATURE_SIZE + 4
STATE_SCALAR_SIZE = 18
STATE_VECTOR_SIZE = ZONE_FEATURE_SIZE * 11 + STATE_SCALAR_SIZE
OPTION_VECTOR_SIZE = len(ACTION_TYPES) + len(factions) + len(ABILITY_FAMILIES) + 7 + CARD_FEATURE_SIZE


def _zero_vector(size: int) -> List[float]:
    return [0.0 for _ in range(size)]


def _zero_matrix(rows: int, cols: int) -> List[List[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def _copy_nested(value: Any) -> Any:
    return copy.deepcopy(value)


def _stable_softmax(logits: Sequence[float], temperature: float = 1.0) -> List[float]:
    if not logits:
        return []
    temp = max(temperature, 1e-3)
    scaled = [logit / temp for logit in logits]
    max_logit = max(scaled)
    exps = [math.exp(logit - max_logit) for logit in scaled]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(logits) for _ in logits]
    return [value / total for value in exps]


def _sample_index(probs: Sequence[float]) -> int:
    draw = random.random()
    cumulative = 0.0
    for idx, prob in enumerate(probs):
        cumulative += prob
        if draw <= cumulative:
            return idx
    return max(0, len(probs) - 1)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    average = _mean(values)
    variance = sum((value - average) * (value - average) for value in values) / len(values)
    return math.sqrt(max(variance, 0.0))


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _lerp(start: float, end: float, progress: float) -> float:
    return start + (end - start) * _clip(progress, 0.0, 1.0)


def _weighted_choice(choices: Sequence[Tuple[Any, float]]) -> Any:
    filtered = [(item, weight) for item, weight in choices if weight > 0]
    if not filtered:
        raise ValueError("At least one positive-weight choice is required.")
    total = sum(weight for _, weight in filtered)
    draw = random.random() * total
    cumulative = 0.0
    for item, weight in filtered:
        cumulative += weight
        if draw <= cumulative:
            return item
    return filtered[-1][0]


def _timestamp() -> float:
    return time.time()


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _safe_slug(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "item"
    cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in raw)
    return cleaned.strip("_") or "item"


def _directml_available() -> bool:
    return torch is not None and torch_directml is not None


def _select_torch_device(preference: str = DEVICE_AUTO) -> Tuple[Any, str]:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for starrealms_selfplay.py. "
            "Run this module from the project's training venv where torch is installed."
        ) from TORCH_IMPORT_ERROR

    normalized = str(preference or DEVICE_AUTO).strip().lower()
    if normalized == DEVICE_DIRECTML:
        if torch_directml is None:
            raise RuntimeError(
                "DirectML was requested, but torch_directml is not installed. "
                "Use Python 3.12 in your training venv and install torch-directml."
            ) from TORCH_DIRECTML_IMPORT_ERROR
        return torch_directml.device(), DEVICE_DIRECTML

    if normalized == DEVICE_CPU:
        return torch.device("cpu"), DEVICE_CPU

    if normalized == DEVICE_CUDA and hasattr(torch, "cuda") and torch.cuda.is_available():
        return torch.device("cuda"), DEVICE_CUDA

    if normalized == DEVICE_MPS and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), DEVICE_MPS

    if normalized == DEVICE_AUTO:
        if torch_directml is not None:
            return torch_directml.device(), DEVICE_DIRECTML
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return torch.device("cuda"), DEVICE_CUDA
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), DEVICE_MPS
        return torch.device("cpu"), DEVICE_CPU

    return torch.device("cpu"), DEVICE_CPU


def runtime_environment() -> Dict[str, Any]:
    directml_error = None
    if TORCH_DIRECTML_IMPORT_ERROR is not None:
        directml_error = f"{type(TORCH_DIRECTML_IMPORT_ERROR).__name__}: {TORCH_DIRECTML_IMPORT_ERROR}"
    training_device, training_backend = _select_torch_device(DEVICE_AUTO) if torch is not None else (None, "unavailable")
    return {
        "python_version": ".".join(str(part) for part in os.sys.version_info[:3]),
        "interpreter": os.sys.executable,
        "torch_available": torch is not None,
        "torch_version": None if torch is None else getattr(torch, "__version__", None),
        "torch_directml_available": _directml_available(),
        "torch_directml_error": directml_error,
        "auto_device": training_backend,
        "auto_device_repr": None if training_device is None else str(training_device),
    }


def resolve_device_backend(preference: str = DEVICE_AUTO) -> str:
    _, backend = _select_torch_device(preference)
    return backend


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2)
    last_error: Optional[Exception] = None
    for attempt in range(8):
        temp_path = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.{attempt}.tmp")
        try:
            temp_path.write_text(content, encoding="utf-8")
            os.replace(temp_path, path)
            _invalidate_json_cache(path)
            return
        except PermissionError as exc:
            last_error = exc
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            time.sleep(0.05 * (attempt + 1))
        except Exception:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            raise
    if last_error is not None:
        raise PermissionError(
            f"Permission error while saving {path}. "
            "Another process may still have the file open; please wait a moment and try again."
        ) from last_error


def _is_card_tuple(value: Any) -> bool:
    return isinstance(value, tuple) and len(value) >= 14 and isinstance(value[0], str)


def _is_play_card(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 5 and _is_card_tuple(value[0])


def _extract_card(value: Any) -> Optional[Tuple[Any, ...]]:
    if _is_card_tuple(value):
        if value[0] == "none":
            return None
        return value
    if _is_play_card(value):
        return value[0]
    return None


def _iter_zone_cards(zone: Any) -> List[Any]:
    if zone is None:
        return []
    if isinstance(zone, dict):
        cards: List[Any] = []
        for items in zone.values():
            cards.extend(items)
        return cards
    if isinstance(zone, list):
        return zone
    return []


@lru_cache(maxsize=None)
def _cached_card_feature_updates(card: Tuple[Any, ...]) -> Tuple[Tuple[int, float], ...]:
    updates: List[Tuple[int, float]] = [
        (0, 1.0),
        (1, card[1] / 8.0),
        (2, card[2] / 8.0),
        (3, card[3] / 8.0),
        (4, card[4] / 8.0),
        (5, card[12] / 8.0),
        (6, card[9] / 5.0),
        (7, card[11] / 5.0),
    ]

    faction_offset = 8
    faction_index = FACTION_TO_INDEX.get(card[5])
    if faction_index is not None:
        updates.append((faction_offset + faction_index, 1.0))

    type_offset = faction_offset + len(factions)
    card_type_index = CARD_TYPE_TO_INDEX.get(card[6])
    if card_type_index is not None:
        updates.append((type_offset + card_type_index, 1.0))

    ability_offset = type_offset + len(CARD_TYPE_TO_INDEX)
    for raw_ability in (card[7], card[8], card[10]):
        ability_index = ABILITY_TO_INDEX.get(_normalize_ability(raw_ability))
        if ability_index is not None:
            updates.append((ability_offset + ability_index, 1.0))

    option_offset = ability_offset + len(ABILITY_FAMILIES)
    if isinstance(card[7], str) and card[7].startswith("-"):
        updates.append((option_offset, 1.0))
    if isinstance(card[8], str) and card[8].startswith("-"):
        updates.append((option_offset + 1, 1.0))
    return tuple(updates)


@lru_cache(maxsize=None)
def _cached_card_feature_vector(card: Tuple[Any, ...]) -> Tuple[float, ...]:
    features = _zero_vector(CARD_FEATURE_SIZE)
    for index, value in _cached_card_feature_updates(card):
        features[index] += value
    return tuple(features)


def _add_single_card_features(features: List[float], card: Tuple[Any, ...], scale: float) -> None:
    if card is None:
        return
    for index, value in _cached_card_feature_updates(card):
        features[index] += value * scale


def _single_card_features(card: Optional[Tuple[Any, ...]]) -> List[float]:
    if card is None:
        return _zero_vector(CARD_FEATURE_SIZE)
    return list(_cached_card_feature_vector(card))


def _zone_features_and_count(zone: Any) -> Tuple[List[float], int]:
    features = _zero_vector(ZONE_FEATURE_SIZE)
    count = 0
    scale = 1.0 / CARD_ZONE_SCALE

    if zone is None:
        return features, count

    if isinstance(zone, dict):
        zone_groups = zone.values()
    elif isinstance(zone, list):
        zone_groups = (zone,)
    else:
        return features, count

    for group in zone_groups:
        for item in group:
            count += 1
            if isinstance(item, tuple) and len(item) >= 14 and isinstance(item[0], str):
                if item[0] != "none":
                    _add_single_card_features(features, item, scale)
                continue
            if isinstance(item, list) and len(item) >= 5:
                card = item[0]
                if isinstance(card, tuple) and len(card) >= 14 and isinstance(card[0], str):
                    if card[0] != "none":
                        _add_single_card_features(features, card, scale)
                    option_state = item[2]
                    if option_state not in (None, "used"):
                        features[CARD_FEATURE_SIZE] += scale
                    if option_state == "used":
                        features[CARD_FEATURE_SIZE + 1] += scale
                    if item[1]:
                        features[CARD_FEATURE_SIZE + 2] += scale
                    if item[4]:
                        features[CARD_FEATURE_SIZE + 3] += scale
    return features, count


def _zone_features(zone: Any) -> List[float]:
    return _zone_features_and_count(zone)[0]


def state_to_vector(state: Dict[str, Any], legal_option_count: int = 0) -> List[float]:
    zone_values = {
        "hand": state.get("hand"),
        "discardPile": state.get("discardPile"),
        "scrambleDeck": state.get("scrambleDeck"),
        "topCards": state.get("topCards"),
        "cardsInPlay": state.get("cardsInPlay"),
        "tradeRow": state.get("tradeRow"),
        "opponentDiscardPile": state.get("opponentDiscardPile"),
        "opponentScrambleDeckAndHand": state.get("opponentScrambleDeckAndHand"),
        "opponentTopCards": state.get("opponentTopCards"),
        "opponentHandCards": state.get("opponentHandCards"),
        "opponentCardsInPlay": state.get("opponentCardsInPlay"),
    }
    zone_order = [
        "hand",
        "discardPile",
        "scrambleDeck",
        "topCards",
        "cardsInPlay",
        "tradeRow",
        "opponentDiscardPile",
        "opponentScrambleDeckAndHand",
        "opponentTopCards",
        "opponentHandCards",
        "opponentCardsInPlay",
    ]
    zone_stats = {name: _zone_features_and_count(zone_values[name]) for name in zone_order}
    features: List[float] = []
    for zone_name in zone_order:
        features.extend(zone_stats[zone_name][0])

    hand_count = zone_stats["hand"][1]
    discard_count = zone_stats["discardPile"][1]
    deck_count = zone_stats["scrambleDeck"][1] + zone_stats["topCards"][1]
    in_play_count = zone_stats["cardsInPlay"][1]
    opponent_known_hand_count = zone_stats["opponentHandCards"][1]
    opponent_unknown_count = zone_stats["opponentScrambleDeckAndHand"][1] + zone_stats["opponentTopCards"][1]
    opponent_discard_count = zone_stats["opponentDiscardPile"][1]
    opponent_in_play_count = zone_stats["opponentCardsInPlay"][1]

    scalars = [
        state.get("authority", 0) / 50.0,
        state.get("opponentAuthority", 0) / 50.0,
        (state.get("authority", 0) - state.get("opponentAuthority", 0)) / 50.0,
        state.get("attack", 0) / 20.0,
        state.get("trade", 0) / 20.0,
        state.get("mustDiscard", 0) / 5.0,
        state.get("opponentMustDiscard", 0) / 5.0,
        1.0 if state.get("nextShipTop") else 0.0,
        state.get("blobPlayCount", 0) / 10.0,
        hand_count / 20.0,
        discard_count / 30.0,
        deck_count / 30.0,
        in_play_count / 15.0,
        opponent_known_hand_count / 10.0,
        opponent_unknown_count / 30.0,
    ]
    scalars.extend(
        [
            opponent_discard_count / 30.0,
            opponent_in_play_count / 15.0,
            legal_option_count / 30.0,
        ]
    )
    return features + scalars


def _option_action_name(option: Sequence[Any]) -> str:
    if not option:
        return "other"
    raw_name = option[0]
    if not isinstance(raw_name, str):
        return "other"
    if raw_name in ACTION_TO_INDEX:
        return raw_name
    return "other"


def _resolve_state_referenced_card(option: Sequence[Any], state: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    action = _option_action_name(option)
    if action in {"abilityOption", "scrapFromPlay"} and len(option) >= 3:
        faction = option[1]
        index = option[2]
        if isinstance(faction, str) and isinstance(index, int):
            cards_in_play = state.get("cardsInPlay", {})
            if faction in cards_in_play and 0 <= index < len(cards_in_play[faction]):
                return _extract_card(cards_in_play[faction][index])
    if action in {"attack", "killbase"} and len(option) >= 3:
        faction = option[1]
        index = option[2]
        if isinstance(faction, str) and isinstance(index, int):
            cards_in_play = state.get("opponentCardsInPlay", {})
            if faction in cards_in_play and 0 <= index < len(cards_in_play[faction]):
                return _extract_card(cards_in_play[faction][index])
    return None


def _resolve_option_card(option: Sequence[Any], state: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    for item in option[1:]:
        card = _extract_card(item)
        if card is not None:
            return card
    return _resolve_state_referenced_card(option, state)


def _resolve_option_ability(option: Sequence[Any]) -> str:
    action = _option_action_name(option)
    if action in {"gainattack", "draw", "trade", "authority"}:
        return action
    for item in option[1:]:
        if isinstance(item, str):
            normalized = _normalize_ability(item)
            if normalized in ABILITY_TO_INDEX and normalized != "none":
                return normalized
    return "none"


def option_to_vector(option: Sequence[Any], state: Dict[str, Any]) -> List[float]:
    features = _zero_vector(OPTION_VECTOR_SIZE)
    offset = 0

    action = _option_action_name(option)
    features[offset + ACTION_TO_INDEX[action]] = 1.0
    offset += len(ACTION_TYPES)

    for item in option[1:]:
        if isinstance(item, str) and item in FACTION_TO_INDEX:
            features[offset + FACTION_TO_INDEX[item]] = 1.0
            break
    offset += len(factions)

    ability_name = _resolve_option_ability(option)
    if ability_name in ABILITY_TO_INDEX:
        features[offset + ABILITY_TO_INDEX[ability_name]] = 1.0
    offset += len(ABILITY_FAMILIES)

    numeric_values: List[float] = []
    for item in option[1:]:
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            numeric_values.append(_clip(float(item) / NUMERIC_OPTION_SCALE, -3.0, 3.0))
    features[offset] = len(option) / 8.0
    features[offset + 1] = 1.0 if action in NOOP_ACTIONS else 0.0
    for idx in range(5):
        if idx < len(numeric_values):
            features[offset + 2 + idx] = numeric_values[idx]
    offset += 7

    resolved_card = _resolve_option_card(option, state)
    if resolved_card is not None:
        features[offset:] = _cached_card_feature_vector(resolved_card)
    return features


@dataclass
class TrainingConfig:
    hidden_size: int = DEEP_MODEL_HIDDEN_SIZE
    model_architecture: str = CURRENT_MODEL_ARCHITECTURE
    device_preference: str = DEVICE_AUTO
    learning_rate: float = 0.0019
    min_learning_rate: float = 0.0004
    ppo_epochs: int = 2
    ppo_minibatch_size: int = 64
    ppo_clip: float = 0.2
    min_ppo_clip: float = 0.1
    value_coef: float = 0.5
    grad_clip: float = 1.0
    epsilon_random: float = 0.07
    min_epsilon_random: float = 0.015
    train_temperature: float = 0.9
    min_train_temperature: float = 0.5
    eval_temperature: float = 0.35
    checkpoint_interval: int = 1
    training_matches_per_iteration: int = 5
    training_games_per_match: int = 16
    training_decisions_per_game: str = TRAINING_DECISIONS_PER_GAME_ALL
    promotion_games: int = 24
    promotion_score_threshold: float = 0.6
    candidate_patience: int = 8
    candidate_reset_threshold: float = 0.35
    league_recent_window: int = 10
    league_champion_weight: float = 0.45
    league_best_weight: float = 0.2
    league_recent_weight: float = 0.25
    league_historical_weight: float = 0.1
    anneal_steps: int = 3000
    plateau_start_iterations_without_promotion: int = 12
    plateau_full_iterations_without_promotion: int = 36
    plateau_learning_rate_boost: float = 1.7
    plateau_epsilon_boost: float = 2.4
    plateau_temperature_boost: float = 1.22
    plateau_max_learning_rate: float = 0.0033
    plateau_max_epsilon_random: float = 0.16
    plateau_max_train_temperature: float = 1.15
    elo_k: float = 24.0
    max_training_samples_per_match: int = 256
    max_turns_per_game: int = 400
    max_actions_per_turn: int = 200
    max_iterations: Optional[int] = None

    def merged(self, overrides: Optional[Dict[str, Any]] = None) -> "TrainingConfig":
        if not overrides:
            return self
        data = asdict(self)
        data.update(overrides)
        return TrainingConfig(**data)


if nn is not None and torch is not None:
    class _LegacyTorchPolicyModule(nn.Module):
        def __init__(self, state_size: int, option_size: int, hidden_size: int) -> None:
            super().__init__()
            self.state_linear = nn.Linear(state_size, hidden_size)
            self.option_linear = nn.Linear(option_size, hidden_size)
            self.joint_bias = nn.Parameter(torch.zeros(hidden_size))
            self.policy_head = nn.Linear(hidden_size, 1)
            self.value_head = nn.Linear(hidden_size, 1)

        def _state_hidden(self, state_tensor: "torch.Tensor") -> "torch.Tensor":
            return torch.tanh(self.state_linear(state_tensor))

        def _joint_hidden(self, state_hidden: "torch.Tensor", option_tensor: "torch.Tensor") -> "torch.Tensor":
            option_hidden = torch.tanh(self.option_linear(option_tensor))
            expanded_state_hidden = state_hidden
            while expanded_state_hidden.dim() < option_hidden.dim():
                expanded_state_hidden = expanded_state_hidden.unsqueeze(-2)
            bias_shape = [1] * (option_hidden.dim() - 1) + [-1]
            return torch.tanh(expanded_state_hidden + option_hidden + self.joint_bias.view(*bias_shape))

        def policy_only(self, state_tensor: "torch.Tensor", option_tensor: "torch.Tensor") -> "torch.Tensor":
            state_hidden = self._state_hidden(state_tensor)
            joint_hidden = self._joint_hidden(state_hidden, option_tensor)
            return self.policy_head(joint_hidden).squeeze(-1)

        def forward(self, state_tensor: "torch.Tensor", option_tensor: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            state_hidden = self._state_hidden(state_tensor)
            joint_hidden = self._joint_hidden(state_hidden, option_tensor)
            logits = self.policy_head(joint_hidden).squeeze(-1)
            value = self.value_head(state_hidden).squeeze(-1)
            return logits, value


    class _DeepTorchPolicyModule(nn.Module):
        def __init__(self, state_size: int, option_size: int, hidden_size: int) -> None:
            super().__init__()
            self.state_linear = nn.Linear(state_size, hidden_size)
            self.option_linear = nn.Linear(option_size, hidden_size)
            self.joint_bias = nn.Parameter(torch.zeros(hidden_size))

            self.state_refine_1 = nn.Linear(hidden_size, hidden_size)
            self.state_refine_2 = nn.Linear(hidden_size, hidden_size)
            self.option_refine_1 = nn.Linear(hidden_size, hidden_size)
            self.option_refine_2 = nn.Linear(hidden_size, hidden_size)
            self.joint_mix_1 = nn.Linear(hidden_size * 4, hidden_size)
            self.joint_mix_2 = nn.Linear(hidden_size, hidden_size)
            self.value_refine = nn.Linear(hidden_size, hidden_size)

            self.policy_head = nn.Linear(hidden_size, 1)
            self.value_head = nn.Linear(hidden_size, 1)

        def _state_hidden(self, state_tensor: "torch.Tensor") -> "torch.Tensor":
            state_hidden = torch.tanh(self.state_linear(state_tensor))
            state_hidden = state_hidden + 0.35 * torch.tanh(self.state_refine_1(state_hidden))
            state_hidden = state_hidden + 0.35 * torch.tanh(self.state_refine_2(state_hidden))
            return state_hidden

        def _joint_hidden(self, state_hidden: "torch.Tensor", option_tensor: "torch.Tensor") -> "torch.Tensor":
            option_hidden = torch.tanh(self.option_linear(option_tensor))
            option_hidden = option_hidden + 0.35 * torch.tanh(self.option_refine_1(option_hidden))
            option_hidden = option_hidden + 0.35 * torch.tanh(self.option_refine_2(option_hidden))

            expanded_state_hidden = state_hidden
            while expanded_state_hidden.dim() < option_hidden.dim():
                expanded_state_hidden = expanded_state_hidden.unsqueeze(-2)
            state_expanded = expanded_state_hidden.expand_as(option_hidden)
            bias_shape = [1] * (option_hidden.dim() - 1) + [-1]
            legacy_joint = torch.tanh(state_expanded + option_hidden + self.joint_bias.view(*bias_shape))
            interaction = state_expanded * option_hidden
            joint_input = torch.cat([legacy_joint, state_expanded, option_hidden, interaction], dim=-1)
            joint_hidden = legacy_joint + 0.35 * torch.tanh(self.joint_mix_1(joint_input))
            joint_hidden = joint_hidden + 0.35 * torch.tanh(self.joint_mix_2(joint_hidden))
            return joint_hidden

        def policy_only(self, state_tensor: "torch.Tensor", option_tensor: "torch.Tensor") -> "torch.Tensor":
            state_hidden = self._state_hidden(state_tensor)
            joint_hidden = self._joint_hidden(state_hidden, option_tensor)
            return self.policy_head(joint_hidden).squeeze(-1)

        def forward(self, state_tensor: "torch.Tensor", option_tensor: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            state_hidden = self._state_hidden(state_tensor)
            joint_hidden = self._joint_hidden(state_hidden, option_tensor)
            value_hidden = state_hidden + 0.35 * torch.tanh(self.value_refine(state_hidden))
            logits = self.policy_head(joint_hidden).squeeze(-1)
            value = self.value_head(value_hidden).squeeze(-1)
            return logits, value
else:
    class _LegacyTorchPolicyModule:  # pragma: no cover - only used when torch is unavailable
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for starrealms_selfplay.py. "
                "Run this module from the project's training venv where torch is installed."
            ) from TORCH_IMPORT_ERROR


    class _DeepTorchPolicyModule:  # pragma: no cover - only used when torch is unavailable
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for starrealms_selfplay.py. "
                "Run this module from the project's training venv where torch is installed."
            ) from TORCH_IMPORT_ERROR


class PolicyNetwork:
    def __init__(
        self,
        state_size: int = STATE_VECTOR_SIZE,
        option_size: int = OPTION_VECTOR_SIZE,
        hidden_size: int = 96,
        architecture: str = CURRENT_MODEL_ARCHITECTURE,
        device_preference: str = DEVICE_AUTO,
        init_seed: Optional[int] = None,
    ) -> None:
        if torch is None or nn is None:
            raise RuntimeError(
                "PyTorch is required for starrealms_selfplay.py. "
                "Run this module from the project's training venv where torch is installed."
            ) from TORCH_IMPORT_ERROR
        if init_seed is not None:
            torch.manual_seed(init_seed)
        self.state_size = state_size
        self.option_size = option_size
        self.hidden_size = hidden_size
        self.architecture = architecture
        self.device_preference = device_preference
        self.device, self.device_backend = _select_torch_device(device_preference)
        if architecture == LEGACY_MODEL_ARCHITECTURE:
            self.model = _LegacyTorchPolicyModule(state_size, option_size, hidden_size).to(self.device)
        elif architecture == CURRENT_MODEL_ARCHITECTURE:
            self.model = _DeepTorchPolicyModule(state_size, option_size, hidden_size).to(self.device)
        else:
            raise ValueError(f"Unknown policy architecture: {architecture}")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0015)

    def clone(self) -> "PolicyNetwork":
        clone = PolicyNetwork(
            state_size=self.state_size,
            option_size=self.option_size,
            hidden_size=self.hidden_size,
            architecture=self.architecture,
            device_preference=self.device_preference,
        )
        clone.model.load_state_dict(self.model.state_dict())
        clone.optimizer = torch.optim.Adam(clone.model.parameters(), lr=self.optimizer.param_groups[0]["lr"])
        return clone

    def to_dict(self, include_optimizer: bool = True) -> Dict[str, Any]:
        state_dict = {
            name: tensor.detach().cpu().tolist()
            for name, tensor in self.model.state_dict().items()
        }
        payload = {
            "backend": "torch",
            "state_size": self.state_size,
            "option_size": self.option_size,
            "hidden_size": self.hidden_size,
            "architecture": self.architecture,
            "device_preference": self.device_preference,
            "state_dict": state_dict,
        }
        if include_optimizer:
            payload["optimizer"] = {
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PolicyNetwork":
        state_size = payload.get("state_size", STATE_VECTOR_SIZE)
        option_size = payload.get("option_size", OPTION_VECTOR_SIZE)
        architecture = payload.get("architecture")
        if architecture is None:
            state_dict = payload.get("state_dict") or {}
            if any(name.startswith("state_refine_") or name.startswith("joint_mix_") or name.startswith("value_refine") for name in state_dict.keys()):
                architecture = CURRENT_MODEL_ARCHITECTURE
            else:
                architecture = LEGACY_MODEL_ARCHITECTURE
        hidden_size = payload.get("hidden_size", 24 if architecture == LEGACY_MODEL_ARCHITECTURE else 96)
        device_preference = payload.get("device_preference", DEVICE_AUTO)
        model = cls(
            state_size=state_size,
            option_size=option_size,
            hidden_size=hidden_size,
            architecture=architecture,
            device_preference=device_preference,
        )

        if payload.get("backend") == "torch" or "state_dict" in payload:
            tensor_state = {
                name: torch.tensor(values, dtype=torch.float32, device=model.device)
                for name, values in payload["state_dict"].items()
            }
            model.model.load_state_dict(tensor_state)
            optimizer_state = payload.get("optimizer") or {}
            learning_rate = float(optimizer_state.get("learning_rate", 0.0015))
            model.optimizer = torch.optim.Adam(model.model.parameters(), lr=learning_rate)
            return model

        params = payload["params"]
        model = cls(
            state_size=state_size,
            option_size=option_size,
            hidden_size=hidden_size,
            architecture=LEGACY_MODEL_ARCHITECTURE,
            device_preference=device_preference,
        )
        converted_state = {
            "state_linear.weight": torch.tensor(params["state_w"], dtype=torch.float32, device=model.device),
            "state_linear.bias": torch.tensor(params["state_b"], dtype=torch.float32, device=model.device),
            "option_linear.weight": torch.tensor(params["option_w"], dtype=torch.float32, device=model.device),
            "option_linear.bias": torch.tensor(params["option_b"], dtype=torch.float32, device=model.device),
            "joint_bias": torch.tensor(params["joint_b"], dtype=torch.float32, device=model.device),
            "policy_head.weight": torch.tensor([params["policy_w"]], dtype=torch.float32, device=model.device),
            "policy_head.bias": torch.tensor([params["policy_b"]], dtype=torch.float32, device=model.device),
            "value_head.weight": torch.tensor([params["value_w"]], dtype=torch.float32, device=model.device),
            "value_head.bias": torch.tensor([params["value_b"]], dtype=torch.float32, device=model.device),
        }
        model.model.load_state_dict(converted_state)
        return model

    def _state_tensor(self, state_vec: Sequence[float]) -> "torch.Tensor":
        return torch.tensor(state_vec, dtype=torch.float32, device=self.device)

    def _state_tensor_batch(self, state_vecs: Sequence[Sequence[float]]) -> "torch.Tensor":
        return torch.tensor(state_vecs, dtype=torch.float32, device=self.device)

    def _option_tensor(self, option_vecs: Sequence[Sequence[float]]) -> "torch.Tensor":
        return torch.tensor(option_vecs, dtype=torch.float32, device=self.device)

    def _padded_option_tensor(
        self,
        option_vec_batches: Sequence[Sequence[Sequence[float]]],
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        if not option_vec_batches:
            raise ValueError("At least one option batch is required.")

        batch_size = len(option_vec_batches)
        max_options = max(len(option_vecs) for option_vecs in option_vec_batches)
        if max_options <= 0:
            raise ValueError("Each state requires at least one legal option.")

        option_tensor = torch.zeros(
            (batch_size, max_options, self.option_size),
            dtype=torch.float32,
            device=self.device,
        )
        option_mask = torch.zeros((batch_size, max_options), dtype=torch.bool, device=self.device)

        for batch_index, option_vecs in enumerate(option_vec_batches):
            option_count = len(option_vecs)
            if option_count <= 0:
                raise ValueError("Each state requires at least one legal option.")
            option_tensor[batch_index, :option_count] = torch.tensor(
                option_vecs,
                dtype=torch.float32,
                device=self.device,
            )
            option_mask[batch_index, :option_count] = True

        return option_tensor, option_mask

    def _masked_logits(
        self,
        logits: "torch.Tensor",
        option_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        if option_mask is None:
            return logits
        return logits.masked_fill(~option_mask, -1e9)

    def _policy_logits(self, state_tensor: "torch.Tensor", option_tensor: "torch.Tensor") -> "torch.Tensor":
        return self.model.policy_only(state_tensor, option_tensor)

    def select_action(
        self,
        state_vec: Sequence[float],
        option_vecs: Sequence[Sequence[float]],
        deterministic: bool = False,
        temperature: float = 1.0,
        epsilon_random: float = 0.0,
        need_log_prob: bool = False,
        need_value: bool = False,
    ) -> Dict[str, Any]:
        if self.model.training:
            self.model.eval()
        with torch.no_grad():
            state_tensor = self._state_tensor(state_vec)
            option_tensor = self._option_tensor(option_vecs)
            result: Dict[str, Any] = {}

            if deterministic and not need_log_prob and not need_value:
                logits = self._policy_logits(state_tensor, option_tensor)
                result["action_index"] = int(torch.argmax(logits).item())
                return result

            value_tensor = None
            if need_value:
                logits, value_tensor = self.model(state_tensor, option_tensor)
            else:
                logits = self._policy_logits(state_tensor, option_tensor)

            scaled_logits = logits / max(temperature, 1e-3)
            probs = torch.softmax(scaled_logits, dim=0)
            if deterministic:
                action_index = int(torch.argmax(scaled_logits).item())
            elif random.random() < epsilon_random:
                action_index = random.randrange(len(option_vecs))
            else:
                action_index = int(torch.multinomial(probs, 1).item())

            result["action_index"] = action_index
            if need_log_prob:
                result["log_prob"] = float(torch.log(probs[action_index].clamp_min(EPSILON)).item())
            if need_value and value_tensor is not None:
                result["value"] = float(value_tensor.item())
            return result

    def select_actions_batch(
        self,
        state_vecs: Sequence[Sequence[float]],
        option_vec_batches: Sequence[Sequence[Sequence[float]]],
        deterministic: bool = False,
        temperature: float = 1.0,
        epsilon_random: float = 0.0,
        need_log_prob: bool = False,
        need_value: bool = False,
    ) -> List[Dict[str, Any]]:
        if not state_vecs:
            return []
        if len(state_vecs) != len(option_vec_batches):
            raise ValueError("state_vecs and option_vec_batches must have the same length.")

        if self.model.training:
            self.model.eval()

        with torch.no_grad():
            state_tensor = self._state_tensor_batch(state_vecs)
            option_tensor, option_mask = self._padded_option_tensor(option_vec_batches)

            if deterministic and not need_log_prob and not need_value:
                logits = self._policy_logits(state_tensor, option_tensor)
                masked_logits = self._masked_logits(logits, option_mask)
                action_indices = torch.argmax(masked_logits, dim=1).detach().cpu().tolist()
                return [{"action_index": int(action_index)} for action_index in action_indices]

            value_tensor = None
            if need_value:
                logits, value_tensor = self.model(state_tensor, option_tensor)
            else:
                logits = self._policy_logits(state_tensor, option_tensor)

            masked_logits = self._masked_logits(logits, option_mask)
            scaled_logits = masked_logits / max(temperature, 1e-3)

            if deterministic:
                action_tensor = torch.argmax(scaled_logits, dim=1)
            else:
                probs = torch.softmax(scaled_logits, dim=1)
                action_tensor = torch.multinomial(probs, 1).squeeze(1)
                if epsilon_random > 0.0:
                    action_indices = action_tensor.detach().cpu().tolist()
                    for index, option_vecs in enumerate(option_vec_batches):
                        if random.random() < epsilon_random:
                            action_indices[index] = random.randrange(len(option_vecs))
                    action_tensor = torch.tensor(action_indices, dtype=torch.int64, device=self.device)

            action_indices = [int(value) for value in action_tensor.detach().cpu().tolist()]
            results: List[Dict[str, Any]] = [{"action_index": action_index} for action_index in action_indices]

            if need_log_prob:
                log_probs = torch.log_softmax(scaled_logits, dim=1)
                chosen_log_probs = log_probs.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
                for result, log_prob in zip(results, chosen_log_probs.detach().cpu().tolist()):
                    result["log_prob"] = float(log_prob)

            if need_value and value_tensor is not None:
                for result, value in zip(results, value_tensor.detach().cpu().tolist()):
                    result["value"] = float(value)

            return results

    def _sample_batch_tensors(
        self,
        samples: Sequence[Dict[str, Any]],
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        state_tensor = self._state_tensor_batch([sample["state_vec"] for sample in samples])
        option_tensor, option_mask = self._padded_option_tensor([sample["option_vecs"] for sample in samples])
        action_tensor = torch.tensor(
            [int(sample["action_index"]) for sample in samples],
            dtype=torch.int64,
            device=self.device,
        )
        old_log_prob_tensor = torch.tensor(
            [float(sample["old_log_prob"]) for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        return_tensor = torch.tensor(
            [float(sample["return"]) for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        advantage_tensor = torch.tensor(
            [float(sample["advantage"]) for sample in samples],
            dtype=torch.float32,
            device=self.device,
        )
        return (
            state_tensor,
            option_tensor,
            option_mask,
            action_tensor,
            old_log_prob_tensor,
            return_tensor,
            advantage_tensor,
        )

    def train_on_samples(self, samples: List[Dict[str, Any]], config: TrainingConfig) -> Dict[str, float]:
        if not samples:
            return {
                "samples": 0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "clip_fraction": 0.0,
                "avg_ratio": 1.0,
                "avg_value_prediction": 0.0,
            }

        self.model.train()
        self.optimizer.param_groups[0]["lr"] = config.learning_rate

        policy_losses: List[float] = []
        value_losses: List[float] = []
        ratios: List[float] = []
        clipped: List[float] = []
        value_predictions: List[float] = []
        minibatch_size = max(1, min(int(config.ppo_minibatch_size), len(samples)))

        for _ in range(config.ppo_epochs):
            random.shuffle(samples)
            for batch_start in range(0, len(samples), minibatch_size):
                sample_batch = samples[batch_start:batch_start + minibatch_size]
                (
                    state_tensor,
                    option_tensor,
                    option_mask,
                    action_tensor,
                    old_log_prob_tensor,
                    return_tensor,
                    advantage_tensor,
                ) = self._sample_batch_tensors(sample_batch)

                logits, value = self.model(state_tensor, option_tensor)
                masked_logits = self._masked_logits(logits, option_mask)
                log_probs = torch.log_softmax(masked_logits, dim=1)
                chosen_log_probs = log_probs.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
                ratio = torch.exp(chosen_log_probs - old_log_prob_tensor)
                clipped_ratio = torch.clamp(ratio, 1.0 - config.ppo_clip, 1.0 + config.ppo_clip)
                surrogate_one = ratio * advantage_tensor
                surrogate_two = clipped_ratio * advantage_tensor
                policy_loss = -torch.min(surrogate_one, surrogate_two).mean()
                value_loss = 0.5 * torch.square(value - return_tensor).mean()
                loss = policy_loss + config.value_coef * value_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip)
                self.optimizer.step()

                ratio_values = ratio.detach().cpu().tolist()
                clipped_flags = (
                    (ratio < (1.0 - config.ppo_clip)) | (ratio > (1.0 + config.ppo_clip))
                ).detach().cpu().tolist()
                value_batch = value.detach().cpu().tolist()

                policy_loss_value = float(policy_loss.item())
                value_loss_value = float(value_loss.item())
                policy_losses.extend([policy_loss_value] * len(sample_batch))
                value_losses.extend([value_loss_value] * len(sample_batch))
                ratios.extend(float(ratio_value) for ratio_value in ratio_values)
                clipped.extend(1.0 if was_clipped else 0.0 for was_clipped in clipped_flags)
                value_predictions.extend(float(prediction) for prediction in value_batch)

        return {
            "samples": float(len(samples)),
            "policy_loss": _mean(policy_losses),
            "value_loss": _mean(value_losses),
            "clip_fraction": _mean(clipped),
            "avg_ratio": _mean(ratios),
            "avg_value_prediction": _mean(value_predictions),
        }


class PolicyActor:
    def __init__(
        self,
        policy: PolicyNetwork,
        deterministic: bool = False,
        temperature: float = 1.0,
        epsilon_random: float = 0.0,
        collector: Optional[List[Dict[str, Any]]] = None,
        decision_callback: Optional[Any] = None,
    ) -> None:
        self.policy = policy
        self.deterministic = deterministic
        self.temperature = temperature
        self.epsilon_random = epsilon_random
        self.collector = collector
        self.decision_callback = decision_callback

    def choose(self, player_name: str, options: Sequence[Sequence[Any]], known_game_state: Dict[str, Any]) -> int:
        state_vec = state_to_vector(known_game_state, legal_option_count=len(options))
        option_vecs = [option_to_vector(option, known_game_state) for option in options]
        collect_details = self.collector is not None
        selection = self.policy.select_action(
            state_vec,
            option_vecs,
            deterministic=self.deterministic,
            temperature=self.temperature,
            epsilon_random=self.epsilon_random,
            need_log_prob=collect_details,
            need_value=collect_details,
        )
        if collect_details:
            self.collector.append(
                {
                    "state_vec": state_vec,
                    "option_vecs": option_vecs,
                    "action_index": selection["action_index"],
                    "old_log_prob": selection["log_prob"],
                    "value": selection["value"],
                    "player_name": player_name,
                }
            )
        if self.decision_callback is not None:
            self.decision_callback(player_name, options, known_game_state, selection["action_index"])
        return selection["action_index"]


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _elo_update(rating_a: float, rating_b: float, score_a: float, k_factor: float) -> Tuple[float, float]:
    expected_a = _elo_expected(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    score_b = 1.0 - score_a
    new_a = rating_a + k_factor * (score_a - expected_a)
    new_b = rating_b + k_factor * (score_b - expected_b)
    return new_a, new_b


class RunFiles:
    def __init__(self, run_name: str) -> None:
        self.run_name = run_name
        self.run_dir = RUNS_DIR / run_name
        self.analysis_dir = self.run_dir / "analysis"
        self.policy_file = self.run_dir / "latest_policy.json"
        self.candidate_policy_file = self.run_dir / "candidate_policy.json"
        self.training_state_file = self.run_dir / "training_state.json"
        self.checkpoints_dir = self.run_dir / "checkpoints"


def _default_candidate_state(base_checkpoint: str, iteration: int) -> Dict[str, Any]:
    return {
        "base_checkpoint": base_checkpoint,
        "base_iteration": iteration,
        "attempts_since_reset": 0,
        "total_attempts": 0,
        "resets": 0,
        "promotions": 0,
        "rating_pass_elo": None,
        "last_score": None,
        "last_result": "initialized",
        "last_reset_reason": "initialization",
        "last_opponents": [],
    }


def model_type_overrides(model_type: str) -> Dict[str, Any]:
    normalized = str(model_type or MODEL_TYPE_DEEP).strip().lower()
    if normalized == MODEL_TYPE_DEEP:
        return {
            "model_architecture": CURRENT_MODEL_ARCHITECTURE,
            "hidden_size": DEEP_MODEL_HIDDEN_SIZE,
        }
    if normalized == MODEL_TYPE_DEFAULT:
        return {
            "model_architecture": LEGACY_MODEL_ARCHITECTURE,
            "hidden_size": DEFAULT_MODEL_HIDDEN_SIZE,
        }
    raise ValueError(f"Unknown model type '{model_type}'. Expected 'deep' or 'default'.")


def _convert_policy_architecture(
    source_policy: PolicyNetwork,
    target_architecture: str,
    target_hidden_size: int,
    target_device_preference: str,
) -> Tuple[PolicyNetwork, str]:
    requested_architecture = str(target_architecture or source_policy.architecture).strip()
    requested_hidden_size = int(target_hidden_size)
    requested_device_preference = str(target_device_preference or DEVICE_AUTO).strip().lower()

    if requested_hidden_size <= 0:
        raise ValueError("target_hidden_size must be positive.")

    if (
        source_policy.architecture == requested_architecture
        and int(source_policy.hidden_size) == requested_hidden_size
        and str(source_policy.device_preference).strip().lower() == requested_device_preference
    ):
        return PolicyNetwork.from_dict(source_policy.to_dict(include_optimizer=False) | {"device_preference": requested_device_preference}), "cloned_exact"

    if (
        source_policy.architecture == requested_architecture
        and int(source_policy.hidden_size) == requested_hidden_size
    ):
        payload = source_policy.to_dict(include_optimizer=False)
        payload["device_preference"] = requested_device_preference
        return PolicyNetwork.from_dict(payload), "cloned_device"

    if (
        source_policy.architecture == LEGACY_MODEL_ARCHITECTURE
        and requested_architecture == CURRENT_MODEL_ARCHITECTURE
        and requested_hidden_size >= int(source_policy.hidden_size)
    ):
        converted = PolicyNetwork(
            state_size=source_policy.state_size,
            option_size=source_policy.option_size,
            hidden_size=requested_hidden_size,
            architecture=requested_architecture,
            device_preference=requested_device_preference,
        )
        source_model = source_policy.model
        target_model = converted.model
        source_hidden_size = int(source_policy.hidden_size)

        with torch.no_grad():
            for parameter in target_model.parameters():
                parameter.zero_()

            target_model.state_linear.weight[:source_hidden_size, :] = source_model.state_linear.weight.detach().to(converted.device)
            target_model.state_linear.bias[:source_hidden_size] = source_model.state_linear.bias.detach().to(converted.device)
            target_model.option_linear.weight[:source_hidden_size, :] = source_model.option_linear.weight.detach().to(converted.device)
            target_model.option_linear.bias[:source_hidden_size] = source_model.option_linear.bias.detach().to(converted.device)
            target_model.joint_bias[:source_hidden_size] = source_model.joint_bias.detach().to(converted.device)

            target_model.policy_head.weight[:, :source_hidden_size] = source_model.policy_head.weight.detach().to(converted.device)
            target_model.policy_head.bias.copy_(source_model.policy_head.bias.detach().to(converted.device))
            target_model.value_head.weight[:, :source_hidden_size] = source_model.value_head.weight.detach().to(converted.device)
            target_model.value_head.bias.copy_(source_model.value_head.bias.detach().to(converted.device))

        return converted, "legacy_to_deep_exact_embed"

    raise ValueError(
        "Unsupported checkpoint architecture conversion: "
        f"{source_policy.architecture}/{source_policy.hidden_size} -> "
        f"{requested_architecture}/{requested_hidden_size}. "
        "Currently supported conversions are same-architecture clones and legacy_v1 -> deep_v2 embedding."
    )


def normalize_training_decisions_per_game(value: Any) -> str:
    normalized = str(value if value is not None else TRAINING_DECISIONS_PER_GAME_ALL).strip().upper()
    if normalized not in TRAINING_DECISIONS_PER_GAME_OPTIONS:
        raise ValueError(
            "training_decisions_per_game must be one of: "
            + ", ".join(TRAINING_DECISIONS_PER_GAME_OPTIONS)
            + "."
        )
    return normalized


def training_decisions_per_game_limit(value: Any) -> Optional[int]:
    normalized = normalize_training_decisions_per_game(value)
    if normalized == TRAINING_DECISIONS_PER_GAME_ALL:
        return None
    return int(normalized)


def normalize_train_temperature(value: Any) -> float:
    try:
        temperature = float(value)
    except (TypeError, ValueError):
        raise ValueError("train_temperature must be a number.")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("train_temperature must be greater than 0.")
    return temperature


def derive_temperature_schedule_overrides(train_temperature: float) -> Dict[str, float]:
    normalized_temperature = normalize_train_temperature(train_temperature)
    min_temperature = min(
        normalized_temperature,
        max(0.35, normalized_temperature * DEFAULT_MIN_TRAIN_TEMPERATURE_RATIO),
    )
    plateau_max_temperature = max(
        normalized_temperature,
        normalized_temperature * DEFAULT_PLATEAU_MAX_TRAIN_TEMPERATURE_RATIO,
    )
    return {
        "train_temperature": round(normalized_temperature, 6),
        "min_train_temperature": round(min_temperature, 6),
        "plateau_max_train_temperature": round(plateau_max_temperature, 6),
    }


def normalize_promotion_score_threshold(value: Any) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        raise ValueError("promotion_score_threshold must be a number.")
    if not math.isfinite(threshold) or threshold < 0.5 or threshold > 1.0:
        raise ValueError("promotion_score_threshold must be between 0.5 and 1.0.")
    return threshold


def new_run_overrides(
    model_type: str = MODEL_TYPE_DEEP,
    training_matches_per_iteration: int = 5,
    training_games_per_match: int = 16,
    training_decisions_per_game: str = TRAINING_DECISIONS_PER_GAME_ALL,
    train_temperature: float = 0.9,
    promotion_games: int = 24,
    promotion_score_threshold: float = 0.6,
    device_preference: str = DEVICE_AUTO,
) -> Dict[str, Any]:
    if int(training_matches_per_iteration) <= 0:
        raise ValueError("training_matches_per_iteration must be positive.")
    if int(training_games_per_match) <= 0:
        raise ValueError("training_games_per_match must be positive.")
    if int(promotion_games) <= 0:
        raise ValueError("promotion_games must be positive.")

    overrides = model_type_overrides(model_type)
    overrides.update(derive_temperature_schedule_overrides(train_temperature))
    overrides.update(
        {
            "device_preference": str(device_preference or DEVICE_AUTO).strip().lower(),
            "training_matches_per_iteration": int(training_matches_per_iteration),
            "training_games_per_match": int(training_games_per_match),
            "training_decisions_per_game": normalize_training_decisions_per_game(training_decisions_per_game),
            "promotion_games": int(promotion_games),
            "promotion_score_threshold": normalize_promotion_score_threshold(promotion_score_threshold),
        }
    )
    return overrides


def _default_training_state(config: TrainingConfig) -> Dict[str, Any]:
    return {
        "status": "idle",
        "run_name": LATEST_RUN_NAME,
        "iteration": 0,
        "total_matches": 0,
        "total_games": 0,
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
        "strategy": "champion_league_v2",
        "strategy_version": 2,
        "current_elo": INITIAL_ELO,
        "best_elo": INITIAL_ELO,
        "best_checkpoint": "checkpoint_000000.json",
        "latest_checkpoint": "checkpoint_000000.json",
        "promotions": 0,
        "last_promotion_iteration": 0,
        "config": asdict(config),
        "last_match": None,
        "last_update": None,
        "last_eval": None,
        "last_rating_pass": None,
        "candidate": _default_candidate_state("checkpoint_000000.json", 0),
        "checkpoints": [
            {
                "name": "checkpoint_000000.json",
                "iteration": 0,
                "elo": INITIAL_ELO,
                "created_at": _timestamp(),
                "note": "initial random policy",
            }
        ],
    }


def _load_json_or_none(path: Path, retries: int = 6, delay_seconds: float = 0.05) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None

    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (PermissionError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == retries - 1:
                break
            time.sleep(delay_seconds * (attempt + 1))

    if last_error is not None:
        raise last_error
    return None


_JSON_CACHE_LOCK = threading.Lock()
_JSON_CACHE: Dict[str, Tuple[Tuple[int, int], Optional[Dict[str, Any]]]] = {}


def _json_cache_path_key(path: Path) -> str:
    return str(path)


def _invalidate_json_cache(path: Path) -> None:
    with _JSON_CACHE_LOCK:
        _JSON_CACHE.pop(_json_cache_path_key(path), None)


def _load_cached_json_or_none(path: Path, retries: int = 6, delay_seconds: float = 0.05) -> Optional[Dict[str, Any]]:
    cache_key = _json_cache_path_key(path)
    try:
        stat = path.stat()
    except FileNotFoundError:
        with _JSON_CACHE_LOCK:
            _JSON_CACHE.pop(cache_key, None)
        return None

    signature = (stat.st_mtime_ns, stat.st_size)
    with _JSON_CACHE_LOCK:
        cached = _JSON_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature:
            payload = cached[1]
            return None if payload is None else dict(payload)

    payload = _load_json_or_none(path, retries=retries, delay_seconds=delay_seconds)
    try:
        refreshed_stat = path.stat()
    except FileNotFoundError:
        with _JSON_CACHE_LOCK:
            _JSON_CACHE.pop(cache_key, None)
        return None if payload is None else dict(payload)

    refreshed_signature = (refreshed_stat.st_mtime_ns, refreshed_stat.st_size)
    with _JSON_CACHE_LOCK:
        _JSON_CACHE[cache_key] = (refreshed_signature, payload)
    return None if payload is None else dict(payload)


def _load_policy_from_path(path: Path) -> Optional[PolicyNetwork]:
    payload = _load_json_or_none(path)
    if payload is None:
        return None
    return PolicyNetwork.from_dict(payload)


def _ensure_checkpoint_entry(state: Dict[str, Any], checkpoint_name: str, iteration: int, elo: float) -> Dict[str, Any]:
    checkpoints = state.setdefault("checkpoints", [])
    for item in checkpoints:
        if item.get("name") == checkpoint_name:
            item["iteration"] = int(item.get("iteration", iteration))
            item["elo"] = float(item.get("elo", elo))
            return item
    entry = {
        "name": checkpoint_name,
        "iteration": iteration,
        "elo": float(elo),
        "created_at": _timestamp(),
    }
    checkpoints.append(entry)
    return entry


def _migrate_run_state(files: RunFiles, policy: PolicyNetwork, state: Dict[str, Any], config: TrainingConfig) -> Tuple[PolicyNetwork, Dict[str, Any], bool]:
    dirty = False
    state["run_name"] = state.get("run_name", files.run_name)
    state["checkpoints"] = list(state.get("checkpoints") or [])

    for checkpoint in state["checkpoints"]:
        if "iteration" in checkpoint:
            checkpoint["iteration"] = int(checkpoint["iteration"])
        if "elo" in checkpoint:
            checkpoint["elo"] = float(checkpoint["elo"])

    best_name = state.get("best_checkpoint") or state.get("latest_checkpoint") or "checkpoint_000000.json"
    latest_name = state.get("latest_checkpoint") or best_name
    strategy_version = int(state.get("strategy_version", 0) or 0)
    migrated_to_v2 = state.get("strategy") != "champion_league_v2" or strategy_version < 2

    if migrated_to_v2:
        latest_name = best_name
        champion_policy = _load_policy_from_path(files.checkpoints_dir / latest_name)
        if champion_policy is not None:
            policy = champion_policy
        champion_entry = _ensure_checkpoint_entry(
            state,
            latest_name,
            int(state.get("iteration", 0)),
            float(state.get("best_elo", state.get("current_elo", INITIAL_ELO))),
        )
        champion_elo = float(champion_entry.get("elo", state.get("best_elo", INITIAL_ELO)))
        state["strategy"] = "champion_league_v2"
        state["strategy_version"] = 2
        state["latest_checkpoint"] = latest_name
        state["best_checkpoint"] = best_name or latest_name
        state["current_elo"] = champion_elo
        state["best_elo"] = max(float(state.get("best_elo", champion_elo)), champion_elo)
        state["promotions"] = int(state.get("promotions", 0))
        state["last_eval"] = state.get("last_eval")
        state["candidate"] = _default_candidate_state(latest_name, int(state.get("iteration", 0)))
        dirty = True
    else:
        state["strategy"] = "champion_league_v2"
        state["strategy_version"] = 2
        state["promotions"] = int(state.get("promotions", 0))
        state["last_eval"] = state.get("last_eval")
        default_candidate = _default_candidate_state(latest_name, int(state.get("iteration", 0)))
        candidate_state = dict(default_candidate)
        candidate_state.update(state.get("candidate") or {})
        if candidate_state != state.get("candidate"):
            state["candidate"] = candidate_state
            dirty = True

    champion_entry = _ensure_checkpoint_entry(
        state,
        state.get("latest_checkpoint", latest_name),
        int(state.get("iteration", 0)),
        float(state.get("current_elo", INITIAL_ELO)),
    )
    state["current_elo"] = float(champion_entry.get("elo", state.get("current_elo", INITIAL_ELO)))
    state["best_checkpoint"] = state.get("best_checkpoint", state.get("latest_checkpoint"))
    if state["best_checkpoint"]:
        best_entry = _ensure_checkpoint_entry(
            state,
            state["best_checkpoint"],
            int(champion_entry.get("iteration", state.get("iteration", 0))),
            float(state.get("best_elo", state.get("current_elo", INITIAL_ELO))),
        )
        state["best_elo"] = max(float(state.get("best_elo", INITIAL_ELO)), float(best_entry.get("elo", INITIAL_ELO)))

    if "last_promotion_iteration" not in state or state.get("last_promotion_iteration") is None:
        if int(state.get("promotions", 0)) > 0:
            state["last_promotion_iteration"] = int(champion_entry.get("iteration", state.get("iteration", 0)))
        else:
            state["last_promotion_iteration"] = 0
        dirty = True
    else:
        state["last_promotion_iteration"] = max(0, int(state.get("last_promotion_iteration", 0)))

    if not files.policy_file.exists() or migrated_to_v2:
        _atomic_write_json(files.policy_file, policy.to_dict(include_optimizer=True))
        dirty = True

    if not files.candidate_policy_file.exists():
        _atomic_write_json(files.candidate_policy_file, policy.to_dict(include_optimizer=True))
        dirty = True

    return policy, state, dirty


def _ensure_run(run_name: str, config_overrides: Optional[Dict[str, Any]] = None) -> RunFiles:
    files = RunFiles(run_name)
    files.run_dir.mkdir(parents=True, exist_ok=True)
    files.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if files.training_state_file.exists() and files.policy_file.exists():
        return files

    config = TrainingConfig().merged(config_overrides)
    policy = PolicyNetwork(
        hidden_size=config.hidden_size,
        architecture=config.model_architecture,
        device_preference=config.device_preference,
    )
    state = _default_training_state(config)
    state["run_name"] = run_name
    _atomic_write_json(files.policy_file, policy.to_dict(include_optimizer=True))
    _atomic_write_json(files.candidate_policy_file, policy.to_dict(include_optimizer=True))
    _atomic_write_json(files.training_state_file, state)
    _atomic_write_json(files.checkpoints_dir / "checkpoint_000000.json", policy.to_dict(include_optimizer=False))
    return files


def run_exists(run_name: str) -> bool:
    files = RunFiles(run_name)
    return files.training_state_file.exists() and files.policy_file.exists()


def _load_policy_and_state(run_name: str, config_overrides: Optional[Dict[str, Any]] = None) -> Tuple[RunFiles, PolicyNetwork, Dict[str, Any], TrainingConfig]:
    files = _ensure_run(run_name, config_overrides=config_overrides)
    state_payload = _load_json_or_none(files.training_state_file)
    if state_payload is None:
        state_payload = _default_training_state(TrainingConfig())
        state_payload["run_name"] = run_name
    policy_payload = _load_json_or_none(files.policy_file)
    original_saved_config = dict(state_payload.get("config", {}) or {})
    saved_config = dict(original_saved_config)
    if "training_games_per_match" not in saved_config:
        saved_config["training_games_per_match"] = 24
    if "training_decisions_per_game" not in saved_config:
        saved_config["training_decisions_per_game"] = TRAINING_DECISIONS_PER_GAME_ALL
    else:
        saved_config["training_decisions_per_game"] = normalize_training_decisions_per_game(
            saved_config.get("training_decisions_per_game")
        )
    if saved_config.get("promotion_games") == 13:
        saved_config["promotion_games"] = 24
    legacy_default_updates = {
        "promotion_score_threshold": ((0.55,), 0.6),
        "learning_rate": ((0.0015,), 0.0019),
        "min_learning_rate": ((0.0003,), 0.0004),
        "epsilon_random": ((0.05,), 0.07),
        "min_epsilon_random": ((0.01,), 0.015),
        "train_temperature": ((1.0, 1.1), 0.9),
        "min_train_temperature": ((0.55, 0.62), 0.5),
        "plateau_learning_rate_boost": ((2.0, 2.15), 1.7),
        "plateau_epsilon_boost": ((3.0, 3.2), 2.4),
        "plateau_temperature_boost": ((1.35, 1.45, 1.28), 1.22),
        "plateau_max_learning_rate": ((0.0035, 0.0042), 0.0033),
        "plateau_max_epsilon_random": ((0.18, 0.22), 0.16),
        "plateau_max_train_temperature": ((1.45, 1.6, 1.35), 1.15),
    }
    for key, (old_values, new_value) in legacy_default_updates.items():
        current_value = saved_config.get(key)
        if current_value is None:
            saved_config[key] = new_value
            continue
        try:
            numeric_value = float(current_value)
        except (TypeError, ValueError):
            continue
        if any(abs(numeric_value - float(old_value)) <= 1e-12 for old_value in old_values):
            saved_config[key] = new_value
    if policy_payload is not None:
        payload_architecture = policy_payload.get("architecture")
        if payload_architecture is None:
            payload_state_dict = policy_payload.get("state_dict") or {}
            if any(name.startswith("state_refine_") or name.startswith("joint_mix_") or name.startswith("value_refine") for name in payload_state_dict.keys()):
                payload_architecture = CURRENT_MODEL_ARCHITECTURE
            else:
                payload_architecture = LEGACY_MODEL_ARCHITECTURE
        saved_config["model_architecture"] = payload_architecture
        if "hidden_size" not in original_saved_config or saved_config.get("hidden_size") != policy_payload.get("hidden_size"):
            saved_config["hidden_size"] = policy_payload.get("hidden_size", saved_config.get("hidden_size", 96))
    state_payload["config"] = saved_config
    config = TrainingConfig().merged(saved_config).merged(config_overrides)
    if policy_payload is None:
        policy = PolicyNetwork(
            hidden_size=config.hidden_size,
            architecture=config.model_architecture,
            device_preference=config.device_preference,
        )
    else:
        policy = PolicyNetwork.from_dict(policy_payload)
    policy, state_payload, dirty = _migrate_run_state(files, policy, state_payload, config)
    dirty = dirty or (saved_config != original_saved_config)
    if dirty:
        _save_policy_and_state(files, policy, state_payload)
    return files, policy, state_payload, config


def _save_policy_and_state(files: RunFiles, policy: PolicyNetwork, state: Dict[str, Any]) -> None:
    state["updated_at"] = _timestamp()
    _atomic_write_json(files.policy_file, policy.to_dict(include_optimizer=True))
    _atomic_write_json(files.training_state_file, state)


def _checkpoint_path(files: RunFiles, iteration: int) -> Path:
    return files.checkpoints_dir / f"champion_{iteration:06d}.json"


def load_policy(run_name: str = LATEST_RUN_NAME, checkpoint: str = "latest") -> PolicyNetwork:
    files, policy, state, _ = _load_policy_and_state(run_name)
    if checkpoint == "latest":
        return policy
    if checkpoint == "candidate":
        candidate_policy = _load_policy_from_path(files.candidate_policy_file)
        return candidate_policy or policy.clone()
    if checkpoint == "best":
        checkpoint_name = state.get("best_checkpoint", state.get("latest_checkpoint"))
    else:
        checkpoint_name = checkpoint
    checkpoint_file = files.checkpoints_dir / checkpoint_name
    payload = _load_json_or_none(checkpoint_file)
    if payload is None:
        return policy
    return PolicyNetwork.from_dict(payload)


def list_runs() -> List[Dict[str, Any]]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runs: List[Dict[str, Any]] = []
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        state_path = run_dir / "training_state.json"
        state = _load_cached_json_or_none(state_path) or {}
        checkpoints = list(state.get("checkpoints") or [])
        runs.append(
            {
                "run_name": state.get("run_name", run_dir.name),
                "status": state.get("status", "idle"),
                "iteration": int(state.get("iteration", 0)),
                "total_matches": int(state.get("total_matches", 0)),
                "total_games": int(state.get("total_games", 0)),
                "current_elo": float(state.get("current_elo", INITIAL_ELO)),
                "best_elo": float(state.get("best_elo", INITIAL_ELO)),
                "best_checkpoint": state.get("best_checkpoint"),
                "latest_checkpoint": state.get("latest_checkpoint"),
                "promotions": int(state.get("promotions", 0)),
                "checkpoint_count": len(checkpoints),
                "created_at": state.get("created_at"),
                "updated_at": state.get("updated_at"),
                "last_match": state.get("last_match"),
                "last_update": state.get("last_update"),
                "last_eval": state.get("last_eval"),
                "last_error": state.get("last_error"),
                "run_dir": str(run_dir),
            }
        )
    runs.sort(key=lambda item: (item.get("updated_at") or 0.0, item["run_name"]), reverse=True)
    return runs


def list_checkpoints(run_name: str = LATEST_RUN_NAME) -> List[Dict[str, Any]]:
    files = RunFiles(run_name)
    state = _load_cached_json_or_none(files.training_state_file) or {}
    best_name = state.get("best_checkpoint")
    latest_name = state.get("latest_checkpoint")
    checkpoints = []
    for item in list(state.get("checkpoints") or []):
        checkpoint = dict(item)
        checkpoint["is_best"] = checkpoint.get("name") == best_name
        checkpoint["is_latest"] = checkpoint.get("name") == latest_name
        checkpoint["is_candidate"] = False
        checkpoints.append(checkpoint)
    if files.candidate_policy_file.exists():
        candidate_state = dict(state.get("candidate") or {})
        candidate_elo = candidate_state.get("rating_pass_elo")
        if candidate_elo is None:
            candidate_elo = state.get("current_elo", INITIAL_ELO)
        checkpoints.append(
            {
                "name": "candidate",
                "iteration": int(state.get("iteration", 0)),
                "elo": float(candidate_elo),
                "is_best": False,
                "is_latest": False,
                "is_candidate": True,
                "base_checkpoint": candidate_state.get("base_checkpoint"),
                "attempts_since_reset": int(candidate_state.get("attempts_since_reset", 0)),
                "last_score": candidate_state.get("last_score"),
            }
        )
    checkpoints.sort(key=lambda item: (int(item.get("iteration", 0)), item.get("name", "")))
    return checkpoints


def get_run_state(run_name: str = LATEST_RUN_NAME) -> Dict[str, Any]:
    files = RunFiles(run_name)
    state = _load_cached_json_or_none(files.training_state_file) or {}
    if not state:
        return {}
    state["run_name"] = state.get("run_name", run_name)
    state["run_dir"] = str(files.run_dir)
    state["candidate_policy_file"] = str(files.candidate_policy_file)
    return state


def _checkpoint_sort_key(item: Dict[str, Any]) -> Tuple[int, str]:
    return int(item.get("iteration", 0)), str(item.get("name", ""))


def _resolve_checkpoint_name(state: Dict[str, Any], checkpoint: str) -> str:
    checkpoint_name = str(checkpoint or "").strip()
    if checkpoint_name == "latest":
        checkpoint_name = str(state.get("latest_checkpoint") or "")
    elif checkpoint_name == "best":
        checkpoint_name = str(state.get("best_checkpoint") or state.get("latest_checkpoint") or "")
    return checkpoint_name


def _pick_latest_checkpoint_entry(checkpoints: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not checkpoints:
        return None
    return max(checkpoints, key=_checkpoint_sort_key)


def _pick_best_checkpoint_entry(checkpoints: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not checkpoints:
        return None
    return max(
        checkpoints,
        key=lambda item: (
            float(item.get("elo", INITIAL_ELO)),
            int(item.get("iteration", 0)),
            str(item.get("name", "")),
        ),
    )


def _scrub_deleted_checkpoint_refs(value: Any, deleted_checkpoint: str, replacement_checkpoint: Optional[str]) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, nested_value in value.items():
            if key == "checkpoint" and nested_value == deleted_checkpoint:
                cleaned[key] = replacement_checkpoint
            elif key in {"base_checkpoint", "best_checkpoint", "latest_checkpoint", "champion_checkpoint"} and nested_value == deleted_checkpoint:
                cleaned[key] = replacement_checkpoint
            elif key == "promoted_checkpoint" and nested_value == deleted_checkpoint:
                cleaned[key] = None
            elif key == "name" and nested_value == deleted_checkpoint and value.get("kind") == "checkpoint":
                continue
            else:
                cleaned[key] = _scrub_deleted_checkpoint_refs(nested_value, deleted_checkpoint, replacement_checkpoint)
        return cleaned
    if isinstance(value, list):
        cleaned_list: List[Any] = []
        for item in value:
            cleaned_item = _scrub_deleted_checkpoint_refs(item, deleted_checkpoint, replacement_checkpoint)
            if isinstance(cleaned_item, dict) and cleaned_item.get("kind") == "checkpoint" and cleaned_item.get("name") is None:
                continue
            cleaned_list.append(cleaned_item)
        return cleaned_list
    return value


def delete_checkpoints(run_name: str = LATEST_RUN_NAME, checkpoints: Sequence[str] = ()) -> Dict[str, Any]:
    files, policy, state, _ = _load_policy_and_state(run_name)
    requested_names = [_resolve_checkpoint_name(state, checkpoint) for checkpoint in checkpoints]
    checkpoint_names = [name for name in requested_names if name]
    if not checkpoint_names:
        raise ValueError("At least one checkpoint name is required.")
    if any(name == "candidate" for name in checkpoint_names):
        raise ValueError("The candidate row is not a checkpoint file. Delete saved checkpoints instead.")

    trainer = _TRAINERS.get(run_name)
    if trainer is not None and trainer.is_running:
        raise RuntimeError(f"Run '{run_name}' is currently training. Interrupt it before deleting checkpoints.")
    if str(state.get("status", "idle")).lower() == "running":
        raise RuntimeError(f"Run '{run_name}' is marked as running. Interrupt training before deleting checkpoints.")

    checkpoint_entries = list(state.get("checkpoints") or [])
    existing_names = {str(item.get("name", "")) for item in checkpoint_entries}
    missing = [name for name in checkpoint_names if name not in existing_names]
    if missing:
        if len(missing) == 1:
            raise FileNotFoundError(f"Checkpoint '{missing[0]}' was not found in run '{run_name}'.")
        raise FileNotFoundError(f"These checkpoints were not found in run '{run_name}': {', '.join(sorted(missing))}.")

    unique_checkpoint_names = sorted(set(checkpoint_names))
    deleted_set = set(unique_checkpoint_names)
    for checkpoint_name in unique_checkpoint_names:
        checkpoint_path = files.checkpoints_dir / checkpoint_name
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    state["checkpoints"] = [
        dict(item)
        for item in checkpoint_entries
        if str(item.get("name", "")) not in deleted_set
    ]

    deleted_latest = str(state.get("latest_checkpoint")) in deleted_set
    deleted_best = str(state.get("best_checkpoint")) in deleted_set

    replacement_checkpoint: Optional[str] = None
    if deleted_latest or not state.get("checkpoints"):
        replacement_checkpoint = "champion_current.json"
        replacement_path = files.checkpoints_dir / replacement_checkpoint
        _atomic_write_json(replacement_path, policy.to_dict(include_optimizer=False))
        replacement_entry = _ensure_checkpoint_entry(
            state,
            replacement_checkpoint,
            int(state.get("iteration", 0)),
            float(state.get("current_elo", INITIAL_ELO)),
        )
        replacement_entry["iteration"] = int(state.get("iteration", 0))
        replacement_entry["elo"] = float(state.get("current_elo", INITIAL_ELO))
        replacement_entry["created_at"] = replacement_entry.get("created_at", _timestamp())
        replacement_entry["note"] = "synthetic current champion snapshot"
        state["latest_checkpoint"] = replacement_checkpoint
    else:
        latest_entry = _pick_latest_checkpoint_entry(list(state.get("checkpoints") or []))
        if latest_entry is not None:
            state["latest_checkpoint"] = latest_entry.get("name")

    latest_entry = next(
        (item for item in state.get("checkpoints", []) if item.get("name") == state.get("latest_checkpoint")),
        None,
    )
    if latest_entry is not None:
        state["current_elo"] = float(latest_entry.get("elo", state.get("current_elo", INITIAL_ELO)))

    if deleted_best or str(state.get("best_checkpoint")) in deleted_set:
        best_entry = _pick_best_checkpoint_entry(list(state.get("checkpoints") or []))
        if best_entry is not None:
            state["best_checkpoint"] = best_entry.get("name")
            state["best_elo"] = float(best_entry.get("elo", state.get("current_elo", INITIAL_ELO)))
        else:
            state["best_checkpoint"] = state.get("latest_checkpoint")
            state["best_elo"] = float(state.get("current_elo", INITIAL_ELO))
    else:
        best_entry = next(
            (item for item in state.get("checkpoints", []) if item.get("name") == state.get("best_checkpoint")),
            None,
        )
        if best_entry is not None:
            state["best_elo"] = float(best_entry.get("elo", state.get("best_elo", INITIAL_ELO)))

    replacement_reference = str(state.get("latest_checkpoint") or state.get("best_checkpoint") or "")
    for checkpoint_name in unique_checkpoint_names:
        state["candidate"] = _scrub_deleted_checkpoint_refs(
            state.get("candidate") or {},
            checkpoint_name,
            replacement_reference,
        )
        state["last_match"] = _scrub_deleted_checkpoint_refs(
            state.get("last_match"),
            checkpoint_name,
            replacement_reference,
        )
        state["last_eval"] = _scrub_deleted_checkpoint_refs(
            state.get("last_eval"),
            checkpoint_name,
            replacement_reference,
        )
        state["last_rating_pass"] = _scrub_deleted_checkpoint_refs(
            state.get("last_rating_pass"),
            checkpoint_name,
            replacement_reference,
        )
        state["last_update"] = _scrub_deleted_checkpoint_refs(
            state.get("last_update"),
            checkpoint_name,
            replacement_reference,
        )
        _POLICY_CACHE.pop((run_name, checkpoint_name), None)

    _POLICY_CACHE.pop((run_name, "best"), None)
    _POLICY_CACHE.pop((run_name, "latest"), None)
    _TRAINERS.pop(run_name, None)

    _save_policy_and_state(files, policy, state)

    return {
        "run_name": run_name,
        "deleted_checkpoints": unique_checkpoint_names,
        "latest_checkpoint": state.get("latest_checkpoint"),
        "best_checkpoint": state.get("best_checkpoint"),
        "current_elo": float(state.get("current_elo", INITIAL_ELO)),
        "best_elo": float(state.get("best_elo", INITIAL_ELO)),
        "checkpoint_count": len(list(state.get("checkpoints") or [])),
        "replacement_checkpoint": replacement_checkpoint,
    }


def delete_checkpoint(run_name: str = LATEST_RUN_NAME, checkpoint: str = "") -> Dict[str, Any]:
    result = delete_checkpoints(run_name=run_name, checkpoints=[checkpoint])
    deleted = list(result.get("deleted_checkpoints") or [])
    result["deleted_checkpoint"] = deleted[0] if deleted else ""
    return result


def _play_match(
    policy_a: PolicyNetwork,
    policy_b: PolicyNetwork,
    config: TrainingConfig,
    collect_a: bool = False,
    deterministic_a: bool = False,
    deterministic_b: bool = False,
    games_per_match: int = 24,
) -> Dict[str, Any]:
    collector: Optional[List[Dict[str, Any]]] = [] if collect_a else None
    trajectory_by_game: Optional[List[List[Dict[str, Any]]]] = [] if collect_a else None
    wins_a = 0
    wins_b = 0
    per_game_winners: List[str] = []
    majority = games_per_match // 2 + 1
    started_at = _timestamp()

    while wins_a < majority and wins_b < majority and len(per_game_winners) < games_per_match:
        game_collector: Optional[List[Dict[str, Any]]] = [] if collect_a else None
        actor_a = PolicyActor(
            policy_a,
            deterministic=deterministic_a,
            temperature=config.eval_temperature if deterministic_a else config.train_temperature,
            epsilon_random=0.0 if deterministic_a else config.epsilon_random,
            collector=game_collector,
        )
        actor_b = PolicyActor(
            policy_b,
            deterministic=deterministic_b,
            temperature=config.eval_temperature if deterministic_b else config.train_temperature,
            epsilon_random=0.0 if deterministic_b else config.epsilon_random,
            collector=None,
        )
        game = Game(
            "policy_a",
            "policy_b",
            p1_choose=actor_a.choose,
            p2_choose=actor_b.choose,
            verbose=False,
            max_turns=config.max_turns_per_game,
            max_actions_per_turn=config.max_actions_per_turn,
        )
        per_game_winners.append(game.winner.name)
        if game.winner.name == "policy_a":
            wins_a += 1
        else:
            wins_b += 1
        if collect_a and collector is not None and trajectory_by_game is not None and game_collector is not None:
            collector.extend(game_collector)
            trajectory_by_game.append(game_collector)

    games_played = len(per_game_winners)
    return_value = (wins_a - wins_b) / max(games_played, 1)
    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "games_played": games_played,
        "return_value": return_value,
        "per_game_winners": per_game_winners,
        "trajectory": collector or [],
        "trajectory_by_game": trajectory_by_game or [],
        "duration_seconds": _timestamp() - started_at,
    }


def _play_balanced_match(
    policy_a: PolicyNetwork,
    policy_b: PolicyNetwork,
    config: TrainingConfig,
    games_per_match: int,
) -> Dict[str, Any]:
    if games_per_match <= 1:
        return _play_match(
            policy_a,
            policy_b,
            config,
            collect_a=False,
            deterministic_a=True,
            deterministic_b=True,
            games_per_match=1,
        )

    games_as_first = max(1, games_per_match // 2)
    games_as_second = max(1, games_per_match - games_as_first)
    started_at = _timestamp()

    first_seat = _play_match(
        policy_a,
        policy_b,
        config,
        collect_a=False,
        deterministic_a=True,
        deterministic_b=True,
        games_per_match=games_as_first,
    )
    second_seat = _play_match(
        policy_b,
        policy_a,
        config,
        collect_a=False,
        deterministic_a=True,
        deterministic_b=True,
        games_per_match=games_as_second,
    )

    wins_a = int(first_seat.get("wins_a", 0)) + int(second_seat.get("wins_b", 0))
    wins_b = int(first_seat.get("wins_b", 0)) + int(second_seat.get("wins_a", 0))
    games_played = int(first_seat.get("games_played", 0)) + int(second_seat.get("games_played", 0))
    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "games_played": games_played,
        "return_value": (wins_a - wins_b) / max(games_played, 1),
        "per_game_winners": list(first_seat.get("per_game_winners", []))
        + [("policy_a" if winner == "policy_b" else "policy_b") for winner in second_seat.get("per_game_winners", [])],
        "trajectory": [],
        "duration_seconds": _timestamp() - started_at,
        "seat_results": [
            {"seat": "a_first", **first_seat},
            {"seat": "a_second", **second_seat},
        ],
    }


class SelfPlayTrainer:
    def __init__(self, run_name: str = LATEST_RUN_NAME, config_overrides: Optional[Dict[str, Any]] = None) -> None:
        files, policy, state, config = _load_policy_and_state(run_name, config_overrides=config_overrides)
        self.files = files
        self.policy = policy
        self.candidate_policy = _load_policy_from_path(files.candidate_policy_file) or policy.clone()
        self.state = state
        self.config = config
        self.run_name = run_name
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _save(self) -> None:
        config_payload = asdict(self.config)
        config_payload["max_iterations"] = None
        self.state["config"] = config_payload
        _save_policy_and_state(self.files, self.policy, self.state)
        _atomic_write_json(self.files.candidate_policy_file, self.candidate_policy.to_dict(include_optimizer=True))

    def _candidate_state(self) -> Dict[str, Any]:
        candidate_state = self.state.get("candidate")
        if candidate_state is None:
            candidate_state = _default_candidate_state(
                self.state.get("latest_checkpoint", "checkpoint_000000.json"),
                int(self.state.get("iteration", 0)),
            )
            self.state["candidate"] = candidate_state
        return candidate_state

    def _scheduled_training_config(self) -> TrainingConfig:
        anneal_steps = max(1, int(self.config.anneal_steps))
        progress = _clip(float(self.state.get("iteration", 0)) / float(anneal_steps), 0.0, 1.0)
        scheduled = self.config.merged(
            {
                "learning_rate": _lerp(self.config.learning_rate, self.config.min_learning_rate, progress),
                "epsilon_random": _lerp(self.config.epsilon_random, self.config.min_epsilon_random, progress),
                "train_temperature": _lerp(self.config.train_temperature, self.config.min_train_temperature, progress),
                "ppo_clip": _lerp(self.config.ppo_clip, self.config.min_ppo_clip, progress),
            }
        )
        boost = self._promotion_drought_boost()
        if boost["progress"] <= 0.0:
            return scheduled
        return scheduled.merged(
            {
                "learning_rate": min(
                    float(self.config.plateau_max_learning_rate),
                    float(scheduled.learning_rate) * float(boost["learning_rate_multiplier"]),
                ),
                "epsilon_random": min(
                    float(self.config.plateau_max_epsilon_random),
                    float(scheduled.epsilon_random) * float(boost["epsilon_multiplier"]),
                ),
                "train_temperature": min(
                    float(self.config.plateau_max_train_temperature),
                    float(scheduled.train_temperature) * float(boost["temperature_multiplier"]),
                ),
            }
        )

    def _last_promotion_iteration(self) -> int:
        latest_name = self.state.get("latest_checkpoint")
        latest_entry = self._find_checkpoint_entry(str(latest_name)) if latest_name else None
        derived_iteration = int(latest_entry.get("iteration", 0)) if latest_entry is not None else 0
        stored_iteration = int(self.state.get("last_promotion_iteration", derived_iteration) or 0)
        return max(0, stored_iteration)

    def _iterations_since_promotion(self) -> int:
        return max(0, int(self.state.get("iteration", 0)) - self._last_promotion_iteration())

    def _promotion_drought_boost(self) -> Dict[str, float]:
        iterations_since_promotion = self._iterations_since_promotion()
        start = max(0, int(self.config.plateau_start_iterations_without_promotion))
        full = max(start + 1, int(self.config.plateau_full_iterations_without_promotion))
        progress = 0.0
        if iterations_since_promotion > start:
            progress = _clip(
                float(iterations_since_promotion - start) / float(max(1, full - start)),
                0.0,
                1.0,
            )
        return {
            "iterations_since_promotion": float(iterations_since_promotion),
            "progress": progress,
            "learning_rate_multiplier": _lerp(1.0, self.config.plateau_learning_rate_boost, progress),
            "epsilon_multiplier": _lerp(1.0, self.config.plateau_epsilon_boost, progress),
            "temperature_multiplier": _lerp(1.0, self.config.plateau_temperature_boost, progress),
        }

    def _rating_pass_pool(self, max_policies: int, include_candidate: bool) -> List[Dict[str, Any]]:
        checkpoints = self._checkpoint_entries_sorted()
        candidate_available = include_candidate and self.files.candidate_policy_file.exists()
        checkpoint_limit = max(0, max_policies - (1 if candidate_available else 0))

        selected_names: List[str] = []
        checkpoint_by_name = {item.get("name"): item for item in checkpoints if item.get("name")}

        if checkpoint_limit >= len(checkpoints):
            selected_names = [str(item["name"]) for item in checkpoints if item.get("name")]
        else:
            for anchor_name in [
                self.state.get("latest_checkpoint"),
                self.state.get("best_checkpoint"),
                checkpoints[0]["name"] if checkpoints else None,
            ]:
                if (
                    checkpoint_limit > 0
                    and anchor_name
                    and anchor_name in checkpoint_by_name
                    and anchor_name not in selected_names
                    and len(selected_names) < checkpoint_limit
                ):
                    selected_names.append(str(anchor_name))

            remaining = [item for item in checkpoints if item.get("name") not in selected_names]
            extra_slots = max(0, checkpoint_limit - len(selected_names))
            if extra_slots >= len(remaining):
                selected_names.extend(str(item["name"]) for item in remaining if item.get("name"))
            elif extra_slots > 0 and remaining:
                chosen_indices = set()
                for index in range(extra_slots):
                    raw_index = int((index + 0.5) * len(remaining) / extra_slots)
                    raw_index = _clip(raw_index, 0, len(remaining) - 1)
                    chosen_indices.add(int(raw_index))
                while len(chosen_indices) < extra_slots:
                    for index in range(len(remaining)):
                        if index not in chosen_indices:
                            chosen_indices.add(index)
                        if len(chosen_indices) >= extra_slots:
                            break
                selected_names.extend(
                    str(remaining[index]["name"])
                    for index in sorted(chosen_indices)
                    if remaining[index].get("name")
                )

        participants: List[Dict[str, Any]] = []
        for checkpoint_name in selected_names:
            checkpoint = checkpoint_by_name.get(checkpoint_name)
            if checkpoint is None:
                continue
            participants.append(
                {
                    "name": checkpoint_name,
                    "kind": "checkpoint",
                    "iteration": int(checkpoint.get("iteration", 0)),
                    "elo": float(checkpoint.get("elo", INITIAL_ELO)),
                }
            )

        if candidate_available and len(participants) < max_policies:
            candidate_state = self._candidate_state()
            candidate_elo = candidate_state.get("rating_pass_elo")
            if candidate_elo is None:
                candidate_elo = self.state.get("current_elo", INITIAL_ELO)
            participants.append(
                {
                    "name": "candidate",
                    "kind": "candidate",
                    "iteration": int(self.state.get("iteration", 0)),
                    "elo": float(candidate_elo),
                    "base_checkpoint": candidate_state.get("base_checkpoint"),
                }
            )

        participants.sort(key=lambda item: (int(item.get("iteration", 0)), item.get("name", "")))
        return participants

    def _policy_for_rating_participant(self, participant: Dict[str, Any]) -> PolicyNetwork:
        if participant.get("kind") == "candidate":
            return self.candidate_policy
        checkpoint_name = participant.get("name")
        if checkpoint_name == self.state.get("latest_checkpoint"):
            return self.policy
        return _cached_policy(self.run_name, str(checkpoint_name))

    def run_rating_pass(
        self,
        max_policies: int = 8,
        games_per_pair: int = 12,
        include_candidate: bool = True,
        ) -> Dict[str, Any]:
        if max_policies < 2:
            raise ValueError("Rating pass needs at least 2 policies.")
        if games_per_pair < 2:
            raise ValueError("Rating pass needs at least 2 games per pairing.")
        status = str(self.state.get("status", "idle"))
        last_updated_at = float(self.state.get("updated_at", 0.0) or 0.0)
        recently_updated = (_timestamp() - last_updated_at) <= 300.0
        if self.is_running or (status == "training" and recently_updated):
            raise RuntimeError("Interrupt training for this run before starting a rating pass.")

        participants = self._rating_pass_pool(max_policies=max_policies, include_candidate=include_candidate)
        if len(participants) < 2:
            raise RuntimeError("Not enough available models in this run to perform a rating pass.")

        rating_map = {item["name"]: float(item.get("elo", INITIAL_ELO)) for item in participants}
        started_at = _timestamp()
        self.state["status"] = "rating"
        self.state["last_error"] = None
        self.state["last_rating_pass"] = {
            "status": "running",
            "participant_count": len(participants),
            "games_per_pair": int(games_per_pair),
            "include_candidate": bool(include_candidate),
            "started_at": started_at,
        }
        self._save()

        try:
            pairings: List[Dict[str, Any]] = []
            for index_a, participant_a in enumerate(participants):
                for index_b in range(index_a + 1, len(participants)):
                    participant_b = participants[index_b]
                    name_a = str(participant_a["name"])
                    name_b = str(participant_b["name"])
                    rating_a_before = float(rating_map[name_a])
                    rating_b_before = float(rating_map[name_b])
                    match_summary = _play_balanced_match(
                        self._policy_for_rating_participant(participant_a),
                        self._policy_for_rating_participant(participant_b),
                        self.config,
                        games_per_match=games_per_pair,
                    )
                    score_a = float(match_summary.get("wins_a", 0)) / max(int(match_summary.get("games_played", 0)), 1)
                    rating_a_after, rating_b_after = _elo_update(
                        rating_a_before,
                        rating_b_before,
                        score_a,
                        self.config.elo_k,
                    )
                    rating_map[name_a] = rating_a_after
                    rating_map[name_b] = rating_b_after
                    pairings.append(
                        {
                            "a": name_a,
                            "b": name_b,
                            "wins_a": int(match_summary.get("wins_a", 0)),
                            "wins_b": int(match_summary.get("wins_b", 0)),
                            "games_played": int(match_summary.get("games_played", 0)),
                            "score_a": score_a,
                            "rating_a_before": rating_a_before,
                            "rating_b_before": rating_b_before,
                            "rating_a_after": rating_a_after,
                            "rating_b_after": rating_b_after,
                            "duration_seconds": float(match_summary.get("duration_seconds", 0.0)),
                        }
                    )

            candidate_state = self._candidate_state()
            for participant in participants:
                name = str(participant["name"])
                rating = float(rating_map[name])
                if participant.get("kind") == "candidate":
                    candidate_state["rating_pass_elo"] = rating
                    continue
                checkpoint_name = name
                checkpoint_entry = self._find_checkpoint_entry(checkpoint_name)
                if checkpoint_entry is None:
                    checkpoint_entry = _ensure_checkpoint_entry(
                        self.state,
                        checkpoint_name,
                        int(participant.get("iteration", 0)),
                        rating,
                    )
                checkpoint_entry["elo"] = rating

            latest_name = self.state.get("latest_checkpoint")
            latest_entry = self._find_checkpoint_entry(str(latest_name)) if latest_name else None
            if latest_entry is not None:
                self.state["current_elo"] = float(latest_entry.get("elo", self.state.get("current_elo", INITIAL_ELO)))

            checkpoint_entries = self._checkpoint_entries_sorted()
            if checkpoint_entries:
                best_entry = max(
                    checkpoint_entries,
                    key=lambda item: (float(item.get("elo", INITIAL_ELO)), int(item.get("iteration", 0))),
                )
                self.state["best_checkpoint"] = best_entry.get("name")
                self.state["best_elo"] = float(best_entry.get("elo", INITIAL_ELO))

            leaderboard = [
                {
                    "name": str(participant["name"]),
                    "kind": str(participant.get("kind", "checkpoint")),
                    "iteration": int(participant.get("iteration", 0)),
                    "elo": float(rating_map[str(participant["name"])]),
                }
                for participant in participants
            ]
            leaderboard.sort(key=lambda item: (item["elo"], item["iteration"], item["name"]), reverse=True)

            summary = {
                "status": "completed",
                "participant_count": len(participants),
                "participants": [
                    {
                        "name": str(participant["name"]),
                        "kind": str(participant.get("kind", "checkpoint")),
                        "iteration": int(participant.get("iteration", 0)),
                    }
                    for participant in participants
                ],
                "include_candidate": bool(include_candidate),
                "games_per_pair": int(games_per_pair),
                "pairings_played": len(pairings),
                "pairings": pairings,
                "leaderboard": leaderboard,
                "duration_seconds": _timestamp() - started_at,
                "completed_at": _timestamp(),
            }
            self.state["last_rating_pass"] = summary
            self.state["status"] = "idle"
            self.state["last_error"] = None
            self._save()
            return summary
        except Exception as exc:
            self.state["status"] = "idle"
            self.state["last_error"] = f"{type(exc).__name__}: {exc}"
            self.state["last_rating_pass"] = {
                "status": "failed",
                "participant_count": len(participants),
                "games_per_pair": int(games_per_pair),
                "include_candidate": bool(include_candidate),
                "duration_seconds": _timestamp() - started_at,
                "error": self.state["last_error"],
            }
            self._save()
            raise

    def _find_checkpoint_entry(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        for item in self.state.get("checkpoints", []):
            if item["name"] == checkpoint_name:
                return item
        return None

    def _checkpoint_entries_sorted(self) -> List[Dict[str, Any]]:
        return sorted(
            list(self.state.get("checkpoints", [])),
            key=lambda item: (int(item.get("iteration", 0)), item.get("name", "")),
        )

    def _sample_league_opponent(self) -> Tuple[PolicyNetwork, Dict[str, Any]]:
        latest_name = self.state.get("latest_checkpoint")
        best_name = self.state.get("best_checkpoint")
        historical_names = [item["name"] for item in self._checkpoint_entries_sorted() if item.get("name")]
        historical_names = [name for name in historical_names if name != latest_name]

        recent_window = max(0, int(self.config.league_recent_window))
        recent_pool = historical_names[-recent_window:] if recent_window > 0 else []
        older_pool = historical_names[:-recent_window] if recent_window > 0 else historical_names

        choices: List[Tuple[Dict[str, Any], float]] = [
            (
                {
                    "label": "champion",
                    "source": "champion",
                    "checkpoint": latest_name,
                },
                float(self.config.league_champion_weight),
            )
        ]

        if best_name and best_name != latest_name:
            choices.append(
                (
                    {
                        "label": "best",
                        "source": "best",
                        "checkpoint": best_name,
                    },
                    float(self.config.league_best_weight),
                )
            )
        if recent_pool:
            choices.append(
                (
                    {
                        "label": "recent",
                        "source": "recent",
                        "checkpoint": random.choice(recent_pool),
                    },
                    float(self.config.league_recent_weight),
                )
            )
        if older_pool:
            choices.append(
                (
                    {
                        "label": "historical",
                        "source": "historical",
                        "checkpoint": random.choice(older_pool),
                    },
                    float(self.config.league_historical_weight),
                )
            )

        selected = _weighted_choice(choices)
        checkpoint_name = selected.get("checkpoint")
        if selected["source"] == "champion" or checkpoint_name in (None, latest_name):
            return self.policy, selected
        return _cached_policy(self.run_name, checkpoint_name), selected

    def _match_to_training_samples(self, match_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        trajectory_by_game = list(match_summary.get("trajectory_by_game") or [])
        if not trajectory_by_game:
            flat_trajectory = list(match_summary.get("trajectory") or [])
            if flat_trajectory:
                trajectory_by_game = [flat_trajectory]
        if not trajectory_by_game:
            return []
        return_value = match_summary["return_value"]
        decisions_per_game_limit = training_decisions_per_game_limit(self.config.training_decisions_per_game)
        selected_trajectory: List[Dict[str, Any]] = []

        for game_trajectory in trajectory_by_game:
            if not game_trajectory:
                continue
            for step in game_trajectory:
                step["return"] = return_value
                step["advantage"] = return_value - step["value"]
            if decisions_per_game_limit is None or len(game_trajectory) <= decisions_per_game_limit:
                selected_trajectory.extend(game_trajectory)
            else:
                selected_trajectory.extend(random.sample(game_trajectory, decisions_per_game_limit))

        if not selected_trajectory:
            return []

        advantages = [step["advantage"] for step in selected_trajectory]
        advantage_mean = _mean(advantages)
        advantage_std = _std(advantages)
        if advantage_std > 1e-6:
            for step in selected_trajectory:
                step["advantage"] = (step["advantage"] - advantage_mean) / advantage_std
        if len(selected_trajectory) > self.config.max_training_samples_per_match:
            selected_trajectory = random.sample(selected_trajectory, self.config.max_training_samples_per_match)
        return selected_trajectory

    def _summarize_training_matches(self, match_summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not match_summaries:
            return {
                "wins": 0,
                "losses": 0,
                "games_played": 0,
                "return_value": 0.0,
                "duration_seconds": 0.0,
                "matches_played": 0,
                "opponents": [],
            }
        return {
            "wins": sum(int(match.get("wins_a", 0)) for match in match_summaries),
            "losses": sum(int(match.get("wins_b", 0)) for match in match_summaries),
            "games_played": sum(int(match.get("games_played", 0)) for match in match_summaries),
            "return_value": _mean([float(match.get("return_value", 0.0)) for match in match_summaries]),
            "duration_seconds": sum(float(match.get("duration_seconds", 0.0)) for match in match_summaries),
            "matches_played": len(match_summaries),
            "opponents": [match.get("opponent") for match in match_summaries],
        }

    def _record_live_match_progress(
        self,
        match_summary: Dict[str, Any],
        kind: str,
        completed_training_matches: Sequence[Dict[str, Any]],
    ) -> None:
        self.state["total_matches"] = int(self.state.get("total_matches", 0)) + 1
        self.state["total_games"] = int(self.state.get("total_games", 0)) + int(match_summary.get("games_played", 0))
        if kind == "training":
            self.state["last_match"] = self._summarize_training_matches(completed_training_matches)
        else:
            self.state["last_eval"] = {
                "wins": match_summary.get("wins_a", 0),
                "losses": match_summary.get("wins_b", 0),
                "games_played": match_summary.get("games_played", 0),
                "score": float(match_summary.get("wins_a", 0)) / max(int(match_summary.get("games_played", 0)), 1),
                "promoted": False,
                "action": "evaluating",
                "promoted_checkpoint": None,
                "champion_checkpoint": self.state.get("latest_checkpoint"),
                "duration_seconds": match_summary.get("duration_seconds", 0.0),
            }
        self._save()

    def _reseed_candidate(
        self,
        base_iteration: int,
        total_attempts: int,
        total_resets: int,
        total_promotions: int,
        last_score: float,
        last_result: str,
        reason: str,
        opponents: Sequence[Dict[str, Any]],
    ) -> None:
        self.candidate_policy = self.policy.clone()
        candidate_state = _default_candidate_state(
            self.state.get("latest_checkpoint", "checkpoint_000000.json"),
            base_iteration,
        )
        candidate_state["total_attempts"] = total_attempts
        candidate_state["resets"] = total_resets
        candidate_state["promotions"] = total_promotions
        candidate_state["last_score"] = last_score
        candidate_state["last_result"] = last_result
        candidate_state["last_reset_reason"] = reason
        candidate_state["last_opponents"] = list(opponents)
        candidate_state["rating_pass_elo"] = float(self.state.get("current_elo", INITIAL_ELO))
        self.state["candidate"] = candidate_state

    def _promote_candidate(self, iteration: int, eval_summary: Dict[str, Any], training_summary: Dict[str, Any]) -> str:
        previous_checkpoint = self.state.get("latest_checkpoint", "checkpoint_000000.json")
        previous_entry = _ensure_checkpoint_entry(
            self.state,
            previous_checkpoint,
            int(self.state.get("iteration", 0)),
            float(self.state.get("current_elo", INITIAL_ELO)),
        )
        previous_elo = float(previous_entry.get("elo", self.state.get("current_elo", INITIAL_ELO)))
        score = float(eval_summary.get("wins_a", 0)) / max(int(eval_summary.get("games_played", 0)), 1)
        new_candidate_elo, updated_previous_elo = _elo_update(previous_elo, previous_elo, score, self.config.elo_k)
        previous_entry["elo"] = updated_previous_elo
        previous_entry["last_defense"] = eval_summary

        checkpoint_path = _checkpoint_path(self.files, iteration)
        checkpoint_name = checkpoint_path.name
        _atomic_write_json(checkpoint_path, self.candidate_policy.to_dict(include_optimizer=False))

        promoted_entry = _ensure_checkpoint_entry(self.state, checkpoint_name, iteration, new_candidate_elo)
        promoted_entry["elo"] = new_candidate_elo
        promoted_entry["created_at"] = _timestamp()
        promoted_entry["promoted_from"] = previous_checkpoint
        promoted_entry["promotion_eval"] = eval_summary
        promoted_entry["training_summary"] = training_summary

        self.policy = self.candidate_policy.clone()
        self.state["latest_checkpoint"] = checkpoint_name
        self.state["current_elo"] = new_candidate_elo
        self.state["last_promotion_iteration"] = int(iteration)
        if new_candidate_elo >= float(self.state.get("best_elo", INITIAL_ELO)):
            self.state["best_elo"] = new_candidate_elo
            self.state["best_checkpoint"] = checkpoint_name
        return checkpoint_name

    def train_iteration(self) -> Dict[str, Any]:
        scheduled_config = self._scheduled_training_config()
        drought_boost = self._promotion_drought_boost()
        candidate_state = self._candidate_state()
        training_matches: List[Dict[str, Any]] = []
        training_samples: List[Dict[str, Any]] = []
        opponent_summaries: List[Dict[str, Any]] = []

        for _ in range(max(1, int(scheduled_config.training_matches_per_iteration))):
            opponent_policy, opponent_meta = self._sample_league_opponent()
            match_summary = _play_match(
                self.candidate_policy,
                opponent_policy,
                scheduled_config,
                collect_a=True,
                deterministic_a=False,
                deterministic_b=False,
                games_per_match=max(1, int(scheduled_config.training_games_per_match)),
            )
            match_summary["opponent"] = opponent_meta
            training_matches.append(match_summary)
            opponent_summaries.append(opponent_meta)
            training_samples.extend(self._match_to_training_samples(match_summary))
            self._record_live_match_progress(match_summary, "training", training_matches)

        update_stats = self.candidate_policy.train_on_samples(training_samples, scheduled_config)
        training_summary = self._summarize_training_matches(training_matches)
        eval_summary = _play_match(
            self.candidate_policy,
            self.policy,
            scheduled_config,
            collect_a=False,
            deterministic_a=True,
            deterministic_b=True,
            games_per_match=max(1, int(scheduled_config.promotion_games)),
        )
        self._record_live_match_progress(eval_summary, "evaluation", training_matches)
        candidate_score = float(eval_summary["wins_a"]) / max(int(eval_summary["games_played"]), 1)
        promoted = (
            eval_summary["wins_a"] > eval_summary["wins_b"]
            and candidate_score >= float(scheduled_config.promotion_score_threshold)
        )
        defending_champion = self.state.get("latest_checkpoint")

        next_iteration = int(self.state.get("iteration", 0)) + 1
        total_attempts = int(candidate_state.get("total_attempts", 0)) + 1
        total_resets = int(candidate_state.get("resets", 0))
        total_promotions = int(candidate_state.get("promotions", 0))
        action = "keep_training"
        promoted_checkpoint: Optional[str] = None

        self.state["iteration"] = next_iteration
        self.state["last_match"] = training_summary

        self.state["last_update"] = {
            **update_stats,
            "learning_rate": scheduled_config.learning_rate,
            "epsilon_random": scheduled_config.epsilon_random,
            "train_temperature": scheduled_config.train_temperature,
            "ppo_clip": scheduled_config.ppo_clip,
            "iterations_since_promotion": int(drought_boost["iterations_since_promotion"]),
            "promotion_drought_progress": round(float(drought_boost["progress"]), 4),
            "learning_rate_multiplier": round(float(drought_boost["learning_rate_multiplier"]), 4),
            "epsilon_multiplier": round(float(drought_boost["epsilon_multiplier"]), 4),
            "temperature_multiplier": round(float(drought_boost["temperature_multiplier"]), 4),
            "matches_collected": len(training_matches),
        }

        if promoted:
            promoted_checkpoint = self._promote_candidate(next_iteration, eval_summary, training_summary)
            total_promotions += 1
            self.state["promotions"] = int(self.state.get("promotions", 0)) + 1
            action = "promoted"
            self._reseed_candidate(
                next_iteration,
                total_attempts,
                total_resets,
                total_promotions,
                candidate_score,
                "promoted",
                f"Promoted to {promoted_checkpoint}",
                opponent_summaries,
            )
        else:
            attempts_since_reset = int(candidate_state.get("attempts_since_reset", 0)) + 1
            should_reset = (
                candidate_score <= float(scheduled_config.candidate_reset_threshold)
                or attempts_since_reset >= max(1, int(scheduled_config.candidate_patience))
            )
            if should_reset:
                total_resets += 1
                reason = (
                    f"Candidate reset after score {candidate_score:.3f} "
                    f"and {attempts_since_reset} attempt(s) since the last reset."
                )
                action = "reset_candidate"
                self._reseed_candidate(
                    next_iteration,
                    total_attempts,
                    total_resets,
                    total_promotions,
                    candidate_score,
                    "reset",
                    reason,
                    opponent_summaries,
                )
            else:
                candidate_state["attempts_since_reset"] = attempts_since_reset
                candidate_state["total_attempts"] = total_attempts
                candidate_state["last_score"] = candidate_score
                candidate_state["last_result"] = "keep_training"
                candidate_state["last_opponents"] = opponent_summaries

        self.state["last_eval"] = {
            "wins": eval_summary["wins_a"],
            "losses": eval_summary["wins_b"],
            "games_played": eval_summary["games_played"],
            "score": candidate_score,
            "promoted": promoted,
            "action": action,
            "promoted_checkpoint": promoted_checkpoint,
            "champion_checkpoint": defending_champion,
            "duration_seconds": eval_summary["duration_seconds"],
        }
        self.state["last_error"] = None
        self._save()
        return {
            "match": training_summary,
            "evaluation": self.state["last_eval"],
            "update": update_stats,
            "iteration": self.state["iteration"],
            "current_elo": self.state.get("current_elo", INITIAL_ELO),
            "action": action,
        }

    def _training_loop(self) -> None:
        with self._lock:
            self.state["status"] = "training"
            self.state["last_error"] = None
            self._save()
        target_iterations = self.config.max_iterations
        try:
            while not self._stop_event.is_set():
                if target_iterations is not None and self.state["iteration"] >= target_iterations:
                    break
                self.train_iteration()
        except Exception as exc:
            with self._lock:
                self.state["status"] = "error"
                self.state["last_error"] = f"{type(exc).__name__}: {exc}"
                self._save()
        finally:
            with self._lock:
                if self.state.get("status") != "error":
                    self.state["status"] = "idle"
                self._save()

    def start(self, background: bool = True) -> Dict[str, Any]:
        with self._lock:
            if self.is_running:
                return {"started": False, "message": "training already running", "status": self.state["status"]}
            self._stop_event.clear()
            if background:
                self._thread = threading.Thread(
                    target=self._training_loop,
                    name=f"starrealms-selfplay-{self.run_name}",
                    daemon=True,
                )
                self._thread.start()
                return {"started": True, "background": True, "run_name": self.run_name}
        self._training_loop()
        return {"started": True, "background": False, "run_name": self.run_name}

    def interrupt(self) -> Dict[str, Any]:
        if not self.is_running:
            self.state["status"] = "idle"
            self._save()
            return {"stopped": False, "message": "training was not running"}
        self.state["status"] = "stop_requested"
        self._save()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._thread = None
        return {"stopped": True, "message": "training stopped after the current match"}

    def progress_summary(self) -> Dict[str, Any]:
        runtime = 0.0
        if "created_at" in self.state and self.state["created_at"] is not None:
            runtime = _timestamp() - float(self.state["created_at"])
        drought_boost = self._promotion_drought_boost()
        scheduled_config = self._scheduled_training_config()
        return {
            "run_name": self.run_name,
            "status": self.state.get("status", "idle"),
            "iteration": self.state.get("iteration", 0),
            "total_matches": self.state.get("total_matches", 0),
            "total_games": self.state.get("total_games", 0),
            "current_elo": round(float(self.state.get("current_elo", INITIAL_ELO)), 2),
            "best_elo": round(float(self.state.get("best_elo", INITIAL_ELO)), 2),
            "best_checkpoint": self.state.get("best_checkpoint"),
            "latest_checkpoint": self.state.get("latest_checkpoint"),
            "promotions": int(self.state.get("promotions", 0)),
            "candidate": self.state.get("candidate"),
            "last_match": self.state.get("last_match"),
            "last_update": self.state.get("last_update"),
            "last_eval": self.state.get("last_eval"),
            "last_rating_pass": self.state.get("last_rating_pass"),
            "last_error": self.state.get("last_error"),
            "runtime": _format_seconds(runtime),
            "run_dir": str(self.files.run_dir),
            "device_backend": getattr(self.policy, "device_backend", DEVICE_CPU),
            "device_repr": str(getattr(self.policy, "device", "cpu")),
            "device_preference": self.config.device_preference,
            "last_promotion_iteration": self._last_promotion_iteration(),
            "iterations_since_promotion": int(drought_boost["iterations_since_promotion"]),
            "promotion_drought_progress": round(float(drought_boost["progress"]), 4),
            "learning_rate_multiplier": round(float(drought_boost["learning_rate_multiplier"]), 4),
            "epsilon_multiplier": round(float(drought_boost["epsilon_multiplier"]), 4),
            "temperature_multiplier": round(float(drought_boost["temperature_multiplier"]), 4),
            "scheduled_learning_rate": float(scheduled_config.learning_rate),
            "scheduled_epsilon_random": float(scheduled_config.epsilon_random),
            "scheduled_train_temperature": float(scheduled_config.train_temperature),
        }


_TRAINERS: Dict[str, SelfPlayTrainer] = {}
_POLICY_CACHE: Dict[Tuple[str, str], Tuple[float, PolicyNetwork]] = {}


def _get_trainer(run_name: str = LATEST_RUN_NAME, config_overrides: Optional[Dict[str, Any]] = None) -> SelfPlayTrainer:
    trainer = _TRAINERS.get(run_name)
    if trainer is None or (config_overrides and trainer.config != trainer.config.merged(config_overrides)):
        trainer = SelfPlayTrainer(run_name, config_overrides=config_overrides)
        _TRAINERS[run_name] = trainer
    return trainer


def summarize_start_training(run_name: str = LATEST_RUN_NAME, **config_overrides: Any) -> str:
    trainer = _get_trainer(run_name, config_overrides or None)
    started = trainer.start(background=True)
    summary = trainer.progress_summary()
    if started.get("started") and started.get("background"):
        return (
            f"Started background training for '{run_name}'. "
            f"Iteration {summary['iteration']}, status {summary['status']}, policy dir {summary['run_dir']}."
        )
    return f"Training for '{run_name}' is already running."


def start_training(
    run_name: str = LATEST_RUN_NAME,
    background: bool = True,
    **config_overrides: Any,
) -> Dict[str, Any]:
    trainer = _get_trainer(run_name, config_overrides or None)
    trainer.start(background=background)
    return trainer.progress_summary()


def summarize_training_progress(run_name: str = LATEST_RUN_NAME) -> str:
    trainer = _get_trainer(run_name)
    summary = trainer.progress_summary()
    last_match = summary.get("last_match") or {}
    last_eval = summary.get("last_eval") or {}
    match_text = "no completed matches yet"
    if last_match:
        match_text = (
            f"last match {last_match.get('wins', 0)}-{last_match.get('losses', 0)} "
            f"over {last_match.get('games_played', 0)} games"
        )
    eval_text = ""
    if last_eval:
        eval_text = (
            f" Last eval {last_eval.get('wins', 0)}-{last_eval.get('losses', 0)} "
            f"({last_eval.get('action', 'keep_training')})."
        )
    boost_text = ""
    if float(summary.get("promotion_drought_progress", 0.0)) > 0.0:
        boost_text = (
            f" Exploration boost active after {summary.get('iterations_since_promotion', 0)} "
            f"non-promotion iteration(s): lr x{summary.get('learning_rate_multiplier', 1.0)}, "
            f"eps x{summary.get('epsilon_multiplier', 1.0)}, temp x{summary.get('temperature_multiplier', 1.0)}."
        )
    return (
        f"Run '{summary['run_name']}' is {summary['status']}. "
        f"Iteration {summary['iteration']}, total matches {summary['total_matches']}, "
        f"Elo {summary['current_elo']} (best {summary['best_elo']}), promotions {summary.get('promotions', 0)}, "
        f"device {summary.get('device_backend', DEVICE_CPU)}, "
        f"{match_text}.{eval_text}{boost_text} "
        f"Artifacts: {summary['run_dir']}."
    )


def training_progress(run_name: str = LATEST_RUN_NAME) -> Dict[str, Any]:
    return _get_trainer(run_name).progress_summary()


def interrupt_training(run_name: str = LATEST_RUN_NAME) -> str:
    trainer = _get_trainer(run_name)
    result = trainer.interrupt()
    if result["stopped"]:
        return f"Training for '{run_name}' has stopped. Latest weights are saved in {trainer.files.run_dir}."
    return f"Training for '{run_name}' was not running."


def continue_training(run_name: str = LATEST_RUN_NAME, **config_overrides: Any) -> str:
    trainer = _get_trainer(run_name, config_overrides or None)
    trainer.start(background=True)
    return summarize_training_progress(run_name)


def train_iterations(
    iterations: int,
    run_name: str = LATEST_RUN_NAME,
    **config_overrides: Any,
) -> Dict[str, Any]:
    trainer = _get_trainer(run_name, config_overrides or None)
    original_max = trainer.config.max_iterations
    trainer.config.max_iterations = trainer.state.get("iteration", 0) + max(0, iterations)
    try:
        trainer.start(background=False)
        return trainer.progress_summary()
    finally:
        trainer.config.max_iterations = original_max


def create_run(
    run_name: str,
    model_type: str = MODEL_TYPE_DEEP,
    device_preference: str = DEVICE_AUTO,
    training_matches_per_iteration: int = 5,
    training_games_per_match: int = 16,
    training_decisions_per_game: str = TRAINING_DECISIONS_PER_GAME_ALL,
    train_temperature: float = 0.9,
    promotion_games: int = 24,
    promotion_score_threshold: float = 0.6,
) -> Dict[str, Any]:
    if not str(run_name).strip():
        raise ValueError("run_name is required.")
    if run_exists(run_name):
        raise ValueError(f"Run '{run_name}' already exists.")

    overrides = new_run_overrides(
        model_type=model_type,
        device_preference=device_preference,
        training_matches_per_iteration=training_matches_per_iteration,
        training_games_per_match=training_games_per_match,
        training_decisions_per_game=training_decisions_per_game,
        train_temperature=train_temperature,
        promotion_games=promotion_games,
        promotion_score_threshold=promotion_score_threshold,
    )
    _ensure_run(run_name, config_overrides=overrides)
    trainer = _get_trainer(run_name, overrides)
    return trainer.progress_summary()


def create_run_from_checkpoint(
    run_name: str,
    source_run_name: str,
    source_checkpoint: str = "latest",
    model_type: str = MODEL_TYPE_DEEP,
    device_preference: str = DEVICE_AUTO,
    training_matches_per_iteration: int = 5,
    training_games_per_match: int = 16,
    training_decisions_per_game: str = TRAINING_DECISIONS_PER_GAME_ALL,
    train_temperature: float = 0.9,
    promotion_games: int = 24,
    promotion_score_threshold: float = 0.6,
) -> Dict[str, Any]:
    if not str(run_name).strip():
        raise ValueError("run_name is required.")
    if run_exists(run_name):
        raise ValueError(f"Run '{run_name}' already exists.")
    if not run_exists(source_run_name):
        raise ValueError(f"Source run '{source_run_name}' does not exist.")

    source_policy = load_policy(source_run_name, source_checkpoint)
    source_state = get_run_state(source_run_name)
    resolved_source_checkpoint = _resolve_checkpoint_name(source_state, source_checkpoint) or source_checkpoint
    target_model = model_type_overrides(model_type)

    config = TrainingConfig().merged(
        {
            "hidden_size": int(target_model["hidden_size"]),
            "model_architecture": str(target_model["model_architecture"]),
            "device_preference": str(device_preference or DEVICE_AUTO).strip().lower(),
            "training_matches_per_iteration": int(training_matches_per_iteration),
            "training_games_per_match": int(training_games_per_match),
            "training_decisions_per_game": normalize_training_decisions_per_game(training_decisions_per_game),
            "promotion_games": int(promotion_games),
            "promotion_score_threshold": normalize_promotion_score_threshold(promotion_score_threshold),
        }
    )
    config = config.merged(derive_temperature_schedule_overrides(train_temperature))

    if config.training_matches_per_iteration <= 0:
        raise ValueError("training_matches_per_iteration must be positive.")
    if config.training_games_per_match <= 0:
        raise ValueError("training_games_per_match must be positive.")
    if config.promotion_games <= 0:
        raise ValueError("promotion_games must be positive.")

    files = RunFiles(run_name)
    files.run_dir.mkdir(parents=True, exist_ok=True)
    files.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    fork_policy, conversion_mode = _convert_policy_architecture(
        source_policy,
        target_architecture=config.model_architecture,
        target_hidden_size=config.hidden_size,
        target_device_preference=config.device_preference,
    )
    fork_policy.optimizer = torch.optim.Adam(fork_policy.model.parameters(), lr=config.learning_rate)

    state = _default_training_state(config)
    state["run_name"] = run_name
    state["forked_from"] = {
        "run_name": source_run_name,
        "checkpoint": source_checkpoint,
        "resolved_checkpoint": resolved_source_checkpoint,
        "source_architecture": source_policy.architecture,
        "source_hidden_size": int(source_policy.hidden_size),
        "target_architecture": config.model_architecture,
        "target_hidden_size": int(config.hidden_size),
        "requested_model_type": str(model_type or MODEL_TYPE_DEEP).strip().lower(),
        "conversion_mode": conversion_mode,
        "created_at": _timestamp(),
    }
    if state.get("checkpoints"):
        state["checkpoints"][0]["note"] = f"forked from {source_run_name}/{source_checkpoint}"
        state["checkpoints"][0]["source_checkpoint"] = resolved_source_checkpoint

    _atomic_write_json(files.policy_file, fork_policy.to_dict(include_optimizer=True))
    _atomic_write_json(files.candidate_policy_file, fork_policy.to_dict(include_optimizer=True))
    _atomic_write_json(files.training_state_file, state)
    _atomic_write_json(files.checkpoints_dir / "checkpoint_000000.json", fork_policy.to_dict(include_optimizer=False))

    trainer = SelfPlayTrainer(run_name)
    _TRAINERS[run_name] = trainer
    return trainer.progress_summary()


def update_run_config(
    run_name: str,
    train_temperature: Optional[float] = None,
    promotion_score_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    if not run_exists(run_name):
        raise ValueError(f"Run '{run_name}' does not exist.")

    trainer = _get_trainer(run_name)
    if trainer.is_running or str(trainer.state.get("status", "idle")).lower() == "running":
        raise RuntimeError(f"Run '{run_name}' is currently training. Interrupt it before changing run settings.")

    overrides: Dict[str, Any] = {}
    if train_temperature is not None:
        overrides.update(derive_temperature_schedule_overrides(train_temperature))
    if promotion_score_threshold is not None:
        overrides["promotion_score_threshold"] = normalize_promotion_score_threshold(promotion_score_threshold)
    if not overrides:
        return trainer.progress_summary()

    trainer.config = trainer.config.merged(overrides)
    trainer._save()
    return trainer.progress_summary()


def run_rating_pass(
    run_name: str = LATEST_RUN_NAME,
    max_policies: int = 8,
    games_per_pair: int = 12,
    include_candidate: bool = True,
) -> Dict[str, Any]:
    trainer = _get_trainer(run_name)
    return trainer.run_rating_pass(
        max_policies=max_policies,
        games_per_pair=games_per_pair,
        include_candidate=include_candidate,
    )


def summarize_rating_pass(
    run_name: str = LATEST_RUN_NAME,
    max_policies: int = 8,
    games_per_pair: int = 12,
    include_candidate: bool = True,
) -> str:
    summary = run_rating_pass(
        run_name=run_name,
        max_policies=max_policies,
        games_per_pair=games_per_pair,
        include_candidate=include_candidate,
    )
    top_entry = (summary.get("leaderboard") or [{}])[0]
    return (
        f"Rating pass for '{run_name}' completed with {summary.get('participant_count', 0)} models "
        f"and {summary.get('pairings_played', 0)} pairings. "
        f"Top rated: {top_entry.get('name', '-')} at {round(float(top_entry.get('elo', INITIAL_ELO)), 2)} Elo."
    )


def _initial_card_elo_ratings() -> Dict[str, float]:
    return {card_name: INITIAL_ELO for card_name in CARD_NAME_ORDER}


def _apply_card_acquire_elo_result(
    ratings: Dict[str, float],
    winner_name: str,
    loser_names: Sequence[str],
    k_factor: float,
) -> int:
    winner_name = str(winner_name or "").strip()
    if not winner_name:
        return 0

    unique_losers: List[str] = []
    seen_losers = set()
    for loser_name in loser_names:
        normalized_name = str(loser_name or "").strip()
        if not normalized_name or normalized_name == winner_name or normalized_name in seen_losers:
            continue
        seen_losers.add(normalized_name)
        unique_losers.append(normalized_name)

    if not unique_losers:
        return 0

    ratings.setdefault(winner_name, INITIAL_ELO)
    winner_rating = float(ratings[winner_name])
    pair_k = float(k_factor) / max(len(unique_losers), 1)
    winner_delta = 0.0
    loser_deltas: Dict[str, float] = {}

    for loser_name in unique_losers:
        ratings.setdefault(loser_name, INITIAL_ELO)
        loser_rating = float(ratings[loser_name])
        expected_winner = _elo_expected(winner_rating, loser_rating)
        delta = pair_k * (1.0 - expected_winner)
        winner_delta += delta
        loser_deltas[loser_name] = loser_deltas.get(loser_name, 0.0) - delta

    ratings[winner_name] = winner_rating + winner_delta
    for loser_name, delta in loser_deltas.items():
        ratings[loser_name] = float(ratings[loser_name]) + delta

    return len(unique_losers)


def _normalized_card_elo_ratings(raw_ratings: Dict[str, float]) -> Tuple[Dict[str, float], float, float]:
    explorer_rating = float(raw_ratings.get("Explorer", INITIAL_ELO))
    if abs(explorer_rating) <= 1e-9:
        normalization_factor = 1.0
    else:
        normalization_factor = explorer_rating / 200.0
    normalized = {
        card_name: float(rating) / normalization_factor
        for card_name, rating in raw_ratings.items()
    }
    return normalized, normalization_factor, explorer_rating


def format_card_acquire_elo_test_report(result: Dict[str, Any]) -> str:
    leaderboard = list(result.get("leaderboard") or [])
    lines = [
        f"Card Acquire Elo Test for {result.get('run_name', '-')}/{result.get('checkpoint', '-')}",
        f"Resolved checkpoint: {result.get('resolved_checkpoint', '-')}",
        f"Games: {result.get('games', 0)}",
        f"Deterministic: {result.get('deterministic', True)}",
        f"Ended by limit: {result.get('ended_by_limit_games', 0)} game(s)",
        f"Turn summaries: {result.get('turn_summaries', 0)}",
        f"Eligible single-acquire turns: {result.get('eligible_single_acquire_turns', 0)}",
        f"Scored decisions: {result.get('scored_decisions', 0)}",
        f"Pairwise comparisons: {result.get('pairwise_comparisons', 0)}",
        f"Explorer raw Elo: {round(float(result.get('explorer_raw_elo', INITIAL_ELO)), 4)}",
        f"Normalization factor: {round(float(result.get('normalization_factor', 1.0)), 6)}",
        f"Duration: {_format_seconds(float(result.get('duration_seconds', 0.0)))}",
        "",
        "Card Rankings",
    ]
    for index, entry in enumerate(leaderboard, start=1):
        lines.append(f"{index:>2}. {entry.get('card_name', '-'):<20} {float(entry.get('elo', 0.0)):>8.2f}")
    return "\n".join(lines)


def run_card_acquire_elo_test(
    run_name: str = LATEST_RUN_NAME,
    checkpoint: str = "latest",
    games: int = CARD_ACQUIRE_ELO_TEST_GAMES,
    deterministic: bool = True,
    k_factor: float = CARD_ACQUIRE_ELO_K_FACTOR,
) -> Dict[str, Any]:
    if games <= 0:
        raise ValueError("games must be positive.")
    if not math.isfinite(float(k_factor)) or float(k_factor) <= 0.0:
        raise ValueError("k_factor must be greater than 0.")

    trainer = _get_trainer(run_name)
    config = trainer.config
    state = trainer.state
    resolved_checkpoint = _resolve_checkpoint_name(state, checkpoint) or str(checkpoint or "latest")
    policy = load_policy(run_name, checkpoint)
    temperature = config.eval_temperature if deterministic else config.train_temperature
    epsilon_random = 0.0 if deterministic else config.epsilon_random
    start_time = time.time()

    ratings = _initial_card_elo_ratings()
    turn_summaries = 0
    eligible_single_acquire_turns = 0
    scored_decisions = 0
    pairwise_comparisons = 0
    ended_by_limit_games = 0

    for _ in range(int(games)):
        actor_a = PolicyActor(
            policy,
            deterministic=deterministic,
            temperature=temperature,
            epsilon_random=epsilon_random,
        )
        actor_b = PolicyActor(
            policy,
            deterministic=deterministic,
            temperature=temperature,
            epsilon_random=epsilon_random,
        )
        game_turn_summaries: List[Dict[str, Any]] = []

        def capture_turn_summary(summary: Dict[str, Any]) -> None:
            game_turn_summaries.append(_copy_nested(summary))

        game = Game(
            "policy_a",
            "policy_b",
            p1_choose=actor_a.choose,
            p2_choose=actor_b.choose,
            verbose=False,
            max_turns=config.max_turns_per_game,
            max_actions_per_turn=config.max_actions_per_turn,
            turn_summary_callback=capture_turn_summary,
        )
        if game.ended_by_limit:
            ended_by_limit_games += 1

        for turn_summary in game_turn_summaries:
            turn_summaries += 1
            acquisition_events = list(turn_summary.get("acquisitionEvents") or [])
            total_acquisitions = int(turn_summary.get("totalAcquisitions", len(acquisition_events)))
            if total_acquisitions != 1 or len(acquisition_events) != 1:
                continue

            event = acquisition_events[0]
            if str(event.get("type", "")) != "acquire":
                continue

            eligible_single_acquire_turns += 1
            winner_name = str(event.get("cardName", "")).strip()
            if not winner_name:
                continue

            total_trade_gained = float(turn_summary.get("totalTradeGained", 0.0))
            loser_names: List[str] = []
            seen_losers = set()
            for trade_card in list(event.get("tradeRowSnapshot") or []):
                if not isinstance(trade_card, (list, tuple)) or len(trade_card) < 2:
                    continue
                card_name = str(trade_card[0] or "").strip()
                if not card_name or card_name == winner_name or card_name in seen_losers:
                    continue
                card_cost = CARD_COST_BY_NAME.get(card_name)
                if card_cost is None:
                    try:
                        card_cost = int(trade_card[1])
                    except (TypeError, ValueError):
                        continue
                if float(card_cost) > total_trade_gained:
                    continue
                seen_losers.add(card_name)
                loser_names.append(card_name)

            comparisons = _apply_card_acquire_elo_result(ratings, winner_name, loser_names, float(k_factor))
            if comparisons <= 0:
                continue
            scored_decisions += 1
            pairwise_comparisons += comparisons

    normalized_ratings, normalization_factor, explorer_raw_elo = _normalized_card_elo_ratings(ratings)
    leaderboard = [
        {
            "card_name": card_name,
            "elo": round(float(normalized_ratings.get(card_name, 0.0)), 4),
            "raw_elo": round(float(ratings.get(card_name, INITIAL_ELO)), 4),
        }
        for card_name in ratings.keys()
    ]
    leaderboard.sort(key=lambda entry: (-float(entry["elo"]), str(entry["card_name"])))

    duration_seconds = time.time() - start_time
    result: Dict[str, Any] = {
        "run_name": run_name,
        "checkpoint": checkpoint,
        "resolved_checkpoint": resolved_checkpoint,
        "games": int(games),
        "deterministic": bool(deterministic),
        "k_factor": float(k_factor),
        "ended_by_limit_games": ended_by_limit_games,
        "turn_summaries": turn_summaries,
        "eligible_single_acquire_turns": eligible_single_acquire_turns,
        "scored_decisions": scored_decisions,
        "pairwise_comparisons": pairwise_comparisons,
        "normalization_factor": normalization_factor,
        "explorer_raw_elo": explorer_raw_elo,
        "leaderboard": leaderboard,
        "duration_seconds": duration_seconds,
    }
    report_text = format_card_acquire_elo_test_report(result)
    result["report_text"] = report_text

    files = trainer.files
    files.analysis_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_slug = _safe_slug(resolved_checkpoint)
    timestamp_slug = time.strftime("%Y%m%d-%H%M%S")
    report_stem = f"card_acquire_elo_{checkpoint_slug}_{timestamp_slug}"
    report_path = files.analysis_dir / f"{report_stem}.txt"
    json_path = files.analysis_dir / f"{report_stem}.json"
    result["report_path"] = str(report_path.resolve())
    result["json_path"] = str(json_path.resolve())
    report_path.write_text(report_text, encoding="utf-8")
    _atomic_write_json(json_path, result)
    return result


def play_self_game(
    run_name: str = LATEST_RUN_NAME,
    checkpoint: str = "latest",
    deterministic: bool = True,
    verbose: bool = True,
    ui_observer: Optional[Any] = None,
) -> Dict[str, Any]:
    policy_a = load_policy(run_name, checkpoint)
    policy_b = load_policy(run_name, checkpoint)
    config = _get_trainer(run_name).config
    if ui_observer is not None:
        ui_observer.start_session(f"Star Realms Self-Play: {run_name} / {checkpoint}")
    actor_a = PolicyActor(
        policy_a,
        deterministic=deterministic,
        temperature=config.eval_temperature if deterministic else config.train_temperature,
        epsilon_random=0.0 if deterministic else config.epsilon_random,
        decision_callback=None if ui_observer is None else ui_observer.policy_choice,
    )
    actor_b = PolicyActor(
        policy_b,
        deterministic=deterministic,
        temperature=config.eval_temperature if deterministic else config.train_temperature,
        epsilon_random=0.0 if deterministic else config.epsilon_random,
        decision_callback=None if ui_observer is None else ui_observer.policy_choice,
    )
    game = Game(
        "policy_a",
        "policy_b",
        p1_choose=actor_a.choose,
        p2_choose=actor_b.choose,
        verbose=verbose,
        max_turns=config.max_turns_per_game,
        max_actions_per_turn=config.max_actions_per_turn,
    )
    result = {
        "winner": game.winner.name,
        "checkpoint": checkpoint,
        "run_name": run_name,
        "ended_by_limit": game.ended_by_limit,
        "turns_taken": game.turnsTaken,
    }
    if ui_observer is not None:
        ui_observer.finish_session(result)
    return result


def play_policy_match(
    run_name_a: str,
    checkpoint_a: str = "latest",
    run_name_b: Optional[str] = None,
    checkpoint_b: str = "latest",
    games_per_match: int = 24,
    deterministic: bool = True,
) -> Dict[str, Any]:
    if not str(run_name_a).strip():
        raise ValueError("run_name_a is required.")
    resolved_run_b = str(run_name_b or run_name_a).strip() or run_name_a
    if games_per_match <= 0:
        raise ValueError("games_per_match must be positive.")

    policy_a = load_policy(run_name_a, checkpoint_a)
    policy_b = load_policy(resolved_run_b, checkpoint_b)
    config_a = _get_trainer(run_name_a).config
    config_b = _get_trainer(resolved_run_b).config
    config = config_a.merged(
        {
            "max_turns_per_game": max(int(config_a.max_turns_per_game), int(config_b.max_turns_per_game)),
            "max_actions_per_turn": max(int(config_a.max_actions_per_turn), int(config_b.max_actions_per_turn)),
            "eval_temperature": min(float(config_a.eval_temperature), float(config_b.eval_temperature)),
        }
    )
    summary = _play_balanced_match(
        policy_a,
        policy_b,
        config,
        games_per_match=max(1, int(games_per_match)),
    )
    wins_a = int(summary.get("wins_a", 0))
    wins_b = int(summary.get("wins_b", 0))
    if wins_a > wins_b:
        winner = "policy_a"
    elif wins_b > wins_a:
        winner = "policy_b"
    else:
        winner = "draw"
    return {
        "policy_a": {"run_name": run_name_a, "checkpoint": checkpoint_a},
        "policy_b": {"run_name": resolved_run_b, "checkpoint": checkpoint_b},
        "games_per_match": int(games_per_match),
        "deterministic": bool(deterministic),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "games_played": int(summary.get("games_played", 0)),
        "score_a": wins_a / max(int(summary.get("games_played", 0)), 1),
        "score_b": wins_b / max(int(summary.get("games_played", 0)), 1),
        "winner": winner,
        "duration_seconds": float(summary.get("duration_seconds", 0.0)),
        "seat_results": list(summary.get("seat_results", [])),
    }


def play_human_game(
    run_name: str = LATEST_RUN_NAME,
    checkpoint: str = "latest",
    human_name: str = "Human",
    policy_name: str = "Policy",
    human_choose_fn: Optional[Any] = None,
    verbose: bool = True,
    ui_observer: Optional[Any] = None,
) -> Dict[str, Any]:
    policy = load_policy(run_name, checkpoint)
    config = _get_trainer(run_name).config
    if ui_observer is not None:
        ui_observer.start_session(f"Star Realms Human Game: {run_name} / {checkpoint}")
    actor = PolicyActor(
        policy,
        deterministic=True,
        temperature=config.eval_temperature,
        epsilon_random=0.0,
        decision_callback=None if ui_observer is None else ui_observer.policy_choice,
    )
    chooser_fn = human_choose if human_choose_fn is None else human_choose_fn
    game = Game(
        human_name,
        policy_name,
        p1_choose=chooser_fn,
        p2_choose=actor.choose,
        verbose=verbose,
        max_turns=config.max_turns_per_game,
        max_actions_per_turn=config.max_actions_per_turn,
    )
    result = {
        "winner": game.winner.name,
        "checkpoint": checkpoint,
        "run_name": run_name,
        "ended_by_limit": game.ended_by_limit,
        "turns_taken": game.turnsTaken,
    }
    if ui_observer is not None:
        ui_observer.finish_session(result)
    return result


def _cached_policy(run_name: str, checkpoint: str) -> PolicyNetwork:
    files, _, state, _ = _load_policy_and_state(run_name)
    if checkpoint == "latest":
        path = files.policy_file
    elif checkpoint == "candidate":
        path = files.candidate_policy_file
    elif checkpoint == "best":
        path = files.checkpoints_dir / state.get("best_checkpoint", state.get("latest_checkpoint"))
    else:
        path = files.checkpoints_dir / checkpoint
    mtime = path.stat().st_mtime if path.exists() else 0.0
    cache_key = (run_name, checkpoint)
    cached = _POLICY_CACHE.get(cache_key)
    if cached and cached[0] == mtime:
        return cached[1]
    policy = load_policy(run_name, checkpoint)
    _POLICY_CACHE[cache_key] = (mtime, policy)
    return policy


def choose_with_saved_policy(
    player_name: str,
    options: Sequence[Sequence[Any]],
    known_game_state: Dict[str, Any],
    run_name: str = LATEST_RUN_NAME,
    checkpoint: str = "latest",
    deterministic: bool = True,
) -> int:
    policy = _cached_policy(run_name, checkpoint)
    config = _get_trainer(run_name).config
    actor = PolicyActor(
        policy,
        deterministic=deterministic,
        temperature=config.eval_temperature if deterministic else config.train_temperature,
        epsilon_random=0.0 if deterministic else config.epsilon_random,
        collector=None,
    )
    return actor.choose(player_name, options, known_game_state)
