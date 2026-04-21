"""Resumable self-play training for the Star Realms simulator.

This module trains a lightweight PPO-style actor-critic without external ML
dependencies. The policy learns through the existing `choose(player, options,
state) -> int` interface used by `sim.py`.
"""

from __future__ import annotations

import copy
import json
import math
import random
import threading
import time
from dataclasses import asdict, dataclass
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


ROOT_DIR = Path(__file__).resolve().parent
RUNS_DIR = ROOT_DIR / "starrealms_policies"
LATEST_RUN_NAME = "default"
INITIAL_ELO = 1000.0
CARD_ZONE_SCALE = 15.0
NUMERIC_OPTION_SCALE = 10.0
EPSILON = 1e-12
NOOP_ACTIONS = {"endTurn", "nodiscard", "nokill", "noRowScrap", "noScrapFromHand", "nocopy"}


def _normalize_ability(raw_ability: Any) -> str:
    if not isinstance(raw_ability, str) or raw_ability == "":
        return "none"
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


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


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
        return list(zone)
    return []


def _add_single_card_features(features: List[float], card: Tuple[Any, ...], scale: float) -> None:
    if card is None:
        return
    features[0] += 1.0 * scale
    features[1] += (card[1] / 8.0) * scale
    features[2] += (card[2] / 8.0) * scale
    features[3] += (card[3] / 8.0) * scale
    features[4] += (card[4] / 8.0) * scale
    features[5] += (card[12] / 8.0) * scale
    features[6] += (card[9] / 5.0) * scale
    features[7] += (card[11] / 5.0) * scale

    faction_offset = 8
    if card[5] in FACTION_TO_INDEX:
        features[faction_offset + FACTION_TO_INDEX[card[5]]] += scale

    type_offset = faction_offset + len(factions)
    if card[6] in CARD_TYPE_TO_INDEX:
        features[type_offset + CARD_TYPE_TO_INDEX[card[6]]] += scale

    ability_offset = type_offset + len(CARD_TYPE_TO_INDEX)
    for raw_ability in (card[7], card[8], card[10]):
        ability_name = _normalize_ability(raw_ability)
        if ability_name in ABILITY_TO_INDEX:
            features[ability_offset + ABILITY_TO_INDEX[ability_name]] += scale

    option_offset = ability_offset + len(ABILITY_FAMILIES)
    if isinstance(card[7], str) and card[7].startswith("-"):
        features[option_offset] += scale
    if isinstance(card[8], str) and card[8].startswith("-"):
        features[option_offset + 1] += scale


def _single_card_features(card: Optional[Tuple[Any, ...]]) -> List[float]:
    features = _zero_vector(CARD_FEATURE_SIZE)
    if card is not None:
        _add_single_card_features(features, card, 1.0)
    return features


def _zone_features(zone: Any) -> List[float]:
    features = _zero_vector(ZONE_FEATURE_SIZE)
    for item in _iter_zone_cards(zone):
        card = _extract_card(item)
        if card is not None:
            _add_single_card_features(features, card, 1.0 / CARD_ZONE_SCALE)
        if _is_play_card(item):
            option_state = item[2]
            if option_state not in (None, "used"):
                features[CARD_FEATURE_SIZE] += 1.0 / CARD_ZONE_SCALE
            if option_state == "used":
                features[CARD_FEATURE_SIZE + 1] += 1.0 / CARD_ZONE_SCALE
            if item[1]:
                features[CARD_FEATURE_SIZE + 2] += 1.0 / CARD_ZONE_SCALE
            if item[4]:
                features[CARD_FEATURE_SIZE + 3] += 1.0 / CARD_ZONE_SCALE
    return features


def state_to_vector(state: Dict[str, Any], legal_option_count: int = 0) -> List[float]:
    zones = [
        state.get("hand"),
        state.get("discardPile"),
        state.get("scrambleDeck"),
        state.get("topCards"),
        state.get("cardsInPlay"),
        state.get("tradeRow"),
        state.get("opponentDiscardPile"),
        state.get("opponentScrambleDeckAndHand"),
        state.get("opponentTopCards"),
        state.get("opponentHandCards"),
        state.get("opponentCardsInPlay"),
    ]
    features: List[float] = []
    for zone in zones:
        features.extend(_zone_features(zone))

    hand_cards = _iter_zone_cards(state.get("hand"))
    discard_cards = _iter_zone_cards(state.get("discardPile"))
    deck_cards = _iter_zone_cards(state.get("scrambleDeck")) + _iter_zone_cards(state.get("topCards"))
    in_play_cards = _iter_zone_cards(state.get("cardsInPlay"))
    opponent_known_hand = _iter_zone_cards(state.get("opponentHandCards"))
    opponent_unknown = _iter_zone_cards(state.get("opponentScrambleDeckAndHand")) + _iter_zone_cards(state.get("opponentTopCards"))
    opponent_discard = _iter_zone_cards(state.get("opponentDiscardPile"))
    opponent_in_play = _iter_zone_cards(state.get("opponentCardsInPlay"))

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
        len(hand_cards) / 20.0,
        len(discard_cards) / 30.0,
        len(deck_cards) / 30.0,
        len(in_play_cards) / 15.0,
        len(opponent_known_hand) / 10.0,
        len(opponent_unknown) / 30.0,
    ]
    scalars.extend(
        [
            len(opponent_discard) / 30.0,
            len(opponent_in_play) / 15.0,
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

    features[offset:] = _single_card_features(_resolve_option_card(option, state))
    return features


@dataclass
class TrainingConfig:
    hidden_size: int = 16
    learning_rate: float = 0.0015
    ppo_epochs: int = 1
    ppo_clip: float = 0.2
    value_coef: float = 0.5
    grad_clip: float = 1.0
    epsilon_random: float = 0.05
    train_temperature: float = 1.0
    eval_temperature: float = 0.35
    checkpoint_interval: int = 5
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


class _TorchPolicyModule(nn.Module):
    def __init__(self, state_size: int, option_size: int, hidden_size: int) -> None:
        super().__init__()
        self.state_linear = nn.Linear(state_size, hidden_size)
        self.option_linear = nn.Linear(option_size, hidden_size)
        self.joint_bias = nn.Parameter(torch.zeros(hidden_size))
        self.policy_head = nn.Linear(hidden_size, 1)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, state_tensor: "torch.Tensor", option_tensor: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        state_hidden = torch.tanh(self.state_linear(state_tensor))
        option_hidden = torch.tanh(self.option_linear(option_tensor))
        joint_hidden = torch.tanh(state_hidden.unsqueeze(0) + option_hidden + self.joint_bias.unsqueeze(0))
        logits = self.policy_head(joint_hidden).squeeze(-1)
        value = self.value_head(state_hidden).squeeze(-1)
        return logits, value


class PolicyNetwork:
    def __init__(
        self,
        state_size: int = STATE_VECTOR_SIZE,
        option_size: int = OPTION_VECTOR_SIZE,
        hidden_size: int = 24,
        init_seed: Optional[int] = None,
    ) -> None:
        if torch is None or nn is None:
            raise RuntimeError(
                "PyTorch is required for starrealms_selfplay.py. "
                "Run this module from the project's .venv where torch is installed."
            ) from TORCH_IMPORT_ERROR
        if init_seed is not None:
            torch.manual_seed(init_seed)
        self.state_size = state_size
        self.option_size = option_size
        self.hidden_size = hidden_size
        self.device = torch.device("cpu")
        self.model = _TorchPolicyModule(state_size, option_size, hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0015)

    def clone(self) -> "PolicyNetwork":
        clone = PolicyNetwork(
            state_size=self.state_size,
            option_size=self.option_size,
            hidden_size=self.hidden_size,
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
        hidden_size = payload.get("hidden_size", 24)
        model = cls(state_size=state_size, option_size=option_size, hidden_size=hidden_size)

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

    def _option_tensor(self, option_vecs: Sequence[Sequence[float]]) -> "torch.Tensor":
        return torch.tensor(option_vecs, dtype=torch.float32, device=self.device)

    def select_action(
        self,
        state_vec: Sequence[float],
        option_vecs: Sequence[Sequence[float]],
        deterministic: bool = False,
        temperature: float = 1.0,
        epsilon_random: float = 0.0,
    ) -> Dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            state_tensor = self._state_tensor(state_vec)
            option_tensor = self._option_tensor(option_vecs)
            logits, value = self.model(state_tensor, option_tensor)
            logits = logits / max(temperature, 1e-3)
            probs = torch.softmax(logits, dim=0)
            if deterministic:
                action_index = int(torch.argmax(probs).item())
            elif random.random() < epsilon_random:
                action_index = random.randrange(len(option_vecs))
            else:
                action_index = int(torch.multinomial(probs, 1).item())
            log_prob = float(torch.log(probs[action_index].clamp_min(EPSILON)).item())
            return {
                "action_index": action_index,
                "log_prob": log_prob,
                "value": float(value.item()),
                "probs": probs.detach().cpu().tolist(),
            }

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

        for _ in range(config.ppo_epochs):
            random.shuffle(samples)
            for sample in samples:
                state_tensor = self._state_tensor(sample["state_vec"])
                option_tensor = self._option_tensor(sample["option_vecs"])
                action_tensor = torch.tensor(sample["action_index"], dtype=torch.int64, device=self.device)
                old_log_prob_tensor = torch.tensor(sample["old_log_prob"], dtype=torch.float32, device=self.device)
                return_tensor = torch.tensor(sample["return"], dtype=torch.float32, device=self.device)
                advantage_tensor = torch.tensor(sample["advantage"], dtype=torch.float32, device=self.device)

                logits, value = self.model(state_tensor, option_tensor)
                log_probs = torch.log_softmax(logits, dim=0)
                log_prob = log_probs[action_tensor]
                ratio = torch.exp(log_prob - old_log_prob_tensor)
                clipped_ratio = torch.clamp(ratio, 1.0 - config.ppo_clip, 1.0 + config.ppo_clip)
                surrogate_one = ratio * advantage_tensor
                surrogate_two = clipped_ratio * advantage_tensor
                policy_loss = -torch.min(surrogate_one, surrogate_two)
                value_loss = 0.5 * torch.square(value - return_tensor)
                loss = policy_loss + config.value_coef * value_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip)
                self.optimizer.step()

                ratio_value = float(ratio.item())
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                ratios.append(ratio_value)
                clipped.append(1.0 if abs(ratio_value - _clip(ratio_value, 1.0 - config.ppo_clip, 1.0 + config.ppo_clip)) > 1e-8 else 0.0)
                value_predictions.append(float(value.item()))

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
    ) -> None:
        self.policy = policy
        self.deterministic = deterministic
        self.temperature = temperature
        self.epsilon_random = epsilon_random
        self.collector = collector

    def choose(self, player_name: str, options: Sequence[Sequence[Any]], known_game_state: Dict[str, Any]) -> int:
        state_vec = state_to_vector(known_game_state, legal_option_count=len(options))
        option_vecs = [option_to_vector(option, known_game_state) for option in options]
        selection = self.policy.select_action(
            state_vec,
            option_vecs,
            deterministic=self.deterministic,
            temperature=self.temperature,
            epsilon_random=self.epsilon_random,
        )
        if self.collector is not None:
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
        self.policy_file = self.run_dir / "latest_policy.json"
        self.training_state_file = self.run_dir / "training_state.json"
        self.checkpoints_dir = self.run_dir / "checkpoints"


def _default_training_state(config: TrainingConfig) -> Dict[str, Any]:
    return {
        "status": "idle",
        "run_name": LATEST_RUN_NAME,
        "iteration": 0,
        "total_matches": 0,
        "total_games": 0,
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
        "current_elo": INITIAL_ELO,
        "best_elo": INITIAL_ELO,
        "best_checkpoint": "checkpoint_000000.json",
        "latest_checkpoint": "checkpoint_000000.json",
        "config": asdict(config),
        "last_match": None,
        "last_update": None,
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


def _load_json_or_none(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_run(run_name: str, config_overrides: Optional[Dict[str, Any]] = None) -> RunFiles:
    files = RunFiles(run_name)
    files.run_dir.mkdir(parents=True, exist_ok=True)
    files.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if files.training_state_file.exists() and files.policy_file.exists():
        return files

    config = TrainingConfig().merged(config_overrides)
    policy = PolicyNetwork(hidden_size=config.hidden_size)
    state = _default_training_state(config)
    state["run_name"] = run_name
    _atomic_write_json(files.policy_file, policy.to_dict(include_optimizer=True))
    _atomic_write_json(files.training_state_file, state)
    _atomic_write_json(files.checkpoints_dir / "checkpoint_000000.json", policy.to_dict(include_optimizer=False))
    return files


def _load_policy_and_state(run_name: str, config_overrides: Optional[Dict[str, Any]] = None) -> Tuple[RunFiles, PolicyNetwork, Dict[str, Any], TrainingConfig]:
    files = _ensure_run(run_name, config_overrides=config_overrides)
    state_payload = _load_json_or_none(files.training_state_file)
    if state_payload is None:
        state_payload = _default_training_state(TrainingConfig())
        state_payload["run_name"] = run_name
    config = TrainingConfig().merged(state_payload.get("config", {})).merged(config_overrides)
    policy_payload = _load_json_or_none(files.policy_file)
    if policy_payload is None:
        policy = PolicyNetwork(hidden_size=config.hidden_size)
    else:
        policy = PolicyNetwork.from_dict(policy_payload)
    return files, policy, state_payload, config


def _save_policy_and_state(files: RunFiles, policy: PolicyNetwork, state: Dict[str, Any]) -> None:
    state["updated_at"] = _timestamp()
    _atomic_write_json(files.policy_file, policy.to_dict(include_optimizer=True))
    _atomic_write_json(files.training_state_file, state)


def _checkpoint_path(files: RunFiles, iteration: int) -> Path:
    return files.checkpoints_dir / f"checkpoint_{iteration:06d}.json"


def load_policy(run_name: str = LATEST_RUN_NAME, checkpoint: str = "latest") -> PolicyNetwork:
    files, policy, state, _ = _load_policy_and_state(run_name)
    if checkpoint == "latest":
        return policy
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
        state = _load_json_or_none(state_path) or {}
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
                "checkpoint_count": len(checkpoints),
                "created_at": state.get("created_at"),
                "updated_at": state.get("updated_at"),
                "last_match": state.get("last_match"),
                "last_update": state.get("last_update"),
                "last_error": state.get("last_error"),
                "run_dir": str(run_dir),
            }
        )
    runs.sort(key=lambda item: (item.get("updated_at") or 0.0, item["run_name"]), reverse=True)
    return runs


def list_checkpoints(run_name: str = LATEST_RUN_NAME) -> List[Dict[str, Any]]:
    files = RunFiles(run_name)
    state = _load_json_or_none(files.training_state_file) or {}
    best_name = state.get("best_checkpoint")
    latest_name = state.get("latest_checkpoint")
    checkpoints = []
    for item in list(state.get("checkpoints") or []):
        checkpoint = dict(item)
        checkpoint["is_best"] = checkpoint.get("name") == best_name
        checkpoint["is_latest"] = checkpoint.get("name") == latest_name
        checkpoints.append(checkpoint)
    checkpoints.sort(key=lambda item: (int(item.get("iteration", 0)), item.get("name", "")))
    return checkpoints


def get_run_state(run_name: str = LATEST_RUN_NAME) -> Dict[str, Any]:
    files = RunFiles(run_name)
    state = _load_json_or_none(files.training_state_file) or {}
    if not state:
        return {}
    state["run_name"] = state.get("run_name", run_name)
    state["run_dir"] = str(files.run_dir)
    return state


def _play_match(
    policy_a: PolicyNetwork,
    policy_b: PolicyNetwork,
    config: TrainingConfig,
    collect_a: bool = False,
    deterministic_a: bool = False,
    deterministic_b: bool = False,
    games_per_match: int = 13,
) -> Dict[str, Any]:
    collector: Optional[List[Dict[str, Any]]] = [] if collect_a else None
    wins_a = 0
    wins_b = 0
    per_game_winners: List[str] = []
    majority = games_per_match // 2 + 1
    started_at = _timestamp()

    while wins_a < majority and wins_b < majority and len(per_game_winners) < games_per_match:
        actor_a = PolicyActor(
            policy_a,
            deterministic=deterministic_a,
            temperature=config.eval_temperature if deterministic_a else config.train_temperature,
            epsilon_random=0.0 if deterministic_a else config.epsilon_random,
            collector=collector,
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

    games_played = len(per_game_winners)
    return_value = (wins_a - wins_b) / max(games_played, 1)
    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "games_played": games_played,
        "return_value": return_value,
        "per_game_winners": per_game_winners,
        "trajectory": collector or [],
        "duration_seconds": _timestamp() - started_at,
    }


class SelfPlayTrainer:
    def __init__(self, run_name: str = LATEST_RUN_NAME, config_overrides: Optional[Dict[str, Any]] = None) -> None:
        files, policy, state, config = _load_policy_and_state(run_name, config_overrides=config_overrides)
        self.files = files
        self.policy = policy
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

    def _record_checkpoint(self) -> None:
        checkpoint_name = _checkpoint_path(self.files, self.state["iteration"]).name
        _atomic_write_json(
            _checkpoint_path(self.files, self.state["iteration"]),
            self.policy.to_dict(include_optimizer=False),
        )
        self.state["latest_checkpoint"] = checkpoint_name
        existing = None
        for item in self.state["checkpoints"]:
            if item["name"] == checkpoint_name:
                existing = item
                break
        if existing is None:
            self.state["checkpoints"].append(
                {
                    "name": checkpoint_name,
                    "iteration": self.state["iteration"],
                    "elo": self.state["current_elo"],
                    "created_at": _timestamp(),
                }
            )

    def _find_checkpoint_entry(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        for item in self.state.get("checkpoints", []):
            if item["name"] == checkpoint_name:
                return item
        return None

    def _evaluate_latest_checkpoint(self) -> None:
        if self.state["iteration"] == 0:
            return

        latest_name = self.state.get("latest_checkpoint")
        if not latest_name:
            return

        best_name = self.state.get("best_checkpoint", latest_name)
        if best_name == latest_name and len(self.state.get("checkpoints", [])) > 1:
            best_name = self.state["checkpoints"][-2]["name"]

        latest_entry = self._find_checkpoint_entry(latest_name)
        best_entry = self._find_checkpoint_entry(best_name)
        if latest_entry is None or best_entry is None:
            return

        latest_policy = load_policy(self.run_name, latest_name)
        best_policy = load_policy(self.run_name, best_name)
        eval_summary = _play_match(
            latest_policy,
            best_policy,
            self.config,
            collect_a=False,
            deterministic_a=True,
            deterministic_b=True,
        )
        score_a = eval_summary["wins_a"] / max(eval_summary["games_played"], 1)
        new_latest_elo, new_best_elo = _elo_update(
            float(latest_entry.get("elo", INITIAL_ELO)),
            float(best_entry.get("elo", INITIAL_ELO)),
            score_a,
            self.config.elo_k,
        )
        latest_entry["elo"] = new_latest_elo
        best_entry["elo"] = new_best_elo
        latest_entry["last_eval"] = eval_summary
        self.state["current_elo"] = new_latest_elo
        if new_latest_elo >= float(self.state.get("best_elo", INITIAL_ELO)):
            self.state["best_elo"] = new_latest_elo
            self.state["best_checkpoint"] = latest_name

    def _match_to_training_samples(self, match_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        trajectory = match_summary["trajectory"]
        if not trajectory:
            return []
        return_value = match_summary["return_value"]
        for step in trajectory:
            step["return"] = return_value
            step["advantage"] = return_value - step["value"]
        advantages = [step["advantage"] for step in trajectory]
        advantage_mean = _mean(advantages)
        advantage_std = _std(advantages)
        if advantage_std > 1e-6:
            for step in trajectory:
                step["advantage"] = (step["advantage"] - advantage_mean) / advantage_std
        if len(trajectory) > self.config.max_training_samples_per_match:
            trajectory = random.sample(trajectory, self.config.max_training_samples_per_match)
        return trajectory

    def train_iteration(self) -> Dict[str, Any]:
        opponent = self.policy.clone()
        match_summary = _play_match(
            self.policy,
            opponent,
            self.config,
            collect_a=True,
            deterministic_a=False,
            deterministic_b=False,
        )
        training_samples = self._match_to_training_samples(match_summary)
        update_stats = self.policy.train_on_samples(training_samples, self.config)

        self.state["iteration"] += 1
        self.state["total_matches"] += 1
        self.state["total_games"] += match_summary["games_played"]
        self.state["last_match"] = {
            "wins": match_summary["wins_a"],
            "losses": match_summary["wins_b"],
            "games_played": match_summary["games_played"],
            "return_value": match_summary["return_value"],
            "duration_seconds": match_summary["duration_seconds"],
        }
        self.state["last_update"] = update_stats

        if self.state["iteration"] % self.config.checkpoint_interval == 0:
            self._record_checkpoint()
            self._evaluate_latest_checkpoint()

        self._save()
        return {
            "match": match_summary,
            "update": update_stats,
            "iteration": self.state["iteration"],
            "current_elo": self.state.get("current_elo", INITIAL_ELO),
        }

    def _training_loop(self) -> None:
        with self._lock:
            self.state["status"] = "training"
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
            "last_match": self.state.get("last_match"),
            "last_update": self.state.get("last_update"),
            "last_error": self.state.get("last_error"),
            "runtime": _format_seconds(runtime),
            "run_dir": str(self.files.run_dir),
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
    match_text = "no completed matches yet"
    if last_match:
        match_text = (
            f"last match {last_match.get('wins', 0)}-{last_match.get('losses', 0)} "
            f"over {last_match.get('games_played', 0)} games"
        )
    return (
        f"Run '{summary['run_name']}' is {summary['status']}. "
        f"Iteration {summary['iteration']}, total matches {summary['total_matches']}, "
        f"Elo {summary['current_elo']} (best {summary['best_elo']}), {match_text}. "
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


def play_self_game(
    run_name: str = LATEST_RUN_NAME,
    checkpoint: str = "latest",
    deterministic: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    policy_a = load_policy(run_name, checkpoint)
    policy_b = load_policy(run_name, checkpoint)
    config = _get_trainer(run_name).config
    actor_a = PolicyActor(
        policy_a,
        deterministic=deterministic,
        temperature=config.eval_temperature if deterministic else config.train_temperature,
        epsilon_random=0.0 if deterministic else config.epsilon_random,
    )
    actor_b = PolicyActor(
        policy_b,
        deterministic=deterministic,
        temperature=config.eval_temperature if deterministic else config.train_temperature,
        epsilon_random=0.0 if deterministic else config.epsilon_random,
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
    return {
        "winner": game.winner.name,
        "checkpoint": checkpoint,
        "run_name": run_name,
        "ended_by_limit": game.ended_by_limit,
        "turns_taken": game.turnsTaken,
    }


def play_human_game(
    run_name: str = LATEST_RUN_NAME,
    checkpoint: str = "latest",
    human_name: str = "Human",
    policy_name: str = "Policy",
    human_choose_fn: Optional[Any] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    policy = load_policy(run_name, checkpoint)
    config = _get_trainer(run_name).config
    actor = PolicyActor(policy, deterministic=True, temperature=config.eval_temperature, epsilon_random=0.0)
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
    return {
        "winner": game.winner.name,
        "checkpoint": checkpoint,
        "run_name": run_name,
        "ended_by_limit": game.ended_by_limit,
        "turns_taken": game.turnsTaken,
    }


def _cached_policy(run_name: str, checkpoint: str) -> PolicyNetwork:
    files, _, state, _ = _load_policy_and_state(run_name)
    if checkpoint == "latest":
        path = files.policy_file
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
