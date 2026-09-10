"""Public-information rollout planning around a frozen checkpoint.

The planner is an experimental policy-improvement operator. It never changes
the frozen policy used in continuations, and its search RNG is independent of
the actual game RNG. Strength must be established by held-out paired games.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .cards import CARD_BY_ID, Faction
from .engine import Action, ActionKind, Decision, DecisionFamily, Game, model_action_indices
from .engine_encoding import EngineEncoder
from .model import NumpyActor


@dataclass(frozen=True)
class PlanningConfig:
    rollouts: int = 8
    max_actions: int = 4
    horizon_turns: int = 4  # zero means terminal rollouts
    min_gain: float = 0.02
    head: int | None = None
    play_first: bool = False
    temperature: float = 0.0
    native: bool = False
    residual_strategy_only: bool = False
    auto_actor_path: str | None = None
    rules_version: int = 1
    # Learned by complete-game optimization, never by action/outcome correlation.
    residual: tuple[float, ...] = ()

    def __post_init__(self):
        if self.rollouts < 0 or self.max_actions < 2 or self.horizon_turns < 0:
            raise ValueError("invalid planning budget")
        if self.min_gain < 0 or self.temperature < 0:
            raise ValueError("gain and temperature must be nonnegative")
        if self.residual and len(self.residual) != 104:
            raise ValueError("residual must contain 104 coefficients")
        if self.rules_version not in (1, 2):
            raise ValueError("rules_version must be 1 or 2")
        if self.rollouts and self.auto_actor_path:
            raise ValueError("hierarchical continuation search has not been implemented")
        if not np.isfinite(self.min_gain) or not np.isfinite(self.temperature):
            raise ValueError("gain and temperature must be finite")
        if self.residual and not np.isfinite(self.residual).all():
            raise ValueError("residual coefficients must be finite")


def residual_features(decision: Decision, action: Action) -> np.ndarray:
    """Small state-dependent decision adapter for direct policy optimization.

    Two independent acquisition tables describe opening and mature decks.
    Four coefficients express same-faction deck density. Two control optional
    in-play scrapping and retaining a card. All coefficients start at zero.
    """
    features = np.zeros(104, dtype=np.float32)
    phase = float(np.clip((decision.observation.turn - 4) / 20, 0, 1))
    if action.kind == ActionKind.ACQUIRE:
        features[action.card_id] = 1 - phase
        features[49 + action.card_id] = phase
        card = CARD_BY_ID[action.card_id]
        faction = list(Faction).index(card.faction)
        if faction:
            own = decision.observation
            deck = [
                *own.hand,
                *own.own_deck,
                *own.own_discard,
                *(item.card for item in own.own_in_play),
            ]
            features[98 + faction - 1] = sum(c.faction == card.faction for c in deck) / max(
                1, len(deck)
            )
    elif action.kind == ActionKind.SCRAP_FOR_ABILITY:
        features[102] = CARD_BY_ID[action.card_id].cost / 8
    elif action.kind == ActionKind.END_TURN:
        features[103] = 1
    return features


def strategic_decision(decision: Decision) -> bool:
    """Keep the frozen low-level policy while cards/bases remain to be played."""
    return decision.family != DecisionFamily.MAIN or not any(
        action.kind in {ActionKind.PLAY_CARD, ActionKind.ACTIVATE_BASE}
        for action in decision.actions
    )


def sample_belief(game: Game, observer: int, seed: int) -> Game:
    """Canonicalize hidden bags, sample, and replace all future chance streams.

    A fixed planning seed must produce the same sampled world for two states
    with the same observation, regardless of their simulator-hidden order or
    opponent hand/deck assignment. Merely shuffling existing lists does not
    provide that property.
    """
    branch = game.fork()
    branch.cancel_hook = None
    branch.decision_hook = None
    own, opponent = branch.players[observer], branch.players[1 - observer]
    known = len(own.known_top)
    unknown = own.deck[:-known] if known else own.deck
    own.deck = sorted(unknown, key=lambda card: card.card_id) + (own.deck[-known:] if known else [])
    known = len(opponent.known_top)
    unknown = opponent.deck[:-known] if known else opponent.deck
    hidden = list(opponent.hand) + list(unknown)
    for card in opponent.revealed_hand:
        hidden.remove(card)
    hidden.sort(key=lambda card: card.card_id)
    slots = len(opponent.hand) - len(opponent.revealed_hand)
    opponent.hand = list(opponent.revealed_hand) + hidden[:slots]
    opponent.deck = hidden[slots:] + (opponent.deck[-known:] if known else [])
    branch.trade_deck.sort(key=lambda card: card.card_id)
    branch.resample_public_belief(observer, seed)
    chance = random.Random(seed ^ 0x5EA2C4)
    for stream in (
        branch.rng_streams.seating,
        branch.rng_streams.market,
        branch.rng_streams.player_0,
        branch.rng_streams.player_1,
    ):
        stream.seed(chance.getrandbits(64))
    return branch


class FrozenChooser:
    def __init__(self, actor: NumpyActor, seed: int, config: PlanningConfig):
        self.actor = actor
        self.encoder = EngineEncoder(version=actor.spec.encoder_version)
        self.rng = np.random.default_rng(seed)
        self.config = config
        self.automatic = None

    def ranked(self, decision: Decision):
        eligible = np.asarray(model_action_indices(decision), dtype=np.int64)
        encoded = self.encoder.encode_decision(decision.observation, decision)
        if self.config.head is None:
            logits = self.actor.predict_options(
                encoded.state, encoded.actions[eligible], int(encoded.family)
            ).mean(axis=1)
        else:
            logits = self.actor.predict_option_head(
                encoded.state, encoded.actions[eligible], int(encoded.family), self.config.head
            )
        apply_residual = not self.config.residual_strategy_only or (
            decision.family == DecisionFamily.MAIN
            and not any(
                action.kind in {ActionKind.PLAY_CARD, ActionKind.ACTIVATE_BASE}
                for action in decision.actions
            )
            and decision.actions[int(eligible[np.argmax(logits)])].kind
            in {ActionKind.ACQUIRE, ActionKind.END_TURN, ActionKind.SCRAP_FOR_ABILITY}
        )
        if self.config.residual and apply_residual:
            coefficients = np.asarray(self.config.residual, dtype=np.float32)
            logits = logits + np.asarray(
                [
                    residual_features(decision, decision.actions[int(index)]) @ coefficients
                    for index in eligible
                ]
            )
        order = np.argsort(-logits, kind="stable")
        if self.config.play_first and decision.family == DecisionFamily.MAIN:
            order = np.asarray(
                sorted(
                    order,
                    key=lambda i: (
                        decision.actions[eligible[i]].kind
                        not in {ActionKind.PLAY_CARD, ActionKind.ACTIVATE_BASE}
                    ),
                )
            )
        return eligible, logits, order

    def __call__(self, _player: int, decision: Decision) -> Action:
        if self.automatic is not None and not strategic_decision(decision):
            return self.automatic(_player, decision)
        eligible, logits, order = self.ranked(decision)
        if self.config.temperature:
            probabilities = np.exp((logits - logits.max()) / self.config.temperature)
            probabilities /= probabilities.sum()
            index = self.rng.choice(len(eligible), p=probabilities)
        else:
            # Match the production actor's random tie break exactly.
            best = np.flatnonzero(logits == logits[order[0]])
            index = int(self.rng.choice(best)) if len(best) > 1 else int(order[0])
        return decision.actions[int(eligible[index])]


class _Leaf(Exception):
    def __init__(self, decision: Decision):
        self.decision = decision


class PlanningChooser(FrozenChooser):
    """Search main-phase strategic choices with shared-randomness rollouts."""

    def __init__(self, actor: NumpyActor, seed: int, config: PlanningConfig):
        super().__init__(actor, seed, config)
        self.game: Game | None = None
        self.searches = self.changes = self.branches = self.truncations = 0

    def __call__(self, player: int, decision: Decision) -> Action:
        if self.automatic is not None and not strategic_decision(decision):
            return self.automatic(player, decision)
        eligible, logits, order = self.ranked(decision)
        chosen = int(order[0])
        default = decision.actions[int(eligible[chosen])]
        if (
            self.config.rollouts == 0
            or decision.family != DecisionFamily.MAIN
            or len(eligible) < 2
            or any(
                action.kind in {ActionKind.PLAY_CARD, ActionKind.ACTIVATE_BASE}
                for action in decision.actions
            )
        ):
            return default
        if self.game is None:
            raise RuntimeError("planner must be attached to its live game")
        candidates = list(order[: self.config.max_actions])
        # Keep the option of retaining cards / declining an acquisition in search.
        end = next(
            (i for i in order if decision.actions[int(eligible[i])].kind == ActionKind.END_TURN),
            None,
        )
        if end is not None and end not in candidates:
            candidates[-1] = end
        root_turn = decision.observation.turn
        values = np.zeros((len(candidates), self.config.rollouts), dtype=np.float64)
        valid = np.ones(self.config.rollouts, dtype=bool)
        for rollout in range(self.config.rollouts):
            seed = int(self.rng.integers(0, 2**63))
            belief = sample_belief(self.game, player, seed)
            for j, index in enumerate(candidates):
                branch = belief.fork()
                continuation = FrozenChooser(self.actor, seed ^ 0xAC70, self.config)

                def choose(pid, next_decision, continuation=continuation):
                    if (
                        self.config.horizon_turns
                        and pid == player
                        and next_decision.family == DecisionFamily.MAIN
                        and next_decision.observation.turn >= root_turn + self.config.horizon_turns
                    ):
                        raise _Leaf(next_decision)
                    return continuation(pid, next_decision)

                branch.choosers = {0: choose, 1: choose}
                try:
                    result = branch.continue_from_main_action(
                        decision.actions[int(eligible[index])]
                    )
                    if result.truncated:
                        valid[rollout] = False
                        self.truncations += 1
                    values[j, rollout] = float(result.winner == player)
                except _Leaf as leaf:
                    state = self.encoder.encode_state(leaf.decision.observation)
                    value_logits = self.actor.predict_values(state, np.asarray([0]))
                    values[j, rollout] = float(np.mean(1 / (1 + np.exp(-value_logits))))
                self.branches += 1
        self.searches += 1
        if not valid.any():
            return default
        means = values[:, valid].mean(axis=1)
        best = int(np.argmax(means))
        if means[best] > means[0] + self.config.min_gain:
            chosen = int(candidates[best])
            self.changes += 1
        return decision.actions[int(eligible[chosen])]
