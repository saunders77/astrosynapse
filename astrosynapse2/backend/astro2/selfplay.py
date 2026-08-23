"""Immutable-engine self-play collection and pickle-friendly CPU workers."""

from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .baselines import make_baseline
from .encoding import DecisionEncoding, DecisionFamily, Encoder
from .engine import (
    Action,
    ActionKind,
    Decision,
    Game,
    GameConfig,
    GameResult,
    Seating,
    model_action_indices,
)
from .engine import DecisionFamily as EngineDecisionFamily
from .engine_encoding import EngineEncoder
from .model import NumpyActor
from .replay import MAX_POLICY_ACTIONS, PolicyItem, PreferenceItem, ReplayItem, make_bootstrap_mask


class EnginePolicy(Protocol):
    def __call__(self, player_id: int, decision: Decision) -> int | Action: ...


class _SearchLeaf(RuntimeError):
    """Internal control flow used to stop a cloned game at a search horizon."""

    def __init__(self, observation: object):
        super().__init__("search horizon reached")
        self.observation = observation


@dataclass(frozen=True, slots=True)
class PlayerExploration:
    """Behavior and bootstrap metadata held fixed for an entire player-game.

    ``head`` remains a real bootstrap head even when ``deployment_policy`` is
    true, so the trajectory keeps a valid required-head mask. In that mode the
    actor itself uses the deployable mean-head, prior-free, greedy policy.
    """

    head: int
    epsilon: float
    bootstrap_mask: np.ndarray
    deployment_policy: bool = False


class ActorPolicy:
    """Adapter from a lightweight NumPy actor to an engine decision policy."""

    def __init__(self, actor: NumpyActor, encoder: Encoder):
        if actor.spec.state_size != encoder.state_size:
            raise ValueError(
                f"actor state size {actor.spec.state_size} != encoder size {encoder.state_size}"
            )
        if actor.spec.action_size != encoder.action_size:
            raise ValueError(
                f"actor action size {actor.spec.action_size} != encoder size {encoder.action_size}"
            )
        if actor.spec.families != len(DecisionFamily):
            raise ValueError("actor family count does not match the stable decision-family table")
        self.actor = actor
        self.encoder = encoder

    @property
    def bootstrap_heads(self) -> int:
        return self.actor.spec.bootstrap_heads

    def score(
        self,
        decision: Decision,
        exploration: PlayerExploration,
        rng: np.random.Generator,
        *,
        exploration_top_k: int = 3,
        randomized_prior_scale: float = 0.0,
    ) -> tuple[Action, DecisionEncoding, float, float]:
        encoded = self.encoder.encode_decision(decision.observation, decision)
        eligible = np.asarray(model_action_indices(decision), dtype=np.int64)
        deployment_policy = exploration.deployment_policy
        local_index, probabilities = self.actor.choose(
            encoded.state,
            encoded.actions[eligible],
            int(encoded.family),
            head=None if deployment_policy else exploration.head,
            epsilon=0.0 if deployment_policy else exploration.epsilon,
            exploration_top_k=exploration_top_k,
            randomized_prior_scale=0.0 if deployment_policy else randomized_prior_scale,
            rng=rng,
        )
        index = int(eligible[local_index])
        epsilon = 0.0 if deployment_policy else exploration.epsilon
        count = (
            len(probabilities)
            if exploration_top_k == 0
            else min(exploration_top_k, len(probabilities))
        )
        top = np.argpartition(probabilities, -count)[-count:]
        best = np.flatnonzero(probabilities == probabilities.max())
        behavior_probability = 0.0
        if local_index in best:
            behavior_probability += (1.0 - epsilon) / len(best)
        if local_index in top:
            behavior_probability += epsilon / count
        return (
            decision.actions[index],
            encoded,
            float(probabilities[local_index]),
            max(float(behavior_probability), np.finfo(np.float32).tiny),
        )

    def select(
        self,
        decision: Decision,
        exploration: PlayerExploration,
        rng: np.random.Generator,
        *,
        exploration_top_k: int = 3,
        randomized_prior_scale: float = 0.0,
    ) -> Action:
        selected, _actions, _next_value, _behavior_probability = self.score(
            decision,
            exploration,
            rng,
            exploration_top_k=exploration_top_k,
            randomized_prior_scale=randomized_prior_scale,
        )
        return selected


@dataclass(frozen=True, slots=True)
class CollectedGame:
    samples: tuple[ReplayItem, ...]
    policy_samples: tuple[PolicyItem, ...]
    preferences: tuple[PreferenceItem, ...]
    result: GameResult
    heads: tuple[int, int]
    epsilons: tuple[float, float]
    bootstrap_masks: tuple[np.ndarray, np.ndarray]

    @property
    def target_by_player(self) -> tuple[float, float]:
        if self.result.winner is None:
            return (0.5, 0.5)
        return (1.0, 0.0) if self.result.winner == 0 else (0.0, 1.0)


@dataclass(frozen=True, slots=True)
class CompactSamples:
    """Contiguous arrays keep ProcessPool transfer overhead low."""

    states: np.ndarray
    actions: np.ndarray
    families: np.ndarray
    targets: np.ndarray
    bootstrap_masks: np.ndarray
    game_ids: np.ndarray
    players: np.ndarray
    steps: np.ndarray
    heads: np.ndarray
    epsilons: np.ndarray
    td_targets: np.ndarray
    td_valid: np.ndarray

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    @classmethod
    def from_items(
        cls,
        items: Sequence[ReplayItem],
        *,
        state_size: int,
        action_size: int,
        bootstrap_heads: int,
    ) -> CompactSamples:
        if not items:
            return cls(
                states=np.empty((0, state_size), dtype=np.float16),
                actions=np.empty((0, action_size), dtype=np.float16),
                families=np.empty(0, dtype=np.uint8),
                targets=np.empty(0, dtype=np.float16),
                bootstrap_masks=np.empty((0, bootstrap_heads), dtype=np.uint8),
                game_ids=np.empty(0, dtype=np.uint64),
                players=np.empty(0, dtype=np.uint8),
                steps=np.empty(0, dtype=np.uint32),
                heads=np.empty(0, dtype=np.uint8),
                epsilons=np.empty(0, dtype=np.float16),
                td_targets=np.empty(0, dtype=np.float16),
                td_valid=np.empty(0, dtype=np.uint8),
            )
        return cls(
            states=np.stack([item.state for item in items]).astype(np.float16),
            actions=np.stack([item.action for item in items]).astype(np.float16),
            families=np.asarray([int(item.family) for item in items], dtype=np.uint8),
            targets=np.asarray([item.target for item in items], dtype=np.float16),
            bootstrap_masks=np.stack([item.bootstrap_mask for item in items]).astype(np.uint8),
            game_ids=np.asarray([int(item.game_id) % (1 << 64) for item in items], dtype=np.uint64),
            players=np.asarray([item.player for item in items], dtype=np.uint8),
            steps=np.asarray([item.step for item in items], dtype=np.uint32),
            heads=np.asarray([item.head for item in items], dtype=np.uint8),
            epsilons=np.asarray([item.epsilon for item in items], dtype=np.float16),
            td_targets=np.asarray([item.td_target for item in items], dtype=np.float16),
            td_valid=np.asarray([item.td_valid for item in items], dtype=np.uint8),
        )

    def replay_items(self) -> list[ReplayItem]:
        return [
            ReplayItem(
                state=self.states[index].astype(np.float32),
                action=self.actions[index].astype(np.float32),
                family=int(self.families[index]),
                target=float(self.targets[index]),
                bootstrap_mask=self.bootstrap_masks[index],
                game_id=int(self.game_ids[index]),
                player=int(self.players[index]),
                step=int(self.steps[index]),
                head=int(self.heads[index]),
                epsilon=float(self.epsilons[index]),
                td_target=float(self.td_targets[index]),
                td_valid=bool(self.td_valid[index]),
            )
            for index in range(len(self))
        ]


@dataclass(frozen=True, slots=True)
class CompactPreferences:
    states: np.ndarray
    preferred_actions: np.ndarray
    disfavored_actions: np.ndarray
    families: np.ndarray
    bootstrap_masks: np.ndarray

    def __len__(self) -> int:
        return int(self.families.shape[0])

    @classmethod
    def from_items(
        cls,
        items: Sequence[PreferenceItem],
        *,
        state_size: int,
        action_size: int,
        bootstrap_heads: int,
    ) -> CompactPreferences:
        if not items:
            return cls(
                states=np.empty((0, state_size), dtype=np.float16),
                preferred_actions=np.empty((0, action_size), dtype=np.float16),
                disfavored_actions=np.empty((0, action_size), dtype=np.float16),
                families=np.empty(0, dtype=np.uint8),
                bootstrap_masks=np.empty((0, bootstrap_heads), dtype=np.uint8),
            )
        return cls(
            states=np.stack([item.state for item in items]).astype(np.float16),
            preferred_actions=np.stack([item.preferred_action for item in items]).astype(
                np.float16
            ),
            disfavored_actions=np.stack([item.disfavored_action for item in items]).astype(
                np.float16
            ),
            families=np.asarray([int(item.family) for item in items], dtype=np.uint8),
            bootstrap_masks=np.stack([item.bootstrap_mask for item in items]).astype(np.uint8),
        )


@dataclass(frozen=True, slots=True)
class CompactPolicySamples:
    """Ragged legal-action sets encoded as one concatenated action matrix."""

    states: np.ndarray
    legal_actions: np.ndarray
    action_offsets: np.ndarray
    selected_indices: np.ndarray
    families: np.ndarray
    targets: np.ndarray
    behavior_probabilities: np.ndarray
    bootstrap_masks: np.ndarray
    game_ids: np.ndarray
    players: np.ndarray
    steps: np.ndarray
    search_policy: np.ndarray
    search_mask: np.ndarray
    search_values: np.ndarray
    search_valid: np.ndarray

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    @classmethod
    def from_items(
        cls,
        items: Sequence[PolicyItem],
        *,
        state_size: int,
        action_size: int,
        bootstrap_heads: int,
    ) -> CompactPolicySamples:
        if not items:
            return cls(
                states=np.empty((0, state_size), dtype=np.float16),
                legal_actions=np.empty((0, action_size), dtype=np.float16),
                action_offsets=np.zeros(1, dtype=np.uint32),
                selected_indices=np.empty(0, dtype=np.uint16),
                families=np.empty(0, dtype=np.uint8),
                targets=np.empty(0, dtype=np.float16),
                behavior_probabilities=np.empty(0, dtype=np.float16),
                bootstrap_masks=np.empty((0, bootstrap_heads), dtype=np.uint8),
                game_ids=np.empty(0, dtype=np.uint64),
                players=np.empty(0, dtype=np.uint8),
                steps=np.empty(0, dtype=np.uint32),
                search_policy=np.empty(0, dtype=np.float16),
                search_mask=np.empty(0, dtype=np.uint8),
                search_values=np.empty(0, dtype=np.float16),
                search_valid=np.empty(0, dtype=np.uint8),
            )
        counts = np.asarray([len(item.legal_actions) for item in items], dtype=np.uint32)
        offsets = np.concatenate((np.zeros(1, dtype=np.uint32), np.cumsum(counts, dtype=np.uint32)))
        return cls(
            states=np.stack([item.state for item in items]).astype(np.float16),
            legal_actions=np.concatenate([item.legal_actions for item in items]).astype(np.float16),
            action_offsets=offsets,
            selected_indices=np.asarray([item.selected_index for item in items], dtype=np.uint16),
            families=np.asarray([int(item.family) for item in items], dtype=np.uint8),
            targets=np.asarray([item.target for item in items], dtype=np.float16),
            behavior_probabilities=np.asarray(
                [item.behavior_probability for item in items], dtype=np.float16
            ),
            bootstrap_masks=np.stack([item.bootstrap_mask for item in items]).astype(np.uint8),
            game_ids=np.asarray([int(item.game_id) % (1 << 64) for item in items], dtype=np.uint64),
            players=np.asarray([item.player for item in items], dtype=np.uint8),
            steps=np.asarray([item.step for item in items], dtype=np.uint32),
            search_policy=np.concatenate(
                [
                    np.asarray(
                        item.search_policy
                        if item.search_policy is not None
                        else np.zeros(len(item.legal_actions)),
                        dtype=np.float16,
                    )
                    for item in items
                ]
            ),
            search_mask=np.concatenate(
                [
                    np.asarray(
                        item.search_mask
                        if item.search_mask is not None
                        else np.zeros(len(item.legal_actions)),
                        dtype=np.uint8,
                    )
                    for item in items
                ]
            ),
            search_values=np.asarray([item.search_value for item in items], dtype=np.float16),
            search_valid=np.asarray([item.search_valid for item in items], dtype=np.uint8),
        )


@dataclass(frozen=True, slots=True)
class WorkerResult:
    samples: CompactSamples
    policy_samples: CompactPolicySamples
    preferences: CompactPreferences
    games: int
    wins: tuple[int, int]
    draws: int
    truncated: int
    turns: int
    decisions: int
    forced_choices: int
    counterfactual_preferences: int = 0
    reanalysis_positions: int = 0


@dataclass(slots=True)
class _PendingSample:
    state: np.ndarray
    action: np.ndarray
    family: DecisionFamily
    player: int
    step: int
    td_target: float = 0.5
    td_valid: bool = False


@dataclass(slots=True)
class _PendingPolicySample:
    state: np.ndarray
    legal_actions: np.ndarray
    selected_index: int
    behavior_probability: float
    family: DecisionFamily
    player: int
    step: int
    search_policy: np.ndarray | None = None
    search_mask: np.ndarray | None = None
    search_value: float = 0.5
    search_valid: bool = False


def _tactical_preference(
    decision: Decision,
    encoded: DecisionEncoding,
    bootstrap_mask: np.ndarray,
) -> PreferenceItem | None:
    """Create one deterministic, exact dominance pair for this decision."""

    if decision.family != EngineDecisionFamily.MAIN:
        return None
    end_indices = [
        index for index, action in enumerate(decision.actions) if action.kind == ActionKind.END_TURN
    ]
    if not end_indices:
        return None
    priorities = (
        {ActionKind.ATTACK_PLAYER, ActionKind.ATTACK_BASE},
        {ActionKind.PLAY_CARD},
        {ActionKind.ACTIVATE_BASE},
    )
    preferred_indices: list[int] = []
    for kinds in priorities:
        preferred_indices = [
            index for index, action in enumerate(decision.actions) if action.kind in kinds
        ]
        if preferred_indices:
            break
    if not preferred_indices:
        return None
    # Rotate among equivalent legal options so a long trajectory supervises
    # more than the first hand card or first base.
    preferred_index = preferred_indices[decision.observation.action_number % len(preferred_indices)]
    return PreferenceItem(
        state=encoded.state,
        preferred_action=encoded.actions[preferred_index],
        disfavored_action=encoded.actions[end_indices[0]],
        family=encoded.family,
        bootstrap_mask=bootstrap_mask.copy(),
    )


def _coerce_pair(value: float | Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        result = (float(value), float(value))
    else:
        if len(value) != 2:
            raise ValueError(f"{name} must contain one value per player")
        result = (float(value[0]), float(value[1]))
    return result


def _coerce_bool_pair(value: bool | Sequence[bool], name: str) -> tuple[bool, bool]:
    if isinstance(value, (bool, np.bool_)):
        return (bool(value), bool(value))
    if len(value) != 2:
        raise ValueError(f"{name} must contain one value per player")
    if any(not isinstance(item, (bool, np.bool_)) for item in value):
        raise TypeError(f"{name} entries must be booleans")
    return (bool(value[0]), bool(value[1]))


def _selected_action(raw: int | Action, decision: Decision) -> Action:
    if isinstance(raw, bool):
        raise TypeError("policy returned bool; return an Action or integer index")
    if isinstance(raw, int):
        if not 0 <= raw < len(decision.actions):
            raise IndexError(f"policy selected illegal action index {raw}")
        return decision.actions[raw]
    if isinstance(raw, Action) and raw in decision.actions:
        return decision.actions[decision.actions.index(raw)]
    raise ValueError("policy returned an action that is not legal in this decision")


def collect_game(
    policies: Sequence[EnginePolicy | ActorPolicy],
    *,
    seed: int,
    encoder: Encoder | None = None,
    bootstrap_heads: int = 3,
    epsilons: float | Sequence[float] = 0.0,
    heads: Sequence[int] | None = None,
    bootstrap_probability: float = 0.8,
    exploration_top_k: int = 3,
    randomized_prior_scale: float = 0.0,
    deployment_policy: bool | Sequence[bool] = False,
    use_bootstrap_targets: bool = True,
    collect_preferences: bool = True,
    collect_policy_decisions: bool = False,
    collect_outcome_decisions: bool = True,
    counterfactual_fraction: float = 0.0,
    counterfactual_max_per_game: int = 1,
    reanalysis_fraction: float = 0.0,
    reanalysis_max_per_game: int = 0,
    reanalysis_max_actions: int = 6,
    reanalysis_rollouts_per_action: int = 2,
    reanalysis_horizon_turns: int = 2,
    reanalysis_policy_temperature: float = 0.35,
    collect_players: Sequence[bool] = (True, True),
    game_id: int | None = None,
    seating: Seating = Seating.FIXED,
    starting_player: int | None = None,
    max_turns: int = 400,
    max_actions_per_turn: int = 200,
    cancel_hook: Callable[[], bool] | None = None,
) -> CollectedGame:
    """Play one game and label each non-forced chosen action at termination."""

    if len(policies) != 2 or len(collect_players) != 2:
        raise ValueError("provide exactly one policy and collection flag per player")
    if bootstrap_heads < 1:
        raise ValueError("bootstrap_heads must be positive")
    if not 0 <= counterfactual_fraction <= 1:
        raise ValueError("counterfactual_fraction must be in [0, 1]")
    if counterfactual_max_per_game < 0:
        raise ValueError("counterfactual_max_per_game must be nonnegative")
    if not 0 <= reanalysis_fraction <= 1:
        raise ValueError("reanalysis_fraction must be in [0, 1]")
    if reanalysis_max_per_game < 0 or reanalysis_max_actions < 2:
        raise ValueError("invalid reanalysis limits")
    if (
        reanalysis_rollouts_per_action < 1
        or reanalysis_horizon_turns < 2
        or reanalysis_policy_temperature <= 0
    ):
        raise ValueError("invalid reanalysis rollout settings")
    encoder = encoder or EngineEncoder()
    requested_epsilon_pair = _coerce_pair(epsilons, "epsilons")
    if any(not 0 <= epsilon <= 1 for epsilon in requested_epsilon_pair):
        raise ValueError("epsilons must be in [0, 1]")
    deployment_pair = _coerce_bool_pair(deployment_policy, "deployment_policy")
    epsilon_pair = tuple(
        0.0 if deployment_pair[player] else requested_epsilon_pair[player] for player in range(2)
    )

    normalized_seed = int(seed) % (1 << 64)
    rngs = [
        np.random.default_rng(np.random.SeedSequence([normalized_seed, 0xA57A, player]))
        for player in range(2)
    ]
    policy_head_counts = tuple(
        policy.bootstrap_heads if isinstance(policy, ActorPolicy) else bootstrap_heads
        for policy in policies
    )
    for player, policy in enumerate(policies):
        if (
            collect_players[player]
            and isinstance(policy, ActorPolicy)
            and policy.bootstrap_heads != bootstrap_heads
        ):
            raise ValueError("collected ActorPolicy and replay bootstrap head counts differ")

    if heads is None:
        # Frozen league opponents may use a legacy head count. Their behavior
        # head must be valid for their own actor, while replay masks retain the
        # current learner's fixed width for the player whose samples we keep.
        head_pair = tuple(
            int(rng.integers(0, policy_head_counts[player])) for player, rng in enumerate(rngs)
        )
    else:
        if len(heads) != 2:
            raise ValueError("heads must contain one head per player")
        head_pair = (int(heads[0]), int(heads[1]))
    if any(not 0 <= head_pair[player] < policy_head_counts[player] for player in range(2)):
        raise ValueError("selected head is outside that policy's bootstrap head range")

    explorations = tuple(
        PlayerExploration(
            head=head_pair[player],
            epsilon=epsilon_pair[player],
            bootstrap_mask=make_bootstrap_mask(
                bootstrap_heads,
                rngs[player],
                inclusion_probability=bootstrap_probability,
                required_head=head_pair[player] if collect_players[player] else None,
            ),
            deployment_policy=deployment_pair[player],
        )
        for player in range(2)
    )
    pending: list[_PendingSample] = []
    pending_policy: list[_PendingPolicySample] = []
    preferences: list[PreferenceItem] = []
    player_steps = [0, 0]
    previous_actor_sample: list[int | None] = [None, None]
    counterfactual_count = 0
    reanalysis_count = 0
    game_ref: list[Game] = []

    def continuation_chooser(
        player: int,
        branch_seed: int,
        *,
        leaf_player: int | None = None,
        leaf_turn: int | None = None,
    ):
        policy = policies[player]
        branch_rng = np.random.default_rng(
            np.random.SeedSequence([normalized_seed, 0xC0FA, branch_seed, player])
        )

        def choose(player_id: int, decision: Decision) -> Action:
            if (
                leaf_player == player_id
                and leaf_turn is not None
                and decision.family == EngineDecisionFamily.MAIN
                and decision.observation.turn >= leaf_turn
            ):
                raise _SearchLeaf(decision.observation)
            if isinstance(policy, ActorPolicy):
                exploration = PlayerExploration(
                    head=head_pair[player],
                    epsilon=0.0,
                    bootstrap_mask=explorations[player].bootstrap_mask,
                    deployment_policy=True,
                )
                return policy.select(decision, exploration, branch_rng)
            return _selected_action(policy(player_id, decision), decision)

        return choose

    def branch_score(
        game: Game,
        player: int,
        action: Action,
        branch_seed: int,
        belief_seed: int | None = None,
        horizon_turns: int | None = None,
    ) -> float | None:
        branch = copy.deepcopy(game)
        branch.cancel_hook = None
        branch.decision_hook = None
        if belief_seed is not None:
            branch.resample_public_belief(player, belief_seed)
        leaf_turn = game.turns + horizon_turns if horizon_turns is not None else None
        branch.choosers = {
            0: continuation_chooser(
                0, branch_seed, leaf_player=player, leaf_turn=leaf_turn
            ),
            1: continuation_chooser(
                1, branch_seed, leaf_player=player, leaf_turn=leaf_turn
            ),
        }
        try:
            result = branch.continue_from_main_action(action)
        except _SearchLeaf as leaf:
            policy = policies[player]
            if isinstance(policy, ActorPolicy) and hasattr(policy.actor, "predict_values"):
                state = policy.encoder.encode_state(leaf.observation)
                logits = policy.actor.predict_values(
                    state,
                    np.asarray([int(DecisionFamily.MAIN)], dtype=np.int64),
                )
                probabilities = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
                return float(np.mean(probabilities))
            own = branch.players[player].authority
            opponent = branch.players[1 - player].authority
            return float(1.0 / (1.0 + np.exp(-(own - opponent) / 10.0)))
        if result.truncated:
            return None
        if result.winner is not None:
            return float(result.winner == player)
        own = branch.players[player].authority
        opponent = branch.players[1 - player].authority
        return float(1.0 / (1.0 + np.exp(-(own - opponent) / 10.0)))

    def reanalysis_hook(player: int, decision: Decision, selected: Action) -> None:
        nonlocal reanalysis_count
        if (
            reanalysis_count >= reanalysis_max_per_game
            or decision.family != EngineDecisionFamily.MAIN
            or not collect_players[player]
            or not pending_policy
            or len(decision.actions) < 2
            or rngs[player].random() >= reanalysis_fraction
        ):
            return
        pending_item = pending_policy[-1]
        if pending_item.player != player or pending_item.step != player_steps[player] - 1:
            return
        eligible = np.asarray(model_action_indices(decision), dtype=np.int64)
        selected_engine_index = decision.actions.index(selected)
        selected_matches = np.flatnonzero(eligible == selected_engine_index)
        if not len(selected_matches):
            return
        selected_index = int(selected_matches[0])
        count = min(reanalysis_max_actions, len(eligible))
        if isinstance(policies[player], ActorPolicy):
            logits = policies[player].actor.predict_options(
                pending_item.state,
                pending_item.legal_actions,
                int(pending_item.family),
            ).mean(axis=1)
            ranked = list(np.argsort(logits)[::-1].astype(int))
        else:
            ranked = list(rngs[player].permutation(len(eligible)).astype(int))
        candidates = [selected_index]
        candidates.extend(index for index in ranked if index != selected_index)
        candidates = candidates[:count]
        scores = np.zeros(len(eligible), dtype=np.float32)
        searched = np.zeros(len(eligible), dtype=np.uint8)
        branch_seed = player_steps[player] + 1_000 * decision.observation.turn
        for local_index in candidates:
            action_scores: list[float] = []
            for rollout in range(reanalysis_rollouts_per_action):
                # The same belief seed is used for every candidate at one
                # rollout index, removing chance/deck noise from comparisons.
                belief_seed = normalized_seed + 7_919 * branch_seed + 104_729 * rollout
                score = branch_score(
                    game_ref[0],
                    player,
                    decision.actions[int(eligible[local_index])],
                    branch_seed + rollout,
                    belief_seed,
                    reanalysis_horizon_turns,
                )
                if score is not None:
                    action_scores.append(score)
            if action_scores:
                scores[local_index] = float(np.mean(action_scores))
                searched[local_index] = 1
        valid_indices = np.flatnonzero(searched)
        if len(valid_indices) < 2:
            return
        scaled = scores[valid_indices] / float(reanalysis_policy_temperature)
        scaled -= float(np.max(scaled))
        probabilities = np.exp(scaled)
        probabilities /= float(np.sum(probabilities))
        target = np.zeros(len(eligible), dtype=np.float32)
        target[valid_indices] = probabilities
        pending_item.search_policy = target
        pending_item.search_mask = searched
        pending_item.search_value = float(np.sum(probabilities * scores[valid_indices]))
        pending_item.search_valid = True
        reanalysis_count += 1

    def counterfactual_hook(player: int, decision: Decision, selected: Action) -> None:
        nonlocal counterfactual_count
        if (
            counterfactual_count >= counterfactual_max_per_game
            or decision.family != EngineDecisionFamily.MAIN
            or len(decision.actions) < 2
            or rngs[player].random() >= counterfactual_fraction
        ):
            return
        eligible = np.asarray(model_action_indices(decision), dtype=np.int64)
        selected_index = decision.actions.index(selected)
        alternatives = eligible[eligible != selected_index]
        if selected_index not in eligible or not len(alternatives):
            return
        # Compare the behavior action with an unbiased eligible alternative.
        # Outcome determines the preference; no card or strategy is privileged.
        first = selected_index
        second = int(rngs[player].choice(alternatives))
        encoded = encoder.encode_decision(decision.observation, decision)
        branch_seed = player_steps[player] + 1_000 * decision.observation.turn
        counterfactual_count += 1
        first_score = branch_score(game_ref[0], player, decision.actions[first], branch_seed)
        second_score = branch_score(game_ref[0], player, decision.actions[second], branch_seed)
        if first_score is None or second_score is None or abs(first_score - second_score) < 1e-6:
            return
        preferred, disfavored = (first, second) if first_score > second_score else (second, first)
        preferences.append(
            PreferenceItem(
                state=encoded.state,
                preferred_action=encoded.actions[preferred],
                disfavored_action=encoded.actions[disfavored],
                family=encoded.family,
                bootstrap_mask=explorations[player].bootstrap_mask.copy(),
            )
        )

    def decision_hook(player: int, decision: Decision, selected: Action) -> None:
        if counterfactual_fraction > 0:
            counterfactual_hook(player, decision, selected)
        if reanalysis_fraction > 0:
            reanalysis_hook(player, decision, selected)

    def make_chooser(player: int):
        def choose(player_id: int, decision: Decision) -> Action:
            if player_id != player:
                raise RuntimeError("engine invoked a chooser for the wrong player")
            policy = policies[player]
            if isinstance(policy, ActorPolicy):
                selected, encoded, next_value, behavior_probability = policy.score(
                    decision,
                    explorations[player],
                    rngs[player],
                    exploration_top_k=exploration_top_k,
                    randomized_prior_scale=randomized_prior_scale,
                )
                previous = previous_actor_sample[player]
                if use_bootstrap_targets and previous is not None:
                    pending[previous].td_target = next_value
                    pending[previous].td_valid = True
            else:
                selected = _selected_action(policy(player_id, decision), decision)
                behavior_probability = 1.0
            if collect_players[player]:
                if not isinstance(policy, ActorPolicy):
                    encoded = encoder.encode_decision(decision.observation, decision)
                selected_index = decision.actions.index(selected)
                if collect_policy_decisions:
                    eligible = np.asarray(model_action_indices(decision), dtype=np.int64)
                    if selected_index not in eligible:
                        eligible = np.arange(len(decision.actions), dtype=np.int64)
                    policy_selected = int(np.flatnonzero(eligible == selected_index)[0])
                    # Exceptionally large legal sets are omitted intact; they
                    # must never be silently truncated into a different policy.
                    if 2 <= len(eligible) <= MAX_POLICY_ACTIONS:
                        pending_policy.append(
                            _PendingPolicySample(
                                state=encoded.state,
                                legal_actions=encoded.actions[eligible],
                                selected_index=policy_selected,
                                behavior_probability=behavior_probability,
                                family=encoded.family,
                                player=player,
                                step=player_steps[player],
                            )
                        )
                if collect_preferences:
                    preference = _tactical_preference(
                        decision,
                        encoded,
                        explorations[player].bootstrap_mask,
                    )
                    if preference is not None:
                        preferences.append(preference)
                if collect_outcome_decisions:
                    pending.append(
                        _PendingSample(
                            state=encoded.state,
                            action=encoded.actions[selected_index],
                            family=encoded.family,
                            player=player,
                            step=player_steps[player],
                        )
                    )
                if isinstance(policy, ActorPolicy) and collect_outcome_decisions:
                    previous_actor_sample[player] = len(pending) - 1
                player_steps[player] += 1
            return selected

        return choose

    game = Game(
        choosers=(make_chooser(0), make_chooser(1)),
        config=GameConfig(
            seed=seed,
            seating=seating,
            starting_player=starting_player,
            max_turns=max_turns,
            max_actions_per_turn=max_actions_per_turn,
        ),
        cancel_hook=cancel_hook,
        decision_hook=(
            decision_hook if counterfactual_fraction > 0 or reanalysis_fraction > 0 else None
        ),
    )
    game_ref.append(game)
    result = game.run()
    targets = (0.5, 0.5)
    if result.winner is not None:
        targets = (1.0, 0.0) if result.winner == 0 else (0.0, 1.0)
    resolved_game_id = seed if game_id is None else game_id
    samples = ()
    policy_samples = ()
    if not result.truncated:
        samples = tuple(
            ReplayItem(
                state=item.state,
                action=item.action,
                family=item.family,
                target=targets[item.player],
                bootstrap_mask=explorations[item.player].bootstrap_mask.copy(),
                game_id=resolved_game_id,
                player=item.player,
                step=item.step,
                head=explorations[item.player].head,
                epsilon=explorations[item.player].epsilon,
                td_target=item.td_target,
                td_valid=item.td_valid,
            )
            for item in pending
        )
        policy_samples = tuple(
            PolicyItem(
                state=item.state,
                legal_actions=item.legal_actions,
                selected_index=item.selected_index,
                family=item.family,
                target=targets[item.player],
                behavior_probability=item.behavior_probability,
                bootstrap_mask=explorations[item.player].bootstrap_mask.copy(),
                game_id=resolved_game_id,
                player=item.player,
                step=item.step,
                search_policy=item.search_policy,
                search_mask=item.search_mask,
                search_value=item.search_value,
                search_valid=item.search_valid,
            )
            for item in pending_policy
        )
    return CollectedGame(
        samples=samples,
        policy_samples=policy_samples,
        preferences=() if result.truncated else tuple(preferences),
        result=result,
        heads=head_pair,
        epsilons=epsilon_pair,
        bootstrap_masks=(
            explorations[0].bootstrap_mask.copy(),
            explorations[1].bootstrap_mask.copy(),
        ),
    )


_ACTOR_CACHE_LIMIT = 4
_ACTOR_CACHE: OrderedDict[str, tuple[int, int, NumpyActor]] = OrderedDict()


def _cached_actor(path: str) -> NumpyActor:
    key = str(Path(path).expanduser().resolve())
    stat = Path(key).stat()
    signature = (stat.st_mtime_ns, stat.st_size)
    cached = _ACTOR_CACHE.get(key)
    if cached is None or cached[:2] != signature:
        actor = NumpyActor.load(key)
        _ACTOR_CACHE[key] = (*signature, actor)
    else:
        actor = cached[2]
    _ACTOR_CACHE.move_to_end(key)
    while len(_ACTOR_CACHE) > _ACTOR_CACHE_LIMIT:
        _ACTOR_CACHE.popitem(last=False)
    return actor


def clear_actor_cache() -> None:
    """Drop process-local actor archives (mainly useful for tests/diagnostics)."""

    _ACTOR_CACHE.clear()


def _compact_policy_transfer(
    items: list[PolicyItem],
    *,
    limit_per_player_game: int,
    seed: int,
) -> list[PolicyItem]:
    """Apply the long-horizon reservoir before process-pool serialization."""

    if limit_per_player_game <= 0:
        return items
    grouped: OrderedDict[tuple[int, int], list[PolicyItem]] = OrderedDict()
    for item in items:
        grouped.setdefault((int(item.game_id), int(item.player)), []).append(item)
    rng = np.random.default_rng(seed + 0xA5705)
    retained: list[PolicyItem] = []
    for episode in grouped.values():
        if len(episode) <= limit_per_player_game:
            retained.extend(episode)
            continue
        selected = [item for item in episode if item.search_valid][:limit_per_player_game]
        selected_ids = {id(item) for item in selected}
        strata: dict[tuple[int, int], list[PolicyItem]] = {}
        for item in episode:
            if id(item) in selected_ids:
                continue
            turn = float(item.state[11]) * 50.0 if len(item.state) > 11 else 0.0
            phase = 0 if turn <= 6 else 1 if turn <= 16 else 2
            strata.setdefault((int(item.family), phase), []).append(item)
        pools = [rows for _key, rows in sorted(strata.items())]
        while len(selected) < limit_per_player_game and any(pools):
            for pool in pools:
                if not pool or len(selected) >= limit_per_player_game:
                    continue
                selected.append(pool.pop(int(rng.integers(0, len(pool)))))
        retained.extend(sorted(selected, key=lambda item: item.step))
    return retained


def collect_worker_batch(
    actor_paths: Sequence[str | None],
    *,
    games: int,
    seed: int,
    epsilons: float | Sequence[float] = 0.0,
    baseline_names: Sequence[str] = ("balanced", "balanced"),
    bootstrap_heads: int = 3,
    collect_players: Sequence[bool] | None = None,
    max_turns: int = 180,
    max_actions_per_turn: int = 160,
    exploration_top_k: int = 3,
    bootstrap_probability: float = 0.8,
    randomized_prior_scale: float = 0.0,
    deployment_policy: bool | Sequence[bool] = False,
    use_bootstrap_targets: bool = True,
    collect_preferences: bool = True,
    collect_policy_decisions: bool = False,
    collect_outcome_decisions: bool = True,
    counterfactual_fraction: float = 0.0,
    counterfactual_max_per_game: int = 1,
    reanalysis_fraction: float = 0.0,
    reanalysis_max_per_game: int = 0,
    reanalysis_max_actions: int = 6,
    reanalysis_rollouts_per_action: int = 2,
    reanalysis_horizon_turns: int = 2,
    reanalysis_policy_temperature: float = 0.35,
    policy_replay_decisions_per_player_game: int = 0,
    encoder_version: int = 1,
) -> WorkerResult:
    """Top-level ProcessPool worker; imports no MLX and caches actor archives."""

    if games < 1:
        raise ValueError("games must be positive")
    if len(actor_paths) != 2 or len(baseline_names) != 2:
        raise ValueError("actor_paths and baseline_names must have two entries")
    encoder = EngineEncoder(version=encoder_version)
    policies: list[EnginePolicy | ActorPolicy] = []
    for player, path in enumerate(actor_paths):
        if path is None:
            policies.append(make_baseline(baseline_names[player], seed + 10_007 * (player + 1)))
        else:
            actor = _cached_actor(path)
            policies.append(
                ActorPolicy(actor, EngineEncoder(version=actor.spec.encoder_version))
            )
    flags = (
        tuple(path is not None for path in actor_paths)
        if collect_players is None
        else collect_players
    )

    all_items: list[ReplayItem] = []
    all_policy_items: list[PolicyItem] = []
    all_preferences: list[PreferenceItem] = []
    wins = [0, 0]
    draws = truncated = turns = decisions = forced_choices = 0
    counterfactual_preferences = 0
    reanalysis_positions = 0
    for game_index in range(games):
        game_seed = seed + game_index
        collected = collect_game(
            policies,
            seed=game_seed,
            encoder=encoder,
            bootstrap_heads=bootstrap_heads,
            epsilons=epsilons,
            collect_players=flags,
            game_id=game_seed,
            starting_player=game_index % 2,
            max_turns=max_turns,
            max_actions_per_turn=max_actions_per_turn,
            exploration_top_k=exploration_top_k,
            bootstrap_probability=bootstrap_probability,
            randomized_prior_scale=randomized_prior_scale,
            deployment_policy=deployment_policy,
            use_bootstrap_targets=use_bootstrap_targets,
            collect_preferences=collect_preferences,
            collect_policy_decisions=collect_policy_decisions,
            collect_outcome_decisions=collect_outcome_decisions,
            counterfactual_fraction=counterfactual_fraction,
            counterfactual_max_per_game=counterfactual_max_per_game,
            reanalysis_fraction=reanalysis_fraction,
            reanalysis_max_per_game=reanalysis_max_per_game,
            reanalysis_max_actions=reanalysis_max_actions,
            reanalysis_rollouts_per_action=reanalysis_rollouts_per_action,
            reanalysis_horizon_turns=reanalysis_horizon_turns,
            reanalysis_policy_temperature=reanalysis_policy_temperature,
        )
        all_items.extend(collected.samples)
        all_policy_items.extend(collected.policy_samples)
        reanalysis_positions += sum(int(item.search_valid) for item in collected.policy_samples)
        all_preferences.extend(collected.preferences)
        if counterfactual_fraction > 0:
            # Tactical preference collection is disabled for generation 4, so
            # every returned preference in that mode is a paired rollout.
            counterfactual_preferences += len(collected.preferences)
        result = collected.result
        if result.truncated:
            truncated += 1
        elif result.winner is None:
            draws += 1
        else:
            wins[result.winner] += 1
        turns += result.turns
        decisions += result.decisions
        forced_choices += result.forced_choices

    transferred_policy_items = _compact_policy_transfer(
        all_policy_items,
        limit_per_player_game=policy_replay_decisions_per_player_game,
        seed=seed,
    )
    return WorkerResult(
        samples=CompactSamples.from_items(
            all_items,
            state_size=encoder.state_size,
            action_size=encoder.action_size,
            bootstrap_heads=bootstrap_heads,
        ),
        policy_samples=CompactPolicySamples.from_items(
            transferred_policy_items,
            state_size=encoder.state_size,
            action_size=encoder.action_size,
            bootstrap_heads=bootstrap_heads,
        ),
        preferences=CompactPreferences.from_items(
            all_preferences,
            state_size=encoder.state_size,
            action_size=encoder.action_size,
            bootstrap_heads=bootstrap_heads,
        ),
        games=games,
        wins=(wins[0], wins[1]),
        draws=draws,
        truncated=truncated,
        turns=turns,
        decisions=decisions,
        forced_choices=forced_choices,
        counterfactual_preferences=counterfactual_preferences,
        reanalysis_positions=reanalysis_positions,
    )


__all__ = [
    "ActorPolicy",
    "CollectedGame",
    "CompactSamples",
    "CompactPreferences",
    "CompactPolicySamples",
    "PlayerExploration",
    "WorkerResult",
    "clear_actor_cache",
    "collect_game",
    "collect_worker_batch",
]
