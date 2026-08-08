"""Immutable-engine self-play collection and pickle-friendly CPU workers."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .baselines import make_baseline
from .encoding import DecisionFamily, Encoder
from .engine import Action, Decision, Game, GameConfig, GameResult, Seating
from .model import NumpyActor
from .replay import ReplayItem, make_bootstrap_mask


class EnginePolicy(Protocol):
    def __call__(self, player_id: int, decision: Decision) -> int | Action: ...


@dataclass(frozen=True, slots=True)
class PlayerExploration:
    """Values sampled once and held fixed for an entire player-game."""

    head: int
    epsilon: float
    bootstrap_mask: np.ndarray


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

    def select(
        self,
        decision: Decision,
        exploration: PlayerExploration,
        rng: np.random.Generator,
    ) -> Action:
        encoded = self.encoder.encode_decision(decision.observation, decision)
        index, _probabilities = self.actor.choose(
            encoded.state,
            encoded.actions,
            int(encoded.family),
            head=exploration.head,
            epsilon=exploration.epsilon,
            rng=rng,
        )
        return decision.actions[index]


@dataclass(frozen=True, slots=True)
class CollectedGame:
    samples: tuple[ReplayItem, ...]
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
            )
        return cls(
            states=np.stack([item.state for item in items]).astype(np.float16),
            actions=np.stack([item.action for item in items]).astype(np.float16),
            families=np.asarray([int(item.family) for item in items], dtype=np.uint8),
            targets=np.asarray([item.target for item in items], dtype=np.float16),
            bootstrap_masks=np.stack([item.bootstrap_mask for item in items]).astype(np.uint8),
            game_ids=np.asarray(
                [int(item.game_id) % (1 << 64) for item in items], dtype=np.uint64
            ),
            players=np.asarray([item.player for item in items], dtype=np.uint8),
            steps=np.asarray([item.step for item in items], dtype=np.uint32),
            heads=np.asarray([item.head for item in items], dtype=np.uint8),
            epsilons=np.asarray([item.epsilon for item in items], dtype=np.float16),
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
            )
            for index in range(len(self))
        ]


@dataclass(frozen=True, slots=True)
class WorkerResult:
    samples: CompactSamples
    games: int
    wins: tuple[int, int]
    draws: int
    truncated: int
    turns: int
    decisions: int
    forced_choices: int


@dataclass(frozen=True, slots=True)
class _PendingSample:
    state: np.ndarray
    action: np.ndarray
    family: DecisionFamily
    player: int
    step: int


def _coerce_pair(value: float | Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        result = (float(value), float(value))
    else:
        if len(value) != 2:
            raise ValueError(f"{name} must contain one value per player")
        result = (float(value[0]), float(value[1]))
    return result


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
    encoder = encoder or Encoder()
    epsilon_pair = _coerce_pair(epsilons, "epsilons")
    if any(not 0 <= epsilon <= 1 for epsilon in epsilon_pair):
        raise ValueError("epsilons must be in [0, 1]")

    normalized_seed = int(seed) % (1 << 64)
    rngs = [
        np.random.default_rng(np.random.SeedSequence([normalized_seed, 0xA57A, player]))
        for player in range(2)
    ]
    if heads is None:
        head_pair = tuple(int(rng.integers(0, bootstrap_heads)) for rng in rngs)
    else:
        if len(heads) != 2:
            raise ValueError("heads must contain one head per player")
        head_pair = (int(heads[0]), int(heads[1]))
    if any(not 0 <= head < bootstrap_heads for head in head_pair):
        raise ValueError("selected head is outside the bootstrap head range")

    explorations = tuple(
        PlayerExploration(
            head=head_pair[player],
            epsilon=epsilon_pair[player],
            bootstrap_mask=make_bootstrap_mask(
                bootstrap_heads,
                rngs[player],
                inclusion_probability=bootstrap_probability,
                required_head=head_pair[player],
            ),
        )
        for player in range(2)
    )
    for policy in policies:
        if isinstance(policy, ActorPolicy) and policy.bootstrap_heads != bootstrap_heads:
            raise ValueError("ActorPolicy and collector bootstrap head counts differ")

    pending: list[_PendingSample] = []
    player_steps = [0, 0]

    def make_chooser(player: int):
        def choose(player_id: int, decision: Decision) -> Action:
            if player_id != player:
                raise RuntimeError("engine invoked a chooser for the wrong player")
            policy = policies[player]
            if isinstance(policy, ActorPolicy):
                selected = policy.select(decision, explorations[player], rngs[player])
            else:
                selected = _selected_action(policy(player_id, decision), decision)
            if collect_players[player]:
                encoded = encoder.encode_decision(decision.observation, decision)
                selected_index = decision.actions.index(selected)
                pending.append(
                    _PendingSample(
                        state=encoded.state,
                        action=encoded.actions[selected_index],
                        family=encoded.family,
                        player=player,
                        step=player_steps[player],
                    )
                )
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
    )
    result = game.run()
    targets = (0.5, 0.5)
    if result.winner is not None:
        targets = (1.0, 0.0) if result.winner == 0 else (0.0, 1.0)
    resolved_game_id = seed if game_id is None else game_id
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
        )
        for item in pending
    )
    return CollectedGame(
        samples=samples,
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
) -> WorkerResult:
    """Top-level ProcessPool worker; imports no MLX and caches actor archives."""

    if games < 1:
        raise ValueError("games must be positive")
    if len(actor_paths) != 2 or len(baseline_names) != 2:
        raise ValueError("actor_paths and baseline_names must have two entries")
    encoder = Encoder()
    policies: list[EnginePolicy | ActorPolicy] = []
    for player, path in enumerate(actor_paths):
        if path is None:
            policies.append(make_baseline(baseline_names[player], seed + 10_007 * (player + 1)))
        else:
            policies.append(ActorPolicy(_cached_actor(path), encoder))
    flags = tuple(path is not None for path in actor_paths) if collect_players is None else collect_players

    all_items: list[ReplayItem] = []
    wins = [0, 0]
    draws = truncated = turns = decisions = forced_choices = 0
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
        )
        all_items.extend(collected.samples)
        result = collected.result
        if result.winner is None:
            draws += 1
        else:
            wins[result.winner] += 1
        truncated += int(result.truncated)
        turns += result.turns
        decisions += result.decisions
        forced_choices += result.forced_choices

    return WorkerResult(
        samples=CompactSamples.from_items(
            all_items,
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
    )


__all__ = [
    "ActorPolicy",
    "CollectedGame",
    "CompactSamples",
    "PlayerExploration",
    "WorkerResult",
    "clear_actor_cache",
    "collect_game",
    "collect_worker_batch",
]
