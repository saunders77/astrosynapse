"""Memory-bounded, decision-stratified replay for Monte-Carlo outcomes."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from .encoding import FAMILY_COUNT, DecisionFamily

DEFAULT_CAPACITY_WEIGHTS: dict[DecisionFamily, float] = {
    DecisionFamily.MAIN: 0.50,
    DecisionFamily.DISCARD: 0.09,
    DecisionFamily.SCRAP: 0.12,
    DecisionFamily.DESTROY_BASE: 0.06,
    DecisionFamily.SCRAP_TRADE_ROW: 0.06,
    DecisionFamily.COPY_SHIP: 0.05,
    DecisionFamily.FREE_ACQUIRE: 0.04,
    DecisionFamily.ABILITY_MODE: 0.08,
}


def make_bootstrap_mask(
    heads: int,
    rng: np.random.Generator,
    *,
    inclusion_probability: float = 0.8,
    required_head: int | None = None,
) -> np.ndarray:
    """Draw a nonempty Bernoulli bootstrap mask.

    ``required_head`` is useful for deep exploration: the head that generated a
    player-game trajectory is guaranteed to learn from that same trajectory.
    The collector draws this once per player-game, not once per decision.
    """

    if heads < 1:
        raise ValueError("heads must be positive")
    if not 0 < inclusion_probability <= 1:
        raise ValueError("inclusion_probability must be in (0, 1]")
    if required_head is not None and not 0 <= required_head < heads:
        raise ValueError("required_head is outside the bootstrap head range")
    mask = (rng.random(heads) < inclusion_probability).astype(np.uint8)
    if required_head is not None:
        mask[required_head] = 1
    elif not mask.any():
        mask[int(rng.integers(0, heads))] = 1
    return mask


def _numeric_game_id(value: int | str) -> np.uint64:
    if isinstance(value, (int, np.integer)):
        return np.uint64(int(value) % (1 << 64))
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return np.frombuffer(digest, dtype="<u8")[0]


@dataclass(frozen=True, slots=True)
class ReplayItem:
    """One chosen action labelled with its acting player's terminal result."""

    state: np.ndarray
    action: np.ndarray
    family: DecisionFamily | int
    target: float
    bootstrap_mask: np.ndarray
    game_id: int | str = 0
    player: int = 0
    step: int = 0
    head: int = 0
    epsilon: float = 0.0


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    states: np.ndarray
    actions: np.ndarray
    families: np.ndarray
    targets: np.ndarray
    bootstrap_mask: np.ndarray
    sample_weights: np.ndarray
    game_ids: np.ndarray
    players: np.ndarray
    steps: np.ndarray
    heads: np.ndarray
    epsilons: np.ndarray
    sequences: np.ndarray

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def learner_inputs(self) -> dict[str, np.ndarray]:
        """Return exactly the arrays consumed by ``bootstrap_bce_loss``."""

        return {
            "states": self.states,
            "actions": self.actions,
            "families": self.families,
            "targets": self.targets,
            "bootstrap_mask": self.bootstrap_mask,
            "sample_weights": self.sample_weights,
        }


class _FamilyRing:
    def __init__(
        self,
        capacity: int,
        state_size: int,
        action_size: int,
        bootstrap_heads: int,
        storage_dtype: np.dtype,
    ):
        self.capacity = capacity
        self.states = np.empty((capacity, state_size), dtype=storage_dtype)
        self.actions = np.empty((capacity, action_size), dtype=storage_dtype)
        self.targets = np.empty(capacity, dtype=np.float16)
        self.bootstrap_masks = np.empty((capacity, bootstrap_heads), dtype=np.uint8)
        self.game_ids = np.empty(capacity, dtype=np.uint64)
        self.players = np.empty(capacity, dtype=np.uint8)
        self.steps = np.empty(capacity, dtype=np.uint32)
        self.heads = np.empty(capacity, dtype=np.uint8)
        self.epsilons = np.empty(capacity, dtype=np.float16)
        self.sequences = np.empty(capacity, dtype=np.uint64)
        self.write_index = 0
        self.size = 0
        self.writes = 0
        self.overwrites = 0

    def add(self, item: ReplayItem, sequence: int) -> None:
        index = self.write_index
        if self.size == self.capacity:
            self.overwrites += 1
        self.states[index] = item.state
        self.actions[index] = item.action
        self.targets[index] = item.target
        self.bootstrap_masks[index] = item.bootstrap_mask
        self.game_ids[index] = _numeric_game_id(item.game_id)
        self.players[index] = item.player
        self.steps[index] = item.step
        self.heads[index] = item.head
        self.epsilons[index] = item.epsilon
        self.sequences[index] = sequence
        self.write_index = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.writes += 1

    def add_arrays(
        self,
        *,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        bootstrap_masks: np.ndarray,
        game_ids: np.ndarray,
        players: np.ndarray,
        steps: np.ndarray,
        heads: np.ndarray,
        epsilons: np.ndarray,
        sequences: np.ndarray,
    ) -> int:
        """Append one family as slices, avoiding per-decision Python objects."""

        incoming_count = int(len(targets))
        if incoming_count == 0:
            return 0
        count = incoming_count
        if count > self.capacity:
            start = count - self.capacity
            states = states[start:]
            actions = actions[start:]
            targets = targets[start:]
            bootstrap_masks = bootstrap_masks[start:]
            game_ids = game_ids[start:]
            players = players[start:]
            steps = steps[start:]
            heads = heads[start:]
            epsilons = epsilons[start:]
            sequences = sequences[start:]
            count = self.capacity

        # Match scalar insertion telemetry even when an oversized block lets us
        # copy only its newest ringful.  The omitted prefix was still written
        # logically and immediately evicted by later rows in the same block.
        evictions = max(0, self.size + incoming_count - self.capacity)
        first = min(count, self.capacity - self.write_index)
        second = count - first

        def copy_slice(destination: np.ndarray, source: np.ndarray) -> None:
            destination[self.write_index : self.write_index + first] = source[:first]
            if second:
                destination[:second] = source[first:]

        copy_slice(self.states, states)
        copy_slice(self.actions, actions)
        copy_slice(self.targets, targets)
        copy_slice(self.bootstrap_masks, bootstrap_masks)
        copy_slice(self.game_ids, game_ids)
        copy_slice(self.players, players)
        copy_slice(self.steps, steps)
        copy_slice(self.heads, heads)
        copy_slice(self.epsilons, epsilons)
        copy_slice(self.sequences, sequences)

        self.write_index = (self.write_index + count) % self.capacity
        self.size = min(self.capacity, self.size + count)
        self.writes += incoming_count
        self.overwrites += evictions
        return incoming_count

    def chronological_indices(self) -> np.ndarray:
        if self.size < self.capacity:
            return np.arange(self.size, dtype=np.int64)
        return np.concatenate(
            (
                np.arange(self.write_index, self.capacity, dtype=np.int64),
                np.arange(0, self.write_index, dtype=np.int64),
            )
        )

    def sample_indices(
        self,
        count: int,
        recent_count: int,
        recent_window_fraction: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if count < 1 or self.size < 1:
            return np.empty(0, dtype=np.int64)
        ordered = self.chronological_indices()
        recent_count = min(count, recent_count)
        window_size = min(
            self.size,
            max(recent_count, 1, int(np.ceil(self.size * recent_window_fraction))),
        )
        recent_pool = ordered[-window_size:]
        recent = rng.choice(
            recent_pool, size=recent_count, replace=recent_count > len(recent_pool)
        ).astype(np.int64, copy=False)

        general_count = count - recent_count
        if general_count == 0:
            return recent
        # Prefer a no-duplicate batch when enough items exist.  Sampling with
        # replacement remains defined during warmup and for tiny rare strata.
        if self.size - len(np.unique(recent)) >= general_count:
            pool = np.setdiff1d(ordered, recent, assume_unique=False)
            general = rng.choice(pool, size=general_count, replace=False)
        else:
            general = rng.choice(ordered, size=general_count, replace=general_count > self.size)
        result = np.concatenate((recent, general)).astype(np.int64, copy=False)
        rng.shuffle(result)
        return result


def _allocate_capacities(
    capacity: int,
    weights: dict[DecisionFamily, float],
) -> dict[DecisionFamily, int]:
    if capacity < FAMILY_COUNT:
        raise ValueError(f"capacity must be at least {FAMILY_COUNT}")
    values = np.array([max(0.0, float(weights.get(family, 0.0))) for family in DecisionFamily])
    if values.sum() <= 0:
        raise ValueError("at least one family capacity weight must be positive")
    # Reserve one slot for every family so a rare chooser can never disappear.
    remaining = capacity - FAMILY_COUNT
    raw = values / values.sum() * remaining
    allocation = np.floor(raw).astype(np.int64) + 1
    for index in np.argsort(-(raw - np.floor(raw)))[: capacity - int(allocation.sum())]:
        allocation[index] += 1
    return {family: int(allocation[int(family)]) for family in DecisionFamily}


def _batch_family_counts(
    batch_size: int,
    available: list[DecisionFamily],
    weights: dict[DecisionFamily, float] | None,
    rng: np.random.Generator,
) -> dict[DecisionFamily, int]:
    if weights is None:
        base, extra = divmod(batch_size, len(available))
        shuffled = list(available)
        rng.shuffle(shuffled)
        counts = {family: base for family in available}
        for family in shuffled[:extra]:
            counts[family] += 1
        return counts

    values = np.array([max(0.0, float(weights.get(family, 0.0))) for family in available])
    if values.sum() <= 0:
        raise ValueError("sampling weights give no mass to a nonempty family")
    values /= values.sum()
    raw = values * batch_size
    counts_array = np.floor(raw).astype(np.int64)
    if batch_size >= len(available):
        counts_array[counts_array == 0] = 1
        while counts_array.sum() > batch_size:
            donors = np.flatnonzero(counts_array > 1)
            counts_array[donors[np.argmax(counts_array[donors] - raw[donors])]] -= 1
    remainder = batch_size - int(counts_array.sum())
    order = np.argsort(-(raw - np.floor(raw)))
    for index in order[:remainder]:
        counts_array[index] += 1
    return {family: int(counts_array[index]) for index, family in enumerate(available)}


class StratifiedReplayBuffer:
    """Independent NumPy rings prevent common decisions evicting rare ones.

    Storage defaults to float16, which keeps the recommended 900k buffer near
    2.7 GB with the base-set encoder and leaves unified-memory headroom on a
    16 GB Mac. Counts used by the encoder are small integers and are represented
    exactly in float16; sampled learner batches are converted back to float32.
    """

    def __init__(
        self,
        *,
        capacity: int,
        state_size: int,
        action_size: int,
        bootstrap_heads: int,
        family_capacity_weights: dict[DecisionFamily, float] | None = None,
        recent_sample_fraction: float = 0.35,
        recent_window_fraction: float = 0.10,
        storage_dtype: np.dtype | type = np.float16,
        seed: int | None = None,
    ):
        if state_size < 1 or action_size < 1 or bootstrap_heads < 1:
            raise ValueError("state_size, action_size, and bootstrap_heads must be positive")
        if not 0 <= recent_sample_fraction <= 1:
            raise ValueError("recent_sample_fraction must be in [0, 1]")
        if not 0 < recent_window_fraction <= 1:
            raise ValueError("recent_window_fraction must be in (0, 1]")
        dtype = np.dtype(storage_dtype)
        if dtype.kind != "f":
            raise ValueError("storage_dtype must be floating point")

        self.capacity = int(capacity)
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.bootstrap_heads = int(bootstrap_heads)
        self.recent_sample_fraction = float(recent_sample_fraction)
        self.recent_window_fraction = float(recent_window_fraction)
        self.storage_dtype = dtype
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()
        self._sequence = 0
        self.sample_calls = 0
        self.samples_drawn = 0

        capacity_weights = family_capacity_weights or DEFAULT_CAPACITY_WEIGHTS
        self.family_capacities = _allocate_capacities(self.capacity, capacity_weights)
        self._rings = {
            family: _FamilyRing(
                self.family_capacities[family],
                self.state_size,
                self.action_size,
                self.bootstrap_heads,
                self.storage_dtype,
            )
            for family in DecisionFamily
        }

    def __len__(self) -> int:
        with self._lock:
            return sum(ring.size for ring in self._rings.values())

    def _validate_item(self, item: ReplayItem) -> ReplayItem:
        family = DecisionFamily(int(item.family))
        state = np.asarray(item.state, dtype=np.float32)
        action = np.asarray(item.action, dtype=np.float32)
        mask = np.asarray(item.bootstrap_mask, dtype=np.uint8)
        if state.shape != (self.state_size,):
            raise ValueError(f"state must have shape {(self.state_size,)}, got {state.shape}")
        if action.shape != (self.action_size,):
            raise ValueError(f"action must have shape {(self.action_size,)}, got {action.shape}")
        if mask.shape != (self.bootstrap_heads,):
            raise ValueError(
                f"bootstrap_mask must have shape {(self.bootstrap_heads,)}, got {mask.shape}"
            )
        if not np.isin(mask, (0, 1)).all() or not mask.any():
            raise ValueError("bootstrap_mask must be binary and nonempty")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("state and action features must be finite")
        if not 0.0 <= float(item.target) <= 1.0:
            raise ValueError("target must be in [0, 1]")
        if not 0 <= int(item.player) <= 255:
            raise ValueError("player must fit in uint8")
        if not 0 <= int(item.head) < self.bootstrap_heads:
            raise ValueError("head is outside the bootstrap head range")
        if not 0 <= float(item.epsilon) <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        return ReplayItem(
            state=state,
            action=action,
            family=family,
            target=float(item.target),
            bootstrap_mask=mask,
            game_id=item.game_id,
            player=int(item.player),
            step=int(item.step),
            head=int(item.head),
            epsilon=float(item.epsilon),
        )

    def add(self, item: ReplayItem) -> None:
        validated = self._validate_item(item)
        with self._lock:
            self._sequence += 1
            self._rings[DecisionFamily(int(validated.family))].add(validated, self._sequence)

    def extend(self, items: list[ReplayItem] | tuple[ReplayItem, ...]) -> int:
        # Validate before mutating so a malformed actor batch is atomic.
        validated = [self._validate_item(item) for item in items]
        with self._lock:
            for item in validated:
                self._sequence += 1
                self._rings[DecisionFamily(int(item.family))].add(item, self._sequence)
        return len(validated)

    def extend_compact(self, compact: Any) -> int:
        """Append a worker's contiguous sample block without object expansion.

        ``CompactSamples`` deliberately lives in :mod:`selfplay`, which imports
        this module.  A small structural interface here avoids that import cycle
        and keeps IPC arrays in float16 until their final replay assignment.
        """

        states = np.asarray(compact.states)
        actions = np.asarray(compact.actions)
        families = np.asarray(compact.families)
        targets = np.asarray(compact.targets)
        masks = np.asarray(compact.bootstrap_masks)
        game_ids = np.asarray(compact.game_ids)
        players = np.asarray(compact.players)
        steps = np.asarray(compact.steps)
        heads = np.asarray(compact.heads)
        epsilons = np.asarray(compact.epsilons)
        count = int(len(targets))
        expected = {
            "states": (count, self.state_size),
            "actions": (count, self.action_size),
            "families": (count,),
            "targets": (count,),
            "bootstrap_masks": (count, self.bootstrap_heads),
            "game_ids": (count,),
            "players": (count,),
            "steps": (count,),
            "heads": (count,),
            "epsilons": (count,),
        }
        actual = {
            "states": states.shape,
            "actions": actions.shape,
            "families": families.shape,
            "targets": targets.shape,
            "bootstrap_masks": masks.shape,
            "game_ids": game_ids.shape,
            "players": players.shape,
            "steps": steps.shape,
            "heads": heads.shape,
            "epsilons": epsilons.shape,
        }
        invalid = [name for name, shape in actual.items() if shape != expected[name]]
        if invalid:
            details = ", ".join(
                f"{name}={actual[name]} expected {expected[name]}" for name in invalid
            )
            raise ValueError(f"invalid compact replay shapes: {details}")
        if count == 0:
            return 0
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("state and action features must be finite")
        if not np.isfinite(targets).all() or np.any((targets < 0) | (targets > 1)):
            raise ValueError("targets must be finite and in [0, 1]")
        if families.dtype.kind not in "iu" or np.any((families < 0) | (families >= FAMILY_COUNT)):
            raise ValueError("compact samples contain an unknown decision family")
        if heads.dtype.kind not in "iu" or np.any((heads < 0) | (heads >= self.bootstrap_heads)):
            raise ValueError("compact samples contain an unknown bootstrap head")
        if players.dtype.kind not in "iu" or np.any((players < 0) | (players > 1)):
            raise ValueError("compact samples contain an unknown player")
        if (
            steps.dtype.kind not in "iu"
            or game_ids.dtype.kind not in "iu"
            or np.any(steps < 0)
            or np.any(game_ids < 0)
        ):
            raise ValueError("compact steps and game IDs must be nonnegative integers")
        if not np.isfinite(epsilons).all() or np.any((epsilons < 0) | (epsilons > 1)):
            raise ValueError("epsilons must be finite and in [0, 1]")
        if not np.isin(masks, (0, 1)).all() or np.any(masks.sum(axis=1) == 0):
            raise ValueError("bootstrap masks must be binary and nonempty")

        with self._lock:
            sequence_values = np.arange(
                self._sequence + 1,
                self._sequence + count + 1,
                dtype=np.uint64,
            )
            for family in DecisionFamily:
                indices = np.flatnonzero(families == int(family))
                if not len(indices):
                    continue
                self._rings[family].add_arrays(
                    states=states[indices],
                    actions=actions[indices],
                    targets=targets[indices],
                    bootstrap_masks=masks[indices],
                    game_ids=game_ids[indices],
                    players=players[indices],
                    steps=steps[indices],
                    heads=heads[indices],
                    epsilons=epsilons[indices],
                    sequences=sequence_values[indices],
                )
            self._sequence += count
        return count

    def sample(
        self,
        batch_size: int,
        *,
        recent_fraction: float | None = None,
        family_weights: dict[DecisionFamily, float] | None = None,
    ) -> ReplayBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        recent = self.recent_sample_fraction if recent_fraction is None else recent_fraction
        if not 0 <= recent <= 1:
            raise ValueError("recent_fraction must be in [0, 1]")

        with self._lock:
            available = [family for family, ring in self._rings.items() if ring.size]
            if not available:
                raise ValueError("cannot sample an empty replay buffer")
            counts = _batch_family_counts(batch_size, available, family_weights, self._rng)
            chunks: list[tuple[DecisionFamily, _FamilyRing, np.ndarray]] = []
            for family in available:
                count = counts.get(family, 0)
                if not count:
                    continue
                recent_count = int(round(count * recent))
                indices = self._rings[family].sample_indices(
                    count,
                    recent_count,
                    self.recent_window_fraction,
                    self._rng,
                )
                chunks.append((family, self._rings[family], indices))

            families = np.concatenate(
                [np.full(len(indices), int(family), dtype=np.int32) for family, _, indices in chunks]
            )
            states = np.concatenate([ring.states[indices] for _, ring, indices in chunks]).astype(
                np.float32
            )
            actions = np.concatenate([ring.actions[indices] for _, ring, indices in chunks]).astype(
                np.float32
            )
            targets = np.concatenate([ring.targets[indices] for _, ring, indices in chunks]).astype(
                np.float32
            )
            masks = np.concatenate(
                [ring.bootstrap_masks[indices] for _, ring, indices in chunks]
            ).astype(np.float32)
            game_ids = np.concatenate([ring.game_ids[indices] for _, ring, indices in chunks])
            players = np.concatenate([ring.players[indices] for _, ring, indices in chunks])
            steps = np.concatenate([ring.steps[indices] for _, ring, indices in chunks])
            heads = np.concatenate([ring.heads[indices] for _, ring, indices in chunks])
            epsilons = np.concatenate([ring.epsilons[indices] for _, ring, indices in chunks]).astype(
                np.float32
            )
            sequences = np.concatenate([ring.sequences[indices] for _, ring, indices in chunks])

            order = self._rng.permutation(len(targets))
            self.sample_calls += 1
            self.samples_drawn += batch_size
            return ReplayBatch(
                states=states[order],
                actions=actions[order],
                families=families[order],
                targets=targets[order],
                bootstrap_mask=masks[order],
                sample_weights=np.ones(batch_size, dtype=np.float32),
                game_ids=game_ids[order],
                players=players[order],
                steps=steps[order],
                heads=heads[order],
                epsilons=epsilons[order],
                sequences=sequences[order],
            )

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            size = sum(ring.size for ring in self._rings.values())
            writes = sum(ring.writes for ring in self._rings.values())
            overwrites = sum(ring.overwrites for ring in self._rings.values())
            return {
                "size": size,
                "capacity": self.capacity,
                "utilization": size / self.capacity,
                "writes": writes,
                "overwrites": overwrites,
                "sample_calls": self.sample_calls,
                "samples_drawn": self.samples_drawn,
                "storage_bytes": sum(
                    array.nbytes
                    for ring in self._rings.values()
                    for array in (
                        ring.states,
                        ring.actions,
                        ring.targets,
                        ring.bootstrap_masks,
                        ring.game_ids,
                        ring.players,
                        ring.steps,
                        ring.heads,
                        ring.epsilons,
                        ring.sequences,
                    )
                ),
                "families": {
                    family.name.lower(): {
                        "id": int(family),
                        "size": ring.size,
                        "capacity": ring.capacity,
                        "utilization": ring.size / ring.capacity,
                        "writes": ring.writes,
                        "overwrites": ring.overwrites,
                    }
                    for family, ring in self._rings.items()
                },
            }


# Concise compatibility alias for trainer code.
ReplayBuffer = StratifiedReplayBuffer
