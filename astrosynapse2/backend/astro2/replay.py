"""Memory-bounded, decision-stratified replay for Monte-Carlo outcomes."""

from __future__ import annotations

import copy
import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .encoding import FAMILY_COUNT, DecisionFamily

MAX_POLICY_ACTIONS = 64

DEFAULT_CAPACITY_WEIGHTS: dict[DecisionFamily, float] = {
    DecisionFamily.MAIN: 0.82,
    DecisionFamily.DISCARD: 0.03,
    DecisionFamily.SCRAP: 0.04,
    DecisionFamily.DESTROY_BASE: 0.02,
    DecisionFamily.SCRAP_TRADE_ROW: 0.02,
    DecisionFamily.COPY_SHIP: 0.015,
    DecisionFamily.FREE_ACQUIRE: 0.015,
    DecisionFamily.ABILITY_MODE: 0.04,
}

# Preserve useful rare-family coverage without letting a family responsible
# for a fraction of a percent of decisions consume one eighth of every update.
DEFAULT_SAMPLING_WEIGHTS: dict[DecisionFamily, float] = {
    DecisionFamily.MAIN: 0.72,
    DecisionFamily.DISCARD: 0.05,
    DecisionFamily.SCRAP: 0.06,
    DecisionFamily.DESTROY_BASE: 0.035,
    DecisionFamily.SCRAP_TRADE_ROW: 0.035,
    DecisionFamily.COPY_SHIP: 0.025,
    DecisionFamily.FREE_ACQUIRE: 0.025,
    DecisionFamily.ABILITY_MODE: 0.05,
}

# A light rare-family floor without the 100x--180x replay amplification seen
# in the stalled run.  The main family still receives most updates, matching
# the information distribution much more closely.
NATURAL_SAMPLING_WEIGHTS: dict[DecisionFamily, float] = {
    DecisionFamily.MAIN: 0.88,
    DecisionFamily.DISCARD: 0.035,
    DecisionFamily.SCRAP: 0.04,
    DecisionFamily.DESTROY_BASE: 0.012,
    DecisionFamily.SCRAP_TRADE_ROW: 0.008,
    DecisionFamily.COPY_SHIP: 0.007,
    DecisionFamily.FREE_ACQUIRE: 0.008,
    DecisionFamily.ABILITY_MODE: 0.01,
}

_RING_ARRAY_NAMES = (
    "states",
    "actions",
    "targets",
    "bootstrap_masks",
    "game_ids",
    "players",
    "steps",
    "heads",
    "epsilons",
    "td_targets",
    "td_valid",
    "sequences",
)
FULL_REPLAY_FORMAT_VERSION = 2


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
    td_target: float = 0.5
    td_valid: bool = False


@dataclass(frozen=True, slots=True)
class PolicyItem:
    """One complete legal-action decision for actor-critic policy learning."""

    state: np.ndarray
    legal_actions: np.ndarray
    selected_index: int
    family: DecisionFamily | int
    target: float
    behavior_probability: float
    bootstrap_mask: np.ndarray
    game_id: int | str = 0
    player: int = 0
    step: int = 0
    search_policy: np.ndarray | None = None
    search_mask: np.ndarray | None = None
    search_value: float = 0.5
    search_valid: bool = False


@dataclass(frozen=True, slots=True)
class PolicyBatch:
    states: np.ndarray
    legal_actions: np.ndarray
    legal_mask: np.ndarray
    selected_indices: np.ndarray
    families: np.ndarray
    targets: np.ndarray
    behavior_probabilities: np.ndarray
    bootstrap_mask: np.ndarray
    sample_weights: np.ndarray
    game_ids: np.ndarray
    players: np.ndarray
    steps: np.ndarray
    search_policy: np.ndarray
    search_mask: np.ndarray
    search_values: np.ndarray
    search_valid: np.ndarray

    def __len__(self) -> int:
        return int(self.targets.shape[0])


@dataclass(frozen=True, slots=True)
class PreferenceItem:
    """One exact, rules-derived action preference for an observed state."""

    state: np.ndarray
    preferred_action: np.ndarray
    disfavored_action: np.ndarray
    family: DecisionFamily | int
    bootstrap_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class PreferenceBatch:
    states: np.ndarray
    preferred_actions: np.ndarray
    disfavored_actions: np.ndarray
    families: np.ndarray
    bootstrap_mask: np.ndarray

    def __len__(self) -> int:
        return int(self.families.shape[0])


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
    td_targets: np.ndarray
    td_valid: np.ndarray
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
        self.td_targets = np.empty(capacity, dtype=np.float16)
        self.td_valid = np.empty(capacity, dtype=np.uint8)
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
        self.td_targets[index] = item.td_target
        self.td_valid[index] = item.td_valid
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
        td_targets: np.ndarray,
        td_valid: np.ndarray,
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
            td_targets = td_targets[start:]
            td_valid = td_valid[start:]
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
        copy_slice(self.td_targets, td_targets)
        copy_slice(self.td_valid, td_valid)
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
        unique_recent = np.unique(recent)
        if self.size - len(unique_recent) >= general_count:
            # ``ordered`` is a permutation of physical ring indices and
            # ``unique_recent`` is explicitly deduplicated.  Telling NumPy
            # this invariant avoids sorting/scanning the full ring to prove it
            # again for every learner update.
            pool = np.setdiff1d(ordered, unique_recent, assume_unique=True)
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
        family_sampling_weights: dict[DecisionFamily, float] | None = None,
        importance_correct_sampling: bool = False,
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
        self.family_samples_drawn = {family: 0 for family in DecisionFamily}
        self.last_recent_sample_items = 0
        self.last_sample_batch_size = 0
        self.last_importance_weight_min = 1.0
        self.last_importance_weight_max = 1.0
        self.last_importance_effective_sample_size = 0.0

        capacity_weights = family_capacity_weights or DEFAULT_CAPACITY_WEIGHTS
        self.family_sampling_weights = family_sampling_weights or DEFAULT_SAMPLING_WEIGHTS
        self.importance_correct_sampling = bool(importance_correct_sampling)
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

    def rng_state(self) -> dict[str, Any]:
        """Return an isolated, JSON-serializable sampling RNG state."""

        with self._lock:
            return copy.deepcopy(self._rng.bit_generator.state)

    def restore_rng_state(self, state: dict[str, Any]) -> None:
        """Restore the sampling stream at a durable checkpoint boundary."""

        with self._lock:
            self._rng.bit_generator.state = copy.deepcopy(state)

    def clear(self) -> None:
        """Reset logical contents without reallocating the large backing arrays."""

        with self._lock:
            self._sequence = 0
            self.sample_calls = 0
            self.samples_drawn = 0
            self.last_recent_sample_items = 0
            self.last_sample_batch_size = 0
            self.last_importance_weight_min = 1.0
            self.last_importance_weight_max = 1.0
            self.last_importance_effective_sample_size = 0.0
            for family, ring in self._rings.items():
                ring.write_index = 0
                ring.size = 0
                ring.writes = 0
                ring.overwrites = 0
                self.family_samples_drawn[family] = 0

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
        if not 0.0 <= float(item.td_target) <= 1.0:
            raise ValueError("td_target must be in [0, 1]")
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
            td_target=float(item.td_target),
            td_valid=bool(item.td_valid),
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
        td_targets = np.asarray(compact.td_targets)
        td_valid = np.asarray(compact.td_valid)
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
            "td_targets": (count,),
            "td_valid": (count,),
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
            "td_targets": td_targets.shape,
            "td_valid": td_valid.shape,
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
        if not np.isfinite(td_targets).all() or np.any((td_targets < 0) | (td_targets > 1)):
            raise ValueError("td_targets must be finite and in [0, 1]")
        if not np.isin(td_valid, (0, 1)).all():
            raise ValueError("td_valid must be binary")
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
                    td_targets=td_targets[indices],
                    td_valid=td_valid[indices],
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
            weights = self.family_sampling_weights if family_weights is None else family_weights
            counts = _batch_family_counts(batch_size, available, weights, self._rng)
            for family, count in counts.items():
                self.family_samples_drawn[family] += count
            chunks: list[tuple[DecisionFamily, _FamilyRing, np.ndarray]] = []
            recent_items = 0
            for family in available:
                count = counts.get(family, 0)
                if not count:
                    continue
                recent_count = int(round(count * recent))
                recent_items += recent_count
                indices = self._rings[family].sample_indices(
                    count,
                    recent_count,
                    self.recent_window_fraction,
                    self._rng,
                )
                chunks.append((family, self._rings[family], indices))

            families = np.concatenate(
                [
                    np.full(len(indices), int(family), dtype=np.int32)
                    for family, _, indices in chunks
                ]
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
            epsilons = np.concatenate(
                [ring.epsilons[indices] for _, ring, indices in chunks]
            ).astype(np.float32)
            td_targets = np.concatenate(
                [ring.td_targets[indices] for _, ring, indices in chunks]
            ).astype(np.float32)
            td_valid = np.concatenate(
                [ring.td_valid[indices] for _, ring, indices in chunks]
            ).astype(np.float32)
            sequences = np.concatenate([ring.sequences[indices] for _, ring, indices in chunks])
            if self.importance_correct_sampling:
                # Stratification is useful for representation learning, but an
                # uncorrected batch changes the policy-value objective.  Weight
                # each stratum back to its observed behavior-policy frequency.
                # Cumulative writes are used instead of ring occupancy because
                # the deliberately unequal ring capacities would otherwise
                # become a second source of sampling bias.
                total_writes = sum(max(0, self._rings[family].writes) for family in available)
                importance_parts: list[np.ndarray] = []
                for _family, ring, indices in chunks:
                    target_share = ring.writes / max(1, total_writes)
                    sampled_share = len(indices) / max(1, batch_size)
                    importance_parts.append(
                        np.full(
                            len(indices),
                            target_share / max(sampled_share, np.finfo(np.float32).tiny),
                            dtype=np.float32,
                        )
                    )
                sample_weights = np.concatenate(importance_parts)
                # Rounding family counts can move the finite-batch mean a few
                # ulps from one; normalization keeps loss scale stable.
                sample_weights /= max(float(sample_weights.mean()), np.finfo(np.float32).tiny)
            else:
                sample_weights = np.ones(batch_size, dtype=np.float32)

            weight_sum = float(sample_weights.sum())
            weight_square_sum = float(np.square(sample_weights).sum())
            self.last_recent_sample_items = recent_items
            self.last_sample_batch_size = batch_size
            self.last_importance_weight_min = float(sample_weights.min())
            self.last_importance_weight_max = float(sample_weights.max())
            self.last_importance_effective_sample_size = (
                weight_sum * weight_sum / max(weight_square_sum, np.finfo(np.float32).tiny)
            )

            order = self._rng.permutation(len(targets))
            self.sample_calls += 1
            self.samples_drawn += batch_size
            return ReplayBatch(
                states=states[order],
                actions=actions[order],
                families=families[order],
                targets=targets[order],
                bootstrap_mask=masks[order],
                sample_weights=sample_weights[order],
                game_ids=game_ids[order],
                players=players[order],
                steps=steps[order],
                heads=heads[order],
                epsilons=epsilons[order],
                td_targets=td_targets[order],
                td_valid=td_valid[order],
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
                        ring.td_targets,
                        ring.td_valid,
                        ring.sequences,
                    )
                ),
                "importance_correct_sampling": self.importance_correct_sampling,
                "recent_sample_fraction_configured": self.recent_sample_fraction,
                "recent_window_fraction": self.recent_window_fraction,
                "recent_sample_fraction_realized": self.last_recent_sample_items
                / max(1, self.last_sample_batch_size),
                "importance_weights": {
                    "minimum": self.last_importance_weight_min,
                    "maximum": self.last_importance_weight_max,
                    "effective_sample_size": self.last_importance_effective_sample_size,
                    "effective_sample_fraction": self.last_importance_effective_sample_size
                    / max(1, self.last_sample_batch_size),
                },
                "families": {
                    family.name.lower(): {
                        "id": int(family),
                        "size": ring.size,
                        "capacity": ring.capacity,
                        "utilization": ring.size / ring.capacity,
                        "writes": ring.writes,
                        "overwrites": ring.overwrites,
                        "write_share": ring.writes / max(1, writes),
                        "configured_sampling_weight": self.family_sampling_weights.get(family, 0.0),
                        "samples_drawn": self.family_samples_drawn[family],
                        "sample_share": self.family_samples_drawn[family]
                        / max(1, self.samples_drawn),
                        "sample_to_write_ratio": self.family_samples_drawn[family]
                        / max(1, ring.writes),
                    }
                    for family, ring in self._rings.items()
                },
            }

    def snapshot(self, path: str | Path, *, max_items: int) -> int:
        """Persist the globally newest replay rows for exact-enough resume.

        Full replay is intentionally optional because the base-set state tensor
        is large.  A bounded recent journal prevents a restart from resuming a
        mature checkpoint with an empty buffer and a low learning rate.
        """

        if max_items < 1:
            return 0
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.stem}.partial{target.suffix}")
        with self._lock:
            family_parts: list[np.ndarray] = []
            index_parts: list[np.ndarray] = []
            sequence_parts: list[np.ndarray] = []
            for family, ring in self._rings.items():
                indices = ring.chronological_indices()
                if not len(indices):
                    continue
                family_parts.append(np.full(len(indices), int(family), dtype=np.uint8))
                index_parts.append(indices)
                sequence_parts.append(ring.sequences[indices])
            if not sequence_parts:
                return 0
            families = np.concatenate(family_parts)
            physical_indices = np.concatenate(index_parts)
            sequences = np.concatenate(sequence_parts)
            keep = min(int(max_items), len(sequences))
            newest = np.argsort(sequences, kind="stable")[-keep:]
            families = families[newest]
            physical_indices = physical_indices[newest]
            sequences = sequences[newest]

            def gather(name: str) -> np.ndarray:
                first = getattr(next(iter(self._rings.values())), name)
                result = np.empty((keep, *first.shape[1:]), dtype=first.dtype)
                for family, ring in self._rings.items():
                    positions = np.flatnonzero(families == int(family))
                    if len(positions):
                        result[positions] = getattr(ring, name)[physical_indices[positions]]
                return result

            arrays = {
                "states": gather("states"),
                "actions": gather("actions"),
                "families": families,
                "targets": gather("targets"),
                "bootstrap_masks": gather("bootstrap_masks"),
                "game_ids": gather("game_ids"),
                "players": gather("players"),
                "steps": gather("steps"),
                "heads": gather("heads"),
                "epsilons": gather("epsilons"),
                "td_targets": gather("td_targets"),
                "td_valid": gather("td_valid"),
                "sequences": sequences,
                "sequence_cursor": np.asarray(self._sequence, dtype=np.uint64),
                "family_writes": np.asarray(
                    [self._rings[family].writes for family in DecisionFamily],
                    dtype=np.uint64,
                ),
                "family_overwrites": np.asarray(
                    [self._rings[family].overwrites for family in DecisionFamily],
                    dtype=np.uint64,
                ),
                "family_samples_drawn": np.asarray(
                    [self.family_samples_drawn[family] for family in DecisionFamily],
                    dtype=np.uint64,
                ),
                "sample_calls": np.asarray(self.sample_calls, dtype=np.uint64),
                "samples_drawn": np.asarray(self.samples_drawn, dtype=np.uint64),
                "last_recent_sample_items": np.asarray(
                    self.last_recent_sample_items,
                    dtype=np.uint64,
                ),
                "last_sample_batch_size": np.asarray(
                    self.last_sample_batch_size,
                    dtype=np.uint64,
                ),
                "last_importance_weight_min": np.asarray(
                    self.last_importance_weight_min,
                    dtype=np.float64,
                ),
                "last_importance_weight_max": np.asarray(
                    self.last_importance_weight_max,
                    dtype=np.float64,
                ),
                "last_importance_effective_sample_size": np.asarray(
                    self.last_importance_effective_sample_size,
                    dtype=np.float64,
                ),
            }
        np.savez_compressed(temporary, **arrays)
        temporary.replace(target)
        return keep

    def snapshot_full(self, path: str | Path) -> int:
        """Persist every replay stratum with bounded extra memory.

        Restart-boundary snapshots use the rings' physical layout directly,
        avoiding the multi-gigabyte gather allocation required by the compact
        recent-journal format. ``numpy`` streams each archive member while the
        trainer is paused, so peak memory remains close to the live buffer.
        """

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.stem}.partial{target.suffix}")
        with self._lock:
            arrays: dict[str, np.ndarray] = {
                "format_version": np.asarray(FULL_REPLAY_FORMAT_VERSION, dtype=np.uint16),
                "capacity": np.asarray(self.capacity, dtype=np.uint64),
                "state_size": np.asarray(self.state_size, dtype=np.uint32),
                "action_size": np.asarray(self.action_size, dtype=np.uint32),
                "bootstrap_heads": np.asarray(self.bootstrap_heads, dtype=np.uint16),
                "sequence_cursor": np.asarray(self._sequence, dtype=np.uint64),
                "sample_calls": np.asarray(self.sample_calls, dtype=np.uint64),
                "samples_drawn": np.asarray(self.samples_drawn, dtype=np.uint64),
                "last_recent_sample_items": np.asarray(
                    self.last_recent_sample_items,
                    dtype=np.uint64,
                ),
                "last_sample_batch_size": np.asarray(
                    self.last_sample_batch_size,
                    dtype=np.uint64,
                ),
                "last_importance_weight_min": np.asarray(
                    self.last_importance_weight_min,
                    dtype=np.float64,
                ),
                "last_importance_weight_max": np.asarray(
                    self.last_importance_weight_max,
                    dtype=np.float64,
                ),
                "last_importance_effective_sample_size": np.asarray(
                    self.last_importance_effective_sample_size,
                    dtype=np.float64,
                ),
                "family_samples_drawn": np.asarray(
                    [self.family_samples_drawn[family] for family in DecisionFamily],
                    dtype=np.uint64,
                ),
            }
            total = 0
            for family, ring in self._rings.items():
                prefix = f"family_{int(family)}"
                size = int(ring.size)
                total += size
                arrays[f"{prefix}_size"] = np.asarray(size, dtype=np.uint64)
                arrays[f"{prefix}_write_index"] = np.asarray(
                    ring.write_index,
                    dtype=np.uint64,
                )
                arrays[f"{prefix}_writes"] = np.asarray(ring.writes, dtype=np.uint64)
                arrays[f"{prefix}_overwrites"] = np.asarray(
                    ring.overwrites,
                    dtype=np.uint64,
                )
                for name in _RING_ARRAY_NAMES:
                    arrays[f"{prefix}_{name}"] = getattr(ring, name)[:size]
            if total == 0:
                return 0
            np.savez_compressed(temporary, **arrays)
        temporary.replace(target)
        return total

    def _restore_full_archive(self, archive: Any) -> int:
        version = int(np.asarray(archive["format_version"]).item())
        if version != FULL_REPLAY_FORMAT_VERSION:
            raise ValueError(f"unsupported full replay format version: {version}")
        expected = {
            "capacity": self.capacity,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "bootstrap_heads": self.bootstrap_heads,
        }
        for name, value in expected.items():
            if int(np.asarray(archive[name]).item()) != value:
                raise ValueError(f"full replay {name} does not match this run")

        restored = 0
        for family, ring in self._rings.items():
            prefix = f"family_{int(family)}"
            size = int(np.asarray(archive[f"{prefix}_size"]).item())
            write_index = int(np.asarray(archive[f"{prefix}_write_index"]).item())
            if not 0 <= size <= ring.capacity:
                raise ValueError(f"full replay {prefix} size exceeds its capacity")
            if not 0 <= write_index < ring.capacity:
                raise ValueError(f"full replay {prefix} write index is invalid")
            if size < ring.capacity and write_index != size:
                raise ValueError(f"partial full replay {prefix} has a wrapped write index")
            for name in _RING_ARRAY_NAMES:
                source = np.asarray(archive[f"{prefix}_{name}"])
                destination = getattr(ring, name)
                if source.shape != (size, *destination.shape[1:]):
                    raise ValueError(f"full replay {prefix}_{name} has an invalid shape")
                destination[:size] = source
            ring.size = size
            ring.write_index = write_index
            ring.writes = max(size, int(np.asarray(archive[f"{prefix}_writes"]).item()))
            ring.overwrites = max(
                0,
                int(np.asarray(archive[f"{prefix}_overwrites"]).item()),
            )
            restored += size

        self._sequence = int(np.asarray(archive["sequence_cursor"]).item())
        self.sample_calls = max(0, int(np.asarray(archive["sample_calls"]).item()))
        self.samples_drawn = max(0, int(np.asarray(archive["samples_drawn"]).item()))
        self.last_recent_sample_items = max(
            0,
            int(np.asarray(archive["last_recent_sample_items"]).item()),
        )
        self.last_sample_batch_size = max(
            0,
            int(np.asarray(archive["last_sample_batch_size"]).item()),
        )
        self.last_importance_weight_min = float(
            np.asarray(archive["last_importance_weight_min"]).item()
        )
        self.last_importance_weight_max = float(
            np.asarray(archive["last_importance_weight_max"]).item()
        )
        self.last_importance_effective_sample_size = float(
            np.asarray(archive["last_importance_effective_sample_size"]).item()
        )
        family_samples = np.asarray(archive["family_samples_drawn"], dtype=np.uint64)
        if family_samples.shape != (FAMILY_COUNT,):
            raise ValueError("full replay family sample counters are invalid")
        for family in DecisionFamily:
            self.family_samples_drawn[family] = int(family_samples[int(family)])
        return restored

    def restore(self, path: str | Path) -> int:
        """Restore a recent journal or a full restart-boundary snapshot."""

        target = Path(path)
        if not target.is_file():
            return 0
        with np.load(target, allow_pickle=False) as archive:
            if "format_version" in archive.files:
                try:
                    return self._restore_full_archive(archive)
                except Exception:
                    self.clear()
                    raise
            compact = SimpleNamespace(
                states=np.asarray(archive["states"]),
                actions=np.asarray(archive["actions"]),
                families=np.asarray(archive["families"]),
                targets=np.asarray(archive["targets"]),
                bootstrap_masks=np.asarray(archive["bootstrap_masks"]),
                game_ids=np.asarray(archive["game_ids"]),
                players=np.asarray(archive["players"]),
                steps=np.asarray(archive["steps"]),
                heads=np.asarray(archive["heads"]),
                epsilons=np.asarray(archive["epsilons"]),
                td_targets=np.asarray(archive["td_targets"]),
                td_valid=np.asarray(archive["td_valid"]),
            )
            sequence_cursor = int(np.asarray(archive["sequence_cursor"]).item())
            saved_sequences = np.asarray(archive["sequences"], dtype=np.uint64)
            family_writes = (
                np.asarray(archive["family_writes"], dtype=np.uint64)
                if "family_writes" in archive.files
                else None
            )
            family_overwrites = (
                np.asarray(archive["family_overwrites"], dtype=np.uint64)
                if "family_overwrites" in archive.files
                else None
            )
            family_samples_drawn = (
                np.asarray(archive["family_samples_drawn"], dtype=np.uint64)
                if "family_samples_drawn" in archive.files
                else None
            )
            sample_calls = (
                int(np.asarray(archive["sample_calls"]).item())
                if "sample_calls" in archive.files
                else None
            )
            samples_drawn = (
                int(np.asarray(archive["samples_drawn"]).item())
                if "samples_drawn" in archive.files
                else None
            )
        restored = self.extend_compact(compact)
        with self._lock:
            self._sequence = max(self._sequence, sequence_cursor)
            for family in DecisionFamily:
                ring = self._rings[family]
                source = saved_sequences[compact.families == int(family)]
                indices = ring.chronological_indices()
                if len(source) == len(indices):
                    ring.sequences[indices] = source
                if family_writes is not None and family_writes.shape == (FAMILY_COUNT,):
                    ring.writes = max(ring.size, int(family_writes[int(family)]))
                if family_overwrites is not None and family_overwrites.shape == (FAMILY_COUNT,):
                    ring.overwrites = max(0, int(family_overwrites[int(family)]))
                if family_samples_drawn is not None and family_samples_drawn.shape == (
                    FAMILY_COUNT,
                ):
                    self.family_samples_drawn[family] = max(
                        0, int(family_samples_drawn[int(family)])
                    )
            if sample_calls is not None:
                self.sample_calls = max(0, sample_calls)
            if samples_drawn is not None:
                self.samples_drawn = max(0, samples_drawn)
        return restored


class PreferenceReplayBuffer:
    """Small recent ring of exact tactical preferences.

    These examples are generated from rules-level dominance relations, not
    game outcomes.  Keeping them separate prevents their repetition rate from
    distorting outcome replay accounting.
    """

    def __init__(
        self,
        *,
        capacity: int,
        state_size: int,
        action_size: int,
        bootstrap_heads: int,
        storage_dtype: np.dtype | type = np.float16,
        seed: int | None = None,
    ):
        if capacity < 1 or state_size < 1 or action_size < 1 or bootstrap_heads < 1:
            raise ValueError("preference replay dimensions must be positive")
        self.capacity = int(capacity)
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.bootstrap_heads = int(bootstrap_heads)
        dtype = np.dtype(storage_dtype)
        self.states = np.empty((capacity, state_size), dtype=dtype)
        self.preferred_actions = np.empty((capacity, action_size), dtype=dtype)
        self.disfavored_actions = np.empty((capacity, action_size), dtype=dtype)
        self.families = np.empty(capacity, dtype=np.uint8)
        self.bootstrap_masks = np.empty((capacity, bootstrap_heads), dtype=np.uint8)
        self.write_index = 0
        self.size = 0
        self.writes = 0
        self.overwrites = 0
        self.samples_drawn = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def clear(self) -> None:
        self.write_index = 0
        self.size = 0

    def snapshot(self, path: str | Path, *, max_items: int = 0) -> int:
        count = self.size if max_items <= 0 else min(self.size, int(max_items))
        if self.size < self.capacity:
            ordered = np.arange(self.size, dtype=np.int64)
        else:
            ordered = np.concatenate(
                (np.arange(self.write_index, self.capacity), np.arange(0, self.write_index))
            )
        indices = ordered[-count:]
        np.savez_compressed(
            Path(path),
            format_version=np.asarray(1, dtype=np.int32),
            states=self.states[indices],
            preferred_actions=self.preferred_actions[indices],
            disfavored_actions=self.disfavored_actions[indices],
            families=self.families[indices],
            bootstrap_masks=self.bootstrap_masks[indices],
        )
        return count

    def restore(self, path: str | Path) -> int:
        with np.load(Path(path), allow_pickle=False) as archive:
            compact = type(
                "PreferenceSnapshot",
                (),
                {
                    "states": archive["states"],
                    "preferred_actions": archive["preferred_actions"],
                    "disfavored_actions": archive["disfavored_actions"],
                    "families": archive["families"],
                    "bootstrap_masks": archive["bootstrap_masks"],
                },
            )
            self.clear()
            return self.extend_compact(compact)

    def extend_compact(self, compact: Any) -> int:
        states = np.asarray(compact.states)
        preferred = np.asarray(compact.preferred_actions)
        disfavored = np.asarray(compact.disfavored_actions)
        families = np.asarray(compact.families)
        bootstrap_masks = np.asarray(compact.bootstrap_masks)
        count = int(len(families))
        expected = {
            "states": (count, self.state_size),
            "preferred_actions": (count, self.action_size),
            "disfavored_actions": (count, self.action_size),
            "families": (count,),
            "bootstrap_masks": (count, self.bootstrap_heads),
        }
        actual = {
            "states": states.shape,
            "preferred_actions": preferred.shape,
            "disfavored_actions": disfavored.shape,
            "families": families.shape,
            "bootstrap_masks": bootstrap_masks.shape,
        }
        invalid = [name for name, shape in actual.items() if shape != expected[name]]
        if invalid:
            raise ValueError(
                "invalid compact preference shapes: "
                + ", ".join(f"{name}={actual[name]} expected {expected[name]}" for name in invalid)
            )
        if count == 0:
            return 0
        if not (
            np.isfinite(states).all()
            and np.isfinite(preferred).all()
            and np.isfinite(disfavored).all()
        ):
            raise ValueError("preference features must be finite")
        if families.dtype.kind not in "iu" or np.any((families < 0) | (families >= FAMILY_COUNT)):
            raise ValueError("preferences contain an unknown decision family")
        if (
            bootstrap_masks.dtype.kind not in "biu"
            or not np.isin(bootstrap_masks, (0, 1)).all()
            or (count and np.any(np.sum(bootstrap_masks, axis=1) == 0))
        ):
            raise ValueError("preference bootstrap masks must be binary and nonempty")

        incoming = count
        if count > self.capacity:
            start = count - self.capacity
            states = states[start:]
            preferred = preferred[start:]
            disfavored = disfavored[start:]
            families = families[start:]
            bootstrap_masks = bootstrap_masks[start:]
            count = self.capacity
        self.overwrites += max(0, self.size + incoming - self.capacity)
        first = min(count, self.capacity - self.write_index)
        second = count - first

        def copy(destination: np.ndarray, source: np.ndarray) -> None:
            destination[self.write_index : self.write_index + first] = source[:first]
            if second:
                destination[:second] = source[first:]

        copy(self.states, states)
        copy(self.preferred_actions, preferred)
        copy(self.disfavored_actions, disfavored)
        copy(self.families, families)
        copy(self.bootstrap_masks, bootstrap_masks)
        self.write_index = (self.write_index + count) % self.capacity
        self.size = min(self.capacity, self.size + count)
        self.writes += incoming
        return incoming

    def sample(self, batch_size: int) -> PreferenceBatch:
        if self.size < 1:
            raise ValueError("cannot sample empty preference replay")
        count = min(max(1, int(batch_size)), self.size)
        indices = self._rng.choice(self.size, size=count, replace=False)
        self.samples_drawn += count
        return PreferenceBatch(
            states=self.states[indices].astype(np.float32),
            preferred_actions=self.preferred_actions[indices].astype(np.float32),
            disfavored_actions=self.disfavored_actions[indices].astype(np.float32),
            families=self.families[indices].astype(np.int32),
            bootstrap_mask=self.bootstrap_masks[indices].astype(np.float32),
        )

    def metrics(self) -> dict[str, int | float]:
        return {
            "size": self.size,
            "capacity": self.capacity,
            "utilization": self.size / self.capacity,
            "writes": self.writes,
            "overwrites": self.overwrites,
            "samples_drawn": self.samples_drawn,
            "storage_bytes": int(
                self.states.nbytes
                + self.preferred_actions.nbytes
                + self.disfavored_actions.nbytes
                + self.families.nbytes
                + self.bootstrap_masks.nbytes
            ),
        }


# Concise compatibility alias for trainer code.
ReplayBuffer = StratifiedReplayBuffer


class GameBalancedPolicyReplayBuffer:
    """Bounded policy replay sampled uniformly by player-game, then decision.

    Evicting whole player-game trajectories and sampling one decision from a
    uniformly selected trajectory prevents a 400-decision game from receiving
    20 times the policy weight of a concise 20-decision game.
    """

    def __init__(
        self,
        capacity: int,
        state_size: int,
        action_size: int,
        bootstrap_heads: int,
        *,
        max_actions: int = MAX_POLICY_ACTIONS,
        max_decisions_per_player_game: int = 0,
        family_balanced: bool = False,
        seed: int = 0,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.bootstrap_heads = int(bootstrap_heads)
        self.max_actions = int(max_actions)
        self.max_decisions_per_player_game = max(0, int(max_decisions_per_player_game))
        self.family_balanced = bool(family_balanced)
        self._episodes: OrderedDict[tuple[int, int], list[PolicyItem]] = OrderedDict()
        self._size = 0
        self._writes = 0
        self._evicted_decisions = 0
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return self._size

    def _validate(self, item: PolicyItem) -> PolicyItem:
        state = np.asarray(item.state, dtype=np.float16)
        actions = np.asarray(item.legal_actions, dtype=np.float16)
        mask = np.asarray(item.bootstrap_mask, dtype=np.uint8)
        search_policy = np.asarray(
            item.search_policy if item.search_policy is not None else np.zeros(len(actions)),
            dtype=np.float16,
        )
        search_mask = np.asarray(
            item.search_mask if item.search_mask is not None else np.zeros(len(actions)),
            dtype=np.uint8,
        )
        family = DecisionFamily(int(item.family))
        if state.shape != (self.state_size,):
            raise ValueError("policy state has the wrong shape")
        if (
            actions.ndim != 2
            or actions.shape[1] != self.action_size
            or len(actions) < 2
            or len(actions) > self.max_actions
        ):
            raise ValueError("policy legal_actions must contain at least two encoded actions")
        if not 0 <= int(item.selected_index) < len(actions):
            raise ValueError("selected policy action is outside the legal set")
        if (
            mask.shape != (self.bootstrap_heads,)
            or not np.isin(mask, (0, 1)).all()
            or not mask.any()
        ):
            raise ValueError("policy bootstrap mask must be binary and nonempty")
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError("policy features must be finite")
        if not 0 <= float(item.target) <= 1:
            raise ValueError("policy target must be in [0, 1]")
        if not 0 < float(item.behavior_probability) <= 1:
            raise ValueError("behavior probability must be in (0, 1]")
        if search_policy.shape != (len(actions),) or search_mask.shape != (len(actions),):
            raise ValueError("search targets must align with the legal action set")
        if not np.isfinite(search_policy).all() or np.any(search_policy < 0):
            raise ValueError("search policy targets must be finite and nonnegative")
        if not np.isin(search_mask, (0, 1)).all():
            raise ValueError("search mask must be binary")
        search_valid = bool(item.search_valid)
        if search_valid:
            mass = float(np.sum(search_policy))
            if not np.any(search_mask) or not np.isfinite(mass) or abs(mass - 1.0) > 0.02:
                raise ValueError("valid search policy targets must sum to one over a nonempty mask")
            if np.any(search_policy[search_mask == 0] > 1e-3):
                raise ValueError("search policy assigns mass to an unsearched action")
            if not 0 <= float(item.search_value) <= 1:
                raise ValueError("search value must be in [0, 1]")
        return PolicyItem(
            state=state,
            legal_actions=actions,
            selected_index=int(item.selected_index),
            family=family,
            target=float(item.target),
            behavior_probability=float(item.behavior_probability),
            bootstrap_mask=mask,
            game_id=item.game_id,
            player=int(item.player),
            step=int(item.step),
            search_policy=search_policy,
            search_mask=search_mask,
            search_value=float(item.search_value),
            search_valid=search_valid,
        )

    def _compact_episode(self, episode: list[PolicyItem]) -> list[PolicyItem]:
        limit = self.max_decisions_per_player_game
        if not limit or len(episode) <= limit:
            return episode
        strata: dict[tuple[int, int], list[PolicyItem]] = {}
        for item in episode:
            turn = float(item.state[11]) * 50.0 if len(item.state) > 11 else 0.0
            phase = 0 if turn <= 6 else 1 if turn <= 16 else 2
            family = int(item.family) if self.family_balanced else 0
            strata.setdefault((family, phase), []).append(item)
        # Round-robin strata prevent common main-phase decisions from erasing
        # rare tactical families. Random selection within each stratum is a
        # true per-game reservoir rather than a fixed early-turn slice.
        selected: list[PolicyItem] = [item for item in episode if item.search_valid][:limit]
        selected_ids = {id(item) for item in selected}
        pools = [
            [item for item in rows if id(item) not in selected_ids]
            for _key, rows in sorted(strata.items())
        ]
        while len(selected) < limit and any(pools):
            for pool in pools:
                if not pool or len(selected) >= limit:
                    continue
                index = int(self._rng.integers(0, len(pool)))
                selected.append(pool.pop(index))
        return sorted(selected, key=lambda item: item.step)

    def extend(self, items: list[PolicyItem] | tuple[PolicyItem, ...]) -> int:
        validated = [self._validate(item) for item in items]
        grouped: OrderedDict[tuple[int, int], list[PolicyItem]] = OrderedDict()
        for item in validated:
            key = (int(_numeric_game_id(item.game_id)), item.player)
            grouped.setdefault(key, []).append(item)
        with self._lock:
            for key, episode in grouped.items():
                episode = self._compact_episode(episode)
                previous = self._episodes.pop(key, None)
                if previous:
                    self._size -= len(previous)
                self._episodes[key] = episode
                self._size += len(episode)
            self._writes += len(validated)
            while self._size > self.capacity and self._episodes:
                _key, removed = self._episodes.popitem(last=False)
                self._size -= len(removed)
                self._evicted_decisions += len(removed)
        return len(validated)

    def extend_compact(self, compact: Any) -> int:
        offsets = np.asarray(compact.action_offsets, dtype=np.int64)
        count = int(len(compact.targets))
        if offsets.shape != (count + 1,) or offsets[0] != 0:
            raise ValueError("invalid compact policy offsets")
        actions = np.asarray(compact.legal_actions)
        if offsets[-1] != len(actions):
            raise ValueError("compact policy offsets do not cover legal actions")
        items = [
            PolicyItem(
                state=np.asarray(compact.states[index], dtype=np.float32),
                legal_actions=np.asarray(
                    actions[offsets[index] : offsets[index + 1]], dtype=np.float32
                ),
                selected_index=int(compact.selected_indices[index]),
                family=int(compact.families[index]),
                target=float(compact.targets[index]),
                behavior_probability=float(compact.behavior_probabilities[index]),
                bootstrap_mask=np.asarray(compact.bootstrap_masks[index], dtype=np.uint8),
                game_id=int(compact.game_ids[index]),
                player=int(compact.players[index]),
                step=int(compact.steps[index]),
                search_policy=np.asarray(
                    compact.search_policy[offsets[index] : offsets[index + 1]],
                    dtype=np.float32,
                ),
                search_mask=np.asarray(
                    compact.search_mask[offsets[index] : offsets[index + 1]],
                    dtype=np.uint8,
                ),
                search_value=float(compact.search_values[index]),
                search_valid=bool(compact.search_valid[index]),
            )
            for index in range(count)
        ]
        return self.extend(items)

    def sample(self, batch_size: int) -> PolicyBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        with self._lock:
            if not self._episodes:
                raise ValueError("cannot sample empty policy replay")
            keys = list(self._episodes)
            chosen_keys = self._rng.choice(
                len(keys), size=batch_size, replace=batch_size > len(keys)
            )
            items: list[PolicyItem] = []
            for key_index in chosen_keys:
                episode = self._episodes[keys[int(key_index)]]
                phases = ([], [], [])
                for item in episode:
                    turn = float(item.state[11]) * 50.0 if len(item.state) > 11 else 0.0
                    phase = 0 if turn <= 6 else 1 if turn <= 16 else 2
                    phases[phase].append(item)
                available_phases = [phase for phase, rows in enumerate(phases) if rows]
                phase = available_phases[int(self._rng.integers(0, len(available_phases)))]
                rows = phases[phase]
                if self.family_balanced:
                    families = sorted({int(item.family) for item in rows})
                    family = families[int(self._rng.integers(0, len(families)))]
                    rows = [item for item in rows if int(item.family) == family]
                items.append(rows[int(self._rng.integers(0, len(rows)))])
        maximum = self.max_actions
        legal_actions = np.zeros((batch_size, maximum, self.action_size), dtype=np.float32)
        legal_mask = np.zeros((batch_size, maximum), dtype=np.float32)
        search_policy = np.zeros((batch_size, maximum), dtype=np.float32)
        search_mask = np.zeros((batch_size, maximum), dtype=np.float32)
        for index, item in enumerate(items):
            count = len(item.legal_actions)
            legal_actions[index, :count] = item.legal_actions
            legal_mask[index, :count] = 1.0
            search_policy[index, :count] = item.search_policy
            search_mask[index, :count] = item.search_mask
        return PolicyBatch(
            states=np.stack([item.state for item in items]).astype(np.float32),
            legal_actions=legal_actions,
            legal_mask=legal_mask,
            selected_indices=np.asarray([item.selected_index for item in items], dtype=np.int32),
            families=np.asarray([int(item.family) for item in items], dtype=np.int32),
            targets=np.asarray([item.target for item in items], dtype=np.float32),
            behavior_probabilities=np.asarray(
                [item.behavior_probability for item in items], dtype=np.float32
            ),
            bootstrap_mask=np.stack([item.bootstrap_mask for item in items]).astype(np.float32),
            sample_weights=np.ones(batch_size, dtype=np.float32),
            game_ids=np.asarray(
                [int(_numeric_game_id(item.game_id)) for item in items], dtype=np.uint64
            ),
            players=np.asarray([item.player for item in items], dtype=np.uint8),
            steps=np.asarray([item.step for item in items], dtype=np.uint32),
            search_policy=search_policy,
            search_mask=search_mask,
            search_values=np.asarray([item.search_value for item in items], dtype=np.float32),
            search_valid=np.asarray([item.search_valid for item in items], dtype=np.float32),
        )

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            lengths = [len(items) for items in self._episodes.values()]
            return {
                "size": self._size,
                "capacity": self.capacity,
                "utilization": self._size / self.capacity,
                "player_games": len(lengths),
                "mean_decisions_per_player_game": float(np.mean(lengths)) if lengths else 0.0,
                "writes": self._writes,
                "evicted_decisions": self._evicted_decisions,
                "sampling": "uniform_player_game_then_turn_phase_then_decision",
                "max_decisions_per_player_game": self.max_decisions_per_player_game,
                "family_balanced": self.family_balanced,
                "searched_decisions": sum(
                    int(item.search_valid)
                    for episode in self._episodes.values()
                    for item in episode
                ),
            }

    def clear(self) -> None:
        with self._lock:
            self._episodes.clear()
            self._size = 0

    def snapshot(self, path: str | Path, *, max_items: int = 0) -> int:
        """Persist a ragged, whole-episode Astro5 replay archive."""

        with self._lock:
            episodes = list(self._episodes.items())
            if max_items > 0:
                selected: list[tuple[tuple[int, int], list[PolicyItem]]] = []
                count = 0
                for key, rows in reversed(episodes):
                    if selected and count + len(rows) > max_items:
                        break
                    selected.append((key, rows))
                    count += len(rows)
                episodes = list(reversed(selected))
            items = [item for _key, rows in episodes for item in rows]
            episode_lengths = np.asarray([len(rows) for _key, rows in episodes], dtype=np.uint16)
            if items:
                counts = np.asarray([len(item.legal_actions) for item in items], dtype=np.uint16)
                offsets = np.concatenate(
                    (np.zeros(1, dtype=np.uint64), np.cumsum(counts, dtype=np.uint64))
                )
                legal_actions = np.concatenate([item.legal_actions for item in items])
                search_policy = np.concatenate([item.search_policy for item in items])
                search_mask = np.concatenate([item.search_mask for item in items])
            else:
                offsets = np.zeros(1, dtype=np.uint64)
                legal_actions = np.empty((0, self.action_size), dtype=np.float16)
                search_policy = np.empty(0, dtype=np.float16)
                search_mask = np.empty(0, dtype=np.uint8)
            # Policy archives are intentionally uncompressed. On the target
            # Mac, ZIP compression made checkpoint pauses CPU-bound for tens
            # of seconds; the project explicitly budgets ample local disk.
            np.savez(
                Path(path),
                format_version=np.asarray(1, dtype=np.int32),
                capacity=np.asarray(self.capacity, dtype=np.int64),
                state_size=np.asarray(self.state_size, dtype=np.int32),
                action_size=np.asarray(self.action_size, dtype=np.int32),
                bootstrap_heads=np.asarray(self.bootstrap_heads, dtype=np.int32),
                episode_game_ids=np.asarray([key[0] for key, _rows in episodes], dtype=np.uint64),
                episode_players=np.asarray([key[1] for key, _rows in episodes], dtype=np.uint8),
                episode_lengths=episode_lengths,
                states=(
                    np.stack([item.state for item in items]).astype(np.float16)
                    if items
                    else np.empty((0, self.state_size), dtype=np.float16)
                ),
                legal_actions=legal_actions.astype(np.float16),
                action_offsets=offsets,
                selected_indices=np.asarray([item.selected_index for item in items], dtype=np.uint16),
                families=np.asarray([int(item.family) for item in items], dtype=np.uint8),
                targets=np.asarray([item.target for item in items], dtype=np.float16),
                behavior_probabilities=np.asarray(
                    [item.behavior_probability for item in items], dtype=np.float16
                ),
                bootstrap_masks=(
                    np.stack([item.bootstrap_mask for item in items]).astype(np.uint8)
                    if items
                    else np.empty((0, self.bootstrap_heads), dtype=np.uint8)
                ),
                steps=np.asarray([item.step for item in items], dtype=np.uint32),
                search_policy=search_policy.astype(np.float16),
                search_mask=search_mask.astype(np.uint8),
                search_values=np.asarray([item.search_value for item in items], dtype=np.float16),
                search_valid=np.asarray([item.search_valid for item in items], dtype=np.uint8),
            )
            return len(items)

    def restore(self, path: str | Path) -> int:
        with np.load(Path(path), allow_pickle=False) as archive:
            if int(archive["state_size"]) != self.state_size or int(archive["action_size"]) != self.action_size:
                raise ValueError("policy replay snapshot dimensions do not match")
            offsets = np.asarray(archive["action_offsets"], dtype=np.int64)
            game_ids = np.asarray(archive["episode_game_ids"], dtype=np.uint64)
            players = np.asarray(archive["episode_players"], dtype=np.uint8)
            lengths = np.asarray(archive["episode_lengths"], dtype=np.int64)
            item_game_ids = np.repeat(game_ids, lengths)
            item_players = np.repeat(players, lengths)
            compact = type(
                "PolicySnapshot",
                (),
                {
                    "states": archive["states"],
                    "legal_actions": archive["legal_actions"],
                    "action_offsets": offsets,
                    "selected_indices": archive["selected_indices"],
                    "families": archive["families"],
                    "targets": archive["targets"],
                    "behavior_probabilities": archive["behavior_probabilities"],
                    "bootstrap_masks": archive["bootstrap_masks"],
                    "game_ids": item_game_ids,
                    "players": item_players,
                    "steps": archive["steps"],
                    "search_policy": archive["search_policy"],
                    "search_mask": archive["search_mask"],
                    "search_values": archive["search_values"],
                    "search_valid": archive["search_valid"],
                },
            )
            self.clear()
            return self.extend_compact(compact)
