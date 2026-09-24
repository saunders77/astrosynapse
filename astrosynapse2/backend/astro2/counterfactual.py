"""Paired terminal action comparisons on public beliefs, without critic targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arena import _ActorChooser, _derived_seed
from .engine import (
    ActionKind,
    DecisionFamily,
    Game,
    GameConfig,
    Seating,
    _TruncateGame,
    model_action_indices,
)
from .engine_encoding import EngineEncoder
from .onpolicy import cached_actor, masked_log_policy
from .planning import sample_belief, strategic_decision


@dataclass(frozen=True)
class ComparisonConfig:
    roots: int = 2
    anchors: int = 6
    rollouts: int = 16
    max_actions: int = 4
    rules_version: int = 1

    def __post_init__(self):
        if min(self.roots, self.anchors, self.rollouts) < 1 or self.max_actions < 2:
            raise ValueError("comparison budgets must be positive, with at least two actions")
        if self.rules_version not in (1, 2):
            raise ValueError("unsupported rules version")


def economic_indices(decision, selected):
    """Compare deck-economy decisions after routine plays, not play permutations."""
    if decision.family != DecisionFamily.MAIN or not strategic_decision(decision):
        return []
    kinds = {ActionKind.ACQUIRE, ActionKind.SCRAP_FOR_ABILITY, ActionKind.END_TURN}
    if selected.kind not in kinds:
        return []
    # Preserve the actually chosen action (including its engine-local locator).
    choices, seen = [selected], {selected.semantic_key}
    for index in model_action_indices(decision):
        action = decision.actions[index]
        if action.kind in kinds and action.semantic_key not in seen:
            choices.append(action)
            seen.add(action.semantic_key)
    return choices if len(choices) > 1 else []


def paired_counts(outcomes):
    """Discard a whole chance sample if ANY candidate truncated; no optimistic fills."""
    outcomes = np.asarray(outcomes, dtype=np.float32)
    if outcomes.ndim != 2 or len(outcomes) < 2:
        raise ValueError("need a reference and at least one alternative")
    valid = np.all(np.isfinite(outcomes), axis=0)
    scores = outcomes[:, valid]
    if np.any((scores != 0) & (scores != 1)):
        raise ValueError("terminal comparisons require binary natural outcomes")
    return (
        np.sum(scores > scores[:1], axis=1).astype(np.float32),
        np.sum(scores < scores[:1], axis=1).astype(np.float32),
        int(valid.sum()),
    )


def _reservoir(items, item, seen, capacity, rng):
    if len(items) < capacity:
        items.append(item())
    else:
        index = int(rng.integers(seen))
        if index < capacity:
            items[index] = item()


def continuation_choosers(actor, opponent, seat, chance_seed):
    """Fresh per-seat RNGs for every alternative, with the correct opponent."""
    return {
        pid: _ActorChooser(
            actor if pid == seat else opponent,
            EngineEncoder(version=(actor if pid == seat else opponent).spec.encoder_version),
            _derived_seed(chance_seed, pid, "continuation"),
        )
        for pid in (0, 1)
    }


def collect_comparisons(task):
    actor_path, opponent_path, seed, game_index, settings = task
    config = ComparisonConfig(**settings)
    actor, opponent = cached_actor(actor_path), cached_actor(opponent_path)
    seat = game_index % 2
    encoder = EngineEncoder(version=actor.spec.encoder_version)
    selector = np.random.default_rng(_derived_seed(seed, game_index, "root-selection"))
    anchor_rng = np.random.default_rng(_derived_seed(seed, game_index, "anchor-selection"))
    roots, anchors = [], []
    root_seen = anchor_seen = 0

    def position(decision):
        encoded = encoder.encode_decision(decision.observation, decision)
        eligible = np.asarray(model_action_indices(decision), dtype=np.int32)
        actions = encoded.actions[eligible]
        logits = actor.predict_options(encoded.state, actions, int(encoded.family)).mean(axis=1)
        return dict(
            state=encoded.state,
            actions=actions,
            family=int(encoded.family),
            teacher_logits=logits,
            wins=np.zeros(len(actions), dtype=np.float32),
            losses=np.zeros(len(actions), dtype=np.float32),
            reference=0,
            alternatives=0,
            samples=0,
            game=game_index,
        ), eligible

    def observe(pid, decision, selected):
        nonlocal root_seen, anchor_seen
        if pid != seat:
            return
        if len(model_action_indices(decision)) > 1:
            anchor_seen += 1
            _reservoir(
                anchors, lambda: position(decision)[0], anchor_seen, config.anchors, anchor_rng
            )
        candidates = economic_indices(decision, selected)
        if candidates:
            root_seen += 1
            _reservoir(
                roots,
                lambda: (game.fork(), decision, candidates),
                root_seen,
                config.roots,
                selector,
            )

    learner = _ActorChooser(actor, encoder, _derived_seed(seed, game_index, "learner"))
    other = _ActorChooser(
        opponent,
        EngineEncoder(version=opponent.spec.encoder_version),
        _derived_seed(seed, game_index, "opponent"),
    )
    game = Game(
        choosers=(learner, other) if seat == 0 else (other, learner),
        config=GameConfig(
            seed=_derived_seed(seed, game_index, "game"),
            seating=Seating.FIXED,
            starting_player=0,
            max_turns=180,
            max_actions_per_turn=160,
            rules_version=config.rules_version,
        ),
        decision_hook=observe,
    )
    played = game.run()
    stats = dict(
        game=game_index,
        eligible_roots=root_seen,
        roots=0,
        branches=0,
        truncated_branches=0,
        discarded_samples=0,
        source_game_truncated=bool(played.truncated),
        discordant=0,
    )
    # These labels do not use the source game's outcome. Sampled roots remain
    # legitimate even if that source continuation happens to truncate.
    positions = list(anchors)
    for root_index, (snapshot, decision, candidates) in enumerate(roots):
        root_seed = _derived_seed(seed, game_index, f"comparisons-{root_index}")
        rng = np.random.default_rng(root_seed)
        if len(candidates) > config.max_actions:
            selected = rng.choice(
                np.arange(1, len(candidates)), config.max_actions - 1, replace=False
            )
            candidates = [candidates[0], *(candidates[int(i)] for i in selected)]
        outcomes = np.full((len(candidates), config.rollouts), np.nan, dtype=np.float32)
        for draw in range(config.rollouts):
            chance_seed = _derived_seed(root_seed, draw, "belief")
            belief = sample_belief(snapshot, seat, chance_seed)
            for index, action in enumerate(candidates):
                branch = belief.fork()
                # Each seat gets its actual frozen policy and its own reset RNG.
                # Never substitute the learner for the opponent in continuations.
                branch.choosers = continuation_choosers(actor, opponent, seat, chance_seed)
                try:
                    result = branch.continue_from_main_action(action)
                    if not result.truncated and result.winner is not None:
                        outcomes[index, draw] = float(result.winner == seat)
                except _TruncateGame:
                    # The historical engine can raise during the initial forced
                    # action, before continue_from_main_action's internal try.
                    pass
                stats["branches"] += 1
                stats["truncated_branches"] += int(not np.isfinite(outcomes[index, draw]))
        wins, losses, samples = paired_counts(outcomes)
        stats["discarded_samples"] += config.rollouts - samples
        if not samples:
            continue
        row, eligible = position(decision)
        # Map by object identity: semantic equality can hide distinct locators.
        local = [
            next(i for i, j in enumerate(eligible) if decision.actions[j] is action)
            for action in candidates
        ]
        row["reference"] = local[0]
        row["wins"][local] = wins
        row["losses"][local] = losses
        row.update(samples=samples, alternatives=len(candidates) - 1)
        positions.append(row)
        stats["roots"] += 1
        stats["discordant"] += int(wins.sum() + losses.sum())
    return positions, stats


def save_positions(path, rows):
    """Portable ragged arrays only; no pickle or simulator-hidden state on disk."""
    counts = np.asarray([len(r["actions"]) for r in rows], dtype=np.int32)
    arrays = {
        key: np.asarray([r[key] for r in rows])
        for key in ["state", "family", "reference", "alternatives", "samples", "game"]
    }
    arrays["offsets"] = np.concatenate([[0], np.cumsum(counts)])
    for key in ["actions", "teacher_logits", "wins", "losses"]:
        arrays[key] = np.concatenate([r[key] for r in rows])
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_positions(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
        rows = []
        for i in range(len(data["state"])):
            start, end = data["offsets"][i : i + 2]
            row = {
                key: data[key][i]
                for key in ["state", "family", "reference", "alternatives", "samples", "game"]
            }
            row.update(
                {
                    key: data[key][start:end]
                    for key in ["actions", "teacher_logits", "wins", "losses"]
                }
            )
            rows.append(row)
        return rows


def training_batch(rows):
    count = max(len(r["actions"]) for r in rows)
    actions = np.zeros((len(rows), count, rows[0]["actions"].shape[1]), dtype=np.float32)
    mask = np.zeros((len(rows), count), dtype=np.float32)
    teacher, wins, losses = (np.zeros_like(mask) for _ in range(3))
    for i, row in enumerate(rows):
        n = len(row["actions"])
        actions[i, :n] = row["actions"]
        mask[i, :n] = 1
        for target, key in [(teacher, "teacher_logits"), (wins, "wins"), (losses, "losses")]:
            target[i, :n] = row[key]
    return (
        np.asarray([r["state"] for r in rows], dtype=np.float32),
        actions,
        mask,
        np.asarray([r["family"] for r in rows], dtype=np.int32),
        teacher,
        wins,
        losses,
        np.asarray([r["reference"] for r in rows], dtype=np.int32),
        np.asarray([r["samples"] for r in rows], dtype=np.float32),
        np.asarray([r["alternatives"] for r in rows], dtype=np.float32),
    )


def comparison_loss(
    model,
    states,
    actions,
    mask,
    families,
    teacher,
    wins,
    losses,
    reference,
    samples,
    alternatives,
    *,
    pair_temperature=0.1,
    anchor_temperature=0.03,
    kl_weight=0.1,
):
    import mlx.core as mx

    logp, _ = masked_log_policy(model, states, actions, mask, families, anchor_temperature)
    reference_logp = mx.take_along_axis(logp, reference[:, None], axis=1)
    gap = (logp - reference_logp) * (anchor_temperature / pair_temperature)
    # Keep BOTH directions of each paired result, rather than converting a
    # noisy sample winner into a certain one-hot search target. Ties give no
    # preference gradient; proposals receive gradients even at tiny old pi(a).
    pair_losses = mx.sum(wins * mx.logaddexp(0, -gap) + losses * mx.logaddexp(0, gap), axis=1)
    roots = (samples > 0).astype(mx.float32)
    ranking = mx.sum(pair_losses / mx.maximum(samples * alternatives, 1)) / mx.maximum(
        mx.sum(roots), 1
    )
    teacher = mx.where(mask > 0, teacher / anchor_temperature, -1e9)
    old = teacher - mx.logsumexp(teacher, axis=1, keepdims=True)
    kl = mx.mean(mx.sum(mx.exp(old) * (old - logp) * mask, axis=1))
    return ranking + kl_weight * kl, dict(ranking_loss=ranking, anchor_kl=kl)
