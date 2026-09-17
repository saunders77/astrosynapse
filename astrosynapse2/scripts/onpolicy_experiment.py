"""Train and screen portable checkpoints without modifying the live registry."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import signal
import sqlite3
import time
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from dataclasses import asdict

import numpy as np
from astro2.experiment_control import code_identity, next_learning_rate
from astro2.model import (
    export_actor,
    load_model,
    load_optimizer_state,
    save_model,
    save_optimizer_state,
)
from astro2.onpolicy import collect_trajectory, masked_log_policy, ppo_loss
from astro2.planning import PlanningConfig
from planning_experiment import pair, summary

STOP = False


def stop(*_):
    global STOP
    STOP = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="5583687405f845fc")
    p.add_argument("--opponent", default="08aa018c672847d9")
    p.add_argument("--iterations", type=int, default=80)
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=2e-6)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-pairs", type=int, default=128)
    p.add_argument("--seed", type=int, default=2026090921)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--train-scope", choices=["full", "heads"], default="full")
    p.add_argument("--strategic-only", action="store_true")
    p.add_argument("--automatic-model")
    p.add_argument("--initial-optimizer")
    p.add_argument("--rules-version", type=int, choices=[1, 2], default=1)
    p.add_argument("--adaptive-kl", action="store_true")
    p.add_argument("--max-learning-rate", type=float, default=0.0002)
    args = p.parse_args()
    if args.temperature <= 0 or args.games < 2 or args.batch_size < 2:
        raise ValueError("invalid training settings")
    if not 0 < args.learning_rate <= args.max_learning_rate or args.target_kl <= 0:
        raise ValueError("invalid learning rate or KL budget")
    out = Path(args.output).resolve()
    out.mkdir(exist_ok=True, parents=True)
    resume_state = None
    manifest_path = out / "manifest.json"
    import astro2

    identity = code_identity(Path(astro2.__file__).parent, Path(__file__).parent)
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("learner_version") != 3:
            raise ValueError("this checkpoint requires an explicit fork into a new experiment")
        if manifest.get("code_identity") != identity:
            raise ValueError(
                "source changed: resume with the frozen runtime or start an explicit fork"
            )
        requested_iterations = args.iterations
        for key in vars(args):
            if key not in {"iterations", "workers", "resume", "output"}:
                setattr(args, key, manifest.get(key, getattr(args, key)))
        args.iterations = max(requested_iterations, manifest["iterations"])
        resume_state = json.loads((out / "state.json").read_text())
    elif manifest_path.exists():
        raise ValueError("use --resume or a new output directory")
    c = sqlite3.connect("file:data/astrosynapse2.sqlite3?mode=ro", uri=True)
    source = c.execute("select path from checkpoints where id=?", (args.model,)).fetchone()
    model_path = source[0] if source else str(Path(args.model).resolve())
    opponent_row = c.execute(
        "select actor_path from checkpoints where id=?", (args.opponent,)
    ).fetchone()
    opponent_path = opponent_row[0] if opponent_row else str(Path(args.opponent).resolve())
    if not args.resume:
        manifest_path.write_text(
            json.dumps(
                {
                    **vars(args),
                    "learner_version": 3,
                    "code_identity": identity,
                    "source_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
                    "opponent_sha256": hashlib.sha256(Path(opponent_path).read_bytes()).hexdigest(),
                },
                indent=2,
            )
        )
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_map

    mx.random.seed(args.seed)
    mx.set_cache_limit(512 * 1024 * 1024)
    model, spec = load_model(resume_state["model"] if resume_state else model_path)
    model.train()
    if args.train_scope == "heads":
        model.freeze()
        for output in model.head_outputs:
            output.unfreeze()
        model.value_output.unfreeze()
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    if resume_state and not load_optimizer_state(optimizer, resume_state["optimizer"]):
        raise RuntimeError("resumable optimizer artifact is missing")
    if (
        not resume_state
        and args.initial_optimizer
        and not load_optimizer_state(optimizer, args.initial_optimizer)
    ):
        raise RuntimeError("initial optimizer artifact is missing")
    current_lr = (
        resume_state.get("learning_rate", args.learning_rate)
        if resume_state
        else args.learning_rate
    )
    optimizer.learning_rate = current_lr
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0xE7A1]))
    started = time.monotonic()
    updates = total_games = 0
    first_iteration = 0
    elapsed_before = 0.0
    if resume_state:
        updates = resume_state["updates"]
        total_games = resume_state["games"]
        first_iteration = resume_state["iteration"] + 1
        elapsed_before = resume_state["elapsed"]
        rng.bit_generator.state = resume_state["rng"]
        eval_rng.bit_generator.state = resume_state["eval_rng"]
    runtime = out / (f"resume-{total_games:08d}.actor.npz" if resume_state else "initial.actor.npz")
    export_actor(model, spec, runtime, compressed=False)
    auto_path = (
        (
            str(Path(args.automatic_model).resolve())
            if args.automatic_model
            else str(out / "initial.actor.npz")
        )
        if args.strategic_only
        else None
    )
    latest = {}

    def loss(*arrays):
        nonlocal latest
        value, latest = ppo_loss(model, *arrays, temperature=args.temperature)
        return value

    gradient = nn.value_and_grad(model, loss)
    with (
        concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool,
        (out / "metrics.jsonl").open("a", buffering=1) as log,
    ):
        for iteration in range(first_iteration, args.iterations):
            if STOP or (out / "STOP").exists():
                break
            iteration_start = time.monotonic()
            seed = int(rng.integers(2**62))
            tasks = [
                (
                    str(runtime),
                    opponent_path,
                    args.temperature,
                    seed,
                    i,
                    auto_path,
                    args.rules_version,
                )
                for i in range(args.games)
            ]
            trajectories = list(pool.map(collect_trajectory, tasks, chunksize=2))
            rollout_seconds = time.monotonic() - iteration_start
            learning_start = time.monotonic()
            valid = [t for t in trajectories if not t.truncated]
            states = np.asarray([s for t in valid for s in t.states], dtype=np.float32)
            actions = [a for t in valid for a in t.actions]
            families = np.asarray([f for t in valid for f in t.families], dtype=np.int32)
            selected = np.asarray([i for t in valid for i in t.selected], dtype=np.int32)
            old_logp = np.asarray([v for t in valid for v in t.log_probabilities], dtype=np.float32)
            old_log_policies = [v for t in valid for v in t.log_policies]
            if len(states) < 2:
                raise RuntimeError("rollout produced no usable decisions")
            # Keep an independent fixed probe of this rollout for measuring the
            # complete update; minibatch-before-update KL can miss the last step.
            probe_indices = np.linspace(0, len(states) - 1, min(1024, len(states)), dtype=int)
            targets = np.asarray([t.target for t in valid for _ in t.states], dtype=np.float32)
            baseline = []
            for offset in range(0, len(states), 512):
                values = mx.sigmoid(
                    model.state_values(
                        mx.array(states[offset : offset + 512]),
                        mx.array(families[offset : offset + 512]),
                    )
                ).mean(axis=1)
                baseline.extend(np.asarray(values))
            advantages = targets - np.asarray(baseline)
            # A single scale over the full fresh rollout keeps the sign and
            # per-decision return weighting; no family/turn replay resampling.
            advantages /= max(float(advantages.std()), 0.1)
            norms = []
            diagnostics = []
            early_kl_stop = False
            # MLX arrays are immutable. Copy the tree containers to retain the
            # exact pre-update tensors and optimizer moments for rollback.
            saved_parameters = tree_map(lambda value: value, model.parameters())
            saved_optimizer = tree_map(lambda value: value, optimizer.state)
            saved_updates = updates
            for _epoch in range(args.epochs):
                order = rng.permutation(len(states))
                for offset in range(0, len(order), args.batch_size):
                    indices = order[offset : offset + args.batch_size]
                    count = max(len(actions[i]) for i in indices)
                    count = ((count + 3) // 4) * 4
                    legal = np.zeros((len(indices), count, spec.action_size), dtype=np.float32)
                    mask = np.zeros((len(indices), count), dtype=np.float32)
                    old_policy = np.zeros((len(indices), count), dtype=np.float32)
                    for j, i in enumerate(indices):
                        legal[j, : len(actions[i])] = actions[i]
                        mask[j, : len(actions[i])] = 1
                        old_policy[j, : len(actions[i])] = old_log_policies[i]
                    arrays = tuple(
                        mx.array(x)
                        for x in (
                            states[indices],
                            legal,
                            mask,
                            families[indices],
                            selected[indices],
                            old_logp[indices],
                            advantages[indices],
                            targets[indices],
                            old_policy,
                        )
                    )
                    value, grads = gradient(*arrays)
                    grads, norm = optim.clip_grad_norm(grads, 1.0)
                    mx.eval(value, norm, *latest.values())
                    stats = {k: float(v.item()) for k, v in latest.items()}
                    if not all(
                        np.isfinite(v)
                        for v in (float(value.item()), float(norm.item()), *stats.values())
                    ):
                        raise RuntimeError("nonfinite loss, gradient norm, or policy diagnostics")
                    if stats["categorical_kl"] > args.target_kl:
                        early_kl_stop = True
                        break
                    optimizer.update(model, grads)
                    mx.eval(model.parameters(), optimizer.state)
                    updates += 1
                    norms.append(float(norm.item()))
                    diagnostics.append(stats)
                if early_kl_stop:
                    break
            probe_count = max(len(actions[i]) for i in probe_indices)
            probe_actions = np.zeros(
                (len(probe_indices), probe_count, spec.action_size), dtype=np.float32
            )
            probe_mask = np.zeros((len(probe_indices), probe_count), dtype=np.float32)
            probe_old = np.zeros_like(probe_mask)
            for j, i in enumerate(probe_indices):
                probe_actions[j, : len(actions[i])] = actions[i]
                probe_mask[j, : len(actions[i])] = 1
                probe_old[j, : len(actions[i])] = old_log_policies[i]
            probe_new, _ = masked_log_policy(
                model,
                mx.array(states[probe_indices]),
                mx.array(probe_actions),
                mx.array(probe_mask),
                mx.array(families[probe_indices]),
                args.temperature,
            )
            post_update_kl = float(
                mx.mean(
                    mx.sum(
                        mx.exp(mx.array(probe_old))
                        * mx.array(probe_mask)
                        * (mx.array(probe_old) - probe_new),
                        axis=1,
                    )
                ).item()
            )
            used_lr = current_lr
            if not np.isfinite(post_update_kl):
                raise RuntimeError("nonfinite post-update KL")
            update_rejected = post_update_kl > 2 * args.target_kl
            if update_rejected:
                model.update(saved_parameters)
                optimizer.state = saved_optimizer
                updates = saved_updates
                mx.eval(model.parameters(), optimizer.state)
            if args.adaptive_kl:
                current_lr = next_learning_rate(
                    current_lr,
                    args.learning_rate,
                    args.max_learning_rate,
                    args.target_kl,
                    post_update_kl,
                    update_rejected,
                )
                optimizer.learning_rate = current_lr
            total_games += args.games
            learning_seconds = time.monotonic() - learning_start
            checkpoint_start = time.monotonic()
            checkpoint = out / f"g{total_games:08d}.safetensors"
            save_model(model, spec, checkpoint)
            runtime = checkpoint.with_suffix(".actor.npz")
            export_actor(model, spec, runtime, compressed=False)
            optimizer_path = checkpoint.with_suffix(".optimizer.npz")
            save_optimizer_state(optimizer, optimizer_path)
            record = dict(
                iteration=iteration,
                games=total_games,
                updates=updates,
                positions=len(states),
                rollout_score=float(np.mean([t.target for t in trajectories])),
                truncated=sum(t.truncated for t in trajectories),
                elapsed=elapsed_before + time.monotonic() - started,
                iteration_seconds=time.monotonic() - iteration_start,
                rollout_seconds=rollout_seconds,
                learning_seconds=learning_seconds,
                checkpoint_seconds=time.monotonic() - checkpoint_start,
                workers=args.workers,
                updates_this_iteration=updates - saved_updates,
                gradient_norm_mean=float(np.mean(norms)) if norms else None,
                early_kl_stop=early_kl_stop,
                checkpoint=str(checkpoint),
                learning_rate=used_lr,
                next_learning_rate=current_lr,
                post_update_kl=post_update_kl,
                update_rejected=update_rejected,
                **(
                    {k: float(np.mean([d[k] for d in diagnostics])) for k in diagnostics[0]}
                    if diagnostics
                    else {}
                ),
            )
            if (iteration + 1) % args.eval_every == 0 or iteration == args.iterations - 1:
                evaluation_start = time.monotonic()
                eval_seed = int(eval_rng.integers(2**62))
                results = list(
                    pool.map(
                        pair,
                        [
                            (
                                str(runtime),
                                opponent_path,
                                asdict(
                                    PlanningConfig(
                                        rollouts=0,
                                        native=True,
                                        auto_actor_path=auto_path,
                                        rules_version=args.rules_version,
                                    )
                                ),
                                eval_seed,
                                i,
                            )
                            for i in range(args.eval_pairs)
                        ],
                        chunksize=2,
                    )
                )
                record["evaluation"] = summary(results)
                record["evaluation_seconds"] = time.monotonic() - evaluation_start
                record["evaluation_seed"] = eval_seed
                (out / f"g{total_games:08d}.pairs.json").write_text(json.dumps(results))
            record["total_iteration_seconds"] = time.monotonic() - iteration_start
            log.write(json.dumps(record) + "\n")
            state = dict(
                iteration=iteration,
                games=total_games,
                updates=updates,
                model=str(checkpoint),
                optimizer=str(optimizer_path),
                rng=rng.bit_generator.state,
                eval_rng=eval_rng.bit_generator.state,
                elapsed=elapsed_before + time.monotonic() - started,
                learning_rate=current_lr,
            )
            temporary = out / "state.tmp"
            temporary.write_text(json.dumps(state, indent=2))
            temporary.replace(out / "state.json")
            print(json.dumps(record), flush=True)
            mx.clear_cache()
    (out / "complete.json").write_text(
        json.dumps(
            dict(
                games=total_games,
                updates=updates,
                checkpoint=str(runtime),
                stopped=STOP or (out / "STOP").exists(),
                auto_actor_path=auto_path,
                elapsed=elapsed_before + time.monotonic() - started,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
