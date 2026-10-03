"""Policy matches with separately selected, architecture-independent critic diagnostics."""

from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .arena import _ActorChooser, _derived_seed
from .autopilot_registry import ValuePredictor, read
from .engine import Game, GameConfig, Seating
from .engine_encoding import EngineEncoder
from .experiment_control import atomic_json
from .onpolicy import cached_actor

_PREDICTORS = {}


def measured_pair(task):
    model_a, model_b, critic_a, critic_b, seed, index = task
    actors = [cached_actor(model_a), cached_actor(model_b)]
    critics = []
    for spec in (critic_a, critic_b):
        if spec is None:
            critics.append(None)
        else:
            key = (spec["kind"], spec["path"])
            if key not in _PREDICTORS:
                _PREDICTORS[key] = ValuePredictor(spec)
            critics.append(_PREDICTORS[key])
    scores = []
    measurements = [[], []]
    truncated = 0
    for seat in (0, 1):
        predictions = [[], []]
        choosers = [
            _ActorChooser(
                actor,
                EngineEncoder(version=actor.spec.encoder_version),
                _derived_seed(seed, index, f"policy-{i}"),
            )
            for i, actor in enumerate(actors)
        ]

        def hook(pid, decision, _action, seat=seat, predictions=predictions):
            model_index = 0 if pid == seat else 1
            critic = critics[model_index]
            if critic is not None:
                predictions[model_index].append(critic.predict(decision))

        game = Game(
            choosers=tuple(choosers if seat == 0 else choosers[::-1]),
            config=GameConfig(
                seed=_derived_seed(seed, index, "game"),
                rules_version=2,
                seating=Seating.FIXED,
                starting_player=0,
                max_turns=180,
                max_actions_per_turn=160,
            ),
            decision_hook=hook,
        )
        result = game.run()
        truncated += result.truncated
        scores.append(0.5 if result.winner is None else float(result.winner == seat))
        if not result.truncated:
            for i, ps in enumerate(predictions):
                if ps:
                    target = float(result.winner == (seat if i == 0 else 1 - seat))
                    p = np.clip(np.array(ps), 1e-7, 1 - 1e-7)
                    measurements[i].append(
                        dict(
                            brier=float(np.mean((p - target) ** 2)),
                            log_loss=float(
                                np.mean(-target * np.log(p) - (1 - target) * np.log(1 - p))
                            ),
                            predicted=float(p.mean()),
                            observed=target,
                            positions=len(ps),
                        )
                    )
    return dict(pair=index, scores=scores, truncated=truncated, critics=measurements)


def run_match(folder):
    folder = Path(folder)
    state = read(folder / "job.json")
    rows = read(folder / "pairs.json", [])
    state.update(status="running")
    atomic_json(folder / "job.json", state)
    start = time.monotonic()
    try:
        with ProcessPoolExecutor(
            max_workers=state["workers"], mp_context=mp.get_context("spawn")
        ) as pool:
            while len(rows) < state["pairs"]:
                if (folder / "STOP").exists():
                    state["status"] = "paused"
                    break
                tasks = [
                    (
                        state["actor_a"],
                        state["actor_b"],
                        state.get("critic_a"),
                        state.get("critic_b"),
                        state["seed"],
                        i,
                    )
                    for i in range(len(rows), min(len(rows) + 32, state["pairs"]))
                ]
                rows.extend(pool.map(measured_pair, tasks, chunksize=1))
                atomic_json(folder / "pairs.json", rows)
                state.update(
                    pairs_completed=len(rows),
                    score=float(np.mean([sum(r["scores"]) / 2 for r in rows])),
                    truncated=sum(r["truncated"] for r in rows),
                )
                reports = []
                for i in (0, 1):
                    games = [g for r in rows for g in r["critics"][i]]
                    reports.append(
                        {
                            k: float(np.mean([g[k] for g in games]))
                            for k in ("brier", "log_loss", "predicted", "observed")
                        }
                        if games
                        else None
                    )
                state["critic_metrics"] = reports
                atomic_json(folder / "job.json", state)
            else:
                state["status"] = "complete"
    except Exception as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        state["seconds"] = state.get("seconds", 0) + time.monotonic() - start
        atomic_json(folder / "job.json", state)
