"""Independent, resumable outcome prediction for a frozen playing policy.

CPU NumPy learner; never imports MLX or edits the policy. Splits are by complete
source game. Truncated games are censored, not labelled as draws.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .arena import _derived_seed, resolve_model
from .card_analysis import _GreedyChooser, _load_actor_encoder
from .engine import Game, GameConfig
from .experiment_control import atomic_json, sha256

ACTIVE = {"queued", "collecting", "fitting", "evaluating"}


@dataclass(frozen=True)
class CriticConfig:
    games: int = 1000
    epochs: int = 20
    hidden_size: int = 128
    seed: int = 20261001
    rules_version: int = 2
    positions_per_game: int = 64
    learning_rate: float = 0.001

    def __post_init__(self):
        if not 20 <= self.games <= 10000 or not 1 <= self.epochs <= 200:
            raise ValueError("Use 20–10,000 games and 1–200 epochs")
        if self.hidden_size not in (32, 64, 128, 256):
            raise ValueError("Hidden width must be 32, 64, 128 or 256")
        if self.rules_version not in (1, 2) or not 0 <= self.seed < 2**53:
            raise ValueError("Invalid rules or seed")
        if not 16 <= self.positions_per_game <= 256:
            raise ValueError("Use 16–256 sampled positions per game")
        if not 0 < self.learning_rate <= 0.01:
            raise ValueError("Learning rate must be in (0, 0.01]")


def save_npz(path, **arrays):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **arrays)
    temporary.replace(path)


def game_splits(games, seed):
    order = np.random.default_rng(seed).permutation(games)
    return {
        "train": order[: int(games * 0.7)].tolist(),
        "validation": order[int(games * 0.7) : int(games * 0.85)].tolist(),
        "test": order[int(games * 0.85) :].tolist(),
    }


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


class IndependentCritic:
    """Two nonlinear hidden layers over the public state and decision family."""

    def __init__(self, size, width, seed=0):
        rng = np.random.default_rng(seed)
        self.weights = {}
        for i, (a, b) in enumerate(((size, width), (width, width // 2), (width // 2, 1))):
            self.weights[f"w{i}"] = (rng.normal(size=(a, b)) / np.sqrt(a)).astype(np.float32)
            self.weights[f"b{i}"] = np.zeros(b, np.float32)
        self.mean = np.zeros(size, np.float32)
        self.scale = np.ones(size, np.float32)
        self.m = {k: np.zeros_like(v) for k, v in self.weights.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.weights.items()}
        self.step = 0

    def forward(self, x):
        x = np.clip((x - self.mean) / self.scale, -10, 10)
        w = self.weights
        h1 = np.tanh(x @ w["w0"] + w["b0"])
        h2 = np.tanh(h1 @ w["w1"] + w["b1"])
        return (h2 @ w["w2"] + w["b2"]).ravel(), (x, h1, h2)

    def predict(self, x):
        return sigmoid(self.forward(np.asarray(x, dtype=np.float32))[0])

    def train_batch(self, x, y, lr):
        logits, (x, h1, h2) = self.forward(x)
        loss = float(np.mean(np.logaddexp(0, logits) - y * logits))
        delta = ((sigmoid(logits) - y) / len(y))[:, None]
        grads = {"w2": h2.T @ delta, "b2": delta.sum(axis=0)}
        delta = (delta @ self.weights["w2"].T) * (1 - h2 * h2)
        grads.update(w1=h1.T @ delta, b1=delta.sum(axis=0))
        delta = (delta @ self.weights["w1"].T) * (1 - h1 * h1)
        grads.update(w0=x.T @ delta, b0=delta.sum(axis=0))
        if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in grads.values()):
            raise ValueError("Nonfinite critic update")
        self.step += 1
        for k, g in grads.items():
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            self.weights[k] -= (
                lr
                * (self.m[k] / (1 - 0.9**self.step))
                / (np.sqrt(self.v[k] / (1 - 0.999**self.step)) + 1e-8)
            )
        return loss

    def save(self, path, metadata, best=None):
        arrays = {
            **self.weights,
            "mean": self.mean,
            "scale": self.scale,
            "metadata": np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
            "step": np.array(self.step),
        }
        arrays.update({f"m_{k}": v for k, v in self.m.items()})
        arrays.update({f"v_{k}": v for k, v in self.v.items()})
        if best is not None:
            arrays.update({f"best_{k}": v for k, v in best.items()})
        save_npz(path, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            model = cls(z["w0"].shape[0], z["w0"].shape[1])
            for k in model.weights:
                model.weights[k] = z[k].copy()
                model.m[k] = z[f"m_{k}"].copy()
                model.v[k] = z[f"v_{k}"].copy()
            model.mean, model.scale, model.step = z["mean"], z["scale"], int(z["step"])
            meta = json.loads(z["metadata"].tobytes())
            best = {k: z[f"best_{k}"].copy() for k in model.weights} if "best_w0" in z else None
        return model, meta, best


def calibration(predictions, targets, weights=None):
    p = np.asarray(predictions, dtype=float)
    y = np.asarray(targets, dtype=float)
    if not len(y):
        return None
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    bins = []
    for i in range(10):
        mask = (p >= i / 10) & (p < (i + 1) / 10)
        bins.append(
            {
                "lower": i / 10,
                "upper": (i + 1) / 10,
                "positions": int(mask.sum()),
                "predicted": float(np.average(p[mask], weights=w[mask])) if mask.any() else None,
                "observed": float(np.average(y[mask], weights=w[mask])) if mask.any() else None,
            }
        )
    return {
        "positions": len(y),
        "brier": float(np.average((p - y) ** 2, weights=w)),
        "log_loss": float(np.average(-y * np.log(p) - (1 - y) * np.log1p(-p), weights=w)),
        "predicted": float(np.average(p, weights=w)),
        "observed": float(np.average(y, weights=w)),
        "bins": bins,
    }


class Paused(Exception):
    pass


def collect_game(folder, index, config, actor, encoder, check):
    rng = np.random.default_rng(_derived_seed(config.seed, index, "critic-sample"))
    samples = []
    seen = 0

    def observe(player, decision, _selected):
        nonlocal seen
        check()
        seen += 1
        slot = len(samples) if seen <= config.positions_per_game else int(rng.integers(seen))
        if slot >= config.positions_per_game:
            return
        encoded = encoder.encode_decision(decision.observation, decision)
        family = int(encoded.family)
        baseline = float(sigmoid(actor.predict_values(encoded.state, np.array([family]))).mean())
        row = (
            np.concatenate((encoded.state, np.eye(actor.spec.families, dtype=np.float32)[family])),
            player,
            baseline,
            len(decision.actions) == 1,
            family,
            decision.observation.turn,
        )
        if slot == len(samples):
            samples.append(row)
        else:
            samples[slot] = row

    choosers = tuple(
        _GreedyChooser(actor, encoder, _derived_seed(config.seed, index, f"critic-seat-{p}"))
        for p in (0, 1)
    )
    game = Game(
        choosers=choosers,
        config=GameConfig(
            seed=_derived_seed(config.seed, index, "critic-game"),
            rules_version=config.rules_version,
            starting_player=index % 2,
            max_turns=240,
            max_actions_per_turn=220,
        ),
        decision_hook=observe,
    )
    result = game.run()
    check()
    size = actor.spec.state_size + actor.spec.families
    usable = samples if not result.truncated else []
    save_npz(
        folder / f"game-{index:05d}.npz",
        x=np.stack([r[0] for r in usable]) if usable else np.empty((0, size), np.float32),
        y=np.array(
            [0.5 if result.winner is None else float(r[1] == result.winner) for r in usable],
            np.float32,
        ),
        baseline=np.array([r[2] for r in usable]),
        forced=np.array([r[3] for r in usable]),
        family=np.array([r[4] for r in usable]),
        turn=np.array([r[5] for r in usable]),
        truncated=np.array(result.truncated),
    )


def evaluate(model, folder, indices, check):
    rows = {k: [] for k in ("prediction", "y", "baseline", "forced", "family", "turn", "weight")}
    completed = 0
    for index in indices:
        check()
        with np.load(folder / f"game-{index:05d}.npz") as z:
            if not len(z["y"]):
                continue
            completed += 1
            rows["prediction"].extend(model.predict(z["x"]))
            rows["weight"].extend(np.full(len(z["y"]), 1 / len(z["y"])))
            for key in ("y", "baseline", "forced", "family", "turn"):
                rows[key].extend(z[key])
    a = {k: np.asarray(v) for k, v in rows.items()}
    if not completed:
        raise ValueError("No completed games in a split; collect a larger dataset")

    def metrics(mask):
        return {
            "critic": calibration(a["prediction"][mask], a["y"][mask], a["weight"][mask]),
            "original": calibration(a["baseline"][mask], a["y"][mask], a["weight"][mask]),
        }

    return {
        "games": completed,
        **metrics(np.ones(len(a["y"]), dtype=bool)),
        "slices": {
            "forced": metrics(a["forced"] == 1),
            "choice": metrics(a["forced"] == 0),
            "early": metrics(a["turn"] < 10),
            "late": metrics(a["turn"] >= 10),
            **{f"family_{i}": metrics(a["family"] == i) for i in range(8)},
        },
    }


def train_job(folder):
    with (folder / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        job = json.loads((folder / "job.json").read_text())
        config = CriticConfig(**job["config"])

        def update(**changes):
            job.update(changes, updated_at=time.time(), pid=os.getpid())
            atomic_json(folder / "job.json", job)

        def check():
            if (folder / "PAUSE").exists():
                raise Paused

        try:
            update(status="collecting", error=None)
            if sha256(folder / "policy.actor.npz") != job["policy_sha256"]:
                raise ValueError("Frozen policy hash mismatch")
            actor, encoder = _load_actor_encoder(str(folder / "policy.actor.npz"))
            if actor.spec.objective_version < 2:
                raise ValueError("Critic training requires an objective-v2 policy")
            splits = game_splits(config.games, config.seed)
            atomic_json(folder / "splits.json", splits)
            truncated = 0
            for index in range(config.games):
                check()
                path = folder / f"game-{index:05d}.npz"
                if not path.exists():
                    collect_game(folder, index, config, actor, encoder, check)
                with np.load(path) as z:
                    truncated += int(z["truncated"])
                update(
                    games_completed=index + 1,
                    truncated_games=truncated,
                    progress=0.65 * (index + 1) / config.games,
                )
            checkpoint = folder / "checkpoint.npz"
            if checkpoint.exists():
                model, meta, best = IndependentCritic.load(checkpoint)
            else:
                model = IndependentCritic(
                    actor.spec.state_size + actor.spec.families, config.hidden_size, config.seed
                )
                count = 0
                total = np.zeros_like(model.mean, dtype=np.float64)
                squared = total.copy()
                for index in splits["train"]:
                    check()
                    with np.load(folder / f"game-{index:05d}.npz") as z:
                        x = z["x"].astype(np.float64)
                    if not len(x):
                        continue
                    # Every game contributes equal mass to preprocessing, loss and metrics.
                    count += 1
                    total += x.mean(axis=0)
                    squared += (x * x).mean(axis=0)
                if not count:
                    raise ValueError("No completed training games")
                model.mean = (total / count).astype(np.float32)
                model.scale = np.maximum(
                    np.sqrt(np.maximum(squared / count - model.mean.astype(float) ** 2, 0)), 0.05
                ).astype(np.float32)
                meta = {"epoch": 0, "history": [], "best_loss": None, "best_epoch": 0}
                best = {k: v.copy() for k, v in model.weights.items()}
                model.save(checkpoint, meta, best)
            update(
                status="fitting",
                epoch=meta["epoch"],
                history=meta["history"],
                parameters=sum(v.size for v in model.weights.values()),
            )
            for epoch in range(meta["epoch"], config.epochs):
                losses = []
                order = np.random.default_rng(
                    _derived_seed(config.seed, epoch, "critic-epoch")
                ).permutation(splits["train"])
                for index in order:
                    check()
                    with np.load(folder / f"game-{index:05d}.npz") as z:
                        if len(z["y"]):
                            losses.append(model.train_batch(z["x"], z["y"], config.learning_rate))
                validation = evaluate(model, folder, splits["validation"], check)
                loss = validation["critic"]["log_loss"]
                if meta["best_loss"] is None or loss < meta["best_loss"]:
                    best = {k: v.copy() for k, v in model.weights.items()}
                    meta.update(best_loss=loss, best_epoch=epoch + 1)
                meta["epoch"] = epoch + 1
                meta["history"].append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": float(np.mean(losses)),
                        "validation_loss": loss,
                        "validation_brier": validation["critic"]["brier"],
                    }
                )
                model.save(checkpoint, meta, best)
                update(
                    epoch=epoch + 1,
                    history=meta["history"],
                    validation=validation,
                    best_epoch=meta["best_epoch"],
                    progress=0.65 + 0.3 * (epoch + 1) / config.epochs,
                )
            check()
            update(status="evaluating", progress=0.96)
            model.weights = best
            result = {
                "validation": evaluate(model, folder, splits["validation"], check),
                "test": evaluate(model, folder, splits["test"], check),
                "best_epoch": meta["best_epoch"],
                "scope": "Frozen-policy self-play under the selected rules; not calibrated for arbitrary human opponents. No strength improvement is claimed.",
            }
            artifact_meta = {
                "schema_version": 1,
                "encoder_version": actor.spec.encoder_version,
                "state_size": actor.spec.state_size,
                "families": actor.spec.families,
                "policy_sha256": job["policy_sha256"],
                "config": asdict(config),
                **meta,
            }
            model.save(folder / "critic.npz", artifact_meta)
            atomic_json(folder / "result.json", result)
            update(status="complete", progress=1, result=result)
        except Paused:
            update(status="paused")
        except Exception as error:
            update(status="failed", error=f"{type(error).__name__}: {error}")
            raise


class CriticManager:
    def __init__(self, store, output_dir):
        self.store = store
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.children = []

    def folder(self, job_id):
        if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise KeyError(job_id)
        folder = self.output_dir / job_id
        if not (folder / "job.json").exists():
            raise KeyError(job_id)
        return folder

    def get(self, job_id):
        folder = self.folder(job_id)
        job = json.loads((folder / "job.json").read_text())
        if job["status"] in ACTIVE and time.time() - job["updated_at"] > 10:
            with (folder / "run.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    job.update(
                        status="interrupted",
                        error="Worker stopped. Resume from saved games and the last completed epoch.",
                    )
                    atomic_json(folder / "job.json", job)
        job["pause_requested"] = (folder / "PAUSE").exists()
        return job

    def list(self):
        self.children = [c for c in self.children if c.poll() is None]
        return sorted(
            (self.get(p.parent.name) for p in self.output_dir.glob("*/job.json")),
            key=lambda j: j["created_at"],
            reverse=True,
        )

    def launch(self, folder):
        env = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
        try:
            with (folder / "worker.log").open("ab") as log:
                self.children.append(
                    subprocess.Popen(
                        [sys.executable, "-m", "astro2.critic", str(folder)],
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
        except Exception as error:
            job = json.loads((folder / "job.json").read_text())
            job.update(status="failed", error=str(error))
            atomic_json(folder / "job.json", job)
            raise

    def create(self, model_id, config):
        with (self.output_dir / "manager.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if any(j["status"] in ACTIVE for j in self.list()):
                raise ValueError("Pause the active critic job before starting another")
            resolved = resolve_model(self.store, model_id)
            actor, _ = _load_actor_encoder(str(resolved.actor_path))
            if actor.spec.objective_version < 2:
                raise ValueError("Choose an objective-v2 model with an existing critic")
            folder = self.output_dir / uuid.uuid4().hex
            folder.mkdir()
            shutil.copyfile(resolved.actor_path, folder / "policy.actor.npz")
            job = {
                "id": folder.name,
                "model_id": model_id,
                "model_label": resolved.label,
                "policy_sha256": sha256(folder / "policy.actor.npz"),
                "config": asdict(config),
                "created_at": time.time(),
                "updated_at": time.time(),
                "status": "queued",
                "progress": 0,
                "games_completed": 0,
                "epoch": 0,
                "history": [],
            }
            atomic_json(folder / "job.json", job)
            self.launch(folder)
            return job

    def control(self, job_id, action):
        with (self.output_dir / "manager.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            job = self.get(job_id)
            folder = self.folder(job_id)
            if action == "pause":
                if job["status"] not in ACTIVE:
                    raise ValueError("Only running jobs can be paused")
                (folder / "PAUSE").touch()
            elif action == "resume":
                if job["status"] not in {"paused", "interrupted", "failed"}:
                    raise ValueError("Only paused, interrupted or failed jobs can resume")
                if any(j["status"] in ACTIVE for j in self.list()):
                    raise ValueError("Another critic job is active")
                # The previous worker can still be unwinding after writing its final status.
                with (folder / "run.lock").open("a") as worker_lock:
                    try:
                        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as error:
                        raise ValueError("Worker is still stopping; retry shortly") from error
                    (folder / "PAUSE").unlink(missing_ok=True)
                    job.update(status="queued", error=None, updated_at=time.time())
                    atomic_json(folder / "job.json", job)
                self.launch(folder)
            else:
                raise ValueError("Unknown control action")
            result = self.get(job_id)
            result["pause_requested"] = (folder / "PAUSE").exists()
            return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    train_job(parser.parse_args().folder)
