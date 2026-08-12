"""Threaded human-vs-model sessions for the local web client."""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import HeuristicChooser
from .cards import ALL_CARDS
from .encoding import Encoder
from .engine import Action, Decision, Game, GameConfig, model_action_indices
from .model import NumpyActor


class ActorChooser:
    def __init__(self, actor_path: str | Path):
        self.actor = NumpyActor.load(actor_path)
        self.encoder = Encoder(
            card_catalog=ALL_CARDS,
            version=self.actor.spec.encoder_version,
        )

    def __call__(self, _player_id: int, decision: Decision) -> int:
        index, _probabilities = self.score(decision)
        return index

    def score(self, decision: Decision) -> tuple[int, np.ndarray]:
        """Return the deployed head-average choice and value for each option."""

        encoded = self.encoder.encode_decision(decision.observation, decision)
        eligible = np.asarray(model_action_indices(decision), dtype=np.int64)
        local_index, eligible_probabilities = self.actor.choose(
            encoded.state,
            encoded.actions[eligible],
            int(encoded.family),
            head=None,
            epsilon=0.0,
        )
        if len(eligible) == len(decision.actions):
            probabilities = eligible_probabilities
        else:
            # Human players still see every rules-legal option. Score the
            # dominance-masked choices honestly for explanation while keeping
            # the model's deployed selection restricted to eligible actions.
            logits = self.actor.predict_options(
                encoded.state,
                encoded.actions,
                int(encoded.family),
            )
            values = logits.mean(axis=1) if logits.ndim > 1 else logits
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))
        return int(eligible[local_index]), probabilities


_BALANCED_FALLBACK = HeuristicChooser("balanced")


def _fallback_opponent(player_id: int, decision: Decision) -> Action:
    """A legal deterministic fallback used before a checkpoint exists."""

    return _BALANCED_FALLBACK(player_id, decision)


class GameSession:
    def __init__(
        self,
        *,
        seed: int,
        human_starts: bool,
        actor_path: str | Path | None,
        model_label: str,
    ):
        self.id = uuid.uuid4().hex[:12]
        self.seed = int(seed)
        self.human_player = 0
        self.model_label = model_label
        self.created_at = time.time()
        self.updated_at = self.created_at
        self._condition = threading.Condition(threading.RLock())
        self._pending: Decision | None = None
        self._selected: int | None = None
        self._cancelled = False
        self._error: str | None = None
        self._action_log: deque[dict[str, Any]] = deque(maxlen=300)
        self._actor_chooser = ActorChooser(actor_path) if actor_path else None
        opponent = self._actor_chooser or _fallback_opponent
        starting_player = self.human_player if human_starts else 1 - self.human_player
        self.game = Game(
            player_names=("You", model_label),
            choosers=(self._human_choose, opponent),
            config=GameConfig(
                seed=self.seed,
                starting_player=starting_player,
                max_turns=240,
                max_actions_per_turn=220,
            ),
            cancel_hook=lambda: self._cancelled,
            decision_hook=self._record_action,
        )
        self._thread = threading.Thread(
            target=self._run,
            name=f"astro2-game-{self.id}",
            daemon=True,
        )
        self._thread.start()

    def _human_choose(self, _player_id: int, decision: Decision) -> int:
        with self._condition:
            self._pending = decision
            self._selected = None
            self.updated_at = time.time()
            self._condition.notify_all()
            while self._selected is None and not self._cancelled:
                self._condition.wait(timeout=0.5)
            selected = 0 if self._selected is None else self._selected
            self._pending = None
            return selected

    def _record_action(self, player_id: int, decision: Decision, action: Action) -> None:
        with self._condition:
            self._action_log.append(
                {
                    "number": len(self._action_log) + 1,
                    "player_id": player_id,
                    "player": "You" if player_id == self.human_player else self.model_label,
                    "turn": decision.observation.turn,
                    "family": decision.family.value,
                    "action": action.to_dict(),
                    "label": action.label,
                }
            )
            self.updated_at = time.time()
            self._condition.notify_all()

    def _run(self) -> None:
        try:
            self.game.run()
        except Exception as error:  # pragma: no cover - session boundary
            self._error = f"{type(error).__name__}: {error}"
        finally:
            with self._condition:
                self.updated_at = time.time()
                self._condition.notify_all()

    def choose(self, action_id: int) -> dict[str, Any]:
        with self._condition:
            if self._pending is None:
                raise RuntimeError("the game is not waiting for a human choice")
            if self._selected is not None:
                raise RuntimeError("a choice was already submitted")
            if action_id < 0 or action_id >= len(self._pending.actions):
                raise ValueError("action_id is not legal for the pending decision")
            self._selected = int(action_id)
            self.updated_at = time.time()
            self._condition.notify_all()
        # Give the engine a short chance to advance to its next stable decision.
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._pending is None or self.game.result is not None or self._error is not None
                ),
                timeout=0.25,
            )
        return self.snapshot()

    def stop(self) -> None:
        with self._condition:
            self._cancelled = True
            if self._pending is not None and self._selected is None:
                self._selected = 0
            self._condition.notify_all()

    def close(self, timeout: float = 1.0) -> None:
        self.stop()
        self._thread.join(max(0.0, timeout))

    def wait_until_ready(self, timeout: float = 2.0) -> None:
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._pending is not None
                    or self.game.result is not None
                    or self._error is not None
                ),
                timeout=timeout,
            )

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            pending = self._pending
            result = self.game.result
            observation = self.game.observation(self.human_player)
            recommendation = None
            option_values: np.ndarray | None = None
            if pending is not None and self._actor_chooser is not None:
                recommendation, option_values = self._actor_chooser.score(pending)
            return {
                "id": self.id,
                "seed": self.seed,
                "status": (
                    "error"
                    if self._error
                    else "complete"
                    if result is not None
                    else "your_turn"
                    if pending is not None
                    else "model_thinking"
                ),
                "model_label": self.model_label,
                "human_player": self.human_player,
                "observation": observation.to_dict(),
                "board": self.game.state_dict(include_hidden=False),
                "card_zones": self.game.unordered_card_zones(self.human_player),
                "decision": (
                    {
                        "family": pending.family.value,
                        "prompt": pending.prompt,
                        "actions": [
                            {
                                "id": index,
                                "label": action.label,
                                "model_value": (
                                    float(option_values[index])
                                    if option_values is not None
                                    else None
                                ),
                                "model_recommended": recommendation == index,
                                **action.to_dict(),
                            }
                            for index, action in enumerate(pending.actions)
                        ],
                    }
                    if pending is not None
                    else None
                ),
                "action_log": list(self._action_log),
                "result": result.to_dict() if result is not None else None,
                "error": self._error,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }


class PlayManager:
    def __init__(self, maximum_sessions: int = 20):
        self.maximum_sessions = maximum_sessions
        self._lock = threading.RLock()
        self._sessions: dict[str, GameSession] = {}

    def create(
        self,
        *,
        seed: int | None = None,
        human_starts: bool = True,
        actor_path: str | Path | None = None,
        model_label: str = "Balanced baseline",
    ) -> dict[str, Any]:
        with self._lock:
            if len(self._sessions) >= self.maximum_sessions:
                oldest = min(self._sessions.values(), key=lambda session: session.updated_at)
                oldest.stop()
                self._sessions.pop(oldest.id, None)
            session = GameSession(
                seed=int(seed if seed is not None else np.random.default_rng().integers(0, 2**63)),
                human_starts=human_starts,
                actor_path=actor_path,
                model_label=model_label,
            )
            self._sessions[session.id] = session
        session.wait_until_ready()
        return session.snapshot()

    def get(self, session_id: str) -> GameSession:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            return self._sessions[session_id]

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [session.snapshot() for session in self._sessions.values()]

    def shutdown(self, timeout: float = 2.0) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        deadline = time.monotonic() + max(0.0, timeout)
        for session in sessions:
            session.close(max(0.0, deadline - time.monotonic()))
