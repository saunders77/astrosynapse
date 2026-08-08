"""Validated configuration for local self-play runs."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunConfig(BaseModel):
    """A complete, checkpointed training recipe.

    The defaults are intentionally the recommended 24-hour M4/16 GB recipe.
    Every field is persisted with the run so a checkpoint can be reproduced.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="M4 24-hour run", min_length=1, max_length=80)
    preset: Literal["m4_24h", "quick", "custom"] = "m4_24h"
    seed: int = 20260807
    duration_minutes: int = Field(default=24 * 60, ge=1, le=7 * 24 * 60)

    actor_processes: int = Field(default_factory=lambda: max(2, min(8, (os.cpu_count() or 4) - 2)), ge=1, le=16)
    games_per_actor_batch: int = Field(default=16, ge=1, le=128)
    max_turns: int = Field(default=180, ge=40, le=500)
    max_actions_per_turn: int = Field(default=160, ge=40, le=500)

    hidden_size: int = Field(default=192, ge=64, le=768)
    residual_blocks: int = Field(default=3, ge=1, le=8)
    bootstrap_heads: int = Field(default=3, ge=1, le=8)
    batch_size: int = Field(default=2048, ge=64, le=8192)
    learning_rate: float = Field(default=3e-4, gt=0, le=0.01)
    min_learning_rate: float = Field(default=3e-5, gt=0, le=0.01)
    weight_decay: float = Field(default=1e-4, ge=0, le=0.1)
    gradient_clip: float = Field(default=5.0, gt=0, le=100)
    updates_per_iteration: int = Field(default=32, ge=1, le=256)
    replay_capacity: int = Field(default=900_000, ge=10_000, le=2_000_000)
    replay_warmup: int = Field(default=50_000, ge=100, le=500_000)
    heuristic_bootstrap_updates: int = Field(default=2_000, ge=0, le=100_000)
    recent_sample_fraction: float = Field(default=0.35, ge=0, le=1)

    epsilon_start: float = Field(default=0.20, ge=0, le=1)
    epsilon_end: float = Field(default=0.025, ge=0, le=1)
    epsilon_decay_games: int = Field(default=1_500_000, ge=1_000)
    current_selfplay_fraction: float = Field(default=0.55, ge=0, le=1)
    league_fraction: float = Field(default=0.30, ge=0, le=1)
    baseline_fraction: float = Field(default=0.15, ge=0, le=1)

    checkpoint_every_games: int = Field(default=100_000, ge=100)
    evaluate_every_games: int = Field(default=500_000, ge=100)
    evaluation_pairs: int = Field(default=5_000, ge=8, le=20_000)
    promotion_confidence: float = Field(default=0.95, ge=0.80, le=0.999)
    promotion_margin: float = Field(default=0.0, ge=0, le=0.25)
    keep_checkpoints: int = Field(default=12, ge=2, le=100)

    device: Literal["auto", "gpu", "cpu"] = "auto"
    metrics_interval_seconds: float = Field(default=1.0, ge=0.25, le=30)

    @model_validator(mode="after")
    def validate_mix_and_schedule(self) -> RunConfig:
        mix = self.current_selfplay_fraction + self.league_fraction + self.baseline_fraction
        if abs(mix - 1.0) > 1e-6:
            raise ValueError("opponent fractions must sum to 1.0")
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end must not exceed epsilon_start")
        if self.replay_warmup >= self.replay_capacity:
            raise ValueError("replay_warmup must be smaller than replay_capacity")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must not exceed learning_rate")
        return self

    @classmethod
    def quick(cls, name: str = "Quick validation run") -> RunConfig:
        return cls(
            name=name,
            preset="quick",
            duration_minutes=5,
            actor_processes=max(1, min(4, (os.cpu_count() or 4) - 1)),
            games_per_actor_batch=2,
            hidden_size=128,
            residual_blocks=2,
            batch_size=256,
            updates_per_iteration=4,
            replay_capacity=25_000,
            replay_warmup=500,
            heuristic_bootstrap_updates=256,
            checkpoint_every_games=1_000,
            evaluate_every_games=2_000,
            evaluation_pairs=16,
        )


SAFE_LIVE_FIELDS = {
    "duration_minutes",
    "updates_per_iteration",
    "epsilon_end",
    "current_selfplay_fraction",
    "league_fraction",
    "baseline_fraction",
    "checkpoint_every_games",
    "evaluate_every_games",
    "evaluation_pairs",
    "promotion_confidence",
    "promotion_margin",
    "metrics_interval_seconds",
}


def preset_config(preset: str) -> RunConfig:
    if preset == "quick":
        return RunConfig.quick()
    if preset == "m4_24h":
        return RunConfig()
    raise ValueError(f"unknown preset: {preset}")
