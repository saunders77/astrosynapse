"""Validated configuration for local self-play runs."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_REPRODUCIBILITY_SEED = (1 << 53) - 1
MINIMUM_PROMOTION_PAIRS = 1_000


class RunConfig(BaseModel):
    """A complete, checkpointed training recipe.

    Bare-model defaults preserve the Astro2 compatibility contract. New work
    should use :meth:`astro4_m4`; every resolved field is persisted with a run.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="M4 24-hour run", min_length=1, max_length=80)
    preset: Literal[
        "astro5_search", "astro4_m4", "astro3_m4", "m4_24h", "quick", "custom"
    ] = "m4_24h"
    training_generation: Literal[2, 3, 4, 5] = 2
    # Keep seeds exactly representable by the JavaScript dashboard as well as
    # Python, so API/UI round-trips cannot silently change an experiment.
    seed: int = Field(default=20260807, ge=0, le=MAX_REPRODUCIBILITY_SEED)
    duration_minutes: int = Field(default=24 * 60, ge=1, le=7 * 24 * 60)
    budget_type: Literal["minutes", "games", "full_evaluations"] = "minutes"
    budget_games: int | None = Field(default=None, ge=100, le=2_000_000_000)
    budget_full_evaluations: int | None = Field(default=None, ge=1, le=1_000)

    actor_processes: int = Field(
        default_factory=lambda: max(2, min(8, (os.cpu_count() or 4) - 2)), ge=1, le=16
    )
    games_per_actor_batch: int = Field(default=16, ge=1, le=128)
    # Multiple small tasks per worker let the executor work-steal around
    # unusually long games and searched positions.
    rollout_tasks_per_actor: int = Field(default=1, ge=1, le=16)
    max_turns: int = Field(default=180, ge=40, le=500)
    max_actions_per_turn: int = Field(default=160, ge=40, le=500)

    hidden_size: int = Field(default=192, ge=64, le=768)
    residual_blocks: int = Field(default=3, ge=1, le=8)
    bootstrap_heads: int = Field(default=3, ge=1, le=8)
    batch_size: int = Field(default=2048, ge=64, le=8192)
    learning_rate: float = Field(default=3e-4, gt=0, le=0.01)
    min_learning_rate: float = Field(default=3e-5, gt=0, le=0.01)
    learning_rate_decay_updates: int = Field(default=400_000, ge=1_000, le=10_000_000)
    learning_rate_schedule: Literal["cosine", "cosine_restarts"] = "cosine"
    learning_rate_restart_updates: int = Field(default=200_000, ge=1_000, le=10_000_000)
    learning_rate_restart_decay: float = Field(default=0.85, gt=0, le=1)
    weight_decay: float = Field(default=1e-4, ge=0, le=0.1)
    gradient_clip: float = Field(default=5.0, gt=0, le=100)
    updates_per_iteration: int = Field(default=32, ge=1, le=256)
    replay_capacity: int = Field(default=900_000, ge=10_000, le=2_000_000)
    replay_warmup: int = Field(default=50_000, ge=100, le=500_000)
    heuristic_bootstrap_updates: int = Field(default=2_000, ge=0, le=100_000)
    recent_sample_fraction: float = Field(default=0.35, ge=0, le=1)
    replay_sampling_profile: Literal["balanced_rare", "natural"] = "balanced_rare"
    importance_correct_replay: bool = False
    terminal_target_weight: float = Field(default=0.60, ge=0, le=1)
    use_bootstrap_targets: bool = True
    preference_replay_capacity: int = Field(default=50_000, ge=1_000, le=250_000)
    preference_batch_size: int = Field(default=256, ge=16, le=2_048)
    preference_loss_weight: float = Field(default=0.15, ge=0, le=10)
    preference_margin: float = Field(default=1.0, ge=0, le=10)
    tactical_preference_training: bool = True
    policy_replay_capacity: int = Field(default=150_000, ge=1_000, le=2_000_000)
    policy_value_loss_weight: float = Field(default=0.5, ge=0, le=10)
    policy_entropy_weight: float = Field(default=0.01, ge=0, le=1)
    policy_importance_clip: float = Field(default=2.0, ge=1, le=20)
    counterfactual_fraction: float = Field(default=0.0, ge=0, le=1)
    counterfactual_max_per_game: int = Field(default=0, ge=0, le=8)
    counterfactual_loss_weight: float = Field(default=0.0, ge=0, le=10)

    # Astro5 learns from complete action-set searches.  A small fraction of
    # naturally encountered states is cloned and rolled forward from several
    # legal actions; the resulting distribution supervises every searched
    # action rather than only reinforcing the behavior action.
    reanalysis_fraction: float = Field(default=0.0, ge=0, le=1)
    reanalysis_max_per_game: int = Field(default=0, ge=0, le=32)
    reanalysis_max_actions: int = Field(default=6, ge=2, le=64)
    reanalysis_rollouts_per_action: int = Field(default=2, ge=1, le=16)
    reanalysis_horizon_turns: int = Field(default=2, ge=2, le=20)
    reanalysis_policy_temperature: float = Field(default=0.35, gt=0, le=5)
    reanalysis_policy_loss_weight: float = Field(default=0.0, ge=0, le=20)
    reanalysis_value_loss_weight: float = Field(default=0.0, ge=0, le=20)
    # Keeping only a phase/family reservoir from each player-game makes a
    # fixed decision budget remember tens of thousands of games instead of a
    # few thousand long trajectories.
    policy_replay_decisions_per_player_game: int = Field(default=0, ge=0, le=128)
    policy_replay_family_balanced: bool = False

    epsilon_start: float = Field(default=0.20, ge=0, le=1)
    epsilon_end: float = Field(default=0.025, ge=0, le=1)
    epsilon_decay_games: int = Field(default=1_500_000, ge=1_000)
    exploration_decision_scale: float = Field(default=0.10, gt=0, le=1)
    # Zero means every eligible action has exploration support.
    exploration_top_k: int = Field(default=3, ge=0, le=128)
    bootstrap_inclusion_probability: float = Field(default=0.80, gt=0, le=1)
    randomized_prior_scale: float = Field(default=0.0, ge=0, le=5)
    adaptive_training: bool = False
    plateau_patience_evaluations: int = Field(default=3, ge=1, le=20)
    plateau_max_exploration_multiplier: float = Field(default=4.0, ge=1, le=20)
    current_selfplay_fraction: float = Field(default=0.55, ge=0, le=1)
    # Fraction of current-v-current games collected with the exact deployable
    # mean-head, prior-free, greedy policy. Generation 2 remains unchanged.
    deployment_policy_selfplay_fraction: float = Field(default=0.0, ge=0, le=1)
    league_fraction: float = Field(default=0.30, ge=0, le=1)
    baseline_fraction: float = Field(default=0.15, ge=0, le=1)
    behavior_policy: Literal["champion", "learner"] = "champion"
    rollback_rejected_candidates: bool = True
    rejected_candidate_action: Literal["continue", "restore_lineage", "queue_branch"] = (
        "restore_lineage"
    )
    checkpoint_diagnostic_games: int = Field(default=6, ge=2, le=100)
    checkpoint_baseline_pairs: int = Field(default=2, ge=1, le=50)
    baseline_regression_tolerance: float = Field(default=0.20, ge=0, le=1)
    gate_baseline_regression: bool = True
    heldout_brier_regression_tolerance: float = Field(default=0.03, ge=0, le=1)
    gate_heldout_brier_regression: bool = True
    # Deprecated strategy-gate fields remain loadable for persisted runs but
    # are intentionally ignored by checkpoint promotion.
    max_tactical_violations: int = Field(default=0, ge=0, le=1_000)
    gate_raw_tactical_preferences: bool = False
    require_early_high_cost_retention: bool = False
    require_resource_efficiency: bool = False
    minimum_head_disagreement_rate: float = Field(default=0.0, ge=0, le=1)
    maximum_heldout_brier: float = Field(default=1.0, ge=0, le=1)

    checkpoint_every_games: int = Field(default=100_000, ge=100)
    evaluate_every_games: int = Field(default=500_000, ge=100)
    canary_every_games: int = Field(default=0, ge=0)
    canary_pairs: int = Field(default=128, ge=8, le=2_000)
    evaluation_pairs: int = Field(default=2_000, ge=8, le=2_000)
    adaptive_evaluation: bool = True
    promotion_confidence: float = Field(default=0.95, ge=0.80, le=0.999)
    promotion_margin: float = Field(default=0.0, ge=0, le=0.25)
    evaluation_early_rejection: bool = False
    evaluation_early_rejection_min_pairs: int = Field(default=512, ge=32, le=2_000)
    evaluation_early_rejection_confidence: float = Field(default=0.995, ge=0.90, le=0.9999)
    evaluation_early_acceptance: bool = False
    evaluation_early_acceptance_min_pairs: int = Field(
        default=MINIMUM_PROMOTION_PAIRS, ge=MINIMUM_PROMOTION_PAIRS, le=2_000
    )
    evaluation_early_acceptance_confidence: float = Field(
        default=0.995, ge=0.90, le=0.9999
    )
    keep_checkpoints: int = Field(default=12, ge=2, le=100)

    # The controller changes only bounded multipliers and records every
    # decision.  It cannot weaken promotion confidence or mutate architecture.
    realtime_governor: bool = False
    governor_interval_games: int = Field(default=25_000, ge=100)
    governor_max_learning_rate_multiplier: float = Field(default=1.5, ge=1, le=4)
    governor_min_learning_rate_multiplier: float = Field(default=0.25, gt=0, le=1)
    governor_target_normalized_entropy: float = Field(default=0.55, ge=0.05, le=0.95)
    governor_max_updates_multiplier: float = Field(default=2.0, ge=1, le=4)
    governor_branch_after_failures: int = Field(default=3, ge=1, le=20)
    natural_diagnostic_positions: int = Field(default=2_000, ge=16, le=50_000)
    checkpoint_kl_limit: float = Field(default=0.35, ge=0, le=10)

    # Optional immutable branch origin.  New branch runs import all available
    # optimizer/replay artifacts from this checkpoint before training starts.
    initial_checkpoint_id: str | None = Field(default=None, min_length=1, max_length=128)
    branch_experiment_id: str | None = Field(default=None, min_length=1, max_length=128)

    # Astro3 can persist enough state to resume the same optimization process.
    # Zero keeps legacy checkpoints small and weight-only.
    persist_optimizer_state: bool = False
    resume_replay_items: int = Field(default=0, ge=0, le=500_000)

    device: Literal["auto", "gpu", "cpu"] = "auto"
    metrics_interval_seconds: float = Field(default=1.0, ge=0.25, le=30)

    @classmethod
    def model_validate_persisted(cls, value: object) -> RunConfig:
        """Validate a stored recipe while repairing obsolete schema limits.

        Older builds allowed promotion evaluations above the dashboard/API's
        current 2,000-pair safety cap.  Runs are durable across upgrades, so a
        value that was valid when written must not make the trainer crash when
        it is resumed.  New input continues to use ``model_validate`` and is
        therefore rejected rather than silently changed.
        """

        if isinstance(value, dict):
            value = dict(value)
            evaluation_pairs = value.get("evaluation_pairs")
            if (
                isinstance(evaluation_pairs, (int, float))
                and not isinstance(evaluation_pairs, bool)
                and evaluation_pairs > 2_000
            ):
                value["evaluation_pairs"] = 2_000
        return cls.model_validate(value)

    @model_validator(mode="after")
    def validate_mix_and_schedule(self) -> RunConfig:
        # Old generation-2 configs did not persist these fields and retain the
        # original hard gates. Generation 3 defaults to diagnostic-only noisy
        # checks, while an explicit caller can still opt into either gate.
        if (
            "gate_baseline_regression" not in self.model_fields_set
            and self.training_generation >= 3
        ):
            self.gate_baseline_regression = False
        if (
            "gate_heldout_brier_regression" not in self.model_fields_set
            and self.training_generation >= 3
        ):
            self.gate_heldout_brier_regression = False
        mix = self.current_selfplay_fraction + self.league_fraction + self.baseline_fraction
        if abs(mix - 1.0) > 1e-6:
            raise ValueError("opponent fractions must sum to 1.0")
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end must not exceed epsilon_start")
        if self.replay_warmup >= self.replay_capacity:
            raise ValueError("replay_warmup must be smaller than replay_capacity")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must not exceed learning_rate")
        if self.training_generation < 3 and self.deployment_policy_selfplay_fraction:
            raise ValueError("deployment_policy_selfplay_fraction requires training_generation>=3")
        if self.training_generation < 4 and (
            self.counterfactual_fraction or self.counterfactual_max_per_game
        ):
            raise ValueError("counterfactual rollout training requires training_generation=4")
        if self.training_generation < 5 and (
            self.reanalysis_fraction
            or self.reanalysis_max_per_game
            or self.reanalysis_policy_loss_weight
            or self.reanalysis_value_loss_weight
            or self.realtime_governor
        ):
            raise ValueError("search reanalysis and the realtime governor require generation 5")
        if self.reanalysis_fraction and not self.reanalysis_max_per_game:
            raise ValueError("reanalysis_max_per_game must be positive when reanalysis is enabled")
        if (
            self.evaluation_early_acceptance
            and self.evaluation_early_acceptance_min_pairs >= self.evaluation_pairs
        ):
            raise ValueError(
                "evaluation_early_acceptance_min_pairs must be smaller than evaluation_pairs"
            )
        if self.canary_every_games and self.canary_every_games < self.checkpoint_every_games:
            raise ValueError("canary_every_games must be zero or at least checkpoint_every_games")
        if self.budget_type == "games" and self.budget_games is None:
            raise ValueError("budget_games is required when budget_type is games")
        if self.budget_type == "full_evaluations" and self.budget_full_evaluations is None:
            raise ValueError(
                "budget_full_evaluations is required when budget_type is full_evaluations"
            )
        return self

    @classmethod
    def quick(cls, name: str = "Quick validation run") -> RunConfig:
        return cls(
            name=name,
            preset="quick",
            training_generation=3,
            duration_minutes=5,
            actor_processes=max(1, min(4, (os.cpu_count() or 4) - 1)),
            games_per_actor_batch=2,
            hidden_size=128,
            residual_blocks=2,
            bootstrap_heads=3,
            batch_size=256,
            updates_per_iteration=4,
            learning_rate=2e-4,
            min_learning_rate=4e-5,
            learning_rate_decay_updates=50_000,
            learning_rate_schedule="cosine_restarts",
            learning_rate_restart_updates=25_000,
            replay_capacity=25_000,
            replay_warmup=500,
            heuristic_bootstrap_updates=256,
            recent_sample_fraction=0.15,
            replay_sampling_profile="natural",
            importance_correct_replay=True,
            terminal_target_weight=1.0,
            use_bootstrap_targets=False,
            preference_replay_capacity=5_000,
            preference_batch_size=64,
            preference_loss_weight=0.0,
            tactical_preference_training=False,
            epsilon_start=0.15,
            epsilon_end=0.05,
            exploration_decision_scale=1.0,
            exploration_top_k=0,
            bootstrap_inclusion_probability=0.35,
            randomized_prior_scale=0.25,
            adaptive_training=True,
            current_selfplay_fraction=0.60,
            deployment_policy_selfplay_fraction=0.20,
            league_fraction=0.30,
            baseline_fraction=0.10,
            behavior_policy="learner",
            rollback_rejected_candidates=False,
            checkpoint_diagnostic_games=2,
            checkpoint_baseline_pairs=1,
            gate_baseline_regression=False,
            gate_heldout_brier_regression=False,
            gate_raw_tactical_preferences=False,
            require_early_high_cost_retention=False,
            checkpoint_every_games=1_000,
            evaluate_every_games=2_000,
            evaluation_pairs=16,
            adaptive_evaluation=False,
            evaluation_early_rejection=False,
            persist_optimizer_state=True,
            resume_replay_items=10_000,
        )

    @classmethod
    def astro3_m4(cls, name: str = "Astro3 M4 self-play") -> RunConfig:
        """Corrected policy-improvement recipe for a base M4/16 GB machine.

        This is deliberately a new generation instead of changing legacy run
        defaults during resume.  The champion remains the deployable model and
        arena anchor, while the learner continues to generate and improve from
        its own trajectories.
        """

        return cls(
            name=name,
            preset="astro3_m4",
            training_generation=3,
            bootstrap_heads=5,
            learning_rate=2e-4,
            min_learning_rate=4e-5,
            learning_rate_schedule="cosine_restarts",
            learning_rate_restart_updates=200_000,
            learning_rate_restart_decay=0.85,
            recent_sample_fraction=0.15,
            replay_sampling_profile="natural",
            importance_correct_replay=True,
            terminal_target_weight=1.0,
            use_bootstrap_targets=False,
            preference_loss_weight=0.0,
            tactical_preference_training=False,
            epsilon_start=0.15,
            epsilon_end=0.05,
            epsilon_decay_games=2_000_000,
            exploration_decision_scale=1.0,
            exploration_top_k=0,
            bootstrap_inclusion_probability=0.35,
            randomized_prior_scale=0.25,
            adaptive_training=True,
            current_selfplay_fraction=0.60,
            deployment_policy_selfplay_fraction=0.20,
            league_fraction=0.30,
            baseline_fraction=0.10,
            behavior_policy="learner",
            rollback_rejected_candidates=False,
            checkpoint_diagnostic_games=24,
            checkpoint_baseline_pairs=8,
            baseline_regression_tolerance=0.12,
            gate_baseline_regression=False,
            heldout_brier_regression_tolerance=0.02,
            gate_heldout_brier_regression=False,
            gate_raw_tactical_preferences=False,
            require_early_high_cost_retention=False,
            evaluate_every_games=500_000,
            evaluation_early_rejection=True,
            persist_optimizer_state=True,
            resume_replay_items=100_000,
        )

    @classmethod
    def astro4_m4(cls, name: str = "Astro4 policy self-play") -> RunConfig:
        """Legal-set actor-critic training with unbiased paired rollouts."""

        base = cls.astro3_m4(name=name).model_dump()
        base.update(
            preset="astro4_m4",
            training_generation=4,
            seed=20260813,
            # Legal-set batches retain up to 64 alternatives per decision;
            # 2,048 such sets is unnecessarily large for unified 16 GB RAM.
            batch_size=256,
            # Keep cross-head replay overlap low; the required behavior head
            # still guarantees that every trajectory trains its owner.
            bootstrap_inclusion_probability=0.20,
            randomized_prior_scale=0.25,
            # The original 150k window held only about 1,500 player-games in
            # the first real run and forgot useful play within one checkpoint.
            policy_replay_capacity=250_000,
            policy_value_loss_weight=0.5,
            policy_entropy_weight=0.03,
            policy_importance_clip=2.0,
            counterfactual_fraction=0.02,
            counterfactual_max_per_game=1,
            # Paired rollouts are an auxiliary ordering signal, not the main
            # objective.  At 0.25 their replayed loss could dominate the much
            # smaller advantage loss and rapidly collapse policy entropy.
            counterfactual_loss_weight=0.05,
            preference_loss_weight=0.0,
            tactical_preference_training=False,
            heuristic_bootstrap_updates=0,
            learning_rate=1e-4,
            min_learning_rate=2e-5,
            gradient_clip=1.0,
            # Keep the deployed champion monotonic. A failed candidate is a
            # branch to abandon, while the next branch still gets fresh
            # learner-driven games and exploration from the champion.
            rollback_rejected_candidates=True,
            gate_heldout_brier_regression=True,
            maximum_heldout_brier=0.24,
            require_early_high_cost_retention=False,
            require_resource_efficiency=False,
            minimum_head_disagreement_rate=0.05,
            # Astro4 collects fewer but richer decisions per second. Evaluate
            # twice as often in game-count terms without weakening pair counts.
            checkpoint_every_games=50_000,
            evaluate_every_games=250_000,
            # Generation-3 replay rows do not contain complete legal sets.
            resume_replay_items=0,
        )
        return cls.model_validate(base)

    @classmethod
    def astro5_search(cls, name: str = "Astro5 search & branching") -> RunConfig:
        """Long-memory action-set learning with live progress control.

        The defaults fit an M4/16 GB learner while deliberately spending disk
        on complete replay/optimizer snapshots.  Promotion remains conservative;
        failed candidates continue as quarantined research branches so a useful
        valley is not destroyed before the next canary can measure it.
        """

        base = cls.astro4_m4(name=name).model_dump()
        base.update(
            preset="astro5_search",
            training_generation=5,
            seed=20260819,
            games_per_actor_batch=4,
            rollout_tasks_per_actor=4,
            # A compact 12-decision reservoir/player-game gives this budget a
            # horizon near 125k player-games instead of ~2.5k full trajectories.
            policy_replay_capacity=1_500_000,
            policy_replay_decisions_per_player_game=12,
            policy_replay_family_balanced=True,
            replay_warmup=12_000,
            batch_size=384,
            reanalysis_fraction=0.0025,
            reanalysis_max_per_game=1,
            reanalysis_max_actions=4,
            reanalysis_rollouts_per_action=1,
            reanalysis_horizon_turns=2,
            reanalysis_policy_temperature=0.35,
            reanalysis_policy_loss_weight=1.0,
            reanalysis_value_loss_weight=0.5,
            # Disable the stale pairwise auxiliary; search targets are attached
            # to the exact state/action set that produced them.
            counterfactual_fraction=0.0,
            counterfactual_max_per_game=0,
            counterfactual_loss_weight=0.0,
            policy_entropy_weight=0.02,
            rejected_candidate_action="continue",
            rollback_rejected_candidates=False,
            checkpoint_every_games=5_000,
            canary_every_games=10_000,
            canary_pairs=64,
            evaluate_every_games=50_000,
            evaluation_early_acceptance=True,
            evaluation_early_acceptance_min_pairs=MINIMUM_PROMOTION_PAIRS,
            evaluation_early_acceptance_confidence=0.995,
            persist_optimizer_state=True,
            resume_replay_items=500_000,
            realtime_governor=True,
            adaptive_training=False,
            governor_interval_games=500,
            governor_target_normalized_entropy=0.55,
            governor_branch_after_failures=3,
            natural_diagnostic_positions=2_000,
            minimum_head_disagreement_rate=0.0,
            gate_heldout_brier_regression=False,
            maximum_heldout_brier=1.0,
            keep_checkpoints=100,
        )
        return cls.model_validate(base)


SAFE_LIVE_FIELDS = {
    "duration_minutes",
    "updates_per_iteration",
    "epsilon_end",
    "current_selfplay_fraction",
    "deployment_policy_selfplay_fraction",
    "league_fraction",
    "baseline_fraction",
    "checkpoint_every_games",
    "evaluate_every_games",
    "evaluation_pairs",
    "adaptive_evaluation",
    "promotion_confidence",
    "promotion_margin",
    "metrics_interval_seconds",
    "games_per_actor_batch",
    "rollout_tasks_per_actor",
    "reanalysis_fraction",
    "reanalysis_max_per_game",
    "reanalysis_max_actions",
    "reanalysis_rollouts_per_action",
    "reanalysis_horizon_turns",
    "policy_entropy_weight",
    "canary_every_games",
    "canary_pairs",
    "realtime_governor",
    "governor_interval_games",
    "evaluation_early_acceptance",
    "evaluation_early_acceptance_min_pairs",
    "evaluation_early_acceptance_confidence",
}


def preset_config(preset: str) -> RunConfig:
    if preset == "astro5_search":
        return RunConfig.astro5_search()
    if preset == "astro4_m4":
        return RunConfig.astro4_m4()
    if preset == "astro3_m4":
        return RunConfig.astro3_m4()
    if preset == "quick":
        return RunConfig.quick()
    if preset == "m4_24h":
        return RunConfig()
    raise ValueError(f"unknown preset: {preset}")
