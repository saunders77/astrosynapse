# Astrosynapse 2 / Astro5 search generation

This repository is a from-scratch Star Realms self-play system built for a 16 GB Apple M4 Mac mini. Astro5 extends legal-set actor-critic training with public-belief action-set search, direct search-policy/value targets, compact long-horizon replay, cheap checkpoint canaries, complete lineage restoration, sequential checkpoint branches, and a bounded realtime training governor. Astro4, Astro3, and Astro2 checkpoint actors remain playable. It includes:

- a deterministic, typed base-set game engine;
- parallel CPU self-play with a compact NumPy actor;
- an MLX/Metal bootstrapped legal-set actor-critic learner;
- versioned flat and relational information-state encoders;
- game-, turn-phase-, and decision-family-balanced legal-set replay with per-game reservoirs;
- rules-level dominance masks without the unsafe global `END_TURN` preference objective;
- learner, accepted champion-history, and corrected heuristic opponents;
- independent head adapters and bootstrapped exploration;
- public-information-set reanalysis with common-random-number action comparisons;
- paired-seed, seat-swapped canary and promotion evaluation with confidence intervals;
- durable multi-branch experiments from any compatible champion/checkpoint;
- a persistent local training dashboard and model registry;
- a browser game for playing against any retained checkpoint actor.

The original `sim.py` supplied the card/rule reference only. Astrosynapse 2 does not reuse the legacy PPO trainer, lossy feature representation, promotion gate, or GUI.

## Install on the M4 Mac mini

1. Open `astrosynapse2` in Finder.
2. Double-click `setup.command` once.
3. If macOS asks, allow the script to open in Terminal.

The setup script installs an isolated Python 3.12 environment with pinned `uv`, installs MLX and the backend dependencies, installs the dashboard packages, and builds the dashboard. It does not alter the system Python.

Node.js 22 or newer is required. If setup reports that Node is missing, install it first (for example, `brew install node`) and run `setup.command` again.

Terminal equivalent:

```bash
cd /Users/michael/Documents/astrosynapse/astrosynapse2
./setup.command
```

## Start the application

Double-click `start.command`, or run:

```bash
cd /Users/michael/Documents/astrosynapse/astrosynapse2
./start.command
```

The dashboard opens at [http://127.0.0.1:3000](http://127.0.0.1:3000). The control API listens only on `127.0.0.1:8765`. Keep the Terminal window open; the launcher also uses `caffeinate` so macOS does not sleep through a long run.

Press Control-C in that Terminal to request a safe stop and final checkpoint. During self-play, a stop can take one actor batch to finish; during final evaluation it cancels the trainer-owned arena at a game boundary, or immediately if the job is still queued. Closing only the browser does **not** stop training; reopen the dashboard and it reconnects to SQLite-backed state and metrics.

## First run

1. Open **Train**.
2. Select **Quick validation** and click **Launch run**. Let this run for about five minutes to verify Metal, workers, replay, checkpoints, and live charts.
3. Check **Diagnostics / Settings** for `MLX / Metal`, actor throughput, memory pressure, truncations, replay family balance, and non-finite/error warnings.
4. If the quick run is healthy, create the **Astro5 search & branching** preset and launch it.
5. When the 24-hour budget ends, wait for the visible **Finalizing evaluation** phase. Natural completion resolves any older trainer-owned job and then the newest due trainer comparison before reporting complete; then inspect the champion and final candidate in **Models & Arena**.

The Astro5 preset is the recommended M4/16 GB starting point: 8 CPU actors, four work-stealing tasks per actor, a 192-wide three-block model, 5 policy heads, and a 1.5-million-decision replay budget compacted to 12 phase/family-balanced decisions per player-game. Search samples at most one position in 0.25% of games, compares four actions with one shared-randomness rollout each, and bootstraps from the network value after a two-turn lookahead instead of simulating every branch to termination. The preset checkpoints and runs a 64-pair canary every 10,000 games, and starts a 2,000-pair promotion evaluation every 50,000 games. A multiplicity-corrected acceptance test may promote an unambiguously strong candidate from 1,000 pairs; ambiguous candidates run all 2,000. If the completed result is positive but its 95% paired interval has a lower bound from 48% through 50%, the same resumable evaluation runs 2,000 additional pairs, capped at 4,000 total. Rejected candidates remain quarantined from deployment but continue learning, which lets them cross temporary performance valleys. Optimization health is reconsidered every 500 games, while strategic governor decisions change only when new canary evidence arrives. Astro4 remains available as the previous legal-set control.

## Dashboard guide

- **Overview** — run phase, elapsed time and ETA, games/s, decisions/s, replay fill, learning quality, hardware, and recent persisted events.
- **Train** — create a run, select Astro5, Astro4, Astro3, quick validation, or Astro2 compatibility, expose search/governor settings, and start, durably pause, resume, stop, or checkpoint at safe boundaries.
- **Branch Lab** — choose a compatible generation-4/5 checkpoint from any run, select any combination of 1–8 optimization/search variants, and optionally run the queue automatically. Source checkpoints are pinned and every branch independently receives copied immutable artifacts plus its own seed stream; branches do not continue from the preceding branch's champion. Per-branch budgets can be expressed in minutes, training games, or completed full evaluations. The Branch runner controls always pause, resume, or stop the actually active trainer even when a queued branch is selected for inspection.
- **Models & Arena** — inspect checkpoint lineage, pin models, download `.actor.npz` exports, compare checkpoints or baselines with exact seed-paired seat swaps, and select any retained candidate for a one-click 1,000-game Scrap Elo or Acquire Elo card-choice probe. Manual comparisons are capped at 2,000 pairs.
- **Play** — select a checkpoint, choose the starting seat, and play through legal semantic actions. Astro4 shows normalized legal-action policy shares; legacy checkpoints show independent outcome estimates. Baseline games do not invent model scores.
- **Diagnostics / Settings** — outcome/search losses, natural-state calibration and ensemble diagnostics, objective gradient norms, clipping, normalized entropy, governor multipliers/reasons, replay health, CPU/RAM/Metal telemetry, and audit events.

The first random checkpoint is only a lineage root and initial deployment anchor, not a trained opponent. After that root, “champion” means the latest model to pass its persisted automatic paired-evaluation contract; adaptive early tiers use fewer pairs than the mature 2,000-pair gate. It is not an absolute Elo claim.

### Branch Lab workflow

1. Open **Branch Lab** and choose any listed generation-4/5 champion or candidate. The selector spans all runs, not only the run currently shown elsewhere in the dashboard.
2. Check any 1–8 variants and choose a budget in minutes, training games, or completed full evaluations per branch. The built-in recipes test balanced search, heavier search, entropy recovery, value emphasis, faster exploitation, broader belief search, low-LR long memory, and higher exploration.
3. Leave **Run queued branches automatically** enabled to use the single Metal device sequentially. If another ordinary run is active, the experiment remains queued and starts after that trainer releases the device.
4. Click **Create & start branch system**. The source is pinned; every branch starts from that same source with a private copy of every available artifact and a deterministic independent seed. Auto-advance changes which independent branch runs next; it does not feed one branch's winner into another.
5. Click any branch card to make it the selected run. Use **Overview** and **Diagnostics / Settings** to watch canary scores, normalized entropy, search-target coverage, clipping, objective gradient norms, and governor changes. **Pause & save** and **Stop** target the active trainer, not merely the selected card; the same controls are also present in Branch Lab.
6. Treat 64-pair canaries as direction/slope signals only. Promotion evidence begins at the 50,000-game full gate. A confidence-safe acceptance look can end that evaluation at 1,000 pairs only when its multiplicity-adjusted lower bound clears the promotion threshold; otherwise it continues to the 2,000-pair cap. A failed canary or promotion quarantines deployment; it does not destroy the learner branch.

## Files and recovery

Runtime state is local and ignored by Git:

```text
data/astrosynapse2.sqlite3       runs, metrics, arena jobs, audit events
data/checkpoints/<run-id>/       safetensors and NumPy actor snapshots
data/analysis/                   completed card-choice Elo text and JSON reports
```

Training survives browser disconnects. If the backend or Mac exits unexpectedly, the run is marked `interrupted` at restart and can resume from its latest compatible model checkpoint. Paired arena jobs retain completed pairs and recover after a clean backend restart. Astro5 checkpoints preserve weights, actor export, optimizer, compact policy replay, auxiliary replay, counters, elapsed time, branch-relative schedule origins, rollout RNG state, league state, and controller state. Pause/final boundaries write the full configured replay archive; scheduled boundaries write the configured recent horizon. Astro4 and older checkpoints retain their original compatibility contracts.

## Command line

```bash
# Verify MLX/Metal and hardware discovery
./.venv/bin/astro2 doctor

# Run without the browser
./.venv/bin/astro2 train --preset quick
./.venv/bin/astro2 train --preset astro5_search --name "Astro5 search run" --seed 20260819
./.venv/bin/astro2 train --preset astro4_m4 --name "Astro4 seed 1" --seed 20260813
./.venv/bin/astro2 train --preset astro4_m4 --name "Astro4 seed 2" --seed 20260814
./.venv/bin/astro2 train --preset astro4_m4 --name "Astro4 seed 3" --seed 20260815
./.venv/bin/astro2 train --preset astro3_m4 --name "Astro3 control" --seed 20260816
./.venv/bin/astro2 train --preset m4_24h --name "Overnight league"

# Run the same card probe offered in Models & Arena
./.venv/bin/astro2 card-analysis --model <checkpoint-id> --kind scrap
./.venv/bin/astro2 card-analysis --model <checkpoint-id> --kind acquire

# Start only the API (dashboard development or automation)
./.venv/bin/astro2 serve

# Development mode with hot reload
./dev.command

# Validation
PYTHONPATH=backend ./.venv/bin/pytest
./.venv/bin/ruff check backend/astro2 backend/tests
npm run build
npm test
```

MLX initializes Metal when imported. A headless or restricted shell may report that no Metal device is available even when `astro2 doctor` works from the normal macOS Terminal.

## Documentation

- [Astro5 search, governor, and branching](docs/ASTRO5_SEARCH_AND_BRANCHING.md)
- [System and algorithm design](docs/DESIGN.md)
- [Training, metrics, and evaluation](docs/TRAINING.md)
- [Forensic plateau analysis and Astro3 roadmap](docs/PLATEAU_ANALYSIS_AND_ASTROSYNAPSE3.md)
- [Checkpoint artifact retention and safety](docs/CHECKPOINT_RETENTION.md)
- [Legacy rule audit and corrected behavior](docs/RULE_AUDIT.md)
- [Local API](docs/API.md)

## Honest scope

This is a corrected training platform, not a pre-trained “excellent” Astro4 model. The audit identifies direct causes of the Astro3 learning weakness and verifies the new system at unit/integration scale; only fresh multi-seed training and held-out paired arenas can establish a large skill gain. Use diverse frozen opponents, paired seeds, seat splits, truncation rates, calibration, and confidence intervals. The Python engine remains the production engine; a future native search engine should replace it only after differential replay tests prove rule equivalence.
 