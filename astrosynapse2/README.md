# Astrosynapse 2 / Astro4 training generation

This repository is a from-scratch Star Realms self-play system built for a 16 GB Apple M4 Mac mini. Astro4 replaces chosen-action outcome fitting with normalized legal-set policy learning, a separate value baseline, game-balanced replay, paired counterfactual rollouts, and competency gates. Retained Astro3 and Astro2 checkpoint actors remain playable. It includes:

- a deterministic, typed base-set game engine;
- parallel CPU self-play with a compact NumPy actor;
- an MLX/Metal bootstrapped legal-set actor-critic learner;
- versioned flat and relational information-state encoders;
- game- and turn-phase-balanced legal-set replay;
- rules-level dominance masks without the unsafe global `END_TURN` preference objective;
- learner, accepted champion-history, and corrected heuristic opponents;
- independent head adapters and bootstrapped exploration;
- paired-seed, seat-swapped arena evaluation with confidence intervals;
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
4. If the quick run is healthy, create the **Astro4 legal-set policy** preset and launch it.
5. When the 24-hour budget ends, wait for the visible **Finalizing evaluation** phase. Natural completion resolves any older trainer-owned job and then the newest due trainer comparison before reporting complete; then inspect the champion and final candidate in **Models & Arena**.

The Astro4 preset is the recommended M4/16 GB starting point: 8 CPU actors, a 192-wide three-block model, 5 head-specific policy adapters, 150,000 game-balanced legal-set decisions, a separate state-value objective, unbiased bounded paired counterfactual rollouts, and conservative promotion evaluations. Ensemble diversity, truncation safety, held-out calibration, and paired arena strength are release gates. Astro3 remains available for controlled chosen-action Monte Carlo comparisons, and **Astro2 compatibility** preserves generation-2 learner settings and checkpoint decoding.

## Dashboard guide

- **Overview** — run phase, elapsed time and ETA, games/s, decisions/s, replay fill, learning quality, hardware, and recent persisted events.
- **Train** — create a run, select Astro4, Astro3, quick validation, or the Astro2 compatibility preset, expose advanced settings, and start, durably pause, resume, stop, or checkpoint at safe boundaries.
- **Models & Arena** — inspect checkpoint lineage, pin models, download `.actor.npz` exports, compare checkpoints or baselines with exact seed-paired seat swaps, and select any retained candidate for a one-click 1,000-game Scrap Elo or Acquire Elo card-choice probe. Use 5,000 manual pairs before treating a head-to-head comparison as release-strength evidence.
- **Play** — select a checkpoint, choose the starting seat, and play through legal semantic actions. Astro4 shows normalized legal-action policy shares; legacy checkpoints show independent outcome estimates. Baseline games do not invent model scores.
- **Diagnostics / Settings** — outcome losses, calibration and ensemble diagnostics, replay write/sample ratios and effective weights, effective exploration, population mix, plateau response, CPU/RAM/Metal telemetry, audit events, and settings that are safe to apply between batches.

The first random checkpoint is only a lineage root and initial deployment anchor, not a trained opponent. After that root, “champion” means the latest model to pass its persisted automatic paired-evaluation contract; adaptive early tiers use fewer pairs than the mature 5,000-pair gate. It is not an absolute Elo claim.

## Files and recovery

Runtime state is local and ignored by Git:

```text
data/astrosynapse2.sqlite3       runs, metrics, arena jobs, audit events
data/checkpoints/<run-id>/       safetensors and NumPy actor snapshots
data/analysis/                   completed card-choice Elo text and JSON reports
```

Training survives browser disconnects. If the backend or Mac exits unexpectedly, the run is marked `interrupted` at restart and can resume from its latest compatible model checkpoint. Paired arena jobs retain completed pairs and recover after a clean backend restart. Astro4 checkpoints preserve compatible weights, optimizer state, counters, elapsed time, rollout RNG state, and league statistics; legal-set replay is deliberately repopulated after restart before learning resumes. Astro3 retains its configured recent replay journal. Legacy Astro2 checkpoints remain weight-only.

## Command line

```bash
# Verify MLX/Metal and hardware discovery
./.venv/bin/astro2 doctor

# Run without the browser
./.venv/bin/astro2 train --preset quick
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

- [System and algorithm design](docs/DESIGN.md)
- [Training, metrics, and evaluation](docs/TRAINING.md)
- [Forensic plateau analysis and Astro3 roadmap](docs/PLATEAU_ANALYSIS_AND_ASTROSYNAPSE3.md)
- [Checkpoint artifact retention and safety](docs/CHECKPOINT_RETENTION.md)
- [Legacy rule audit and corrected behavior](docs/RULE_AUDIT.md)
- [Local API](docs/API.md)

## Honest scope

This is a corrected training platform, not a pre-trained “excellent” Astro4 model. The audit identifies direct causes of the Astro3 learning weakness and verifies the new system at unit/integration scale; only fresh multi-seed training and held-out paired arenas can establish a large skill gain. Use diverse frozen opponents, paired seeds, seat splits, truncation rates, calibration, and confidence intervals. The Python engine remains the production engine; a future native search engine should replace it only after differential replay tests prove rule equivalence.
