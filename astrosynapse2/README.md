# Astrosynapse 2

Astrosynapse 2 is a from-scratch Star Realms self-play system built for a 16 GB Apple M4 Mac mini. It includes:

- a deterministic, typed base-set game engine;
- parallel CPU self-play with a compact NumPy actor;
- an MLX/Metal bootstrapped action-value learner;
- stratified replay that protects rare decision families;
- frozen-opponent league training and heuristic anchors;
- paired-seed, seat-swapped arena evaluation with confidence intervals;
- a persistent local training dashboard and model registry;
- a browser game for playing against any saved checkpoint.

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

Press Control-C in that Terminal to request a safe stop and final checkpoint. A stop can take one actor batch to finish. Closing only the browser does **not** stop training; reopen the dashboard and it reconnects to SQLite-backed state and metrics.

## First run

1. Open **Train**.
2. Select **Quick validation** and click **Launch run**. Let this run for about five minutes to verify Metal, workers, replay, checkpoints, and live charts.
3. Check **Diagnostics / Settings** for `MLX / Metal`, actor throughput, memory pressure, truncations, replay family balance, and non-finite/error warnings.
4. If the quick run is healthy, create the **M4 24-hour** preset and launch it.
5. When the 24-hour budget ends, open **Models & Arena**. If the final candidate was created after the last scheduled gate, compare it with the champion for 5,000 pairs before deciding which actor to export.

The 24-hour preset is a starting point tuned for the base M4/16 GB target: 8 CPU actors, a 192-wide three-block model, 3 bootstrap heads, 900,000 replay decisions, 2,048-sample GPU batches, and conservative 5,000-pair promotion evaluations. Actual throughput and strength depend on learned game length, memory pressure, and the opponents encountered, so the GUI reports measured rates rather than promising a fixed result.

## Dashboard guide

- **Overview** — run phase, elapsed time and ETA, games/s, decisions/s, replay fill, learning quality, hardware, and recent persisted events.
- **Train** — create a run, select the recommended or quick preset, expose advanced settings, and start, pause, resume, stop, or checkpoint at safe boundaries.
- **Models & Arena** — inspect checkpoint lineage, pin models, download `.actor.npz` exports, and compare checkpoints or baselines with exact seed-paired seat swaps. Use 5,000 pairs before treating a result as promotion evidence.
- **Play** — select a checkpoint, choose the starting seat, and play through legal semantic actions. Checkpoint games show the model's value for each of your legal choices; baseline games do not invent model scores.
- **Diagnostics / Settings** — outcome BCE, Brier score, explained variance, bootstrap uncertainty, decision-family mix, CPU/RAM/Metal telemetry, audit events, and settings that are safe to apply between batches.

The first random checkpoint is only a lineage root. It is not a trained opponent. “Champion” means the latest model to pass a full conservative paired comparison, not an absolute Elo claim.

## Files and recovery

Runtime state is local and ignored by Git:

```text
data/astrosynapse2.sqlite3       runs, metrics, arena jobs, audit events
data/checkpoints/<run-id>/       safetensors and NumPy actor snapshots
```

Training survives browser disconnects. If the backend or Mac exits unexpectedly, the run is marked `interrupted` at restart and can resume from its latest model checkpoint. Paired arena jobs retain completed pairs and resume after a clean backend restart. Replay contents and AdamW optimizer moments are intentionally not checkpointed in this release, so a resumed learner refills replay and restarts optimizer state before updating again.

## Command line

```bash
# Verify MLX/Metal and hardware discovery
./.venv/bin/astro2 doctor

# Run without the browser
./.venv/bin/astro2 train --preset quick
./.venv/bin/astro2 train --preset m4_24h --name "Overnight league"

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
- [Legacy rule audit and corrected behavior](docs/RULE_AUDIT.md)
- [Local API](docs/API.md)

## Honest scope

This release is a complete working training system, not a pre-trained “excellent” model. No self-play algorithm can certify strength from elapsed time alone. Use held-out baselines, frozen champions, paired seeds, seat splits, truncation rates, and confidence intervals to decide whether a 24-hour result is genuinely better. The Python engine is currently the production engine; a future Rust vector engine should replace it only after differential replay tests prove rule equivalence.
