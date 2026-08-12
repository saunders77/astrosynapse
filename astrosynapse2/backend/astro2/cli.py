"""Command-line entry points for setup checks, the API, and unattended runs."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from .arena import ArenaManager
from .config import RunConfig, preset_config
from .hardware import system_snapshot
from .storage import Store
from .supervisor import Supervisor

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def command_doctor(_args: argparse.Namespace) -> int:
    report = {"system": system_snapshot(), "project_root": str(PROJECT_ROOT)}
    try:
        from .hardware import mlx_snapshot

        report["mlx"] = mlx_snapshot()
        report["ok"] = bool(report["mlx"]["metal_available"])
    except Exception as error:
        report["mlx"] = {"error": f"{type(error).__name__}: {error}"}
        report["ok"] = False
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


def command_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "astro2.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        access_log=args.access_log,
    )
    return 0


def command_train(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir).expanduser().resolve()
    store = Store(data_dir / "astrosynapse2.sqlite3")
    arena = ArenaManager(store)
    supervisor = Supervisor(store, PROJECT_ROOT, evaluation_manager=arena)
    config = preset_config(args.preset)
    overrides = config.model_dump()
    overrides["name"] = args.name or config.name
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.minutes is not None:
        overrides["duration_minutes"] = args.minutes
        overrides["preset"] = "custom"
    run = supervisor.create_run(RunConfig.model_validate(overrides))
    supervisor.start(run["id"])

    def request_stop(_signum, _frame):
        supervisor.stop(run["id"])

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print(f"Run {run['id']} started. Press Ctrl-C for a safe stop.")
    try:
        while True:
            current = store.get_run(run["id"])
            if current["status"] in {"stopped", "complete", "failed"}:
                print(json.dumps(current, indent=2, default=str))
                return 0 if current["status"] != "failed" else 1
            time.sleep(1.0)
    finally:
        supervisor.shutdown()
        arena.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astro2", description="Astrosynapse 2 control CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="verify Apple-silicon and MLX access")
    doctor.set_defaults(func=command_doctor)

    serve = subparsers.add_parser("serve", help="run the local control API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--reload", action="store_true")
    serve.add_argument("--access-log", action="store_true")
    serve.set_defaults(func=command_serve)

    train = subparsers.add_parser("train", help="run training without the web interface")
    train.add_argument(
        "--preset",
        choices=["astro3_m4", "m4_24h", "quick"],
        default="astro3_m4",
    )
    train.add_argument("--name")
    train.add_argument(
        "--seed",
        type=int,
        help="reproducibility seed (use a distinct value for each independent run)",
    )
    train.add_argument("--minutes", type=int)
    train.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    train.set_defaults(func=command_train)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
