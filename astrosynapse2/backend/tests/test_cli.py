from argparse import Namespace

from astro2 import cli


def test_headless_training_uses_one_recovering_arena_without_global_drain(tmp_path, monkeypatch):
    calls: list[object] = []
    created_configs = []

    class FakeStore:
        def __init__(self, path):
            calls.append(("store", path))

        def get_run(self, run_id):
            assert run_id == "run-1"
            return {"id": run_id, "status": "complete"}

    class FakeArena:
        def __init__(self, store):
            self.store = store
            calls.append("arena_created")

        def wait_for_idle(self):
            raise AssertionError("trainer completion must not wait for unrelated arena jobs")

        def shutdown(self):
            calls.append("arena_shutdown")

    class FakeSupervisor:
        def __init__(self, store, _project_root, *, evaluation_manager):
            assert evaluation_manager.store is store
            calls.append("shared_arena")

        def create_run(self, config):
            created_configs.append(config)
            return {"id": "run-1"}

        def start(self, run_id):
            assert run_id == "run-1"
            calls.append("training_started")

        def stop(self, _run_id):
            calls.append("training_stopped")

        def shutdown(self):
            calls.append("supervisor_shutdown")

    monkeypatch.setattr(cli, "Store", FakeStore)
    monkeypatch.setattr(cli, "ArenaManager", FakeArena)
    monkeypatch.setattr(cli, "Supervisor", FakeSupervisor)
    monkeypatch.setattr(cli.signal, "signal", lambda *_args: None)

    result = cli.command_train(
        Namespace(
            data_dir=str(tmp_path),
            preset="quick",
            name=None,
            seed=314159,
            minutes=1,
        )
    )

    assert result == 0
    assert created_configs[0].seed == 314159
    assert calls.index("shared_arena") < calls.index("training_started")
    assert calls[-2:] == ["supervisor_shutdown", "arena_shutdown"]


def test_headless_signal_stops_trainer_before_arena_shutdown(tmp_path, monkeypatch):
    calls: list[object] = []
    handlers = {}

    class FakeStore:
        def __init__(self, _path):
            self.stop_requested = False

        def get_run(self, run_id):
            assert run_id == "run-1"
            if not self.stop_requested:
                handlers[cli.signal.SIGINT](cli.signal.SIGINT, None)
                assert "arena_shutdown" not in calls
                self.stop_requested = True
            return {"id": run_id, "status": "stopped"}

    class FakeArena:
        def __init__(self, _store):
            pass

        def wait_for_idle(self):
            raise AssertionError("an explicitly stopped run must not drain final evaluation")

        def shutdown(self):
            calls.append("arena_shutdown")

    class FakeSupervisor:
        def __init__(self, store, _project_root, *, evaluation_manager):
            self.store = store
            assert evaluation_manager is not None

        def create_run(self, _config):
            return {"id": "run-1"}

        def start(self, _run_id):
            calls.append("training_started")

        def stop(self, _run_id):
            calls.append("training_stopped")

        def shutdown(self):
            calls.append("supervisor_shutdown")

    monkeypatch.setattr(cli, "Store", FakeStore)
    monkeypatch.setattr(cli, "ArenaManager", FakeArena)
    monkeypatch.setattr(cli, "Supervisor", FakeSupervisor)
    monkeypatch.setattr(
        cli.signal, "signal", lambda signum, handler: handlers.setdefault(signum, handler)
    )

    result = cli.command_train(
        Namespace(data_dir=str(tmp_path), preset="quick", name=None, seed=None, minutes=1)
    )

    assert result == 0
    assert calls == [
        "training_started",
        "training_stopped",
        "supervisor_shutdown",
        "arena_shutdown",
    ]


def test_train_parser_accepts_an_explicit_seed():
    args = cli.build_parser().parse_args(["train", "--preset", "astro3_m4", "--seed", "20260812"])

    assert args.seed == 20260812


def test_card_analysis_parser_defaults_to_one_thousand_games():
    args = cli.build_parser().parse_args(
        ["card-analysis", "--model", "candidate-42", "--kind", "acquire"]
    )

    assert args.model == "candidate-42"
    assert args.kind == "acquire"
    assert args.games == 1_000
