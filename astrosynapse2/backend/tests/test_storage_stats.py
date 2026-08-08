from astro2.config import RunConfig
from astro2.hardware import RateMeter
from astro2.stats import elo_delta, wilson_interval
from astro2.storage import Store


def test_wilson_interval_behaves_at_small_sample_sizes():
    interval = wilson_interval(15, 24)
    assert interval.estimate == 0.625
    assert interval.low < 0.5 < interval.high
    assert elo_delta(0.5) == 0.0


def test_rate_meter_preserves_last_completed_batch_rate(monkeypatch):
    readings = iter((10.0, 12.0, 12.5))
    monkeypatch.setattr("astro2.hardware.time.monotonic", lambda: next(readings))
    meter = RateMeter.start()
    measured = meter.sample(20, 400)
    final = meter.sample(20, 400)
    assert measured["games_per_second"] == 10.0
    assert measured["decisions_per_second"] == 200.0
    assert final["games_per_second"] == measured["games_per_second"]
    assert final["decisions_per_second"] == measured["decisions_per_second"]


def test_rate_meter_includes_idle_time_between_batched_results(monkeypatch):
    readings = iter((10.0, 11.0, 14.0))
    monkeypatch.setattr("astro2.hardware.time.monotonic", lambda: next(readings))
    meter = RateMeter.start()
    idle = meter.sample(0, 0)
    measured = meter.sample(40, 800)
    assert idle["games_per_second"] == 0.0
    assert measured["games_per_second"] == 10.0
    assert measured["decisions_per_second"] == 200.0


def test_store_round_trip(tmp_path):
    store = Store(tmp_path / "astro2.sqlite3")
    run = store.create_run(RunConfig.quick())
    assert run["status"] == "ready"
    assert run["config"]["preset"] == "quick"

    store.append_metric(run["id"], 1, {"games": 12, "loss": 0.4})
    assert store.metrics(run["id"])[0]["games"] == 12
    store.append_metric(run["id"], 2, {"games": 24})
    store.append_metric(run["id"], 3, {"games": 36})
    assert [item["seq"] for item in store.metrics(run["id"], limit=2)] == [2, 3]
    assert [item["seq"] for item in store.metrics(run["id"], after=1)] == [2, 3]

    updated = store.update_run(run["id"], status="running", games=12)
    assert updated["games"] == 12
    assert store.events(run["id"])[0]["kind"] == "run_created"

    checkpoint = store.add_checkpoint(
        run_id=run["id"],
        label="Candidate",
        path=str(tmp_path / "candidate.safetensors"),
        actor_path=str(tmp_path / "candidate.actor.npz"),
        games=12,
    )
    assert checkpoint["is_pinned"] is False
    assert store.set_checkpoint_pinned(checkpoint["id"], True)["is_pinned"] is True
