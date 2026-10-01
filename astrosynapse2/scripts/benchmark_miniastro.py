"""Compare MiniAstro budgets on one immutable source dataset, sequentially."""

import argparse
import time
from pathlib import Path

from astro2.acquire_student import ACTIVE, StudentConfig, StudentManager
from astro2.experiment_control import atomic_json
from astro2.storage import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1000, 10000, 100000])
    args = parser.parse_args()
    manager = StudentManager(
        Store(args.data_dir / "astrosynapse2.sqlite3"), args.data_dir / "acquire_students"
    )
    source = manager.artifact(args.source)
    output = manager.output_dir / f"miniastro-benchmark-{int(time.time())}.json"
    report = {"source_student_id": args.source, "status": "running", "results": []}
    atomic_json(output, report)
    for budget in args.budgets:
        job = manager.create(
            source["model_id"],
            StudentConfig(
                games=source["config"]["games"],
                seed=source["config"]["seed"],
                rules_version=source["config"]["rules_version"],
                student_type="miniastro",
                parameter_budget=budget,
                epochs=args.epochs,
                source_student_id=args.source,
            ),
        )
        report["active_student"] = job["id"]
        atomic_json(output, report)
        print(f"Started {budget:,}-parameter budget: {job['id']}", flush=True)
        while job["status"] in ACTIVE:
            time.sleep(3)
            manager.list()  # Reap completed worker processes.
            job = manager.get(job["id"])
        if job["status"] != "complete":
            report.update(status=job["status"], error=job.get("error"))
            atomic_json(output, report)
            return
        result = job["result"]
        report["results"].append(
            {
                "id": job["id"],
                "parameter_count": result["parameter_count"],
                "metrics": result["metrics"],
                "best_epoch": result["best_epoch"],
            }
        )
        print(report["results"][-1], flush=True)
        atomic_json(output, report)
    report.update(status="complete", active_student=None)
    atomic_json(output, report)


if __name__ == "__main__":
    main()
