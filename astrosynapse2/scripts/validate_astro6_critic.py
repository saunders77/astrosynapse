"""Bounded validation; install the paused runtime revision only after checks pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    campaign, out = args.campaign.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result = {"status": "running", "steps": []}

    def save():
        (out / "result.json").write_text(json.dumps(result, indent=2))

    def run(name, command, runtime=None):
        print(name, flush=True)
        environment = {
            **os.environ,
            "PYTHONPATH": str(runtime or project / "backend"),
            "VECLIB_MAXIMUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        with (out / f"{name}.log").open("w") as log:
            subprocess.run(
                command,
                cwd=project,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        result["steps"].append(name)
        save()

    def compare(left, right):
        for suffix in [".actor.npz", ".optimizer.npz", ".critic-optimizer.npz"]:
            a, b = left.with_suffix(suffix), right.with_suffix(suffix)
            if not a.exists() and not b.exists():
                continue
            with np.load(a) as one, np.load(b) as two:
                assert set(one.files) == set(two.files), suffix
                for name in one.files:
                    np.testing.assert_array_equal(one[name], two[name], err_msg=f"{suffix}: {name}")

    save()
    try:
        state = json.loads((campaign / "state.json").read_text())
        stage = campaign / f"stage-{state['stage']:03d}"
        tip = json.loads((stage / "state.json").read_text())["model"]
        champion = state["champion"]
        runtime = out / "runtime"
        shutil.copytree(
            campaign / "runtime",
            runtime,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".cache"),
        )
        sources = {
            "scripts/onpolicy_experiment.py": project / "scripts/onpolicy_experiment.py",
            "scripts/progressive_training.py": project / "scripts/progressive_training.py",
            "astro2/onpolicy.py": project / "backend/astro2/onpolicy.py",
        }
        hashes = {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()
        }
        for name, path in sources.items():
            shutil.copy2(path, runtime / name)
        result.update(source=tip, champion=champion, validated_sources=hashes)
        save()
        run(
            "unit-tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "backend/tests/test_onpolicy.py",
                "backend/tests/test_progressive_supervisor.py",
                "backend/tests/test_native_actor.py",
            ],
        )

        def train(name, iterations, *, separate=True, resume=False, old=False, experiment=False):
            code = campaign / "runtime" if old else runtime
            destination = out / name
            command = [
                sys.executable,
                str(code / "scripts/onpolicy_experiment.py"),
                "--model",
                tip,
                "--opponent",
                champion,
                "--output",
                str(destination),
                "--iterations",
                str(iterations),
                "--games",
                "128" if experiment else "4",
                "--workers",
                "4" if experiment else "2",
                "--batch-size",
                "512",
                "--epochs",
                "2",
                "--temperature",
                "0.03",
                "--learning-rate",
                "0.00001",
                "--max-learning-rate",
                "0.00001",
                "--target-kl",
                "0.015",
                "--adaptive-kl",
                "--eval-every",
                "1000000",
                "--eval-pairs",
                "1024" if experiment else "8",
                "--seed",
                "2026091917",
                "--initial-optimizer",
                str(Path(tip).with_suffix(".optimizer.npz")),
            ]
            if not old:
                command += ["--separate-critic" if separate else "--no-separate-critic"]
            if resume:
                command += ["--resume"]
            run(name + ("-resume" if resume else ""), command, code)
            return Path(json.loads((destination / "state.json").read_text())["model"])

        complete = train("uninterrupted", 2)
        train("split", 1)
        resumed = train("split", 2, resume=True)
        compare(complete, resumed)
        result["steps"].append("exact-model-and-both-optimizers-resume")
        old = train("historical-control", 2, old=True)
        legacy = train("compatibility-control", 2, separate=False)
        compare(old, legacy)
        result["steps"].append("exact-historical-behavior-with-correction-disabled")
        save()

        for name, actor in [
            ("champion-value-audit", champion),
            ("stalled-learner-value-audit", str(Path(tip).with_suffix(".actor.npz"))),
        ]:
            run(
                name,
                [
                    sys.executable,
                    str(project / "scripts/audit_astro6_values.py"),
                    "--actor",
                    actor,
                    "--opponent",
                    champion,
                    "--output",
                    str(out / f"{name}.json"),
                ],
                runtime,
            )

        # A bounded comparison supplies calibration and throughput evidence.
        # Its screening scores do not authorize a promotion or prove strength.
        train("ab-control", 32, separate=False, experiment=True)
        train("ab-corrected", 32, experiment=True)
        result["experiment_scope"] = (
            "4096 training games per arm; 1024 paired evaluation seeds per arm; exploratory, no promotion"
        )
        for name, path in sources.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]:
                raise RuntimeError(
                    "source changed during validation; refusing runtime installation"
                )
        run(
            "install-paused-runtime",
            [
                sys.executable,
                "scripts/maintain_progressive_runtime.py",
                "--output",
                str(campaign),
                "--separate-critic",
                "--reason",
                "September 19 critic correction: train forced positions, isolate critic Adam and gradient clipping, persist critic state, measure pre-fit calibration. Historical models, game rules and promotion evidence retained. Validation: "
                + str(out),
            ],
        )
        result["status"] = "passed_and_installed"
    except BaseException:
        result.update(status="failed_not_installed", error=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
