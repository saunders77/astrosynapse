"""Run focused checks, a real-game smoke branch, then the bounded main trial."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(exist_ok=True, parents=True)
    report = dict(status="running", steps=[])
    env = {
        **os.environ,
        "PYTHONPATH": str(project / "backend"),
        "VECLIB_MAXIMUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }

    def save():
        temporary = out / "validation.tmp"
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(out / "validation.json")

    def run(name, command):
        report["active_step"] = name
        save()
        print(name, flush=True)
        with (out / f"{name}.log").open("w") as log:
            subprocess.run(
                command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        report["steps"].append(name)
        save()

    save()
    try:
        run(
            "unit-tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "backend/tests/test_counterfactual.py",
                "backend/tests/test_planning.py",
                "backend/tests/test_native_actor.py",
                "backend/tests/test_onpolicy.py",
                "backend/tests/test_progressive_supervisor.py",
            ],
        )
        common = [
            sys.executable,
            "scripts/counterfactual_experiment.py",
            "--campaign",
            str(args.campaign.resolve()),
        ]
        run(
            "smoke",
            [
                *common,
                "--output",
                str(out / "smoke"),
                "--games",
                "10",
                "--workers",
                "2",
                "--rollouts",
                "2",
                "--roots",
                "1",
                "--anchors",
                "2",
                "--epochs",
                "1",
                "--eval-pairs",
                "8",
                "--hours",
                "0.5",
            ],
        )
        smoke = json.loads((out / "smoke/state.json").read_text())
        if smoke["phase"] not in {"complete", "no_heldout_improvement", "insufficient_signal"}:
            raise RuntimeError(f"smoke branch did not finish: {smoke['phase']}")
        # Verify that the learned actor retained the state representation,
        # critic, and all intermediate features byte for byte.
        if smoke.get("model"):
            import numpy as np
            from safetensors.numpy import load_file

            source = load_file(str(out / "smoke/source.safetensors"))
            candidate = load_file(smoke["model"])
            for name in source:
                if not name.startswith("head_outputs."):
                    np.testing.assert_array_equal(source[name], candidate[name], err_msg=name)
            report["steps"].append("unchanged-critic-and-feature-tensors")
        run(
            "install-forced-state-coverage",
            [
                sys.executable,
                "scripts/maintain_progressive_runtime.py",
                "--output",
                str(args.campaign.resolve()),
                "--separate-critic",
                "--reason",
                "September 20 follow-up: record critic examples through the decision hook, including single legal actions that bypass the chooser. Champion, policy updates, game rules and pending evaluation retained. Validation: "
                + str(out),
            ],
        )
        run("trial", [*common, "--output", str(out / "trial")])
        state = json.loads((out / "trial/state.json").read_text())
        report.update(status=state["phase"], trial=str(out / "trial"), active_step=None)
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
