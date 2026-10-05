"""Extend an unfinished anytime gate while preserving its exact evidence prefix.

Changes only validated resource settings while the worker is paused. No runtime,
checkpoint, alpha, attempt, seed, or existing result is changed. The API service
can remain running; resume reloads configuration in the existing frozen worker.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import time
from pathlib import Path

from astro2.autopilot import CampaignConfig
from astro2.experiment_control import atomic_json, code_identity, sha256


def extend(folder, *, pairs, policy_fraction):
    folder = Path(folder).resolve()
    with (folder / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((folder / "state.json").read_text())
        pending = state.get("pending") or {}
        if (
            state["status"] != "paused"
            or pending.get("kind") != "policy"
            or pending.get("finished")
            or not pending.get("attempt")
            or state.get("active_job") != pending.get("key", "") + "-gate"
            or state.get("child_pid")
        ):
            raise ValueError("Extension requires a paused, unfinished policy gate")
        old = json.loads((folder / "config.json").read_text())
        if pairs <= old["gate_pairs"]:
            raise ValueError("The new ceiling must exceed the existing gate budget")
        config = CampaignConfig(**{**old, "gate_pairs": pairs, "policy_fraction": policy_fraction})
        runtime = folder / "runtime"
        if code_identity(runtime / "astro2", runtime / "scripts") != json.loads(
            (folder / "code.json").read_text()
        ):
            raise ValueError("Frozen runtime identity changed")
        gate = folder / "tasks" / state["active_job"]
        if (gate / "result.json").exists():
            raise ValueError("A completed gate cannot be reopened")
        rows = json.loads((gate / "pairs.json").read_text())
        if [r["pair"] for r in rows] != list(range(len(rows))) or len(rows) >= old["gate_pairs"]:
            raise ValueError("Gate must have an unfinished contiguous evidence prefix")
        if any(r.get("truncated", 0) for r in rows):
            raise ValueError("A truncated gate cannot qualify for promotion")
        candidate = next(
            m
            for m in state["models"]
            if m["actor_path"] == pending.get("selection", {}).get("actor")
        )
        protected = [
            folder / "state.json",
            folder / "jobs.json",
            folder / "code.json",
            folder / "inputs.json",
            gate / "pairs.json",
            Path(candidate["actor_path"]),
            Path(candidate["path"]),
            Path(pending["champion_actor"]),
            Path(pending["critic_path"]),
        ]
        hashes = {str(p): sha256(p) for p in protected}
        backup = folder / "budget-revisions" / str(time.time_ns())
        backup.mkdir(parents=True)
        for name in ("config.json", "state.json", "jobs.json"):
            shutil.copy2(folder / name, backup / name)
        report = dict(
            created_at=time.time(),
            backup=str(backup),
            key=pending["key"],
            candidate=candidate["id"],
            attempt=pending["attempt"],
            alpha=0.025 / (pending["attempt"] * (pending["attempt"] + 1)),
            completed_pairs=len(rows),
            old_gate_pairs=old["gate_pairs"],
            gate_pairs=pairs,
            old_policy_fraction=old["policy_fraction"],
            policy_fraction=policy_fraction,
            protected_hashes=hashes,
            note="Same anytime test and evidence prefix; no spent attempt is reset.",
        )
        try:
            atomic_json(folder / "config.json", config.model_dump())
            if hashes != {str(p): sha256(p) for p in protected}:
                raise RuntimeError("Protected gate artifacts changed")
            atomic_json(backup / "revision.json", report)
        except BaseException:
            shutil.copy2(backup / "config.json", folder / "config.json")
            raise
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--policy-fraction", type=float, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            extend(args.folder, pairs=args.pairs, policy_fraction=args.policy_fraction), indent=2
        )
    )
