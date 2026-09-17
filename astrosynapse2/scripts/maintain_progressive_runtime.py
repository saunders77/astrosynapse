"""Apply audited operational fixes to a paused campaign, retaining its old runtime."""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import time
from pathlib import Path

from astro2.experiment_control import atomic_json, code_identity

PATCHABLE = ("scripts/progressive_training.py", "scripts/onpolicy_experiment.py")


def maintain(out: Path, project: Path, reason: str):
    out = out.resolve()
    with (out / "manager.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((out / "state.json").read_text())
        if state["phase"] != "paused":
            raise ValueError("runtime maintenance requires a safely paused campaign")
        runtime = out / "runtime"
        manifest = json.loads((out / "manifest.json").read_text())
        before = code_identity(runtime / "astro2", runtime / "scripts")
        if before != manifest["code_identity"]:
            raise ValueError("frozen runtime changed; refusing to conceal an unknown modification")
        stage_manifest = out / f"stage-{state['stage']:03d}" / "manifest.json"
        active = json.loads(stage_manifest.read_text()) if stage_manifest.exists() else None
        if active and active["code_identity"] != before:
            raise ValueError("active learner runtime identity differs from supervisor")
        changes = {}
        for name in PATCHABLE:
            content = (project / name).read_bytes()
            compile(content, name, "exec")
            if content != (runtime / name).read_bytes():
                changes[name] = content
        if not changes:
            return {"changed_files": []}
        backup = out / "runtime-revisions" / str(time.time_ns())
        backup.mkdir(parents=True)
        shutil.copytree(
            runtime, backup / "runtime", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        shutil.copy2(out / "manifest.json", backup / "manifest.json")
        shutil.copy2(out / "state.json", backup / "state.json")
        if active:
            shutil.copy2(stage_manifest, backup / "stage-manifest.json")
        try:
            for name, content in changes.items():
                target = runtime / name
                temporary = target.with_suffix(".maintenance.tmp")
                temporary.write_bytes(content)
                temporary.replace(target)
            after = code_identity(runtime / "astro2", runtime / "scripts")
            revision = dict(
                created_at=time.time(),
                reason=reason,
                backup=str(backup),
                changed_files={
                    name: {"before": before[name], "after": after[name]} for name in changes
                },
            )
            atomic_json(backup / "revision.json", revision)
            if active:
                active["code_identity"] = after
                active.setdefault("runtime_revisions", []).append(revision)
                atomic_json(stage_manifest, active)
            manifest["code_identity"] = after
            manifest.setdefault("runtime_revisions", []).append(revision)
            atomic_json(out / "manifest.json", manifest)
        except BaseException:
            for name in changes:
                shutil.copy2(backup / "runtime" / name, runtime / name)
            if active:
                shutil.copy2(backup / "stage-manifest.json", stage_manifest)
            shutil.copy2(backup / "manifest.json", out / "manifest.json")
            raise
        return revision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            maintain(args.output, Path(__file__).resolve().parents[1], args.reason), indent=2
        )
    )
