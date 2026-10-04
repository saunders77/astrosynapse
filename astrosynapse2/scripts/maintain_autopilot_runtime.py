"""Install an audited recovery revision at a paused Autopilot block boundary."""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import time
from pathlib import Path

from astro2.autopilot import CampaignConfig
from astro2.experiment_control import atomic_json, code_identity, sha256

PATCHABLE = ("astro2/autopilot.py", "astro2/autopilot_data.py", "scripts/autopilot_worker.py")


def maintain(folder, project, settings):
    folder, project = Path(folder).resolve(), Path(project).resolve()
    with (folder / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((folder / "state.json").read_text())
        if state["status"] != "paused" or state.get("pending") is not None:
            raise ValueError("Maintenance requires a paused campaign at a completed block boundary")
        runtime = folder / "runtime"
        before = code_identity(runtime / "astro2", runtime / "scripts")
        if before != json.loads((folder / "code.json").read_text()):
            raise ValueError("Frozen runtime changed before maintenance")
        config = CampaignConfig(**{**json.loads((folder / "config.json").read_text()), **settings})
        champion = next(m for m in state["models"] if m["id"] == state["champion_id"])
        critic = next(c for c in state["critics"] if c["id"] == state["critic_id"])
        protected = [folder / "state.json", folder / "jobs.json", folder / "inputs.json"]
        protected += [Path(champion[k]) for k in ("path", "actor_path")]
        protected += [Path(critic["path"])]
        protected += list((folder / "tasks").glob("*/pairs.json"))
        protected += list((folder / "tasks").glob("*/result.json"))
        hashes = {str(p): sha256(p) for p in protected}
        revision = folder / "runtime-revisions" / str(time.time_ns())
        revision.mkdir(parents=True)
        shutil.copytree(runtime, revision / "runtime", ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("config.json", "code.json", "state.json", "jobs.json"):
            shutil.copy2(folder / name, revision / name)
        try:
            for name in PATCHABLE:
                source = project / ("backend" if name.startswith("astro2/") else "") / name
                content = source.read_bytes()
                compile(content, name, "exec")
                target = runtime / name
                temporary = target.with_suffix(".maintenance.tmp")
                temporary.write_bytes(content)
                temporary.replace(target)
            after = code_identity(runtime / "astro2", runtime / "scripts")
            atomic_json(folder / "config.json", config.model_dump())
            atomic_json(folder / "code.json", after)
            if hashes != {str(p): sha256(p) for p in protected}:
                raise RuntimeError("Protected campaign artifacts changed during maintenance")
            report = dict(
                created_at=time.time(),
                backup=str(revision),
                settings=settings,
                changed_files={k: dict(before=before[k], after=after[k]) for k in PATCHABLE},
                protected_hashes=hashes,
                champion_id=state["champion_id"],
                critic_id=state["critic_id"],
                attempt=state["attempt"],
                critic_attempt=state["critic_attempt"],
                note="Existing evidence and attempts preserved; new settings apply to future tasks only.",
            )
            atomic_json(revision / "revision.json", report)
            return report
        except BaseException:
            for name in PATCHABLE:
                shutil.copy2(revision / "runtime" / name, runtime / name)
            for name in ("config.json", "code.json"):
                shutil.copy2(revision / name, folder / name)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            maintain(
                args.folder,
                Path(__file__).resolve().parents[1],
                json.loads(args.settings.read_text()),
            ),
            indent=2,
        )
    )
