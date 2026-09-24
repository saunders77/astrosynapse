"""Read-only discovery of retained Astro6 actors, outside the trainer's league."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def progressive_models(data_dir: Path) -> list[dict[str, Any]]:
    root = (data_dir / "progressive").resolve()
    result = []
    for state_path in sorted(root.glob("*/state.json")):
        folder = state_path.parent.resolve()
        if not folder.is_relative_to(root):
            continue
        try:
            state = json.loads(state_path.read_text())
        except (OSError, ValueError):
            continue
        run_name = f"{state.get('name', 'Astro6')} · {folder.name}"
        screens = {str(Path(h["checkpoint"]).resolve()): h for h in state.get("history", [])}
        for gate in state.get("gates", []):
            if gate.get("model"):
                screens.setdefault(str(Path(gate["model"]).resolve()), gate)
        pending = state.get("pending_gate") or {}
        if pending.get("model"):
            screens.setdefault(
                str(Path(pending["model"]).resolve()), {"games": state.get("games", 0)}
            )
        promotions = {
            str(Path(p["actor"]).resolve()): index + 1
            for index, p in enumerate(state.get("promotions", []))
        }
        stage_offsets = {0: 0}
        stage_offsets.update(
            {
                p["stage"] + 1: p["games"]
                for p in state.get("promotions", [])
                if "stage" in p and "games" in p
            }
        )
        actors = [
            folder / "original-champion.actor.npz",
            folder / "source.actor.npz",
            *folder.glob("stage-*/*.actor.npz"),
            *folder.glob("branches/*/g????????.actor.npz"),
        ]
        for actor in actors:
            actor = actor.resolve()
            if not actor.is_relative_to(folder) or not actor.is_file():
                continue
            relative = actor.relative_to(root).as_posix()
            model = actor.with_name(actor.name.removesuffix(".actor.npz") + ".safetensors")
            screen = screens.get(str(model), {})
            stage_name = actor.parent.name.removeprefix("stage-")
            local_games = model.stem.removeprefix("g")
            games = stage_offsets.get(int(stage_name), 0) if stage_name.isdigit() else 0
            if local_games.isdigit():
                games += int(local_games)
            generation = promotions.get(str(actor))
            checkpoint_name = actor.relative_to(folder).as_posix().removesuffix(".actor.npz")
            label = f"Astro6 · {folder.name} · {checkpoint_name}"
            if generation is not None:
                label += f" · Generation {generation}"
            result.append(
                dict(
                    id="astro6-" + hashlib.sha256(relative.encode()).hexdigest()[:24],
                    run_id=f"astro6-{folder.name}",
                    run_name=run_name,
                    label=label,
                    checkpoint_name=checkpoint_name,
                    generation=generation,
                    path=str(model),
                    actor_path=str(actor),
                    games=screen.get("games", games),
                    created_at=actor.stat().st_mtime,
                    parent_id=None,
                    is_champion=str(actor) == state.get("champion"),
                    was_champion=(
                        generation is not None
                        or actor.name in {"original-champion.actor.npz", "source.actor.npz"}
                    ),
                    is_pinned=False,
                    external=True,
                    training_generation=6,
                    training_rules_version=1,
                    evaluation={},
                )
            )
    return sorted(result, key=lambda item: item["created_at"], reverse=True)
