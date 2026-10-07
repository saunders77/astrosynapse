"""Read-only prepared acquisition-impact reports and sampled positions."""

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from .acquisition_impact import position_detail, safe_folder

router = APIRouter(prefix="/api/acquisition-impact")


def root(request):
    return request.app.state.acquire_students.output_dir.parent / "miniastro_regret"


@router.get("")
def reports(request: Request):
    results = []
    for path in root(request).glob("*/analysis.json"):
        report = json.loads(path.read_text())
        results.append({k: report[k] for k in ("id", "reference", "games", "completed_at")})
    return sorted(results, key=lambda r: r["completed_at"], reverse=True)


def folder(request, run_id):
    try:
        return safe_folder(root(request), run_id)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(404, "Analysis not found") from error


@router.get("/{run_id}")
def report(run_id: str, request: Request):
    return FileResponse(folder(request, run_id) / "analysis.json", media_type="application/json")


@router.get("/{run_id}/download")
def download(run_id: str, request: Request):
    return FileResponse(
        folder(request, run_id) / "analysis.json",
        media_type="application/json",
        filename=f"{run_id}-analysis.json",
    )


@router.get("/{run_id}/positions/{game_id}/{root_id}")
def position(run_id: str, game_id: int, root_id: int, request: Request):
    try:
        if game_id < 0 or root_id < 0:
            raise ValueError("Invalid position")
        return position_detail(folder(request, run_id), game_id, root_id)
    except (ValueError, FileNotFoundError, StopIteration) as error:
        raise HTTPException(404, "Position not found") from error
