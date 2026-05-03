"""Scan trigger endpoints."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, status
from sqlalchemy.exc import OperationalError

from app.api.deps import require_state
from app.models.job import ProcessingJob
from app.services.processing.pipeline import LibraryJobBusyError


router = APIRouter(tags=["scan"])


@router.post("/scan")
async def trigger_scan(
    request: Request,
    full_scan: bool = Query(default=False),
    source_root: Optional[List[str]] = Query(default=None),
    source_roots: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    pipeline = require_state(request, "pipeline")
    requested_roots = _parse_source_roots(source_root=source_root, source_roots=source_roots)
    try:
        summary = pipeline.submit_scan_job(
            full_scan=full_scan,
            run_now=True,
            trigger="api",
            source_roots=requested_roots,
        )
    except LibraryJobBusyError as exc:
        _raise_active_job_conflict(exc)
    return {"job": asdict(summary)}


@router.post("/scan/async", status_code=status.HTTP_202_ACCEPTED)
async def trigger_scan_async(
    request: Request,
    background_tasks: BackgroundTasks,
    full_scan: bool = Query(default=False),
    source_root: Optional[List[str]] = Query(default=None),
    source_roots: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    pipeline = require_state(request, "pipeline")
    requested_roots = _parse_source_roots(source_root=source_root, source_roots=source_roots)
    try:
        summary = pipeline.submit_scan_job(
            full_scan=full_scan,
            run_now=False,
            trigger="api-async",
            source_roots=requested_roots,
        )
    except LibraryJobBusyError as exc:
        _raise_active_job_conflict(exc)
    except OperationalError as exc:
        _raise_job_submission_busy(exc)
    background_tasks.add_task(pipeline.run_scan_job, summary.job_id)
    return {"job": asdict(summary)}


@router.get("/scan/jobs/{job_id}")
async def read_scan_job(request: Request, job_id: str) -> dict[str, Any]:
    database = require_state(request, "database")
    with database.session_factory() as session:
        job = session.get(ProcessingJob, job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="scan job not found")
        return {
            "job": {
                "job_id": job.id,
                "job_kind": job.job_kind,
                "status": job.status,
                "payload": job.payload_json,
                "result": job.result_json,
                "error_stage": job.error_stage,
                "error_message": job.error_message,
                "attempts": job.attempts,
                "enqueued_at": job.enqueued_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "updated_at": job.updated_at,
            }
        }


@router.post("/scan/semantic-backfill")
async def trigger_semantic_backfill(
    request: Request,
    batch_size: int = Query(default=50, ge=1, le=10000),
) -> dict[str, Any]:
    """Generate CLIP embeddings for any media that missed the semantic pass."""
    pipeline = require_state(request, "pipeline")
    try:
        result = pipeline.run_semantic_backfill(batch_size=batch_size)
    except LibraryJobBusyError as exc:
        _raise_active_job_conflict(exc)
    return result


@router.post("/scan/semantic-backfill/async", status_code=status.HTTP_202_ACCEPTED)
async def trigger_semantic_backfill_async(
    request: Request,
    background_tasks: BackgroundTasks,
    batch_size: int = Query(default=50, ge=1, le=10000),
) -> dict[str, Any]:
    pipeline = require_state(request, "pipeline")
    try:
        summary = pipeline.submit_semantic_backfill_job(
            batch_size=batch_size,
            run_now=False,
            trigger="api-async",
        )
    except LibraryJobBusyError as exc:
        _raise_active_job_conflict(exc)
    except OperationalError as exc:
        _raise_job_submission_busy(exc)
    background_tasks.add_task(pipeline.run_semantic_job, summary.job_id)
    return {"job": asdict(summary)}


@router.post("/scan/semantic-maintenance")
async def trigger_semantic_maintenance(
    request: Request,
    batch_size: int = Query(default=100, ge=1, le=10000),
) -> dict[str, Any]:
    """Refresh Phase 2 search documents for only stale or missing rows."""
    pipeline = require_state(request, "pipeline")
    try:
        return pipeline.run_semantic_maintenance(batch_size=batch_size)
    except LibraryJobBusyError as exc:
        _raise_active_job_conflict(exc)


@router.post("/scan/semantic-maintenance/async", status_code=status.HTTP_202_ACCEPTED)
async def trigger_semantic_maintenance_async(
    request: Request,
    background_tasks: BackgroundTasks,
    batch_size: int = Query(default=100, ge=1, le=10000),
) -> dict[str, Any]:
    pipeline = require_state(request, "pipeline")
    try:
        summary = pipeline.submit_semantic_maintenance_job(
            batch_size=batch_size,
            run_now=False,
            trigger="api-async",
        )
    except LibraryJobBusyError as exc:
        _raise_active_job_conflict(exc)
    except OperationalError as exc:
        _raise_job_submission_busy(exc)
    background_tasks.add_task(pipeline.run_semantic_job, summary.job_id)
    return {"job": asdict(summary)}


def _parse_source_roots(*, source_root: Optional[List[str]], source_roots: Optional[str]) -> tuple[Path, ...] | None:
    raw_values: list[str] = []
    if source_root:
        raw_values.extend(source_root)
    if source_roots:
        raw_values.extend(source_roots.replace("\n", ",").split(","))

    resolved_roots: list[Path] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        cleaned = raw_value.strip()
        if not cleaned:
            continue
        path = Path(cleaned).expanduser().resolve()
        key = str(path)
        if key in seen:
            continue
        if not path.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Source root does not exist: {path}",
            )
        if not path.is_dir():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Source root is not a directory: {path}",
            )
        seen.add(key)
        resolved_roots.append(path)

    return tuple(resolved_roots) if resolved_roots else None


def _raise_job_submission_busy(exc: OperationalError) -> None:
    message = str(getattr(exc, "orig", exc)).lower()
    if "database is locked" in message:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another library job is still writing to the catalog. Wait a moment and try again.",
        ) from exc
    raise exc


def _raise_active_job_conflict(exc: LibraryJobBusyError) -> None:
    active = exc.active_job
    kind = str(active.get("job_kind") or "job")
    status_name = str(active.get("status") or "queued")
    if kind == "scan":
        detail = "Phase 1 scan is already active. Wait for it to finish before starting Phase 2."
    elif kind in {"semantic_backfill", "semantic_maintenance"}:
        detail = "Phase 2 semantic work is already active. Wait for it to finish before starting Phase 1."
    else:
        detail = f"Another library job is active ({kind}/{status_name}). Wait for it to finish."
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=detail,
    ) from exc
