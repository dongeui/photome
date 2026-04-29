"""Scan trigger endpoints."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request, status

from app.api.deps import require_state


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
    summary = pipeline.submit_scan_job(
        full_scan=full_scan,
        run_now=True,
        trigger="api",
        source_roots=requested_roots,
    )
    return {"job": asdict(summary)}


@router.post("/scan/semantic-backfill")
async def trigger_semantic_backfill(
    request: Request,
    batch_size: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    """Generate CLIP embeddings for any media that missed the semantic pass."""
    pipeline = require_state(request, "pipeline")
    result = pipeline.run_semantic_backfill(batch_size=batch_size)
    return result


@router.post("/scan/semantic-maintenance")
async def trigger_semantic_maintenance(
    request: Request,
    batch_size: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    """Refresh Phase 2 search documents for only stale or missing rows."""
    pipeline = require_state(request, "pipeline")
    return pipeline.run_semantic_maintenance(batch_size=batch_size)


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
