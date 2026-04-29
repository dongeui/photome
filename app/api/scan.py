"""Scan trigger endpoints."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Query, Request

from app.api.deps import require_state


router = APIRouter(tags=["scan"])


@router.post("/scan")
async def trigger_scan(
    request: Request,
    full_scan: bool = Query(default=False),
) -> dict[str, Any]:
    pipeline = require_state(request, "pipeline")
    summary = pipeline.submit_scan_job(full_scan=full_scan, run_now=True, trigger="api")
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
