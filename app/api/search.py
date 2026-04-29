"""Search endpoints."""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, Optional

from fastapi import APIRouter, Query, Request

from app.api.deps import require_state
from app.services.search import HybridSearchService
from app.services.search.backend import SqlAlchemyHybridSearchBackend
from app.services.search.benchmark import run_benchmark_suite


router = APIRouter(tags=["search"])


@router.get("/search")
async def search_media(
    request: Request,
    q: str = Query(default=""),
    mode: str = Query(default="hybrid"),
    place: Optional[str] = Query(default=None),
    date_from: Optional[date] = Query(default=None),
    date_to: Optional[date] = Query(default=None),
    limit: int = Query(default=40, ge=1, le=100),
    debug: bool = Query(default=False),
    w_ocr: Optional[float] = Query(default=None, ge=0),
    w_clip: Optional[float] = Query(default=None, ge=0),
    w_shadow: Optional[float] = Query(default=None, ge=0),
) -> dict[str, Any]:
    return _search_payload(
        request,
        q=q,
        mode=mode,
        place=place,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        debug=debug,
        weight_overrides=_weight_overrides(w_ocr=w_ocr, w_clip=w_clip, w_shadow=w_shadow),
    )


@router.get("/search/debug")
async def search_media_debug(
    request: Request,
    q: str = Query(default=""),
    mode: str = Query(default="hybrid"),
    place: Optional[str] = Query(default=None),
    date_from: Optional[date] = Query(default=None),
    date_to: Optional[date] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    w_ocr: Optional[float] = Query(default=None, ge=0),
    w_clip: Optional[float] = Query(default=None, ge=0),
    w_shadow: Optional[float] = Query(default=None, ge=0),
) -> dict[str, Any]:
    return _search_payload(
        request,
        q=q,
        mode=mode,
        place=place,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        debug=True,
        weight_overrides=_weight_overrides(w_ocr=w_ocr, w_clip=w_clip, w_shadow=w_shadow),
    )


@router.get("/search/benchmark")
async def search_benchmark(
    request: Request,
    limit: int = Query(default=10, ge=1, le=50),
    w_ocr: Optional[float] = Query(default=None, ge=0),
    w_clip: Optional[float] = Query(default=None, ge=0),
    w_shadow: Optional[float] = Query(default=None, ge=0),
) -> dict[str, Any]:
    database = require_state(request, "database")
    settings = require_state(request, "settings")
    with database.session_factory() as session:
        backend = SqlAlchemyHybridSearchBackend(session, embeddings_root=settings.embeddings_root)
        service = HybridSearchService(backend)
        return run_benchmark_suite(
            service,
            limit=limit,
            weight_overrides=_weight_overrides(w_ocr=w_ocr, w_clip=w_clip, w_shadow=w_shadow),
        )


def _search_payload(
    request: Request,
    *,
    q: str,
    mode: str,
    place: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    limit: int = 40,
    debug: bool = False,
    weight_overrides: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    if not q.strip():
        return {
            "items": [],
            "total": 0,
            "query": q,
            "meta": {
                "effective_mode": mode,
                "intent_reason": "empty",
                "weight_overrides": weight_overrides or {},
            },
        }

    database = require_state(request, "database")
    settings = require_state(request, "settings")
    with database.session_factory() as session:
        backend = SqlAlchemyHybridSearchBackend(session, embeddings_root=settings.embeddings_root)
        service = HybridSearchService(backend)
        items, meta = service.search_with_meta(
            q,
            limit=limit,
            place_filter=place,
            date_from=_start_of_day(date_from),
            date_to=_end_of_day(date_to),
            mode=mode,
            debug=debug,
            weight_overrides=weight_overrides,
        )
    return {"items": items, "total": len(items), "query": q, "meta": meta}


def _start_of_day(value: Optional[date]) -> Optional[datetime]:
    return None if value is None else datetime.combine(value, time.min)


def _end_of_day(value: Optional[date]) -> Optional[datetime]:
    return None if value is None else datetime.combine(value, time.max)


def _weight_overrides(
    *,
    w_ocr: Optional[float],
    w_clip: Optional[float],
    w_shadow: Optional[float],
) -> dict[str, float]:
    overrides: dict[str, float] = {}
    if w_ocr is not None:
        overrides["ocr"] = w_ocr
    if w_clip is not None:
        overrides["clip"] = w_clip
    if w_shadow is not None:
        overrides["shadow"] = w_shadow
    return overrides
