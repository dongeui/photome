"""Search endpoints."""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from app.api.deps import require_state
from app.services.search import HybridSearchService
from app.services.search.backend import SqlAlchemyHybridSearchBackend


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
) -> dict[str, Any]:
    return _search_payload(
        request,
        q=q,
        mode=mode,
        place=place,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
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
) -> dict[str, Any]:
    if not q.strip():
        return {"items": [], "total": 0, "query": q, "meta": {"effective_mode": mode, "intent_reason": "empty"}}

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
        )
    return {"items": items, "total": len(items), "query": q, "meta": meta}


@router.get("/search-ui", response_class=HTMLResponse)
async def search_ui(request: Request, q: str = Query(default=""), mode: str = Query(default="hybrid")) -> HTMLResponse:
    payload = _search_payload(request, q=q, mode=mode)
    cards = "\n".join(_render_result_card(item) for item in payload["items"])
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>photome search</title>
  <style>
    body {{ margin: 0; font-family: Inter, system-ui, sans-serif; background: #f7f4ee; color: #142028; }}
    main {{ width: min(1120px, calc(100vw - 28px)); margin: 0 auto; padding: 24px 0 48px; }}
    header {{ display: flex; justify-content: space-between; gap: 16px; align-items: center; margin-bottom: 18px; }}
    h1 {{ margin: 0; font-size: 2rem; }}
    a {{ color: inherit; }}
    form {{ display: grid; grid-template-columns: 1fr 150px 96px; gap: 8px; margin-bottom: 18px; }}
    input, select, button {{ min-height: 42px; border: 1px solid rgba(20,32,40,.16); border-radius: 8px; padding: 0 12px; font: inherit; background: white; }}
    button {{ background: #142028; color: white; cursor: pointer; }}
    .meta {{ margin-bottom: 14px; color: #60707b; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }}
    .card {{ background: white; border: 1px solid rgba(20,32,40,.1); border-radius: 8px; padding: 12px; }}
    .name {{ font-weight: 700; word-break: break-word; }}
    .small {{ color: #60707b; font-size: .88rem; margin-top: 6px; }}
    .score {{ margin-top: 8px; font-size: .86rem; }}
    @media (max-width: 640px) {{ form {{ grid-template-columns: 1fr; }} header {{ display: block; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>photome search</h1>
      <nav><a href="/gallery">Gallery</a> · <a href="/dashboard">Dashboard</a></nav>
    </header>
    <form method="get" action="/search-ui">
      <input name="q" value="{_escape(q)}" placeholder="Search filename, OCR text, place, person">
      <select name="mode">
        <option value="hybrid" {"selected" if mode == "hybrid" else ""}>Hybrid</option>
        <option value="ocr" {"selected" if mode == "ocr" else ""}>OCR</option>
        <option value="semantic" {"selected" if mode == "semantic" else ""}>Semantic</option>
      </select>
      <button type="submit">Search</button>
    </form>
    <div class="meta">{payload["total"]} result(s), mode {payload["meta"]["effective_mode"]} / {payload["meta"]["intent_reason"]}</div>
    <section class="grid">{cards}</section>
  </main>
</body>
</html>"""
    return HTMLResponse(html)


def _render_result_card(item: dict[str, Any]) -> str:
    text = str(item.get("ocr_text") or "").replace("\n", " ")
    if len(text) > 140:
        text = text[:137] + "..."
    return f"""
      <article class="card">
        <div class="name">{_escape(str(item.get("filename") or item.get("file_id")))}</div>
        <div class="small">{_escape(str(item.get("relative_path") or ""))}</div>
        <div class="score">{_escape(str(item.get("match_explanation") or item.get("match_reason")))} · score {round(float(item.get("rank_score") or 0) * 100)}</div>
        <div class="small">{_escape(text)}</div>
      </article>
    """


def _start_of_day(value: Optional[date]) -> Optional[datetime]:
    return None if value is None else datetime.combine(value, time.min)


def _end_of_day(value: Optional[date]) -> Optional[datetime]:
    return None if value is None else datetime.combine(value, time.max)


def _escape(value: str) -> str:
    import html

    return html.escape(value, quote=True)
