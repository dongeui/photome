"""Server-rendered gallery views backed by the media catalog."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time
from html import escape
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy import Select, exists, false, func, or_, select

from app.api.deps import require_state
from app.models.annotation import MediaAnnotation
from app.models.asset import DerivedAsset
from app.models.media import MediaFile
from app.models.tag import Tag
from app.services.search import HybridSearchService
from app.services.search.backend import SqlAlchemyHybridSearchBackend


router = APIRouter(tags=["gallery"])

PERSON_TAG_TYPES = ("person", "people", "face")
PLACE_TAG_TYPES = ("place", "location")
PAGE_SIZE = 48
GALLERY_SEARCH_LIMIT = 500


@router.get("/", response_class=HTMLResponse)
async def home_page(
    request: Request,
    media_type: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    person: Optional[str] = Query(default=None),
    place: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None),
    page: int = Query(default=1, ge=1),
) -> HTMLResponse:
    return await gallery_page(
        request,
        media_type=media_type,
        date_from=date_from,
        date_to=date_to,
        person=person,
        place=place,
        q=q,
        page=page,
    )


@router.get("/gallery", response_class=HTMLResponse)
async def gallery_page(
    request: Request,
    media_type: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    person: Optional[str] = Query(default=None),
    place: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None),
    page: int = Query(default=1, ge=1),
) -> HTMLResponse:
    database = require_state(request, "database")
    offset = (page - 1) * PAGE_SIZE
    parsed_date_from = _parse_date(date_from)
    parsed_date_to = _parse_date(date_to)
    search_meta: dict[str, str] | None = None
    ranked_ids: list[str] | None = None

    with database.session_factory() as session:
        if q and q.strip():
            settings = require_state(request, "settings")
            backend = SqlAlchemyHybridSearchBackend(session, embeddings_root=settings.embeddings_root)
            service = HybridSearchService(backend)
            search_results, search_meta = service.search_with_meta(
                q,
                limit=GALLERY_SEARCH_LIMIT,
                place_filter=place,
                date_from=_start_of_day(parsed_date_from),
                date_to=_end_of_day(parsed_date_to),
                mode="hybrid",
            )
            ranked_ids = [str(item["file_id"]) for item in search_results]

        ids_query = _build_gallery_ids_query(
            media_type=media_type,
            date_from=_start_of_day(parsed_date_from),
            date_to=_end_of_day(parsed_date_to),
            person=person,
            place=place,
            query=None if ranked_ids is not None else q,
            file_ids=ranked_ids,
        )
        if ranked_ids is not None:
            rank_index = {file_id: index for index, file_id in enumerate(ranked_ids)}
            matched_ids = list(session.scalars(ids_query))
            matched_ids.sort(key=lambda file_id: rank_index.get(file_id, len(rank_index)))
            total = len(matched_ids)
            file_ids = matched_ids[offset:offset + PAGE_SIZE]
        else:
            total = int(session.scalar(select(func.count()).select_from(ids_query.subquery())) or 0)
            file_ids = list(session.scalars(ids_query.limit(PAGE_SIZE).offset(offset)))

        items: list[MediaFile] = []
        annotation_map: dict[str, MediaAnnotation] = {}
        asset_map: dict[str, list[DerivedAsset]] = defaultdict(list)
        tag_map: dict[str, list[Tag]] = defaultdict(list)
        if file_ids:
            items = list(
                session.scalars(
                    select(MediaFile)
                    .where(MediaFile.file_id.in_(file_ids))
                    .order_by(MediaFile.last_seen_at.desc(), MediaFile.file_id.desc())
                )
            )
            if ranked_ids is not None:
                page_rank = {file_id: index for index, file_id in enumerate(file_ids)}
                items.sort(key=lambda item: page_rank.get(item.file_id, len(page_rank)))
            for asset in session.scalars(
                select(DerivedAsset)
                .where(DerivedAsset.file_id.in_(file_ids))
                .order_by(DerivedAsset.file_id.asc(), DerivedAsset.created_at.asc(), DerivedAsset.id.asc())
            ):
                asset_map[asset.file_id].append(asset)
            for tag in session.scalars(
                select(Tag)
                .where(Tag.file_id.in_(file_ids))
                .order_by(Tag.tag_type.asc(), Tag.tag_value.asc())
            ):
                tag_map[tag.file_id].append(tag)
            for annotation in session.scalars(
                select(MediaAnnotation).where(MediaAnnotation.file_id.in_(file_ids))
            ):
                annotation_map[annotation.file_id] = annotation

        person_options = _list_tag_values(session, PERSON_TAG_TYPES)
        place_options = _list_tag_values(session, PLACE_TAG_TYPES)

    current_url = request.url.path
    if request.url.query:
        current_url += f"?{request.url.query}"
    cards = [
        _render_card(
            media_file=item,
            asset=_select_card_asset(asset_map.get(item.file_id, [])),
            tags=tag_map.get(item.file_id, []),
            annotation=annotation_map.get(item.file_id),
            index=index,
            next_url=f"{current_url}#card-{item.file_id}",
        )
        for index, item in enumerate(items)
    ]

    page_count = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    has_prev = page > 1
    has_next = offset + len(items) < total
    person_available = bool(person_options)
    place_available = bool(place_options)
    active_filter_summary = _render_active_filter_summary(
        q=q,
        search_meta=search_meta,
        media_type=media_type,
        date_from=date_from,
        date_to=date_to,
        person=person,
        place=place,
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>photome gallery</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f2ea;
      --paper: rgba(255, 252, 247, 0.78);
      --panel: rgba(255, 255, 255, 0.88);
      --panel-strong: rgba(255, 255, 255, 0.96);
      --text: #142028;
      --muted: #60707b;
      --line: rgba(20, 32, 40, 0.1);
      --line-strong: rgba(20, 32, 40, 0.16);
      --accent: #d05d33;
      --accent-deep: #8a3a22;
      --accent-soft: rgba(208, 93, 51, 0.12);
      --shadow: 0 18px 50px rgba(20, 32, 40, 0.08);
      --shadow-tight: 0 10px 28px rgba(20, 32, 40, 0.06);
      --hero-gradient: linear-gradient(135deg, rgba(255, 255, 255, 0.94), rgba(255, 247, 238, 0.82) 52%, rgba(237, 244, 248, 0.78));
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Instrument Serif", "Iowan Old Style", "Palatino Linotype", serif;
      background:
        radial-gradient(circle at top left, rgba(208, 93, 51, 0.18), transparent 24%),
        radial-gradient(circle at top right, rgba(64, 102, 140, 0.12), transparent 26%),
        linear-gradient(180deg, #fbf8f2 0%, var(--bg) 46%, #f1ede4 100%);
      color: var(--text);
    }}
    a {{ color: inherit; }}
    .shell {{
      width: min(1360px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 20px 0 48px;
    }}
    .hero {{
      position: relative;
      isolation: isolate;
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(240px, .8fr);
      gap: 18px;
      margin-bottom: 18px;
      padding: 26px;
      border: 1px solid var(--line);
      border-radius: 30px;
      background: var(--hero-gradient);
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -8% -28% auto;
      width: 280px;
      height: 280px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(208, 93, 51, 0.18), transparent 66%);
      z-index: -1;
      filter: blur(8px);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(2.2rem, 4vw, 4.2rem);
      line-height: 0.96;
      letter-spacing: -0.03em;
    }}
    .eyebrow {{
      display: inline-flex;
      margin-bottom: 12px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(20, 32, 40, 0.06);
      color: var(--accent-deep);
      font-family: "Inter", "Helvetica Neue", sans-serif;
      font-size: 0.74rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .hero-copy p {{
      margin: 0;
      color: var(--muted);
      max-width: 58ch;
      font-family: "Inter", "Helvetica Neue", sans-serif;
      font-size: 1rem;
      line-height: 1.55;
    }}
    .hero-stats {{
      display: grid;
      gap: 10px;
      align-content: end;
    }}
    .hero-links {{
      display: flex;
      gap: 10px;
      margin-top: 18px;
      flex-wrap: wrap;
    }}
    .stat-card {{
      padding: 14px 16px;
      border: 1px solid rgba(20, 32, 40, 0.08);
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.72);
      backdrop-filter: blur(10px);
      box-shadow: var(--shadow-tight);
    }}
    .stat-label {{
      display: block;
      margin-bottom: 4px;
      color: var(--muted);
      font-family: "Inter", "Helvetica Neue", sans-serif;
      font-size: 0.76rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .stat-value {{
      display: block;
      font-family: "Inter", "Helvetica Neue", sans-serif;
      font-size: clamp(1.2rem, 2vw, 1.7rem);
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .toolbar {{
      position: sticky;
      top: 12px;
      z-index: 20;
      margin-bottom: 18px;
    }}
    form.filters {{
      display: grid;
      grid-template-columns: 1.2fr repeat(5, minmax(120px, .6fr)) auto;
      gap: 10px;
      padding: 14px;
      border: 1px solid rgba(20, 32, 40, 0.08);
      border-radius: 22px;
      background: color-mix(in srgb, var(--paper) 72%, white);
      backdrop-filter: blur(16px) saturate(140%);
      box-shadow: var(--shadow-tight);
    }}
    label {{
      display: grid;
      gap: 5px;
      font-family: "Inter", "Helvetica Neue", sans-serif;
      font-size: 0.76rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    input, select {{
      width: 100%;
      min-height: 44px;
      padding: 11px 13px;
      border: 1px solid rgba(20, 32, 40, 0.12);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.9);
      color: var(--text);
      font: 500 0.95rem "Inter", "Helvetica Neue", sans-serif;
    }}
    .actions {{
      display: flex;
      gap: 10px;
      align-items: end;
      flex-wrap: wrap;
    }}
    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 44px;
      padding: 10px 16px;
      border-radius: 999px;
      border: 1px solid transparent;
      background: linear-gradient(135deg, var(--accent), var(--accent-deep));
      color: white;
      font-family: "Inter", "Helvetica Neue", sans-serif;
      font-weight: 600;
      text-decoration: none;
      cursor: pointer;
      box-shadow: 0 10px 24px rgba(208, 93, 51, 0.22);
    }}
    .button.secondary {{
      border-color: var(--line-strong);
      background: rgba(255, 255, 255, 0.7);
      color: var(--text);
      box-shadow: none;
    }}
    .meta-bar {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin: 8px 0 18px;
      color: var(--muted);
      font-family: "Inter", "Helvetica Neue", sans-serif;
      font-size: 0.92rem;
      flex-wrap: wrap;
    }}
    .active-filters {{
      display: grid;
      gap: 8px;
      margin: 0 0 16px;
    }}
    .active-filters-title {{
      color: var(--muted);
      font-family: "Inter", "Helvetica Neue", sans-serif;
      font-size: 0.8rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .active-filters-list {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      font-family: "Inter", "Helvetica Neue", sans-serif;
    }}
    .filter-chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 10px;
      border-radius: 999px;
      border: 1px solid rgba(20, 32, 40, 0.08);
      background: rgba(255, 255, 255, 0.72);
      color: #31424c;
      font-size: 0.88rem;
    }}
    .filter-chip strong {{
      color: var(--accent-deep);
      font-weight: 700;
    }}
    .control-note {{
      margin: 0;
      color: var(--muted);
      font-family: "Inter", "Helvetica Neue", sans-serif;
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0;
      text-transform: none;
    }}
    .control-unavailable input {{
      background: rgba(20, 32, 40, 0.04);
      color: rgba(20, 32, 40, 0.45);
      cursor: not-allowed;
    }}
    .meta-pillset {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .meta-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 10px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.65);
      border: 1px solid rgba(20, 32, 40, 0.08);
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 18px;
    }}
    .card {{
      display: flex;
      flex-direction: column;
      min-height: 100%;
      overflow: hidden;
      border-radius: 24px;
      border: 1px solid rgba(20, 32, 40, 0.08);
      background: var(--panel-strong);
      box-shadow: var(--shadow-tight);
      transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
      content-visibility: auto;
      contain-intrinsic-size: 360px;
    }}
    .card:hover {{
      transform: translateY(-2px);
      box-shadow: 0 18px 34px rgba(20, 32, 40, 0.1);
      border-color: rgba(20, 32, 40, 0.14);
    }}
    .thumb {{
      position: relative;
      display: block;
      aspect-ratio: 4 / 5;
      background:
        linear-gradient(180deg, rgba(20,32,40,0.02), rgba(20,32,40,0.12)),
        linear-gradient(135deg, rgba(20,32,40,0.12), rgba(20,32,40,0.04));
      overflow: hidden;
      cursor: zoom-in;
    }}
    .thumb::after {{
      content: "";
      position: absolute;
      inset: auto 0 0;
      height: 34%;
      background: linear-gradient(180deg, transparent, rgba(20, 32, 40, 0.16));
      pointer-events: none;
    }}
    .thumb img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
      transform: scale(1.01);
    }}
    .placeholder {{
      display: grid;
      place-items: center;
      height: 100%;
      padding: 20px;
      color: var(--muted);
      font-size: 0.95rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .body {{
      display: grid;
      gap: 9px;
      padding: 14px 15px 16px;
    }}
    .row {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: start;
    }}
    .filename {{
      margin: 0;
      font-size: 1.02rem;
      line-height: 1.2;
      letter-spacing: -0.02em;
      word-break: break-word;
    }}
    .kind {{
      white-space: nowrap;
      padding: 5px 8px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-family: "Inter", "Helvetica Neue", sans-serif;
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .summary, .detail, .tags, .pathline {{
      margin: 0;
      color: var(--muted);
      font-family: "Inter", "Helvetica Neue", sans-serif;
      font-size: 0.88rem;
      line-height: 1.42;
    }}
    .detail {{
      font-size: 0.8rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .summary {{
      color: #2e3c45;
    }}
    .pathline {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 0.8rem;
    }}
    .tags {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }}
    .tag {{
      padding: 4px 8px;
      border-radius: 999px;
      background: rgba(24, 32, 38, 0.07);
      border: 1px solid rgba(20, 32, 40, 0.06);
    }}
    .edit-panel {{
      border-top: 1px solid rgba(20, 32, 40, 0.08);
      padding-top: 8px;
      font-family: "Inter", "Helvetica Neue", sans-serif;
    }}
    .edit-panel summary {{
      color: var(--accent-deep);
      cursor: pointer;
      font-size: 0.82rem;
      font-weight: 700;
    }}
    .edit-form {{
      display: grid;
      gap: 8px;
      margin-top: 8px;
    }}
    .edit-form input, .edit-form textarea {{
      width: 100%;
      min-height: 36px;
      padding: 8px 10px;
      border: 1px solid rgba(20, 32, 40, 0.12);
      border-radius: 10px;
      background: rgba(255, 255, 255, 0.92);
      color: var(--text);
      font: 500 0.84rem "Inter", "Helvetica Neue", sans-serif;
    }}
    .edit-form textarea {{
      min-height: 66px;
      resize: vertical;
    }}
    .edit-form button {{
      justify-self: start;
      min-height: 34px;
      padding: 7px 12px;
      border: 0;
      border-radius: 999px;
      background: var(--text);
      color: white;
      font: 700 0.82rem "Inter", "Helvetica Neue", sans-serif;
      cursor: pointer;
    }}
    .pagination {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-top: 22px;
      align-items: center;
      flex-wrap: wrap;
      font-family: "Inter", "Helvetica Neue", sans-serif;
    }}
    .lightbox {{
      position: fixed;
      inset: 0;
      z-index: 100;
      display: none;
      place-items: center;
      padding: 24px;
      background: rgba(10, 15, 18, 0.78);
      backdrop-filter: blur(14px);
    }}
    .lightbox:target {{
      display: grid;
    }}
    .lightbox-backdrop {{
      position: absolute;
      inset: 0;
      cursor: zoom-out;
    }}
    .lightbox-panel {{
      position: relative;
      z-index: 1;
      display: grid;
      gap: 10px;
      max-width: min(92vw, 1120px);
      max-height: 92vh;
    }}
    .lightbox img {{
      display: block;
      max-width: 100%;
      max-height: calc(92vh - 58px);
      object-fit: contain;
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.08);
      box-shadow: 0 24px 70px rgba(0, 0, 0, 0.35);
    }}
    .lightbox-caption {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      color: white;
      font-family: "Inter", "Helvetica Neue", sans-serif;
      font-size: 0.9rem;
    }}
    .lightbox-close {{
      flex: 0 0 auto;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.14);
      color: white;
      text-decoration: none;
    }}
    @media (max-width: 1100px) {{
      .hero {{ grid-template-columns: 1fr; }}
      form.filters {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    }}
    @media (max-width: 720px) {{
      .shell {{ width: min(100vw - 18px, 1360px); padding-top: 14px; }}
      .hero {{ padding: 22px; border-radius: 24px; }}
      .toolbar {{ top: 8px; }}
      form.filters {{ grid-template-columns: 1fr 1fr; padding: 12px; }}
      .gallery {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
      .card {{ border-radius: 18px; }}
      .thumb {{ aspect-ratio: 1 / 1.15; }}
    }}
    @media (max-width: 540px) {{
      form.filters {{ grid-template-columns: 1fr; }}
      .gallery {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 420px) {{
      .gallery {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="hero-copy">
        <span class="eyebrow">Thumbnail-first browser</span>
        <h1>Fast paths into your photo archive.</h1>
        <p>Server-rendered, derived-thumbnail gallery tuned for quick scanning. The first screen prioritizes visible cards, the rest stays lightweight until needed.</p>
        <div class="hero-links">
          <a class="button secondary" href="/dashboard">Service Dashboard</a>
        </div>
      </div>
      <div class="hero-stats">
        <div class="stat-card">
          <span class="stat-label">Visible Catalog</span>
          <span class="stat-value">{total} items</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">Current View</span>
          <span class="stat-value">{len(items)} cards</span>
        </div>
      </div>
    </section>
    <div class="toolbar">
      <form class="filters" method="get" action="/gallery">
        <label>
          Search
          <input type="search" name="q" value="{escape(q or '')}" placeholder="face, baby, receipt, 어르굴, filename">
        </label>
        <label>
          Media Type
          <select name="media_type">
            {_render_media_type_options(media_type)}
          </select>
        </label>
        <label>
          Date From
          <input type="date" name="date_from" value="{escape(date_from or '')}">
        </label>
        <label>
          Date To
          <input type="date" name="date_to" value="{escape(date_to or '')}">
        </label>
        <label class="{'control-unavailable' if not person_available else ''}">
          Person
          <input type="text" name="person" value="{escape(person or '')}" list="person-options" placeholder="No person tags indexed yet"{" disabled" if not person_available else ""}>
          <span class="control-note">{'Person tag filter is unavailable in this catalog.' if not person_available else 'Matches person-style tags when indexed.'}</span>
        </label>
        <label class="{'control-unavailable' if not place_available else ''}">
          Place
          <input type="text" name="place" value="{escape(place or '')}" list="place-options" placeholder="No place tags indexed yet"{" disabled" if not place_available else ""}>
          <span class="control-note">{'Place tag filter is unavailable in this catalog.' if not place_available else 'Matches place/location tags when indexed.'}</span>
        </label>
        <div class="actions">
          <button class="button" type="submit">Apply</button>
          <a class="button secondary" href="/gallery">Reset</a>
        </div>
        <datalist id="person-options">{_render_datalist_options(person_options)}</datalist>
        <datalist id="place-options">{_render_datalist_options(place_options)}</datalist>
      </form>
    </div>
    <section class="active-filters">
      <span class="active-filters-title">Active filters</span>
      <div class="active-filters-list">{active_filter_summary}</div>
    </section>
    <div class="meta-bar">
      <div class="meta-pillset">
        <span class="meta-pill">{total} items{_render_filter_hint(person, place)}</span>
        {_render_search_mode_pill(search_meta)}
        <span class="meta-pill">Page {page} of {page_count}</span>
      </div>
      <span>{'Showing ' + str(offset + 1) + '–' + str(offset + len(items)) if items else 'Showing 0 items'}</span>
    </div>
    <section id="gallery" class="gallery">
      {''.join(cards) if cards else '<article class="card"><div class="body"><p class="summary">No media matched the current filters.</p></div></article>'}
    </section>
    <nav class="pagination">
      <span>Use filters to narrow the catalog without leaving the page.</span>
      <div class="actions">
        {_render_page_link('Previous', _page_url(request, page - 1), enabled=has_prev)}
        {_render_page_link('Next', _page_url(request, page + 1), enabled=has_next)}
      </div>
    </nav>
  </main>
</body>
</html>"""
    return HTMLResponse(html)


@router.get("/gallery/assets/{asset_id}")
async def gallery_asset(request: Request, asset_id: int) -> FileResponse:
    database = require_state(request, "database")
    settings = require_state(request, "settings")
    with database.session_factory() as session:
        asset = session.get(DerivedAsset, asset_id)
        if asset is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="derived asset not found")

    path = _resolve_asset_path(settings.derived_root, asset.derived_path)
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="derived asset file missing")
    return FileResponse(path, media_type=asset.content_type or "image/jpeg")


def _build_gallery_ids_query(
    *,
    media_type: str | None,
    date_from: datetime | None,
    date_to: datetime | None,
    person: str | None,
    place: str | None,
    query: str | None,
    file_ids: list[str] | None = None,
) -> Select:
    statement = select(MediaFile.file_id).where(
        MediaFile.status.not_in(("missing", "replaced"))
    )
    if file_ids is not None:
        if not file_ids:
            return statement.where(false())
        statement = statement.where(MediaFile.file_id.in_(file_ids))

    if media_type:
        statement = statement.where(MediaFile.media_kind == media_type)
    if query:
        like_query = f"%{query}%"
        statement = statement.where(
            or_(
                MediaFile.current_path.ilike(like_query),
                MediaFile.relative_path.ilike(like_query),
                MediaFile.filename.ilike(like_query),
                MediaFile.file_id.ilike(like_query),
            )
        )

    captured_at_expr = func.coalesce(MediaFile.exif_datetime, MediaFile.processed_at, MediaFile.last_seen_at)
    if date_from is not None:
        statement = statement.where(captured_at_expr >= date_from)
    if date_to is not None:
        statement = statement.where(captured_at_expr <= date_to)
    if person:
        statement = statement.where(_tag_exists_clause(person, PERSON_TAG_TYPES))
    if place:
        statement = statement.where(_tag_exists_clause(place, PLACE_TAG_TYPES))

    return statement.order_by(MediaFile.last_seen_at.desc(), MediaFile.file_id.desc())


def _tag_exists_clause(tag_value: str, tag_types: tuple[str, ...]):
    normalized = tag_value.strip()
    return exists(
        select(Tag.id).where(
            Tag.file_id == MediaFile.file_id,
            Tag.tag_type.in_(tag_types),
            func.lower(Tag.tag_value) == normalized.lower(),
        )
    )


def _list_tag_values(session, tag_types: tuple[str, ...]) -> list[str]:
    statement = (
        select(Tag.tag_value)
        .where(Tag.tag_type.in_(tag_types))
        .group_by(Tag.tag_value)
        .order_by(func.lower(Tag.tag_value).asc())
        .limit(200)
    )
    return [value for value in session.scalars(statement)]


def _select_card_asset(assets: list[DerivedAsset]) -> DerivedAsset | None:
    preferred_order = {"thumb": 0, "keyframe": 1, "preview": 2}
    if not assets:
        return None
    return min(
        assets,
        key=lambda asset: (preferred_order.get(asset.asset_kind, 99), asset.id),
    )


def _render_card(
    *,
    media_file: MediaFile,
    asset: DerivedAsset | None,
    tags: list[Tag],
    annotation: MediaAnnotation | None,
    index: int,
    next_url: str,
) -> str:
    eager = index < 6
    loading_attr = "eager" if eager else "lazy"
    fetchpriority_attr = "high" if index < 4 else "auto"
    image_html = (
        f'<img src="/gallery/assets/{asset.id}" alt="{escape(media_file.filename)}" '
        f'loading="{loading_attr}" decoding="async" fetchpriority="{fetchpriority_attr}" '
        f'sizes="(max-width: 420px) 100vw, (max-width: 720px) 50vw, (max-width: 1100px) 33vw, 24vw" '
        f'width="{media_file.width or 512}" height="{media_file.height or 640}">'
        if asset is not None
        else f'<div class="placeholder">{escape(media_file.media_kind)}</div>'
    )
    title = _display_title(media_file, annotation)
    description = _display_description(media_file, tags, annotation)
    custom_tags = ", ".join(tag.tag_value for tag in tags if tag.tag_type == "custom")
    tag_html = "".join(
        f'<span class="tag">{escape(tag.tag_type)}: {escape(tag.tag_value)}</span>'
        for tag in tags[:4]
    )
    preview_id = f"preview-{asset.id}" if asset is not None else ""
    thumb_href = f"#{preview_id}" if asset is not None else "#gallery"
    lightbox_html = (
        f"""
      <div id="{preview_id}" class="lightbox" aria-label="{escape(title)} preview">
        <a class="lightbox-backdrop" href="#gallery" aria-label="Close preview"></a>
        <div class="lightbox-panel">
          <img src="/gallery/assets/{asset.id}" alt="{escape(title)} enlarged preview">
          <div class="lightbox-caption">
            <span>{escape(title)}</span>
            <a class="lightbox-close" href="#gallery">Close</a>
          </div>
        </div>
      </div>
        """
        if asset is not None
        else ""
    )
    return f"""
      <article id="card-{escape(media_file.file_id)}" class="card">
        <a class="thumb" href="{thumb_href}" aria-label="Open {escape(title)} preview">{image_html}</a>
        <div class="body">
          <div class="row">
            <h2 class="filename">{escape(title)}</h2>
            <span class="kind">{escape(media_file.media_kind)}</span>
          </div>
          <p class="detail">{escape(_display_date(media_file))}</p>
          <p class="summary">{escape(description)}</p>
          <p class="pathline">{escape(media_file.relative_path)}</p>
          {f'<p class="tags">{tag_html}</p>' if tag_html else ''}
          <details class="edit-panel">
            <summary>Edit name, description, tags</summary>
            <form class="edit-form" method="post" action="/media/{escape(media_file.file_id)}/annotation">
              <input name="title" value="{escape(annotation.title if annotation and annotation.title else '')}" placeholder="Display name">
              <textarea name="description" placeholder="Description">{escape(annotation.description if annotation and annotation.description else '')}</textarea>
              <input name="tags" value="{escape(custom_tags)}" placeholder="Tags, comma separated">
              <input type="hidden" name="next" value="{escape(next_url)}">
              <button type="submit">Save</button>
            </form>
          </details>
        </div>
      </article>
      {lightbox_html}
    """


def _display_title(media_file: MediaFile, annotation: MediaAnnotation | None) -> str:
    if annotation and annotation.title:
        return annotation.title
    return media_file.filename


def _display_description(media_file: MediaFile, tags: list[Tag], annotation: MediaAnnotation | None) -> str:
    if annotation and annotation.description:
        return annotation.description
    return _summary_text(media_file, tags)


def _summary_text(media_file: MediaFile, tags: list[Tag]) -> str:
    parts: list[str] = []
    if media_file.width and media_file.height:
        parts.append(f"{media_file.width}x{media_file.height}")
    if media_file.duration_seconds:
        minutes, seconds = divmod(int(round(media_file.duration_seconds)), 60)
        parts.append(f"{minutes}:{seconds:02d}")
    if media_file.mime_type:
        parts.append(media_file.mime_type)
    elif media_file.status:
        parts.append(media_file.status)
    if tags:
        parts.append(", ".join(f"{tag.tag_type}:{tag.tag_value}" for tag in tags[:2]))
    return " · ".join(parts) if parts else media_file.current_path


def _display_date(media_file: MediaFile) -> str:
    for value in (media_file.exif_datetime, media_file.processed_at, media_file.last_seen_at):
        if value is not None:
            return value.strftime("%Y-%m-%d %H:%M")
    return "date unknown"


def _render_media_type_options(selected: str | None) -> str:
    options = [("", "All"), ("image", "Image"), ("video", "Video")]
    rendered: list[str] = []
    for value, label in options:
        is_selected = ' selected' if selected == value or (selected is None and value == "") else ""
        rendered.append(f'<option value="{escape(value)}"{is_selected}>{escape(label)}</option>')
    return "".join(rendered)


def _render_datalist_options(values: list[str]) -> str:
    return "".join(f'<option value="{escape(value)}"></option>' for value in values)


def _render_filter_hint(person: str | None, place: str | None) -> str:
    hints: list[str] = []
    if person:
        hints.append(f"person={escape(person)}")
    if place:
        hints.append(f"place={escape(place)}")
    if not hints:
        return ""
    return " filtered by " + ", ".join(hints)


def _render_active_filter_summary(
    *,
    q: str | None,
    search_meta: dict[str, str] | None = None,
    media_type: str | None,
    date_from: str | None,
    date_to: str | None,
    person: str | None,
    place: str | None,
) -> str:
    filters: list[tuple[str, str]] = []
    if q:
        filters.append(("Search", q))
        if search_meta:
            filters.append(("Mode", f"{search_meta.get('effective_mode', 'hybrid')} / {search_meta.get('intent_reason', 'fallback')}"))
    if media_type:
        filters.append(("Media", media_type))
    if date_from:
        filters.append(("From", date_from))
    if date_to:
        filters.append(("To", date_to))
    if person:
        filters.append(("Person", person))
    if place:
        filters.append(("Place", place))

    if not filters:
        return '<span class="filter-chip"><strong>Active filters</strong> None</span>'

    return "".join(
        f'<span class="filter-chip"><strong>{escape(label)}</strong> {escape(value)}</span>'
        for label, value in filters
    )


def _render_search_mode_pill(search_meta: dict[str, str] | None) -> str:
    if not search_meta:
        return ""
    mode = search_meta.get("effective_mode", "hybrid")
    reason = search_meta.get("intent_reason", "fallback")
    return f'<span class="meta-pill">Search {escape(mode)} / {escape(reason)}</span>'


def _page_url(request: Request, page: int) -> str:
    params = dict(request.query_params)
    params["page"] = str(max(1, page))
    return f"/gallery?{urlencode(params)}"


def _render_page_link(label: str, href: str, *, enabled: bool) -> str:
    if enabled:
        return f'<a class="button secondary" href="{escape(href)}">{escape(label)}</a>'
    return f'<span class="button secondary" style="opacity:.45; pointer-events:none;">{escape(label)}</span>'


def _resolve_asset_path(derived_root: Path, derived_path: str) -> Path:
    candidate = Path(derived_path)
    if candidate.is_absolute():
        return candidate

    resolved_root = derived_root.resolve()
    resolved_path = (resolved_root / candidate).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="derived asset file missing") from exc
    return resolved_path


def _parse_date(value: Optional[str]) -> Optional[date]:
    if value is None or value.strip() == "":
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _start_of_day(value: Optional[date]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.combine(value, time.min)


def _end_of_day(value: Optional[date]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.combine(value, time.max)
