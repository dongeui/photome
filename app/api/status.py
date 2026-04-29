"""Runtime status endpoint and server-rendered dashboard."""

from __future__ import annotations

from html import escape
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app.api.deps import require_state
from app.api.serializers import serialize_scheduler_snapshot
from app.services.processing.registry import MediaCatalog
from app.models.job import ProcessingJob
from app.models.semantic import SearchDocument
from sqlalchemy import select
from sqlalchemy import func


router = APIRouter(tags=["status"])


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    payload = await status(request)
    scheduler = payload["scheduler"]
    semantic = payload["semantic"]
    catalog = payload["catalog"]
    jobs = payload["jobs"]
    health = payload["health"]
    source_roots = payload["storage"]["source_roots"]
    source_roots_text = escape("\n".join(source_roots))

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>photome dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4efe7;
      --paper: rgba(255,255,255,0.88);
      --panel: rgba(255,255,255,0.95);
      --line: rgba(22,31,39,0.10);
      --text: #13202a;
      --muted: #61717c;
      --accent: #cc5f32;
      --accent-soft: rgba(204,95,50,0.12);
      --ok: #2f8f5b;
      --warn: #b46a15;
      --shadow: 0 18px 45px rgba(19,32,42,0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Inter", "Helvetica Neue", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(204,95,50,0.18), transparent 25%),
        linear-gradient(180deg, #fbf8f3 0%, var(--bg) 100%);
      color: var(--text);
    }}
    .shell {{
      width: min(1280px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 20px 0 48px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.2fr .8fr;
      gap: 16px;
      margin-bottom: 18px;
      padding: 24px;
      border-radius: 28px;
      border: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(255,255,255,0.94), rgba(255,246,237,0.82));
      box-shadow: var(--shadow);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(2rem, 4vw, 3.8rem);
      line-height: .95;
      letter-spacing: -0.04em;
      font-family: "Instrument Serif", "Iowan Old Style", serif;
    }}
    .eyebrow {{
      display: inline-flex;
      margin-bottom: 12px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(19,32,42,0.06);
      color: var(--accent);
      font-size: .74rem;
      font-weight: 700;
      letter-spacing: .08em;
      text-transform: uppercase;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
      max-width: 64ch;
    }}
    .hero-links {{
      display: flex;
      gap: 10px;
      margin-top: 16px;
      flex-wrap: wrap;
    }}
    .link-btn {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 42px;
      padding: 10px 16px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      text-decoration: none;
      font-weight: 600;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 16px;
    }}
    .card {{
      grid-column: span 6;
      padding: 18px;
      border-radius: 24px;
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: 0 10px 26px rgba(19,32,42,0.06);
    }}
    .card.full {{ grid-column: 1 / -1; }}
    .card h2 {{
      margin: 0 0 10px;
      font-size: 1.05rem;
      letter-spacing: -0.02em;
    }}
    .sub {{
      margin: 0 0 14px;
      color: var(--muted);
      font-size: .92rem;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .metric {{
      padding: 12px 13px;
      border-radius: 18px;
      background: rgba(19,32,42,0.04);
      border: 1px solid rgba(19,32,42,0.06);
    }}
    .metric strong {{
      display: block;
      margin-top: 4px;
      font-size: 1.05rem;
      letter-spacing: -0.02em;
    }}
    .pill-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 10px;
      border-radius: 999px;
      background: rgba(19,32,42,0.05);
      border: 1px solid rgba(19,32,42,0.07);
      font-size: .88rem;
    }}
    .status-ok {{ color: var(--ok); }}
    .status-warn {{ color: var(--warn); }}
    .list {{
      display: grid;
      gap: 10px;
    }}
    .row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-radius: 16px;
      background: rgba(19,32,42,0.04);
      font-size: .92rem;
    }}
    .scan-form {{
      display: grid;
      gap: 10px;
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
    }}
    .scan-form label {{
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: .86rem;
      font-weight: 700;
    }}
    .scan-form textarea {{
      min-height: 86px;
      resize: vertical;
      padding: 11px 12px;
      border: 1px solid rgba(19,32,42,0.14);
      border-radius: 14px;
      background: rgba(255,255,255,0.9);
      color: var(--text);
      font: .86rem "SFMono-Regular", "Menlo", monospace;
      line-height: 1.45;
    }}
    .scan-actions {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .scan-actions label {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      color: var(--text);
      font-size: .9rem;
      font-weight: 600;
    }}
    .scan-actions button {{
      min-height: 40px;
      padding: 9px 14px;
      border: 0;
      border-radius: 999px;
      background: var(--accent);
      color: white;
      font-weight: 800;
      cursor: pointer;
    }}
    .scan-actions button:disabled {{
      opacity: .62;
      cursor: progress;
    }}
    .scan-card.is-running {{
      border-color: rgba(204,95,50,0.32);
    }}
    .scan-card.is-running .scan-title::after {{
      content: "";
      display: inline-block;
      width: 14px;
      height: 14px;
      margin-left: 8px;
      border-radius: 50%;
      border: 2px solid rgba(204,95,50,0.22);
      border-top-color: var(--accent);
      vertical-align: -2px;
      animation: spin 850ms linear infinite;
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}
    .scan-result {{
      display: none;
      margin: 0;
      padding: 10px 12px;
      overflow: auto;
      border-radius: 14px;
      background: rgba(19,32,42,0.05);
      color: var(--text);
      font: .8rem "SFMono-Regular", "Menlo", monospace;
      line-height: 1.45;
      white-space: pre-wrap;
    }}
    .scan-result.visible {{ display: block; }}
    .debug-form {{
      display: grid;
      gap: 10px;
      margin-top: 16px;
    }}
    .debug-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) repeat(5, minmax(112px, .42fr)) auto;
      gap: 8px;
    }}
    .debug-grid input, .debug-grid select {{
      min-height: 40px;
      padding: 9px 12px;
      border: 1px solid rgba(19,32,42,0.14);
      border-radius: 14px;
      background: rgba(255,255,255,0.9);
      color: var(--text);
      font: .9rem "Inter", "Helvetica Neue", sans-serif;
    }}
    .debug-grid button {{
      min-height: 40px;
      padding: 9px 14px;
      border: 0;
      border-radius: 999px;
      background: #174f49;
      color: white;
      font-weight: 800;
      cursor: pointer;
    }}
    .debug-result {{
      margin: 0;
      padding: 10px 12px;
      min-height: 260px;
      overflow: auto;
      border-radius: 14px;
      background: rgba(19,32,42,0.05);
      color: var(--text);
      font: .8rem "SFMono-Regular", "Menlo", monospace;
      line-height: 1.45;
      white-space: pre-wrap;
    }}
    .benchmark-actions {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 8px;
    }}
    .benchmark-actions button {{
      min-height: 40px;
      padding: 9px 14px;
      border: 0;
      border-radius: 999px;
      background: #13202a;
      color: white;
      font-weight: 800;
      cursor: pointer;
    }}
    .benchmark-summary {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 8px;
      color: var(--muted);
      font-size: .9rem;
    }}
    code {{
      font-family: "SFMono-Regular", "Menlo", monospace;
      font-size: .84rem;
    }}
    @media (max-width: 920px) {{
      .hero {{ grid-template-columns: 1fr; }}
      .card {{ grid-column: 1 / -1; }}
      .metric-grid {{ grid-template-columns: 1fr 1fr; }}
      .debug-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 560px) {{
      .metric-grid {{ grid-template-columns: 1fr; }}
      .debug-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div>
        <span class="eyebrow">Local Service Status</span>
        <h1>Two loops, one library.</h1>
        <p>Phase 1 keeps polling NAS originals into a stable local catalog. Phase 2 keeps scheduling semantic enrichment on top of cached media so new files and version changes keep flowing through without manual resets.</p>
        <div class="hero-links">
          <a class="link-btn" href="/gallery">Open Gallery</a>
        </div>
      </div>
      <div class="metric-grid">
        <div class="metric">
          Phase 1 Catalog
          <strong>{catalog["total"]} items</strong>
        </div>
        <div class="metric">
          Waiting Stable
          <strong>{health["waiting_stable"]}</strong>
        </div>
        <div class="metric">
          Errors
          <strong>{health["error"]}</strong>
        </div>
      </div>
    </section>

    <section class="grid">
      <article class="card scan-card" id="phase1-card">
        <h2 class="scan-title">Phase 1 Polling Loop</h2>
        <p class="sub">Source-driven ingest from NAS originals into local cache and derived assets.</p>
        <div class="pill-row">
          <span class="pill"><strong>Enabled</strong> <span class="{'status-ok' if scheduler['enabled'] else 'status-warn'}">{escape(str(scheduler['enabled']))}</span></span>
          <span class="pill"><strong>Running</strong> <span class="{'status-ok' if scheduler['running'] else 'status-warn'}">{escape(str(scheduler['running']))}</span></span>
          <span class="pill"><strong>Poll</strong> {scheduler['poll_interval_seconds']}s</span>
          <span class="pill"><strong>Full Scan</strong> {scheduler['daily_full_scan_hour']:02d}:{scheduler['daily_full_scan_minute']:02d}</span>
        </div>
        <div class="list" style="margin-top:14px;">
          <div class="row"><span>Last poll</span><span>{escape(str(scheduler['last_poll_at']))}</span></div>
          <div class="row"><span>Next poll</span><span>{escape(str(scheduler['next_poll_at']))}</span></div>
          <div class="row"><span>Last full scan</span><span>{escape(str(scheduler['last_full_scan_at']))}</span></div>
          <div class="row"><span>Next full scan</span><span>{escape(str(scheduler['next_full_scan_at']))}</span></div>
        </div>
        <form class="scan-form" id="phase1-scan-form">
          <label>
            Source roots
            <textarea id="phase1-source-roots" name="source_roots" spellcheck="false">{source_roots_text}</textarea>
          </label>
          <div class="scan-actions">
            <button type="submit" id="phase1-scan-button">Run Phase 1 Scan</button>
            <label><input type="checkbox" id="phase1-full-scan" name="full_scan"> Full scan</label>
          </div>
          <pre class="scan-result" id="phase1-scan-result" aria-live="polite"></pre>
        </form>
      </article>

      <article class="card">
        <h2>Phase 2 Semantic Loop</h2>
        <p class="sub">Catalog-driven semantic maintenance for tags, OCR, captions, embeddings, and search versions.</p>
        <div class="pill-row">
          <span class="pill"><strong>Enabled</strong> <span class="{'status-ok' if semantic['scheduler_enabled'] else 'status-warn'}">{escape(str(semantic['scheduler_enabled']))}</span></span>
          <span class="pill"><strong>Interval</strong> {semantic['scheduler_interval_seconds']}s</span>
          <span class="pill"><strong>Face Analysis</strong> <span class="{'status-ok' if semantic['runtime']['face_analysis_enabled'] else 'status-warn'}">{escape(str(semantic['runtime']['face_analysis_enabled']))}</span></span>
          <span class="pill"><strong>Place Precision</strong> {semantic['runtime']['place_tag_precision']}</span>
        </div>
        <div class="list" style="margin-top:14px;">
          <div class="row"><span>Last semantic maintenance</span><span>{escape(str(scheduler.get('last_semantic_maintenance_at')))}</span></div>
          <div class="row"><span>Next semantic maintenance</span><span>{escape(str(scheduler.get('next_semantic_maintenance_at')))}</span></div>
        </div>
      </article>

      <article class="card full">
        <h2>Semantic Version Set</h2>
        <p class="sub">These versions define when cached semantic outputs can be rebuilt without touching Stage 1 identity or core cache.</p>
        <div class="pill-row">
          <span class="pill"><strong>Place</strong> <code>{escape(semantic['versions']['place'])}</code></span>
          <span class="pill"><strong>Person</strong> <code>{escape(semantic['versions']['person'])}</code></span>
          <span class="pill"><strong>OCR</strong> <code>{escape(semantic['versions']['ocr'])}</code></span>
          <span class="pill"><strong>Caption</strong> <code>{escape(semantic['versions']['caption'])}</code></span>
          <span class="pill"><strong>Embedding</strong> <code>{escape(semantic['versions']['embedding'])}</code></span>
          <span class="pill"><strong>Auto Tags</strong> <code>{escape(semantic['versions']['auto_tags'])}</code></span>
          <span class="pill"><strong>Search</strong> <code>{escape(semantic['versions']['search'])}</code></span>
        </div>
      </article>

      <article class="card full">
        <h2>Source and Storage</h2>
        <p class="sub">Native local runtime with localhost web UI, reading originals from NAS and writing rebuildable derived data locally.</p>
        <div class="list">
          <div class="row"><span>Source roots</span><span>{'<br>'.join(escape(path) for path in source_roots)}</span></div>
          <div class="row"><span>Derived root</span><span><code>{escape(payload['storage']['derived_root'])}</code></span></div>
          <div class="row"><span>Database</span><span><code>{escape(payload['storage']['database_url'])}</code></span></div>
          <div class="row"><span>Recent jobs tracked</span><span>{len(jobs['recent'])}</span></div>
        </div>
      </article>

      <article class="card full">
        <h2>Search Tuning</h2>
        <p class="sub">Inspect query planning, channel candidates, weights, and fused ranking without leaving the dashboard.</p>
        <form class="debug-form" id="search-debug-form">
          <div class="debug-grid">
            <input id="search-debug-query" name="q" placeholder="Search query" value="작년 여름 바다에서 가족이랑 찍은 사진">
            <select id="search-debug-mode" name="mode">
              <option value="hybrid" selected>hybrid</option>
              <option value="ocr">ocr</option>
              <option value="semantic">semantic</option>
            </select>
            <input id="search-debug-place" name="place" placeholder="Place filter">
            <input id="search-debug-w-ocr" name="w_ocr" placeholder="w_ocr" inputmode="decimal">
            <input id="search-debug-w-clip" name="w_clip" placeholder="w_clip" inputmode="decimal">
            <input id="search-debug-w-shadow" name="w_shadow" placeholder="w_shadow" inputmode="decimal">
            <button type="submit">Inspect Search</button>
          </div>
          <pre class="debug-result" id="search-debug-result" aria-live="polite"></pre>
        </form>
        <div class="benchmark-actions">
          <button type="button" id="search-benchmark-run">Run Synthetic Benchmark</button>
          <div class="benchmark-summary" id="search-benchmark-summary"></div>
        </div>
        <pre class="debug-result" id="search-benchmark-result" aria-live="polite"></pre>
      </article>
    </section>
  </main>
  <script>
    const scanForm = document.getElementById("phase1-scan-form");
    const scanResult = document.getElementById("phase1-scan-result");
    const scanCard = document.getElementById("phase1-card");
    const scanButton = document.getElementById("phase1-scan-button");
    const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
    function renderScanJob(job) {{
      const summary = job?.result?.summary || {{}};
      const processed = job?.result?.processed || {{}};
      const lines = [
        `status: ${{job?.status || "unknown"}}`,
      ];
      if (job?.status === "queued" || job?.status === "running") {{
        lines.push(`job: ${{job?.job_id || ""}}`);
        if (job?.started_at) lines.push(`started: ${{job.started_at}}`);
        return lines.join("\\n");
      }}
      lines.push(
        `scanned: ${{summary.scanned ?? 0}}`,
        `created: ${{summary.created ?? 0}}`,
        `updated: ${{summary.updated ?? 0}}`,
        `moved: ${{summary.moved ?? 0}}`,
        `missing: ${{summary.missing ?? 0}}`,
        `failed: ${{summary.failed ?? 0}}`,
        `processed: ${{processed.succeeded ?? 0}} succeeded, ${{processed.failed ?? 0}} failed`,
      );
      if (job?.error_message) lines.push(`error: ${{job.error_message}}`);
      return lines.join("\\n");
    }}
    async function pollScanJob(jobId) {{
      while (true) {{
        const response = await fetch(`/scan/jobs/${{encodeURIComponent(jobId)}}`, {{ cache: "no-store" }});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || `HTTP ${{response.status}}`);
        const job = payload.job;
        scanResult.textContent = renderScanJob(job);
        if (job.status !== "queued" && job.status !== "running") return job;
        await sleep(1200);
      }}
    }}
    scanForm?.addEventListener("submit", async (event) => {{
      event.preventDefault();
      scanResult.classList.add("visible");
      scanCard.classList.add("is-running");
      scanButton.disabled = true;
      scanResult.textContent = "Starting scan...";
      const sourceRoots = document.getElementById("phase1-source-roots").value;
      const fullScan = document.getElementById("phase1-full-scan").checked;
      const params = new URLSearchParams();
      if (sourceRoots.trim()) params.set("source_roots", sourceRoots);
      if (fullScan) params.set("full_scan", "true");
      try {{
        const response = await fetch(`/scan/async?${{params.toString()}}`, {{ method: "POST" }});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || `HTTP ${{response.status}}`);
        scanResult.textContent = renderScanJob(payload.job);
        await pollScanJob(payload.job.job_id);
      }} catch (error) {{
        scanResult.textContent = `error: ${{error.message}}`;
      }} finally {{
        scanCard.classList.remove("is-running");
        scanButton.disabled = false;
      }}
    }});

    const searchDebugForm = document.getElementById("search-debug-form");
    const searchDebugResult = document.getElementById("search-debug-result");
    searchDebugForm?.addEventListener("submit", async (event) => {{
      event.preventDefault();
      searchDebugResult.textContent = "Inspecting search...";
      const params = new URLSearchParams();
      const query = document.getElementById("search-debug-query").value;
      const mode = document.getElementById("search-debug-mode").value;
      const place = document.getElementById("search-debug-place").value;
      const wOcr = document.getElementById("search-debug-w-ocr").value;
      const wClip = document.getElementById("search-debug-w-clip").value;
      const wShadow = document.getElementById("search-debug-w-shadow").value;
      if (query.trim()) params.set("q", query);
      if (mode) params.set("mode", mode);
      if (place.trim()) params.set("place", place);
      if (wOcr.trim()) params.set("w_ocr", wOcr);
      if (wClip.trim()) params.set("w_clip", wClip);
      if (wShadow.trim()) params.set("w_shadow", wShadow);
      try {{
        const response = await fetch(`/search/debug?${{params.toString()}}`);
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || `HTTP ${{response.status}}`);
        searchDebugResult.textContent = JSON.stringify(payload.meta, null, 2);
      }} catch (error) {{
        searchDebugResult.textContent = `error: ${{error.message}}`;
      }}
    }});

    const benchmarkRun = document.getElementById("search-benchmark-run");
    const benchmarkSummary = document.getElementById("search-benchmark-summary");
    const benchmarkResult = document.getElementById("search-benchmark-result");
    benchmarkRun?.addEventListener("click", async () => {{
      benchmarkSummary.textContent = "";
      benchmarkResult.textContent = "Running benchmark...";
      try {{
        const params = new URLSearchParams();
        const wOcr = document.getElementById("search-debug-w-ocr").value;
        const wClip = document.getElementById("search-debug-w-clip").value;
        const wShadow = document.getElementById("search-debug-w-shadow").value;
        if (wOcr.trim()) params.set("w_ocr", wOcr);
        if (wClip.trim()) params.set("w_clip", wClip);
        if (wShadow.trim()) params.set("w_shadow", wShadow);
        const response = await fetch(`/search/benchmark?${{params.toString()}}`);
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || `HTTP ${{response.status}}`);
        const overrideText = Object.keys(payload.weight_overrides || {{}}).length
          ? `, overrides ${{JSON.stringify(payload.weight_overrides)}}`
          : "";
        const failedChecks = Object.keys(payload.summary?.failed_checks || {{}}).length
          ? `, failed checks ${{JSON.stringify(payload.summary.failed_checks)}}`
          : "";
        benchmarkSummary.textContent = `passed ${{payload.passed}} / ${{payload.total}}, failed ${{payload.failed}}${{overrideText}}${{failedChecks}}`;
        benchmarkResult.textContent = JSON.stringify({{
          summary: payload.summary,
          cases: payload.cases,
        }}, null, 2);
      }} catch (error) {{
        benchmarkResult.textContent = `error: ${{error.message}}`;
      }}
    }});
  </script>
</body>
</html>"""
    return HTMLResponse(html)


@router.get("/status")
async def status(request: Request) -> dict[str, Any]:
    settings = require_state(request, "settings")
    database = require_state(request, "database")
    pipeline = require_state(request, "pipeline")
    scheduler = require_state(request, "scheduler")

    with database.session_factory() as session:
        catalog = MediaCatalog(session)
        pipeline_snapshot = pipeline.status_snapshot()
        recent_jobs = session.execute(
            select(ProcessingJob).order_by(ProcessingJob.updated_at.desc(), ProcessingJob.enqueued_at.desc()).limit(10)
        ).scalars().all()

        return {
            "app": {
                "name": settings.app_name,
                "version": settings.app_version,
            },
            "storage": {
                "data_root": str(settings.data_root),
                "derived_root": str(settings.derived_root),
                "source_roots": [str(path) for path in settings.source_roots],
                "database_url": settings.database_url,
            },
            "catalog": pipeline_snapshot["media"],
            "jobs": {
                **pipeline_snapshot["jobs"],
                "recent": [
                    {
                        "id": job.id,
                        "job_kind": job.job_kind,
                        "status": job.status,
                        "payload_json": job.payload_json,
                        "result_json": job.result_json,
                        "error_stage": job.error_stage,
                        "error_message": job.error_message,
                        "attempts": job.attempts,
                        "enqueued_at": job.enqueued_at,
                        "started_at": job.started_at,
                        "finished_at": job.finished_at,
                        "updated_at": job.updated_at,
                    }
                    for job in recent_jobs
                ],
            },
            "scheduler": serialize_scheduler_snapshot(scheduler.snapshot()),
            "semantic": {
                "scheduler_enabled": settings.semantic_scheduler_enabled,
                "scheduler_interval_seconds": settings.semantic_scheduler_interval_seconds,
                "versions": {
                    "place": settings.semantic_place_version,
                    "person": settings.semantic_person_version,
                    "ocr": settings.semantic_ocr_version,
                    "caption": settings.semantic_caption_version,
                    "embedding": settings.semantic_embedding_version,
                    "auto_tags": settings.semantic_auto_tag_version,
                    "search": settings.semantic_search_version,
                },
                "runtime": {
                    "face_analysis_enabled": settings.face_analysis_enabled,
                    "place_tag_precision": settings.place_tag_precision,
                },
                "search_documents": {
                    "total": int(session.scalar(select(func.count()).select_from(SearchDocument)) or 0),
                    "version": settings.semantic_search_version,
                },
            },
            "health": {
                "database_configured": database.configured,
                "waiting_stable": catalog.count_observations(status="waiting_stable"),
                "error": catalog.count_media(status="error"),
            },
        }
