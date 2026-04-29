# photome

Local-first personal photo home server with natural language search.

Merges the production-shaped `photomine` backend with OCR + CLIP + hybrid search from `photomem`.

## Features

- **Natural language search** — "작년 여름 바다", "생일 케이크", "아기 사진", "receipt" all work
- **Hybrid search** — RRF fusion of CLIP semantic, OCR full-text, and tag/annotation results
- **Phase 2 semantic cycle** — only missing, stale, or version-mismatched search documents are rebuilt
- **Search documents + FTS5** — OCR, tags, people, places, annotations, and signals are materialized into SQLite FTS
- **Korean-first** — 90+ lexicon entries, typo correction, filler word stripping, date extraction
- **Query planning** — rule-based keyword/OCR/person/place/date/visual intent extraction
- **Auto-tagging** — 22 CLIP visual categories + signal-based tags (screenshot, document, receipt)
- **Tag synonyms** — Korean ↔ English synonym expansion (아기 ↔ baby, 여행 ↔ travel, …)
- **Gallery UI** — search-first workspace with quick-search chips, thumbnail overlay, and annotations
- **OCR** — Tesseract text extraction indexed for full-text search + Korean n-gram boosting
- **Face detection** — optional OpenCV-based face analysis with person clustering

## Quick Start

```bash
pip install -e .
# optional: enable CLIP visual search (~350MB model download on first run)
export PHOTOME_CLIP_ENABLED=true
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 → gallery with search bar.

Trigger a scan to index your photos:

```
POST /scan
```

Scan a runtime path without restarting the server:

```
POST /scan?source_roots=/path/to/photos
```

Refresh Phase 2 search documents for stale or missing semantic rows:

```
POST /scan/semantic-maintenance
```

Inspect query planning and ranking channels:

```
GET /search/debug?q=작년 여름 바다에서 가족이랑 찍은 사진
```

Run the synthetic benchmark suite for query planning and channel behavior:

```
GET /search/benchmark
```

After enabling CLIP, backfill existing media:

```
POST /scan/semantic-backfill
```

## Search Examples

| Query | What happens |
|-------|-------------|
| `아기` | tag synonym expansion → also matches `baby`, `infant` |
| `작년 여름` | auto date filter: last year June–August |
| `이번달 음식` | date filter: this month + food visual/tag search |
| `영수증` | OCR text match + receipt auto-tag |
| `결혼식` | CLIP semantic match → `wedding` concept |
| `바다 여행` | semantic mode → `beach`, `travel` CLIP concepts |
| `스크린샷 오류` | OCR mode → screen/error text search |

## Configuration

All settings read from environment variables (prefix `PHOTOME_` or `PHOTOMINE_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `PHOTOME_CLIP_ENABLED` | `false` | Enable CLIP visual search |
| `PHOTOME_OCR_ENABLED` | `true` | Enable Tesseract OCR indexing |
| `PHOTOME_SOURCE_ROOTS` | `./photos` | Comma-separated photo directories |
| `PHOTOME_DATA_ROOT` | `./data` | Database and model storage |
| `PHOTOME_DERIVED_ROOT` | `./derived_root` | Thumbnails, embeddings |

See `app/core/settings.py` for the full list.

## Architecture

```
app/
  api/          – FastAPI routes (gallery, search, scan, media, status)
  services/
    search/     – hybrid.py (RRF fusion), backend.py (SQLAlchemy), query_translate.py
                – planner.py (query intent), vector.py (replaceable vector backend)
    semantic/   – search document materialization + Phase 2 maintenance catalog
    analysis/   – auto_tags.py (CLIP concepts), image_signals.py
    embedding/  – clip.py (ViT-B/32 via open_clip)
    ocr/        – Tesseract wrapper
    processing/ – pipeline.py (scan→thumbnail→OCR→CLIP→tags→search documents)
  models/       – SQLAlchemy ORM (MediaFile, Tag, MediaOCR, MediaEmbedding, SearchDocument, …)
```

See [docs/engineering/PHASE2_SEARCH_TECH_REVIEW.md](docs/engineering/PHASE2_SEARCH_TECH_REVIEW.md) for the Phase 2 search direction and [docs/INTEGRATION_PLAN.md](docs/INTEGRATION_PLAN.md) for integration history.
