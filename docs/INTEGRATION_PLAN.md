# photome Integration Plan

`photome` is the integration workspace for the two source projects:

- `photomine`: production-shaped local-first media catalog, pipeline, jobs, scheduler, gallery, and dashboard
- `photomem`: working OCR + CLIP + hybrid search experiment

## Direction

Use `photomine` as the base application and migrate `photomem` search capabilities into it.

The stable identity key is `MediaFile.file_id`; `photomem`'s old integer `photo_id` schema should not be carried forward.

## Imported So Far

- OCR extraction service: `app/services/ocr`
- CLIP image/text encoder wrapper: `app/services/embedding`
- Korean-first CLIP query expansion: `app/services/search/query_translate.py`
- Hybrid OCR/semantic rank fusion shell: `app/services/search/hybrid.py`
- Image text/screenshot/document signal extraction: `app/services/analysis/image_signals.py`
- Semantic persistence models: `MediaOCR`, `MediaOCRBlock`, `MediaOCRGram`, `MediaAnalysisSignal`, `MediaEmbedding`

## Integration Status — All Steps Complete ✓

| # | Task | Status |
|---|------|--------|
| 1 | Wire OCR + image-signal generation into pipeline after thumbnails | ✅ Done (`_materialize_image_semantics`) |
| 2 | CLIP embedding as versioned `DerivedAsset` under `embeddings/clip/<version>` | ✅ Done (registered via `catalog.register_derived_asset` + `SemanticCatalog`) |
| 3 | SQLAlchemy search backend for `HybridSearchService` | ✅ Done (`app/services/search/backend.py`) |
| 4 | `/search` JSON API | ✅ Done (`app/api/search.py`) |
| 5 | Fold search controls into gallery UI | ✅ Done (gallery is now search-first) |
| 6 | Port `photomem` tests to `file_id`-based fixtures | ✅ Done (`tests/`) |

## Phase 2 — Search Quality Improvements (Complete)

### Natural Language Search
- Korean ↔ English lexicon expanded to 90+ entries (life events, places, family, seasons, holidays)
- Filler phrase stripping before CLIP encoding (`찍은 사진`, `에서 찍은` etc.)
- `extract_date_range()`: parses `작년`, `올해`, `이번달`, `여름`, `가을` into datetime filters
  automatically applied when the caller does not specify a date range
- Travel + celebration queries routed to semantic mode (`auto-travel`, `auto-celebration`)

### CLIP Visual Concepts
- Expanded from 9 → 22 categories: beach, mountain, nature, sky, cake, coffee,
  celebration, wedding, travel, sunset, animal, group
- Up to 5 auto-tags per image (was 3)

### Tag Synonym Expansion
- `TAG_SYNONYMS` (40+ Korean↔English pairs): searching `아기` surfaces `baby`/`infant`
  tags and vice versa; multi-token queries each contribute synonyms

### Korean OCR n-gram Boost
- `search_by_ocr` supplements full-text LIKE results with the `media_ocr_grams` index
  when Korean characters appear in the query and main hits are sparse

### Semantic Backfill
- `ProcessingPipeline.run_semantic_backfill()` + `POST /scan/semantic-backfill` to
  generate CLIP embeddings for media processed before CLIP was enabled

## Enabling CLIP (Optional — ~350 MB model download)

CLIP is opt-in to avoid large downloads on small servers:

```bash
export PHOTOME_CLIP_ENABLED=true
```

Run backfill after enabling to generate embeddings for existing media:

```
POST /scan/semantic-backfill
```

## Dependency Notes

`photome` keeps `photomine`'s modular structure but adds `photomem`'s heavier semantic dependencies:

- `open_clip_torch`
- `torch`
- `torchvision`
- `pytesseract`

For deployment, install PyTorch CPU wheels deliberately to avoid accidentally pulling CUDA builds on small local servers.
