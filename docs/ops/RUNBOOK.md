# Runbook

## If GitHub Webhook Delivery Fails

- check `AGENT_WEBHOOK_URL` and `AGENT_WEBHOOK_SECRET`
- inspect workflow run `Agent PR Webhook`
- verify Orchestrator endpoint HMAC validation logs
- do not advance PR stage manually without leaving a comment

## If PR Is Stuck In Wrong Stage

- remove stale stage label
- add the single correct stage label
- leave a comment explaining who owns the next action
- update `STATUS_SNAPSHOT` only if this changes project state, not just one PR state

## If QA Rejects

- set `agent:changes-requested`
- comment with reproduction steps and expected vs actual
- return ownership to Developer

## If Planner Rejects

- set `agent:changes-requested`
- comment with spec mismatch, not implementation advice only
- keep PR open; do not squash the history until fixed

## If NAS Is Offline In Runtime

- do not delete records
- keep existing metadata readable
- record scan failure and retry on next polling cycle

## Phase 1 Media Fact Extraction

Phase 1 owns facts that come from filesystem observation or the media file
itself. During scan/per-media refresh, verify these are populated before treating
Phase 2 search behavior as broken:

- filesystem: `size_bytes`, `mtime_ns`, source root, relative path, current path
- identity: `file_id`, partial hash, media kind
- image/video metadata: MIME type, width, height, duration, codec
- EXIF/container time: `exif_datetime`
- location: raw GPS payload in `metadata_json`, coordinate tags, and reverse-geocoded place tags when enabled

Phase 2 can rebuild `search_documents` from these facts, but should not be the
first place where file size, dimensions, capture time, or GPS are extracted.

Operator-facing status should stay at the Phase 1 / Phase 2 level. Internal job
kinds such as semantic maintenance or backfill are implementation details; the
dashboard should show whether Phase 1 or Phase 2 is running and what work is
currently progressing.

## If Local AI Pack Is Missing

- keep the base app running
- keep Phase 1 scan, gallery, OCR/tag/date/place search, and dashboard usable
- show local AI image search as `not installed` or `model missing`
- do not attempt model download when `PHOTOME_OFFLINE_MODE=1`
- skip CLIP embedding work in Phase 2 and record the skip reason
- install/repair the optional local AI pack only through the online preparation flow
- inspect current state without loading the model:
  - `photome-local-ai-pack status`

## If Local AI Pack Is Installed

- set model cache paths explicitly:
  - `HF_HOME=<photome-data>/models/hf`
  - `TORCH_HOME=<photome-data>/models/torch`
- verify offline load before enabling scheduled Phase 2 CLIP work
- online preparation:
  - `photome-local-ai-pack --cache-root <photome-data>/models prepare`
- offline verification:
  - `photome-local-ai-pack --cache-root <photome-data>/models verify-offline`
- run with:
  - `PHOTOME_CLIP_ENABLED=1`
  - `PHOTOME_OFFLINE_MODE=1`
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`
- if the model/provider changes, bump embedding version and rebuild embeddings

## Reverse Geocoding And Place Tags

- default standard mode: enabled unless `PHOTOME_GEOCODING_ENABLED=0`
- offline mode: disabled even if the default would otherwise enable it
- custom provider endpoint: `PHOTOME_NOMINATIM_URL=<self-hosted-nominatim-url>`
- cache key precision follows `PHOTOME_PLACE_TAG_PRECISION` (default `3`)
- Phase 1 creates:
  - `place=<rounded-lat>,<rounded-lon>`
  - `place_detail=<lat>,<lon>`
  - `place=<city/region/country>` from cached/provider reverse geocoding
- Existing libraries are caught up by semantic maintenance after a place/search version bump; this is a migration path, not the steady-state owner of GPS extraction.

## CLIP embedding source policy (A — default)

- **Primary input:** encode from `MediaFile.current_path` (NAS or source_roots; read-only observation). No requirement to copy full-resolution originals to the derived SSD for CLIP.
- **Fallback:** if the primary path cannot be read or decode fails, use the Phase 1 thumbnail under `derived_root` (`thumb/v1/...`) when that file exists.
- **Video input:** video files are not passed directly to CLIP. Phase 1/Phase 2 first materializes a video thumbnail with `ffmpeg`, then encodes that thumbnail image.
- **When it runs:** With `PHOTOME_CLIP_ENABLED=1`, CLIP embedding and CLIP-derived auto-tags run during **Phase 1** image processing (`_refresh_media_assets` → `_materialize_image_semantics`) alongside thumbnail generation. **Phase 2** (scheduled semantic maintenance / backfill) catches up missing embeddings, stale search documents, or version skew — not a second pass for “better” vectors on unchanged pixels.
- **Phase 1 / Phase 2 jobs:** Only one library job writes the catalog at a time; that serialization is unrelated to CLIP source choice. Policy A does not require the thumbnail to exist before CLIP runs, but the fallback works best if Phase 1 has already written the thumb when the source path fails.

## Search and auto-tag vocabulary changes

- YAML-backed vocabularies are loaded and cached in process:
  - `app/services/analysis/clip_concepts.yaml`: CLIP prompts, thresholds, and aliases.
  - `app/services/analysis/filename_tags.yaml`: filename keyword auto-tags.
  - `app/services/search/tag_synonyms.yaml`: search-time tag synonyms.
  - `app/services/search/vocab_seed.yaml`: planner terms, translations, templates, and typo correction.
- Operational default: restart the service after editing these files.
- If CLIP prompts, thresholds, concept aliases, filename tag rules, or receipt/signal heuristics change behavior, bump `semantic_auto_tag_version` and rerun semantic maintenance/backfill.
- If search document composition changes, bump `semantic_search_version`.
- Code has `clear_auto_tag_caches()` / `clear_tag_synonyms_cache()` helpers for tests or a future admin reload endpoint, but production hot-reload is not the default contract.
