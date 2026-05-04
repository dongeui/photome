# Architecture

Phase 2 검색 종착지 설계와 기술 선택 기준은 `docs/engineering/PHASE2_SEARCH_TECH_REVIEW.md`를 따른다.

## Topology

```text
[NAS - original media, read-only]
        |
        v
[Server host - FastAPI, scanner, DB, workers, scheduler]
        |
        v
[External SSD - thumbnails, previews, keyframes, embeddings]
```

## Runtime and Access

- primary runtime: native local service on the host machine
- primary access: `http://127.0.0.1:<port>` localhost web UI
- macOS: optional menu bar shell controls the local service and opens the web UI
- Windows: localhost web UI is the default operator surface
- Docker / Compose remains an optional packaging and deployment path
- distribution artifacts are split into base app and optional local AI pack
- the base app must start and remain useful when the local AI pack is absent

## Source of Truth

- original binary: NAS
- file identity: `file_id`
- current location: DB `current_path`
- derived outputs: external SSD

## Identity Rules

- `current_path` is an observation, not identity
- `file_id` is stable across rename/move when fingerprint matches
- `content_hash` is optional secondary proof, not the first gate

## Service Boundaries

Stage 1 — ingest and caching:

- `scanner`: file discovery and raw path observation
- `fingerprint`: stable id and moved-file detection
- `metadata`: file/container/EXIF facts — size, mtime, MIME, width/height, duration, codec, captured time, raw metadata JSON, GPS
- `geocoding`: EXIF GPS 좌표 → 지명 계층 태그, 결과 캐시. 온라인 standard mode에서 Phase 1 place fact로 materialize
- `thumbnail`: eager thumbnail creation
- `video`: lazy keyframe extraction
- `analysis`: OpenCV face detection + embedding + person clustering
- `processing`: derived asset registry and pipeline state changes
- `workers`: queued execution and retries
- `scheduler`: polling scan + daily full scan
- `api`: list/detail/filter/scan/status endpoints

Stage 2 — semantic enrichment and NL search:

- `embedding`: 이미지 멀티모달 임베딩(CLIP/SigLIP) 추출 및 저장; enabled
  only when the optional local AI pack and model cache are ready
- `ocr`: 이미지 내 텍스트 추출
- `caption` (선택): VLM 기반 자연어 캡션 생성
- `semantic`: OCR/태그/사람/장소/신호/임베딩 ref를 `search_documents`로 집계
- `search.planner`: 쿼리를 keyword/OCR/person/place/date/visual intent로 분해
- `search.vector`: `VectorIndexBackend` 인터페이스와 `LocalNumpyVectorIndex` 기본 구현
- `search`: FTS5 keyword search + OCR/tag/shadow document + CLIP vector 후보를 RRF로 결합

## Phase 1 Fact Contract

Phase 1 owns facts that already exist in the original media or in the filesystem
observation. These facts are materialized during scan/per-media refresh, not
deferred to semantic search maintenance.

- filesystem facts: `size_bytes`, `mtime_ns`, source root, relative path, current path
- identity facts: `file_id`, partial hash, fingerprint version, media kind
- container/image facts: MIME type, width, height, duration, codec
- EXIF facts: captured datetime, raw EXIF/metadata JSON, GPS latitude/longitude
- location facts: coordinate tags (`place`, `place_detail`) and reverse-geocoded place names when `PHOTOME_GEOCODING_ENABLED=1` and the runtime is not offline
- deterministic local tags: filename/date-derived tags and face/person tags produced during the per-media refresh

Phase 2 may rebuild search documents from these facts or backfill stale place
tags after a version bump, but it does not own the source-of-truth extraction
for file size, capture time, dimensions, or GPS-derived place facts.

## CLIP embedding input and phase split

- **Policy A (default):** CLIP encodes from `current_path` first (NAS/source read). Thumbnail on derived storage is used only as a decode fallback. Originals are not copied wholesale to derived disk for embedding.
- **Phase 1 (scan / per-media refresh):** For each image, after metadata and thumbnail generation, `ProcessingPipeline._materialize_image_semantics` runs when configured: OCR (prefers existing thumbnail via `_analysis_source_path` when present), optional CLIP embedding + CLIP auto-tags, and `search_documents` update. So the first full scan can already produce vectors and semantic rows while CLIP is enabled.
- **Phase 2 (semantic maintenance):** Scheduled or manual `run_semantic_maintenance` / backfill fills gaps: missing embeddings, outdated `search_documents`, version mismatches, or stale Phase 1-derived place tags after a place/search version bump. Re-encoding the same pixels under the same model/version is maintenance, not an accuracy multiplier.

## Phase 2 Cycle Contract

- Phase 2는 Phase 1과 독립된 사이클로 돈다 (라이브러리 잡은 동시에 하나만 실행되어 DB 단일 writer를 보호).
- Phase 1이 새 파일을 안정화하고 `thumb_done` 또는 `analysis_done`으로 만들면, 그 시점에서 이미 시맨틱 산출물이 있을 수 있다(CLIP/OCR 활성 시). Phase 2는 여전히 누락·스테일·버전 불일치를 배치로 보완한다.
- Phase 2는 매 사이클마다 아래 항목만 처리한다.
  - `search_documents`가 없는 media
  - `SearchDocument.version`이 현재 `semantic_search_version`과 다른 media
  - `SearchDocument.source_updated_at < MediaFile.updated_at`인 media
- `ProcessingPipeline.run_semantic_maintenance()`는 non-blocking lock을 사용해 중복 사이클 실행을 막는다.
- 수동 검증용 endpoint는 `POST /scan/semantic-maintenance`다.

## Query Planning Contract

- `QueryPlanner`는 deterministic rule-based planner로 시작한다.
- output은 검색 응답 `meta.query_plan`에 남긴다.
- 현재 분류 필드:
  - `keyword_query`
  - `visual_queries`
  - `date_from`, `date_to`
  - `person_terms`
  - `place_terms`
  - `ocr_terms`
  - `visual_terms`
  - `intent`
- 이미지 검색 품질은 planner가 어떤 channel을 강하게 호출할지 결정하는 데서 나온다. 모델 기반 planner는 이후 optional provider로 붙인다.

## Pipeline States

Stage 1 (현재):

`discovered -> waiting_stable -> queued -> metadata_done -> thumb_done -> preview_done -> analysis_done`

Stage 2:

`thumb_done|analysis_done -> semantic indexed`

Semantic indexed는 별도 media status가 아니라 `search_documents` 및 관련 semantic output row의 존재와 version으로 판단한다.

Terminal or divergent states:

- `missing`
- `error`

## Storage Layout

- DB tracks metadata and asset registry
- SSD stores files by asset type and version, not by original path
- derived layout:
  - `thumb/v1/ab/abcdef....jpg`
  - `preview/v1/ab/abcdef....jpg`
  - `keyframe/v1/ab/abcdef....jpg`
  - `embeddings/faces/v1/ab/<file_id>-face-<idx>.json`
  - `embeddings/people/v1/person-<id>.json`
  - `embeddings/clip/<model_version>/ab/<file_id>.npy` (T17 이후)
- SQLite semantic tables:
  - `media_ocr`
  - `media_analysis_signals`
  - `media_embeddings`
  - `media_auto_tag_states`
  - `search_documents`
  - `search_documents_fts` (SQLite FTS5 virtual table)

## External Dependencies (Stage 2)

- 임베딩 모델: SigLIP2 또는 OpenCLIP, HuggingFace weight 오프라인 캐시.
  배포 시 기본 앱에 포함하지 않고 optional local AI pack으로 분리한다.
- LLM 쿼리 파서: 로컬 Ollama(Qwen 2.5) 또는 외부 API, JSON 스키마 강제
- 벡터 인덱스: `VectorIndexBackend` 기준. 현재 local NumPy exact search, 확장 시 FAISS/LanceDB/Qdrant adapter

## External Dependencies (Stage 1)

- 역지오코딩: 기본 provider는 Nominatim이며 `PHOTOME_NOMINATIM_URL`로 셀프호스트 endpoint를 지정할 수 있다. `PHOTOME_OFFLINE_MODE=1`이면 자동 비활성화하고, `PHOTOME_GEOCODING_ENABLED=0`으로 standard mode에서도 끌 수 있다.

## Packaging Boundary

Photome stays one source repository. Packaging is split by capability:

- `photome-base`: local service, web UI, Phase 1, OCR/tag/date/place/shadow search,
  DB, scheduler, and dashboard
- `photome-local-ai-pack`: PyTorch/OpenCLIP runtime, approved model weights, cache
  verification, and AI model notices

Versioning is multi-dimensional:

- app version: server/UI/API behavior
- semantic search version: `search_documents` composition
- embedding version: vector provider/model/dimensions
- auto tag version: concept thresholds and aliases
- model pack version: bundled local AI runtime and model cache

Changing model/provider/dimensions requires a new embedding version. Changing
concept aliases or thresholds requires a new auto tag version. The base app must
never require the local AI pack to scan, browse, or serve existing metadata.

Operational packaging details live in `docs/ops/PACKAGING_STRATEGY.md`.
