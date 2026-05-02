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
- `metadata`: EXIF, width/height, duration, ffprobe
- `thumbnail`: eager thumbnail creation
- `video`: lazy keyframe extraction
- `analysis`: OpenCV face detection + embedding + person clustering
- `processing`: derived asset registry and pipeline state changes
- `workers`: queued execution and retries
- `scheduler`: polling scan + daily full scan
- `api`: list/detail/filter/scan/status endpoints

Stage 2 — semantic enrichment and NL search:

- `geocoding`: GPS 좌표 → 지명 계층 태그, 결과 캐시
- `embedding`: 이미지 멀티모달 임베딩(CLIP/SigLIP) 추출 및 저장; enabled
  only when the optional local AI pack and model cache are ready
- `ocr`: 이미지 내 텍스트 추출
- `caption` (선택): VLM 기반 자연어 캡션 생성
- `semantic`: OCR/태그/사람/장소/신호/임베딩 ref를 `search_documents`로 집계
- `search.planner`: 쿼리를 keyword/OCR/person/place/date/visual intent로 분해
- `search.vector`: `VectorIndexBackend` 인터페이스와 `LocalNumpyVectorIndex` 기본 구현
- `search`: FTS5 keyword search + OCR/tag/shadow document + CLIP vector 후보를 RRF로 결합

## Phase 2 Cycle Contract

- Phase 2는 Phase 1과 독립된 사이클로 돈다.
- Phase 1이 새 파일을 안정화하고 `thumb_done` 또는 `analysis_done`으로 만들면 Phase 2 대상이 된다.
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

- 역지오코딩: Nominatim 셀프호스트 또는 Kakao 로컬 API 중 택1, 환경변수 스위치
- 임베딩 모델: SigLIP2 또는 OpenCLIP, HuggingFace weight 오프라인 캐시.
  배포 시 기본 앱에 포함하지 않고 optional local AI pack으로 분리한다.
- LLM 쿼리 파서: 로컬 Ollama(Qwen 2.5) 또는 외부 API, JSON 스키마 강제
- 벡터 인덱스: `VectorIndexBackend` 기준. 현재 local NumPy exact search, 확장 시 FAISS/LanceDB/Qdrant adapter

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
