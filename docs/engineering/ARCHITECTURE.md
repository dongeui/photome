# Architecture

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

Stage 2 — semantic enrichment and NL search (planned, T16~T24):

- `geocoding`: GPS 좌표 → 지명 계층 태그, 결과 캐시
- `embedding`: 이미지 멀티모달 임베딩(CLIP/SigLIP) 추출 및 저장
- `ocr` (선택): 이미지 내 텍스트 추출
- `caption` (선택): VLM 기반 자연어 캡션 생성
- `search`: 자연어 쿼리 파서 + 하이브리드 검색(메타 필터 + 벡터 랭킹)

## Pipeline States

Stage 1 (현재):

`discovered -> waiting_stable -> queued -> metadata_done -> thumb_done -> preview_done -> analysis_done`

Stage 2 (T17 이후 확장):

`analysis_done -> embedding_done` (이미지 한정, VLM 캡션/OCR은 병행 상태로 취급)

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

## External Dependencies (Stage 2)

- 역지오코딩: Nominatim 셀프호스트 또는 Kakao 로컬 API 중 택1, 환경변수 스위치
- 임베딩 모델: SigLIP2 또는 OpenCLIP, HuggingFace weight 오프라인 캐시
- LLM 쿼리 파서: 로컬 Ollama(Qwen 2.5) 또는 외부 API, JSON 스키마 강제
- 벡터 인덱스: 초기 in-memory NumPy, 확장 시 LanceDB 또는 Qdrant
