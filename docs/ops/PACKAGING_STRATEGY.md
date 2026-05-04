# Packaging Strategy

Photome ships as a small local-first app by default. Heavy local AI assets are
split into an optional model pack so the base install stays practical.

## Package Lines

### photome base

Purpose: ingest, cache, browse, OCR/search where local tools are installed, and
run the localhost UI.

Includes:

- FastAPI app, dashboard, gallery, search UI
- SQLite schema and local processing pipeline
- Stage 1 scan/cache/media-fact pipeline: filesystem facts, MIME, dimensions, duration, capture time, EXIF GPS, coordinate/place tags
- Stage 2 search document, OCR, tag, date, place, feedback, and FTS logic
- lightweight Python dependencies

Excludes:

- CLIP/OpenCLIP weights
- large PyTorch runtime when the installer format can separate extras
- generated user data: SQLite DB, thumbnails, previews, keyframes, embeddings
- NAS photos and any derived library cache

Expected size target:

- native macOS base: hundreds of MB to about 1 GB depending on runtime bundling
- Docker base: larger, but still without model weights

### photome local-ai-pack

Purpose: enable natural-language visual search such as `바다에서 찍은 사진`,
`baby beach`, `여자`, `아기`, and similar semantic image queries.

Includes:

- `open_clip_torch`
- CPU PyTorch/TorchVision runtime for the target platform
- approved CLIP/OpenCLIP model cache
- cache verification script
- model/license notices

Initial recommended model:

- `ViT-B-32 / openai`

Future candidate:

- `ViT-B-32 / laion2b_s34b_b79k`

Expected size target:

- AI pack: roughly 2 GB to 5 GB depending on OS and PyTorch wheel size
- model weights alone: roughly 400 MB to 800 MB per model

## Runtime Modes

### Standard base mode

- `PHOTOME_CLIP_ENABLED=0`
- app remains fully usable without local AI pack
- Phase 1 still extracts file size, dimensions, capture time, EXIF GPS, and place facts
- search uses filename, OCR, custom tags, auto signal tags, date, GPS/place,
  annotations, and FTS/shadow documents
- Phase 2 skips CLIP embedding generation

### AI-enabled online preparation mode

Used once during setup when internet is available.

- installs or verifies local-ai-pack
- downloads model weights into `data/models`
- runs a model load smoke test
- records readiness in dashboard/status

### AI-enabled offline operation mode

Used for normal operation after model preparation.

- `PHOTOME_CLIP_ENABLED=1`
- `PHOTOME_OFFLINE_MODE=1`
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `HF_HOME=<photome-data>/models/hf`
- `TORCH_HOME=<photome-data>/models/torch`

If the model cache is missing, startup/search must not fail. Dashboard should
show `AI image search: not installed` or `model missing`, and Phase 2 should
skip CLIP work while keeping the rest of the library usable.

## Distribution Targets

### macOS

- primary: native local process controlled by a small shell/menu-bar app
- UI remains localhost web UI
- base installer should not include AI pack by default
- local-ai-pack is installed by an explicit "Enable local AI image search" flow

### Windows

- primary: localhost web UI backed by Docker or a packaged local process
- Docker images should be split:
  - `photome-base`
  - optional `photome-local-ai`
- user media, DB, derived cache, and model cache are mounted volumes

## Versioning Policy

The project does not need to split into separate source repositories now.
Use one repo with package profiles.

Version dimensions:

- app version: server/UI/API behavior, e.g. `0.2.0`
- semantic search version: search document schema/logic, e.g. `search-v3`
- embedding version: vector generation contract, e.g. `embedding-v1`
- auto tag version: concept/alias set, e.g. `auto-v3`
- model pack version: bundled local AI runtime/weights, e.g. `ai-pack-2026.05`

Compatibility rules:

- base app must run without local-ai-pack
- local-ai-pack version must declare compatible app version range
- model changes should not silently reuse incompatible embeddings
- changing model/provider/dimensions requires a new embedding version
- changing concept thresholds/aliases requires a new auto tag version
- changing search document composition requires a new search version

## Work Required To Split Versions

Keep one codebase. Add packaging boundaries instead of forking the project.

### 완료된 항목 (2026-05-04)

- ✅ runtime status fields: CLIP 패키지 설치 여부·모델 캐시 여부·offline 플래그 — `/status` JSON 및 dashboard에 노출
- ✅ model preparation CLI (`app/cli/local_ai_pack.py`):
  - `photome-local-ai-pack status` — 패키지·캐시 상태 출력
  - `photome-local-ai-pack prepare` — 온라인 모델 다운로드 및 로드 테스트
  - `photome-local-ai-pack verify-offline` — 오프라인 캐시 검증
- ✅ dashboard AI pack 3-step 설치 UI (`/ai-pack/status|prepare|progress`):
  - Step 1: 패키지 설치 안내 (pip 명령 + Copy 버튼)
  - Step 2: 모델 다운로드 버튼 → 백그라운드 다운로드 → 폴링 진행 표시
  - Step 3: 활성화 env var 안내 (PHOTOME_CLIP_ENABLED=1 + Copy 버튼)
  - 베이스 앱의 scan/gallery/search는 AI pack 없이도 차단되지 않음

### 남은 항목

- CLIP provider 설정 분리: model name / pretrained / cache root를 settings에서 통합 관리 (현재 CLI와 app settings가 별도 경로 사용)
- build profiles 실체화:
  - base 패키지 (no torch/open_clip)
  - local-ai-pack 패키지 (torch + open_clip + model cache)
  - Docker base 이미지
  - Docker AI 이미지 또는 AI 레이어
- release metadata:
  - app version
  - model pack version
  - supported OS/architecture
  - bundled dependency versions
- smoke tests:
  - base app starts with no CLIP package installed
  - CLIP enabled but model missing degrades cleanly
  - offline mode never attempts network model download
  - AI pack ready enables Phase 2 embedding and semantic visual search

## Decision

Do not split the source repository yet.

Split distribution artifacts:

- `photome-base`: default install
- `photome-local-ai-pack`: optional local AI capability

This keeps product complexity low while preserving the security claim:

> Natural-language AI image search is optional. When installed, models and
> analysis run locally and can operate fully inside an offline local network.
