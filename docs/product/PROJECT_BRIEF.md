# Project Brief

## Product

photomine is a local-first personal photo home server for media stored on a NAS.

## Goal

Build a modular backend that can index, track, process, and serve metadata for photos and videos without mutating originals.

## Core Value

- stable file identity even when paths change
- resilient processing despite NAS downtime
- rebuildable derived storage
- backend-first structure for future web and mobile apps

## Runtime Model

- primary runtime: native local service on the host machine
- primary access surface: localhost web UI
- macOS preferred shell: menu bar controller + localhost web UI
- Windows baseline shell: localhost web UI
- Docker is supported as an optional deployment mode, not the primary runtime
- default distribution is `photome base`; local AI image search is an optional
  `photome local-ai-pack`, not bundled into the base installer
- the source repository remains unified; versions are split by release artifact
  and runtime capability, not by project fork

## In Scope

Stage 1 — 캐싱 기반 (T1~T15):

- FastAPI backend
- scanner
- file identity and path remap
- metadata db
- thumbnail generation
- lazy previews
- video keyframes
- polling scheduler
- face detection, embedding, person clustering
- EXIF GPS place 태그

Stage 2 — 의미 가공 및 자연어 검색 (T16~T24):

- GPS 역지오코딩 (좌표 → 지명 계층 태그)
- CLIP/SigLIP 이미지 임베딩 (optional local-ai-pack)
- 벡터 검색 레이어
- 사람 라벨/그룹 관리 API
- 자연어 쿼리 파서
- 하이브리드 검색 엔드포인트 (`/search`)
- (선택) OCR, VLM 자동 캡션

## Out of Scope

- mutating originals
- full video transcoding
- path-based identity
- cloud-first dependency
- mandatory AI model bundle in the base install
- 클라이언트 앱(웹/모바일 UI)은 본 리포지터리 범위 아님 (백엔드 우선)
- 외부 사진 공유/소셜 기능
- Docker-only deployment strategy
