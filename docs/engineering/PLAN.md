# Engineering Plan

## Delivery Phases

### Phase 0 - Collaboration Ops

- [x] Git 저장소 초기화
- [x] 멀티에이전트 규칙 작성
- [x] GitHub PR stage loop 정의
- [x] webhook/dispatch 문서화

### Phase 1 - Foundation

- [x] T1 project scaffold
- [x] T2 config system
- [x] T3 SQLAlchemy models + DB bootstrap

### Phase 2 - Ingest Core

- [x] T4 NAS scanner
- [x] T5 fingerprint + file identity
- [x] T6 metadata extraction

### Phase 3 - Processing

- [x] T7 thumbnail generation
- [x] T8 video keyframes
- [x] T9 derived asset registry

### Phase 4 - Orchestration

- [x] T10 incremental scan logic
- [x] T11 background job pipeline
- [x] T12 scheduler

### Phase 5 - API and Reliability

- [x] T13 FastAPI endpoints
- [x] T14 status/reporting
- [ ] T15 QA scenario coverage  (active)

### Phase 6 - Semantic Enrichment

자연어 검색을 위한 2차 가공 파이프라인을 확장한다. 기존 캐시(T1~T14) 위에 의미 신호를 얹는 단계다.

- [ ] T16 reverse geocoding (좌표 → 지명 계층 태그)
- [ ] T17 CLIP image embedding (멀티모달 임베딩)
- [ ] T18 vector search layer (텍스트 → top-K 벡터 조회)
- [ ] T19 person labeling API (자동 person 라벨 + 그룹)
- [x] T20 OCR extraction (이미지 내 한국어 텍스트)
- [ ] T21 VLM auto-caption (이미지별 한국어 캡션)
- [x] T21a search document materialization (OCR/태그/사람/장소/신호/임베딩 ref 집계)
- [x] T21b Phase 2 semantic maintenance cycle (missing/stale/version mismatch만 처리)
- [x] T21c SQLite FTS5 keyword index for search documents
- [x] T21d vector index backend abstraction (`LocalNumpyVectorIndex` 기본 구현)

### Phase 7 - Natural Language Search

쿼리를 해석하고 하이브리드 검색으로 결과를 내는 단계다.

- [x] T22 NL query parser (쿼리 → 구조화 JSON)
- [x] T23 hybrid NL search endpoint (메타 필터 + 벡터 랭킹)
- [ ] T24 NL QA scenario expansion (회귀 방지)

## Task Split Ready Table

| Task | Scope | Primary Role | Review Gate | Paths | Can Run In Parallel After |
|---|---|---|---|---|---|
| T1 | package scaffold and app entry layout | Developer | Planner | `app/**` | now |
| T2 | settings and path config | Developer | Planner | `app/core/**`, `config/**` | T1 |
| T3 | SQLAlchemy models and db session bootstrap | Developer | Planner | `app/db/**`, `app/models/**` | T1 |
| T4 | recursive NAS scan and media detection | Developer | QA | `app/services/scanner/**` | T2, T3 |
| T5 | fingerprint, `file_id`, path remap primitives | Developer | QA | `app/services/fingerprint/**` | T2, T3 |
| T6 | image/video metadata extraction | Developer | QA | `app/services/metadata/**` | T2, T3 |
| T7 | thumbnail generation and derived path layout | Developer | QA | `app/services/thumbnail/**` | T5, T6 |
| T8 | video keyframe extraction | Developer | QA | `app/services/video/**` | T6 |
| T9 | derived asset registration | Developer | QA | `app/services/processing/**`, `app/models/**` | T3, T7, T8 |
| T10 | new/modified/moved/deleted detection | Developer | QA | `app/services/scanner/**`, `app/services/fingerprint/**` | T4, T5 |
| T11 | stateful job pipeline | Developer | QA | `app/workers/**`, `app/services/processing/**` | T7, T9, T10 |
| T12 | polling/full scan scheduler | Developer | QA | `app/scheduler/**` | T10, T11 |
| T13 | `/media`, `/media/{id}`, `/scan`, `/status` | Developer | QA + Planner | `app/api/**` | T3, T11 |
| T14 | health/status aggregation | Developer | QA | `app/api/**`, `app/core/**` | T11, T12 |
| T15 | regression and edge-case suite | QA | Planner | `tests/**` | T4 onward |
| T16 | GPS → 지명 계층 태그 확장 | Developer | QA | `app/services/geocoding/**`, `app/services/processing/pipeline.py`, `config/**` | T6, T11 |
| T17 | CLIP 이미지 임베딩 추출 + derived 레지스트리 연동 | Developer | QA | `app/services/embedding/**`, `app/services/processing/pipeline.py`, `app/core/contracts.py` | T7 |
| T18 | 벡터 top-K 검색 레이어 | Developer | QA | `app/services/search/vector.py`, `app/services/embedding/**` | T17 |
| T19 | person 라벨/그룹 관리 API | Developer | QA + Planner | `app/api/people.py`, `app/services/processing/registry.py`, `app/models/person.py` | T11 |
| T20 | OCR 텍스트 추출 및 태그화 | Developer | QA | `app/services/ocr/**`, `app/services/processing/pipeline.py` | T7 |
| T21 | VLM 자동 캡션 | Developer | QA | `app/services/caption/**`, `app/services/processing/pipeline.py` | T7, T17 |
| T22 | 자연어 쿼리 파서 | Developer | QA | `app/services/search/parser.py`, `config/**` | 없음 |
| T23 | `/search` 하이브리드 검색 엔드포인트 | Developer | QA + Planner | `app/api/search.py`, `app/services/search/hybrid.py` | T16, T17, T18, T19, T22 |
| T24 | NL 검색 회귀 시나리오 자동화 | QA | Planner | `tests/**` | T23 |

## PR Slicing Rules

- PR 하나에는 task 하나를 기본으로 한다.
- `T4 scanner`와 `T5 fingerprint`는 분리한다.
- `T7 thumbnail`과 `T8 video keyframes`는 분리한다.
- `T13 API`와 `T12 scheduler`는 분리한다.
- `T16 geocoding`과 `T17 embedding`은 파이프라인 훅이 겹치지 않으므로 병렬 PR 허용.
- `T17 embedding`과 `T18 vector search`는 분리한다 (데이터 생성 vs. 쿼리 레이어).
- `T22 parser`와 `T23 /search endpoint`는 분리한다.
- `T19 people API`는 schema 영향이 있으므로 단독 PR로 한다.
- Planner 리뷰 없이 Developer가 범위를 키우지 않는다.

## First Execution Order

1. `T1` scaffold
2. `T2` config
3. `T3` db
4. `T4` scanner
5. `T5` fingerprint
6. `T6` metadata
7. `T7` thumbnail
8. `T10` incremental scan
9. `T11` jobs
10. `T13` API

## Second Execution Order

Phase 2는 Phase 1과 별도 사이클로 돈다. Phase 1이 새 사진을 `thumb_done`/`analysis_done` 상태로 만든 뒤, Phase 2는 매 사이클마다 아직 semantic 결과가 없거나 버전이 맞지 않거나 원본 semantic source가 갱신된 항목만 처리한다.

1. [x] `search_documents` 정규화 테이블 생성
2. [x] `run_semantic_maintenance()` 사이클 구현 및 중복 실행 lock 추가
3. [x] SQLite FTS5 keyword index 연결
4. [x] `T18` vector index abstraction: local NumPy → FAISS/LanceDB/Qdrant 교체 가능 구조
5. [x] `T22` structured query planner: keyword/OCR/person/place/date/visual intent 분리
6. [ ] `T24` 이미지 없이도 돌 수 있는 synthetic NL scenario QA 확장
7. [ ] `T19` people/person group API
8. [ ] `T16` reverse geocoding provider/cache
9. [ ] `T21` VLM caption adapter

## Phase 6/7 Risk Notes

- Phase 2 검색 종착지 설계는 `docs/engineering/PHASE2_SEARCH_TECH_REVIEW.md`를 기준으로 한다.
- 모델 배포: 초기에는 서버 호스트(Mac mini)에서 CPU 추론 가정. GPU 워커 분리는 T21 이후 리소스 병목이 확인되면 별도 태스크로 분할.
- 스토리지: CLIP 임베딩은 기존 `embeddings_root` 하위에 shard 구조로 저장한다. 얼굴 임베딩(`embeddings/faces/v1/**`)과 디렉터리 네임스페이스를 분리한다 (`embeddings/clip/<version>/**`).
- 재처리: Phase 2는 전량 반복 실행하지 않는다. `SearchDocument.version`, `source_updated_at`, 각 semantic output version을 기준으로 missing/stale/version mismatch만 처리한다. 전량 재인덱싱은 명시적 rebuild 작업으로 분리한다.
- 검색 인덱스: `search_documents`가 canonical semantic 검색 문서이고, SQLite FTS5는 acceleration layer다. FTS5 실패 시 LIKE fallback을 유지한다.
- 쿼리 계획: 초기 `QueryPlanner`는 deterministic rule-based다. 쿼리를 keyword/OCR/person/place/date/visual intent로 분해하고 검색 meta에 노출한다. LLM planner는 schema와 회귀가 안정화된 뒤 optional provider로 붙인다.
- 벡터 검색: `VectorIndexBackend` 인터페이스를 기준으로 둔다. 현재 기본 구현은 exact local NumPy이며, 성능 임계점 확인 후 FAISS/LanceDB/Qdrant adapter를 추가한다.
- AGENTS.md 범위: 실행 정책은 `T1~T24`로 이미 확장되어 있다. Phase 6/7 도중 범위 이탈(T25 이상) 요청이 발생하면 Orchestrator가 먼저 `AGENTS.md`를 갱신한 뒤 Planner가 신규 태스크 카드를 추가한다.
