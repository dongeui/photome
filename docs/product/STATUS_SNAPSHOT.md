# Status Snapshot
updated 2026-05-04 (검색 강화 Round 1~3 + B안 태그 서브타입 + AI pack UI 완료)

이 문서는 아래 4가지만 남긴다.

1. 확정된 사실
2. 현재 작업
3. 남은 작업
4. 참고 파일

지난 토론 과정과 중간 추론은 남기지 않는다.

## 확정된 사실

- `T1~T15` 스테이지-1 캐싱 파이프라인은 로컬에서 동작 확인을 마쳤다. `FastAPI`, `DB`, `scanner`, `fingerprint`, `metadata`, `thumbnail`, `video keyframe`, `incremental scan`, `job pipeline`, `scheduler`, `API`, `status`가 연결돼 있다.
- 실제 NAS 샘플 100개 스모크에서 `created=100`, `processed=100`, `errors=0`을 확인했다. 결과는 이미지 `thumb_done` 45개, 비디오 `analysis_done` 55개였다.
- 사람/장소 태그는 썸네일이 아니라 `원본 이미지 + EXIF GPS`, 비디오는 `keyframe` 기준으로 생성한다.
- 얼굴 검출·임베딩은 OpenCV YuNet + SFace를 사용하고, centroid 기반 사람 클러스터링으로 같은 얼굴은 다른 `file_id`에서도 같은 `person-XXXXXX` 태그를 재사용한다.
- EXIF GPS 기반 `place`(coarse)와 `place_detail`(exact) 태그 생성이 구현돼 있다.
- 스테이지-2 구현은 `T16~T23`까지 코드가 들어가 있다. 범위는 reverse geocoding, embedding/search, OCR, caption, NL query parser, `/search` 하이브리드 검색이다.
- `Phase 1 polling`과 `Phase 2 semantic scheduling`은 설정과 `/status` 응답에 반영돼 있다. semantic 버전 세트는 `place/person/ocr/caption/embedding/auto_tags/search` 기준으로 노출된다.
- Phase 1 dashboard에서 런타임 source root를 입력해 `POST /scan?source_roots=...`로 스캔할 수 있다. 경로 미입력 시 환경변수 source root를 그대로 사용한다.
- Phase 2는 독립 semantic maintenance cycle로 돈다. `run_semantic_maintenance()`는 `search_documents`가 없거나 stale이거나 version mismatch인 media만 처리하며, non-blocking lock으로 중복 실행을 막는다.
- `search_documents`는 Phase 2 canonical search document다. 파일명, 상대 경로, annotation, OCR, tag, person, place, analysis signal, embedding ref를 집계한다.
- SQLite FTS5 virtual table이 keyword search acceleration layer로 붙었다. unicode61 테이블(`search_documents_fts`)과 trigram 테이블(`search_documents_fts_ko`)을 이중 운영한다. FTS 실패 시 기존 LIKE fallback을 유지한다.
- 수동 Phase 2 검증 endpoint는 `POST /scan/semantic-maintenance`다.
- `QueryPlanner`가 추가됐다. 쿼리를 keyword/OCR/person/place/date/visual intent로 분해하고 `/search` 응답 `meta.query_plan`에 노출한다.
- `VectorIndexBackend` 추상화와 `LocalNumpyVectorIndex` 기본 구현이 추가됐다. FAISS/LanceDB/Qdrant는 adapter로 확장한다.
- Phase 2 검색 종착지 기술 검토 문서는 `docs/engineering/PHASE2_SEARCH_TECH_REVIEW.md`다.
- 로컬 서비스 상태를 보는 `/dashboard` 페이지가 추가됐다. 현재 샘플 DB 기준 미리보기 서버는 `127.0.0.1:8001`에서 확인 가능하다.
- 기본 실행 모델은 `native local service + localhost web UI`다. macOS는 메뉴바 셸을 얹고, Docker는 선택 배포 옵션으로 둔다.
- 배포 모델은 `photome-base`와 optional `photome-local-ai-pack`으로 나눈다. 기본 앱에는 CLIP/OpenCLIP 모델 weight를 포함하지 않는다.
- 자연어 이미지 검색(`바다에서 찍은 사진`, `baby beach` 등)은 local AI pack 설치 후 활성화한다. 설치 후 분석은 로컬/오프라인 모드에서 실행 가능해야 한다.
- HEIC 처리에는 macOS `sips` fallback을 사용한다.
- 멀티에이전트 운영체계는 `Orchestrator / Planner / Developer / QA` 4역할로 유지한다.
- 실행 정책은 `AGENTS.md`, `AGENTS_LIGHT.md`, `.codex/context/ALL_TASKS.md` 3계층으로 정리됐다.
- 태스크 범위는 `T1~T24`다. `T1~T15`는 캐싱(스테이지-1), `T16~T24`는 의미 가공·자연어 검색(스테이지-2)이다.
- GitHub PR stage는 `agent:dev -> agent:qa -> agent:planner-review -> agent:ready-to-merge` 순서다.
- GitHub live 자동 루프는 아직 미연결 상태다. 원인은 webhook secret/label/branch protection 미설정이다.

### 검색 강화 (Phase 2 Round 1~3) — 완료

- **DB-driven 태그 어휘** (`TagVocabularyCache`): 5분 TTL로 DB Tag 테이블을 읽어 플래너에 주입. 하드코딩 PLACE_TERMS 의존 제거.
- **KoNLPy 형태소 분석** (`tokenizer.py`): `korean_nouns()` — konlpy 설치 시 Okt/Mecab, 미설치 시 heuristic fallback. `PHOTOME_TOKENIZER` 환경변수로 제어.
- **복합 토큰 분해** (`_split_compound_token`): "제주도갔던사진" → ["제주도", "사진"] 형식 heuristic 분해.
- **멀티채널 RRF 보정**: 2채널 매칭 +15%, 3채널 +30% 보너스.
- **FAISS 스레드 안전**: `update_session()` classmethod로 세션 교체 시 lock 보호.
- **N+1 제거** (`_batch_load_supplements`): tags/face_count/OCR/analysis를 파일 목록 단위 배치 로드.
- **근접 중복 제거** (`remove_near_duplicates`): 3초 내 연속 촬영은 최고 점수 1장만 유지.
- **날짜 soft scoring**: 쿼리 날짜 범위 내 OCR/shadow 결과에 +0.08 보너스.
- **zero-result fallback**: 날짜 파싱으로 결과 0건 시 날짜 필터 제거 후 재검색.
- **FTS 쓰기 최적화**: content hash 비교로 텍스트 미변경 시 FTS 재작성 스킵.
- **캐시 스레드 안전**: `_cache_lock`으로 query cache 동시 쓰기 보호.
- **퇴화 쿼리 차단**: 유의미한 글자 없는 쿼리 조기 반환.
- **CLIP 텍스트 길이 제한**: 200자 초과 시 truncate (77 토큰 한도 대응).
- **FTS 누락 경고**: 프로세스당 1회 WARNING 로그 (DB 마이그레이션 미실행 감지용).
- **구조적 검색 로그**: `logger.debug`로 쿼리·채널별 결과 수 기록.
- **TagVocabularyCache.invalidate() classmethod**: 세션 인스턴스 없이 호출 가능하도록 수정.
- **Fuzzy Korean correction**: 0결과 시 DB 태그 bigram Jaccard(≥0.5)로 토큰 교정 후 재검색. 하드코딩 없음.
- **CLIP weight 재분배**: clip_results가 비면 clip 가중치를 shadow로 이동 — CLIP 미설치/임베딩 미생성 상황 자동 대응.
- **SearchEvent 테이블**: 모든 검색 쿼리·모드·의도·결과수·fallback 경로 기록. 미래 자동 가중치 튜닝 데이터 축적.
- **결과 다양성 cap**: 동일 날짜 최대 5장 제한 — 같은 이벤트가 결과 상단 독점 방지.
- **FAISS fetch_k 동적 조정**: 날짜 범위 폭에 따라 fetch multiplier를 2~20배 자동 산정.
- **HINT 세트 YAML화**: `face_hints`, `text_hints`, `screen_hints`, `travel_hints`, `celebration_hints`를 `vocab_seed.yaml`로 이동. 코드 하드코딩 제거.

### B안 auto tag 서브타입 — 완료

- **`tag_type="auto"` → 5개 서브타입 분리**: `auto_person`, `auto_scene`, `auto_object`, `auto_event`, `auto_screen`.
- **`ClipConcept.tag_type` 필드 추가**: 25개 CLIP 개념에 각각 서브타입 할당.
- **`AUTO_TAG_TYPES` 상수**: `app/services/analysis/auto_tags.py`에서 export. pipeline에서 "auto" 문자열 하드코딩 제거.
- **TagVocabularyCache 분류 확장**: `auto_scene` → `_PLACE_TAG_TYPES`, `auto_person` → `_PERSON_TAG_TYPES`. QueryPlanner가 CLIP 기반 장소/인물 태그를 인식.
- **`search_documents` people/places 필드**: `auto_person` → `people_json`, `auto_scene` → `places_json` 포함.
- **`semantic_auto_tag_version`**: 기본값 `auto-v1` → `auto-v2`. 다음 semantic maintenance 사이클에서 전체 재처리 트리거.
- **`_migrate_auto_tag_types()`**: 서버 시작 시 자동 실행. 기존 `tag_type="auto"` 행을 서브타입으로 변환.
- **`ix_tags_type_value` 복합 인덱스**: `tags(tag_type, tag_value)`. 기존 DB는 `_ensure_tag_indexes()`로 시작 시 자동 적용.

### AI pack 웹 UI — 완료

- **`/ai-pack/status`** `GET`: 패키지 설치 여부·모델 캐시 여부·stage 반환.
- **`/ai-pack/prepare`** `POST`: 백그라운드 스레드로 모델 다운로드 시작. 중복 실행 방지.
- **`/ai-pack/progress`** `GET`: 폴링용 — `needs_packages | needs_download | downloading | ready | error`.
- **Dashboard 3-step UI**: Step 1 패키지 안내, Step 2 모델 다운로드(버튼→스피너→완료), Step 3 활성화 env var. 2초 폴링, 페이지 새로고침 없음.

## 현재 작업

- active branch는 `main`이다.
- 로컬 서버는 `http://127.0.0.1:8000/gallery`와 `http://127.0.0.1:8000/dashboard`에서 확인 가능하다.
- 서버 실행 설정은 Phase 1 polling off, Phase 2 semantic scheduler on, CLIP off, face analysis off다.
- 배포/모델팩 분리 정책은 `docs/ops/PACKAGING_STRATEGY.md`에 정리됐다.

## 남은 작업

- synthetic NL QA 확장: 실제 이미지 없이 planner/channel/ranking 회귀를 강화
- vector adapter 후보 성능 검토: FAISS/LanceDB/Qdrant 중 첫 외부 adapter 결정
- CLIP provider 설정을 model name/pretrained/cache root/env 기반으로 분리
- people/person group API 구현
- reverse geocoding provider/cache 구현
- VLM caption adapter 구현
- `T24` NL 회귀 시나리오 자동화 마감
- `SCENARIO_VALIDATION_MATRIX`와 최종 QA 결과 동기화
- GitHub label 초기 세팅
- `AGENT_WEBHOOK_URL`, `AGENT_WEBHOOK_SECRET` secret 등록
- branch protection에서 `Agent PR Gate` 체크 필수화
- GitHub live 루프 연결 후 planner review / merge 루프 정식화
- 전체 라이브러리 full ingest는 semantic 스택 검증 완료 후 재개
- 빌드 프로파일 실체화: base / local-ai-pack 패키지 분리, Docker 이미지 분리
- 릴리즈 메타데이터 및 스모크 테스트 스위트 (CLIP 없는 시작, offline download 차단 검증)
- ARCHITECTURE.md auto tag 서브타입 분류 체계 및 bootstrap 마이그레이션 섹션 추가

## 참고 파일

- [AGENTS.md](/Users/dongeui/Desktop/chance/photome/AGENTS.md)
- [AGENTS_LIGHT.md](/Users/dongeui/Desktop/chance/photome/AGENTS_LIGHT.md)
- [docs/engineering/PLAN.md](/Users/dongeui/Desktop/chance/photome/docs/engineering/PLAN.md)
- [docs/engineering/ARCHITECTURE.md](/Users/dongeui/Desktop/chance/photome/docs/engineering/ARCHITECTURE.md)
- [docs/engineering/PHASE2_SEARCH_TECH_REVIEW.md](/Users/dongeui/Desktop/chance/photome/docs/engineering/PHASE2_SEARCH_TECH_REVIEW.md)
- [docs/ops/PACKAGING_STRATEGY.md](/Users/dongeui/Desktop/chance/photome/docs/ops/PACKAGING_STRATEGY.md)
- [docs/qa/SCENARIO_VALIDATION_MATRIX.md](/Users/dongeui/Desktop/chance/photome/docs/qa/SCENARIO_VALIDATION_MATRIX.md)
- [docs/integrations/GITHUB_AGENT_WEBHOOKS.md](/Users/dongeui/Desktop/chance/photome/docs/integrations/GITHUB_AGENT_WEBHOOKS.md)
- [app/api/scan.py](/Users/dongeui/Desktop/chance/photome/app/api/scan.py)
- [app/api/status.py](/Users/dongeui/Desktop/chance/photome/app/api/status.py)
- [app/services/processing/pipeline.py](/Users/dongeui/Desktop/chance/photome/app/services/processing/pipeline.py)
- [app/services/semantic/catalog.py](/Users/dongeui/Desktop/chance/photome/app/services/semantic/catalog.py)
- [app/services/search/backend.py](/Users/dongeui/Desktop/chance/photome/app/services/search/backend.py)
