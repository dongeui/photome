# Product History
updated 2026-04-29 (세션 3)

## 2026-04-22

- 프로젝트 운영 방식을 `멀티에이전트 + GitHub PR 상태 머신`으로 확정했다.
- 역할은 `Orchestrator / Planner / Developer / QA` 4개로 최소화했다.
- Planner는 처음 스펙만 쓰는 역할이 아니라 QA 이후 `기획대로 구현됐는지`를 다시 확인하는 최종 리뷰 역할도 맡는다.
- PR merge 기준은 리뷰 개수보다 `stage label + gate check`를 우선으로 두기로 했다.
- PR 이벤트는 GitHub Action이 외부 webhook으로 보내고, 오케스트레이터는 `repository_dispatch`로 stage를 되돌리는 구조를 채택했다.
- 문서 운영은 `상태 스냅샷` 중심으로 유지하고, 토론 과정은 남기지 않기로 했다.

## 2026-04-23

- 실행 정책을 `AGENTS.md + AGENTS_LIGHT.md + .codex/context/ALL_TASKS.md` 구조로 정리했다.
- 역할별 모델은 모두 `gpt-5.4`로 고정하고, Orchestrator는 `high`, 나머지 역할은 `medium` reasoning으로 맞추기로 했다.
- 실제 NAS 샘플 100개 기준 스모크 검증으로 `T1~T14` 범위의 로컬 구현 동작을 확인했다.
- 다음 active task를 `T15 QA suite`로 고정했다.
- 제품 방향을 스테이지-2(의미 가공 + 자연어 검색)까지 확장하기로 결정했다. 핵심 유스케이스는 "작년에 가족들이랑 제주도갔던사진" 류의 복합 자연어 쿼리 대응이다.
- 태스크 범위를 `T1~T15`에서 `T1~T24`로 확장했다. `T16~T21`은 Phase 6 Semantic Enrichment(지오코딩, CLIP 임베딩, 벡터 검색, 사람 라벨링, OCR, VLM 캡션), `T22~T24`는 Phase 7 Natural Language Search(쿼리 파서, 하이브리드 `/search`, NL 회귀)다.
- 문서 다이어트: `.codex/RULES.md`를 canonical 포인터로 축소하고 규정은 `AGENTS.md`와 `AGENTS_LIGHT.md`로 일원화했다.
- `AGENTS_LIGHT.md`의 "Runtime Order"를 "Policy Precedence"로 개명해 "세션 시작 순서"와 개념 분리했다.
- 배포/실행 모델은 `native local service + localhost web UI`를 기본으로 두고, macOS는 메뉴바 셸을 우선 UX로, Docker는 선택 배포 옵션으로 두기로 했다.
- `Phase 1 polling + Phase 2 semantic scheduling` 이중 루프를 제품 운영 모델로 확정하고, semantic 버전 세트(`place/person/ocr/caption/embedding/search`)를 상태 API에 노출하기로 했다.
- `/dashboard`를 추가해 로컬 서비스에서 Phase 1/Phase 2 루프와 semantic 버전 세트를 시각적으로 확인할 수 있게 했다.

## 2026-04-29

- `/dashboard` Phase 1 카드에서 런타임 source root를 입력해 스캔할 수 있게 했다. 경로가 다른 환경에서도 앱 재시작 없이 `POST /scan?source_roots=...`로 테스트한다.
- Phase 2 운영 원칙을 “사이클마다 새 항목/누락 항목/stale 항목/version mismatch 항목만 처리”로 확정했다. 전량 재처리는 명시적 rebuild 작업으로 분리한다.
- `search_documents`를 Phase 2 canonical search document로 채택했다. OCR, tag, person, place, annotation, filename/path, analysis signal, embedding ref를 한 row로 집계한다.
- SQLite FTS5 `search_documents_fts`를 keyword search acceleration layer로 채택했다. FTS5는 canonical source가 아니며 실패 시 `search_documents` LIKE fallback을 유지한다.
- Phase 2 semantic maintenance cycle은 non-blocking lock으로 중복 실행을 막는다.
- 수동 검증 endpoint는 `POST /scan/semantic-maintenance`로 둔다.
- 초기 `QueryPlanner`는 rule-based로 두기로 했다. 쿼리를 keyword/OCR/person/place/date/visual intent로 분해하고 `/search` 응답 meta에 남겨 검색 품질 튜닝의 기준 데이터로 사용한다.
- 벡터 검색은 `VectorIndexBackend` 인터페이스를 기준으로 분리했다. 기본 구현은 local NumPy exact search이며, FAISS/LanceDB/Qdrant는 후속 adapter로 검토한다.
- Phase 2의 종착지 기술 방향을 `PHASE2_SEARCH_TECH_REVIEW.md`로 정리했다. 핵심 원칙은 `search_documents + FTS5 + vector adapter + rule-based planner`를 중심축으로 두고, OCR/caption/person/place/reranker를 provider 형태로 확장하는 것이다.

## 2026-04-29 (세션 2) — Phase 2 자연어 검색 고도화

목표: 자연어 검색을 더 자연스럽고 빠르게. S1~S5 + Fix 1~4 순서로 전체 구현.

### S1 — 쿼리 이해력 강화
- `query_translate.py`: 날짜/시간 표현 대폭 확장. 추가된 패턴: `지난주`, `저번주`, `이번주말`, `지난주말`, `N달전`, `N주전`, `N년전`, `어제`, `오늘`, `1월~12월`, `설날`, `추석`, `크리스마스`
- `query_translate.py`: LEXICON 확장. 가족 관계(형/오빠/누나/언니/동생 등), 서울 주요 지역(강남/홍대/명동 등), 활동(소풍/나들이/드라이브 등) 추가
- `planner.py`: PERSON_TERMS에 형/오빠/누나/언니/조카/남편/아내 등 추가
- `planner.py`: PLACE_TERMS에 강남/홍대/명동/해수욕장/동물원 등 추가
- `planner.py`: OCR_TERMS에 카카오톡/캡처/캡쳐/갈무리/인스타 등 추가
- `planner.py`: VISUAL_TERMS에 소풍/벚꽃/졸업식/돌잔치 등 추가
- `planner.py`: DATE_STOP_TERMS에 지난주/이번주말/설날/추석/오늘/어제 등 추가
- `planner.py`: `_intent()` 수정 — person+place 동시 존재 시 visual, date+person/place 조합도 visual로 분류. 기존에 "엄마랑 카페" 가 hybrid로 빠지던 edge case 수정
- `hybrid.py`: 동일 쿼리에 60초 TTL 인메모리 캐시 추가 (최대 256개, LRU 방출). debug/weight_overrides/date_from 제공 시 캐시 미적용

### S3 — 검색 속도 개선
- `hybrid.py`: shadow + CLIP 채널을 `ThreadPoolExecutor(max_workers=2)`로 병렬 실행. OCR은 effective_mode 결정에 필요하므로 여전히 먼저 실행. 예상 효과: 300~500ms → 100~200ms
- `vector.py`: `FaissVectorIndex` 추가 — lazy index build, thread-safe rebuild, normalized IndexFlatIP inner-product search. `faiss` 미설치 시 자동 skip
- `vector.py`: `build_vector_index()` 팩토리 추가. `PHOTOME_VECTOR_BACKEND=auto|faiss|numpy` 환경변수로 선택
- `backend.py`: `build_vector_index()` 팩토리 사용으로 교체
- `pyproject.toml`: `[faiss]` optional extra 추가 (`faiss-cpu>=1.7`)

### S2 — VLM Caption 통합
- `models/semantic.py`: `MediaCaption` 테이블 추가 (short_caption, objects_json, activities_json, setting, provider, version)
- `services/caption/__init__.py`: `CaptionResult` dataclass + `CaptionProvider` Protocol 정의
- `services/caption/moondream.py`: `Moondream2Provider` 구현 (2B int8, local, lazy-loaded, thread-safe, `PHOTOME_CAPTION_PROVIDER=moondream` 환경변수로 활성화)
- `services/caption/registry.py`: `get_caption_provider()` 팩토리
- `catalog.py`: `upsert_caption()` 메서드 추가. `upsert_search_document()`에서 caption 텍스트를 `semantic_text`에 포함
- `pipeline.py`: `_materialize_image_semantics()`에서 caption provider 호출
- `pyproject.toml`: `[caption]` optional extra 추가 (`moondream>=0.0.5`)

### S4 — Person/Place 컨텍스트 강화
- `models/semantic.py`: `GeocodingCache` 테이블 추가 (key, country, region, city, place, display_name)
- `services/geocoding/__init__.py`: `NominatimProvider` (OSM 공개 API, 1req/s 레이트 리밋, `PHOTOME_NOMINATIM_URL` 커스텀 가능)
- `services/geocoding/cached.py`: `CachedGeocodingService` — DB 캐시 래퍼 (precision=3 좌표 키)
- `pipeline.py`: `_materialize_place_tags()`에 geocoding 통합. `geocoding_enabled=True` + session 전달 시 도시/지역 이름 태그 추가. 기본값 off
- `api/people.py`: Person 관리 API 신규 — `GET /people`, `GET /people/{id}`, `PATCH /people/{id}`, `GET /people/{id}/media`
- `backend.py`: `_resolve_person_tag_ids()` 추가 — Person.display_name과 쿼리 토큰 매칭 → person 태그 ID로 변환해 이름으로 사람 검색 가능

### S5 — 랭킹 고도화
- `models/semantic.py`: `SearchWeightProfile` 테이블 추가 (intent, reason, w_ocr, w_clip, w_shadow)
- `api/search.py`: weight profile API 추가 — `GET /search/weights`, `PUT /search/weights/{intent}/{reason}`, `DELETE /search/weights/{intent}/{reason}`, `GET /search/weights/defaults`
- `backend.py`: `load_persisted_weights()` — DB에서 intent+reason 가중치 조회
- `hybrid.py`: persisted DB 가중치 → built-in default보다 우선. 요청별 override가 최우선
- `hybrid.py`: `set_match_explanations()` 강화 — 한국어 설명, 매칭 태그명, 얼굴 수, 장소 이름, 연/월 포함

### Fix 1 — FTS 한국어 품질 개선
- `bootstrap.py`: `search_documents_fts_ko` trigram FTS virtual table 추가 (SQLite 3.34+, 미지원 시 자동 skip)
- `catalog.py`: `_upsert_search_document_fts()`에서 unicode61 + trigram 두 테이블 동기 업데이트
- `backend.py`: `_search_by_fts_document()` 두 FTS 테이블 쿼리 후 file_id별 최소(최적) BM25 점수로 머지. "카카오"로 "카카오톡" 검색 가능

### Fix 2 — 쿼리 캐시 무효화
- `hybrid.py`: `clear_query_cache()` 함수 추가
- `pipeline.py`: semantic maintenance 완료(succeeded>0) 시 `clear_query_cache()` 호출 → 새 콘텐츠 즉시 검색 가능

### Fix 3 — FAISS 인덱스 자동 갱신
- `vector.py`: `_global_faiss_index` process-level 싱글턴 + `invalidate_global_vector_index()` 추가
- `build_vector_index()`에서 FAISS 선택 시 싱글턴에 저장; 매 호출마다 session 참조 갱신
- `pipeline.py`: semantic maintenance 완료 시 `invalidate_global_vector_index()` 호출 → 다음 검색에서 FAISS 재빌드

### Fix 4 — 복합 조건 검색
- `hybrid.py`: `_search_clip_variants()`에서 `plan.place_terms` 전체를 OR 필터로 반복 실행 (명시적 place_filter 없을 때). 제주+바다 같이 여러 장소어 포함 쿼리에서 더 많은 후보 확보
- `hybrid.py`: `apply_context_filter_boost()` 추가 — plan.place_terms/person_terms와 태그가 일치하는 결과에 rank_score 보너스 적용 (+0.08/장소, +0.06/사람)

### 변경된 파일 목록
`app/services/search/`: query_translate.py, planner.py, hybrid.py, backend.py, vector.py  
`app/services/semantic/`: catalog.py  
`app/services/processing/`: pipeline.py  
`app/services/caption/`: __init__.py (신규), moondream.py (신규), registry.py (신규)  
`app/services/geocoding/`: __init__.py (신규), cached.py (신규)  
`app/models/`: semantic.py, __init__.py  
`app/api/`: search.py, people.py (신규), router.py  
`app/db/`: bootstrap.py  
`pyproject.toml`

## 2026-04-29 (세션 3) — 버그 수정 + R3/R4 기능 추가

### P0/P1 버그 수정
- `hybrid.py`: `result.get("signals").get("face_count")` → `result.get("face_count")` (signals 키 존재 안 함)
- `hybrid.py`: `result.get("exif_datetime")` → `result.get("captured_at")` (_result_dict 실제 키명과 불일치)
- `backend.py`: FTS `rows` 타입 혼재 — `list[tuple[str, float]]`로 초기화부터 명시적 타입 강제
- `hybrid.py`: 사용되지 않는 `as_completed` 임포트 제거

### R3 — 유저 피드백 신호 + 재랭커 인터페이스
- `models/semantic.py`: `SearchFeedback` 테이블 추가 (file_id, action, query_hint, tag_correction)
  - action: `hide` | `promote` | `correct_tag`
  - query_hint: 비어있으면 전역, 값 있으면 해당 쿼리에 한정
- `api/search.py`: 피드백 API 추가
  - `POST /search/feedback`: 피드백 기록
  - `GET /search/feedback`: 피드백 목록 (action 필터 가능)
  - `DELETE /search/feedback/{id}`: 피드백 제거 (hide/promote 취소)
- `backend.py`: `load_feedback_sets()` — DB에서 (hidden_ids, promoted_ids) 로드
- `hybrid.py`: `search_with_meta()`에서 hidden 파일 스코어링 전 제거, promoted 파일에 +0.15 부스트
- `hybrid.py`: `RerankerProtocol` + `PassThroughReranker` 추가 — 플러그인 방식 재랭커 인터페이스
  - `HybridSearchService(backend, reranker=...)` 형태로 커스텀 재랭커 주입 가능

### R4 — 인덱스 재빌드 CLI + 헬스 페이지
- `api/search.py`: 인덱스 관리 API 추가
  - `POST /search/index/rebuild`: FTS 테이블(unicode61 + trigram) DROP → CREATE → 재적재, 쿼리 캐시 초기화, FAISS 싱글턴 무효화
  - `GET /search/index/status`: search_documents 수, FTS 로우 수, embedding 수, FAISS 로드 상태/ntotal 반환

### 변경된 파일 목록
`app/services/search/`: hybrid.py, backend.py  
`app/models/`: semantic.py, __init__.py  
`app/api/`: search.py
