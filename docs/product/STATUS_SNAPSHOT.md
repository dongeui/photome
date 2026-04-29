# Status Snapshot
updated 2026-04-29

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
- `Phase 1 polling`과 `Phase 2 semantic scheduling`은 설정과 `/status` 응답에 반영돼 있다. semantic 버전 세트는 `place/person/ocr/caption/embedding/search` 기준으로 노출된다.
- Phase 1 dashboard에서 런타임 source root를 입력해 `POST /scan?source_roots=...`로 스캔할 수 있다. 경로 미입력 시 환경변수 source root를 그대로 사용한다.
- Phase 2는 독립 semantic maintenance cycle로 돈다. `run_semantic_maintenance()`는 `search_documents`가 없거나 stale이거나 version mismatch인 media만 처리하며, non-blocking lock으로 중복 실행을 막는다.
- `search_documents`는 Phase 2 canonical search document다. 파일명, 상대 경로, annotation, OCR, tag, person, place, analysis signal, embedding ref를 집계한다.
- SQLite FTS5 virtual table `search_documents_fts`가 keyword search acceleration layer로 붙었다. FTS 실패 시 기존 LIKE fallback을 유지한다.
- 수동 Phase 2 검증 endpoint는 `POST /scan/semantic-maintenance`다.
- `QueryPlanner`가 추가됐다. 쿼리를 keyword/OCR/person/place/date/visual intent로 분해하고 `/search` 응답 `meta.query_plan`에 노출한다.
- `VectorIndexBackend` 추상화와 `LocalNumpyVectorIndex` 기본 구현이 추가됐다. FAISS/LanceDB/Qdrant는 adapter로 확장한다.
- Phase 2 검색 종착지 기술 검토 문서는 `docs/engineering/PHASE2_SEARCH_TECH_REVIEW.md`다.
- 로컬 서비스 상태를 보는 `/dashboard` 페이지가 추가됐다. 현재 샘플 DB 기준 미리보기 서버는 `127.0.0.1:8001`에서 확인 가능하다.
- 기본 실행 모델은 `native local service + localhost web UI`다. macOS는 메뉴바 셸을 얹고, Docker는 선택 배포 옵션으로 둔다.
- HEIC 처리에는 macOS `sips` fallback을 사용한다.
- 멀티에이전트 운영체계는 `Orchestrator / Planner / Developer / QA` 4역할로 유지한다.
- 실행 정책은 `AGENTS.md`, `AGENTS_LIGHT.md`, `.codex/context/ALL_TASKS.md` 3계층으로 정리됐다.
- 태스크 범위는 `T1~T24`다. `T1~T15`는 캐싱(스테이지-1), `T16~T24`는 의미 가공·자연어 검색(스테이지-2)이다.
- GitHub PR stage는 `agent:dev -> agent:qa -> agent:planner-review -> agent:ready-to-merge` 순서다.
- GitHub live 자동 루프는 아직 미연결 상태다. 원인은 webhook secret/label/branch protection 미설정이다.

## 현재 작업

- active branch는 `main`이다.
- 현재 포커스는 Phase 2 검색 기반 강화다.
- 로컬 서버는 `http://127.0.0.1:8000/gallery`와 `http://127.0.0.1:8000/dashboard`에서 확인 가능하다.
- 서버 실행 설정은 Phase 1 polling off, Phase 2 semantic scheduler on, CLIP off, face analysis off다.

## 남은 작업

- synthetic NL QA 확장: 실제 이미지 없이 planner/channel/ranking 회귀를 강화
- vector adapter 후보 성능 검토: FAISS/LanceDB/Qdrant 중 첫 외부 adapter 결정
- people/person group API 구현
- reverse geocoding provider/cache 구현
- VLM caption adapter 구현
- `T24` NL 회귀 시나리오 자동화 마감
- Phase 2 변경분 커밋 및 원격 푸시
- `SCENARIO_VALIDATION_MATRIX`와 최종 QA 결과 동기화
- GitHub label 초기 세팅
- `AGENT_WEBHOOK_URL`, `AGENT_WEBHOOK_SECRET` secret 등록
- branch protection에서 `Agent PR Gate` 체크 필수화
- GitHub live 루프 연결 후 planner review / merge 루프 정식화
- 전체 라이브러리 full ingest는 semantic 스택 검증 완료 후 재개

## 참고 파일

- [AGENTS.md](/Users/dongeui/Desktop/chance/photome/AGENTS.md)
- [AGENTS_LIGHT.md](/Users/dongeui/Desktop/chance/photome/AGENTS_LIGHT.md)
- [docs/engineering/PLAN.md](/Users/dongeui/Desktop/chance/photome/docs/engineering/PLAN.md)
- [docs/engineering/ARCHITECTURE.md](/Users/dongeui/Desktop/chance/photome/docs/engineering/ARCHITECTURE.md)
- [docs/engineering/PHASE2_SEARCH_TECH_REVIEW.md](/Users/dongeui/Desktop/chance/photome/docs/engineering/PHASE2_SEARCH_TECH_REVIEW.md)
- [docs/qa/SCENARIO_VALIDATION_MATRIX.md](/Users/dongeui/Desktop/chance/photome/docs/qa/SCENARIO_VALIDATION_MATRIX.md)
- [docs/integrations/GITHUB_AGENT_WEBHOOKS.md](/Users/dongeui/Desktop/chance/photome/docs/integrations/GITHUB_AGENT_WEBHOOKS.md)
- [app/api/scan.py](/Users/dongeui/Desktop/chance/photome/app/api/scan.py)
- [app/api/status.py](/Users/dongeui/Desktop/chance/photome/app/api/status.py)
- [app/services/processing/pipeline.py](/Users/dongeui/Desktop/chance/photome/app/services/processing/pipeline.py)
- [app/services/semantic/catalog.py](/Users/dongeui/Desktop/chance/photome/app/services/semantic/catalog.py)
- [app/services/search/backend.py](/Users/dongeui/Desktop/chance/photome/app/services/search/backend.py)
