# Status Snapshot
updated 2026-04-23

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
- 로컬 서비스 상태를 보는 `/dashboard` 페이지가 추가됐다. 현재 샘플 DB 기준 미리보기 서버는 `127.0.0.1:8001`에서 확인 가능하다.
- 기본 실행 모델은 `native local service + localhost web UI`다. macOS는 메뉴바 셸을 얹고, Docker는 선택 배포 옵션으로 둔다.
- HEIC 처리에는 macOS `sips` fallback을 사용한다.
- 멀티에이전트 운영체계는 `Orchestrator / Planner / Developer / QA` 4역할로 유지한다.
- 실행 정책은 `AGENTS.md`, `AGENTS_LIGHT.md`, `.codex/context/ALL_TASKS.md` 3계층으로 정리됐다.
- 태스크 범위는 `T1~T24`다. `T1~T15`는 캐싱(스테이지-1), `T16~T24`는 의미 가공·자연어 검색(스테이지-2)이다.
- GitHub PR stage는 `agent:dev -> agent:qa -> agent:planner-review -> agent:ready-to-merge` 순서다.
- GitHub live 자동 루프는 아직 미연결 상태다. 원인은 webhook secret/label/branch protection 미설정이다.

## 현재 작업

- active branch는 `feat/t1-t10-foundation`이다.
- 현재 포커스는 스테이지-2 코드(`T16~T23`) 검증과 `T24` QA 회귀 정리다.
- 샘플 서버는 `http://127.0.0.1:8001/gallery`와 `http://127.0.0.1:8001/dashboard`에서 확인 가능하다.
- 다음 세션 시작 시 해야 할 첫 작업은 스테이지-2 변경 파일과 QA 결과를 점검하고, 안정화되면 커밋/푸시하는 것이다.

## 남은 작업

- `T16~T23` 변경 파일 검증
- `T24` NL 회귀 시나리오 자동화 마감
- 스테이지-2 변경분 커밋 및 원격 푸시
- `SCENARIO_VALIDATION_MATRIX`와 최종 QA 결과 동기화
- GitHub label 초기 세팅
- `AGENT_WEBHOOK_URL`, `AGENT_WEBHOOK_SECRET` secret 등록
- branch protection에서 `Agent PR Gate` 체크 필수화
- GitHub live 루프 연결 후 planner review / merge 루프 정식화
- 전체 라이브러리 full ingest는 semantic 스택 검증 완료 후 재개

## 참고 파일

- [AGENTS.md](/Users/dongeui/Desktop/chance/photomine/AGENTS.md)
- [AGENTS_LIGHT.md](/Users/dongeui/Desktop/chance/photomine/AGENTS_LIGHT.md)
- [.codex/context/ALL_TASKS.md](/Users/dongeui/Desktop/chance/photomine/.codex/context/ALL_TASKS.md)
- [docs/engineering/PLAN.md](/Users/dongeui/Desktop/chance/photomine/docs/engineering/PLAN.md)
- [docs/engineering/ARCHITECTURE.md](/Users/dongeui/Desktop/chance/photomine/docs/engineering/ARCHITECTURE.md)
- [docs/qa/SCENARIO_VALIDATION_MATRIX.md](/Users/dongeui/Desktop/chance/photomine/docs/qa/SCENARIO_VALIDATION_MATRIX.md)
- [docs/integrations/GITHUB_AGENT_WEBHOOKS.md](/Users/dongeui/Desktop/chance/photomine/docs/integrations/GITHUB_AGENT_WEBHOOKS.md)
- [app/api/status.py](/Users/dongeui/Desktop/chance/photomine/app/api/status.py)
- [app/api/gallery.py](/Users/dongeui/Desktop/chance/photomine/app/api/gallery.py)
