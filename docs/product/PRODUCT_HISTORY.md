# Product History
updated 2026-04-23

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
