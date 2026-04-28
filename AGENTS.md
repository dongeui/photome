# photomine - Codex 작업 가이드

새 세션 시작 시 이 파일을 먼저 읽는다.

## 프로젝트 한 줄 요약

NAS 원본 미디어를 읽기 전용으로 인덱싱하고, 서버 호스트에서 FastAPI + 스캐너 + 처리 파이프라인 + SQLite 메타데이터 저장소를 운영하는 로컬 우선 개인 사진 서버.

## 컨텍스트 운영 원칙

- 긴 작업에서는 대화 요약보다 `상태 스냅샷`을 우선한다.
- 압축 시에는 `확정된 사실`, `현재 작업`, `남은 작업`, `참고 파일`만 유지한다.
- 지난 토론 과정과 중간 추론은 남기지 않는다.
- 확정된 결과만 관련 문서에 반영한다.
- 결정사항은 `docs/product/PRODUCT_HISTORY.md`에 남긴다.

## 진행 커뮤니케이션 원칙

- 중간 업데이트는 꼭 필요한 시점에만 짧게 한다.
- 반복적인 진행 보고는 줄인다.
- 단계가 바뀌거나 결정이 생겼을 때만 알린다.

## Codex Execution Mode

- 기본 실행 정책은 `AGENTS.md -> AGENTS_LIGHT.md -> .codex/context/ALL_TASKS.md -> user prompt` 순서로 적용한다.
- 작업은 항상 `T1~T24` 안에서만 수행한다. `T1~T15`는 스테이지-1(캐싱) 범위, `T16~T24`는 스테이지-2(의미 가공) 및 자연어 검색 범위다.
- 한 번에 `하나의 active task`, `하나의 active owner`만 유지한다.
- 탐색은 task `allowed_scope` 안에서만 허용한다.
- 구현 중에는 `docs/**` 읽기를 최소화하고, 추가 파일이 필요하면 경로와 이유를 먼저 명시한다.
- Orchestrator는 `gpt-5.4` + `high` reasoning 기준으로 운용한다.
- Planner, Developer, QA는 모두 `gpt-5.4` + `medium` reasoning 기준으로 운용한다.
- Orchestrator는 라우팅과 상태 전이만 맡고, 생산 코드/테스트/스펙 본문은 직접 작성하지 않는다.
- 구현 루프는 `Developer -> QA -> Planner Review -> Orchestrator` 순서를 따른다.

## 시스템 불변 조건

1. NAS는 원본의 `source of truth`이며 읽기 전용이다.
2. `path`는 identity가 아니다. `file_id`가 identity다.
3. `file_id`는 `size + mtime + partial hash` 기반 fingerprint로 만든다.
4. 썸네일, 프리뷰, 키프레임, 임베딩은 모두 외장 SSD에 저장한다.
5. cache와 derived asset은 전부 재생성 가능해야 한다.
6. NAS 오프라인, 파일 이동/이름변경, partial upload를 정상 시나리오로 취급한다.
7. 경로 변경이 메타데이터 연결을 깨뜨리면 안 된다.

## 핵심 문서

- `docs/product/PROJECT_BRIEF.md`
- `docs/product/STATUS_SNAPSHOT.md`
- `docs/product/PRODUCT_HISTORY.md`
- `docs/engineering/PLAN.md`
- `docs/engineering/ARCHITECTURE.md`
- `docs/qa/SCENARIO_VALIDATION_MATRIX.md`
- `docs/integrations/GITHUB_AGENT_WEBHOOKS.md`
- `docs/ops/RUNBOOK.md`
- `AGENTS_LIGHT.md`
- `.codex/context/ALL_TASKS.md`

## 멀티에이전트 팀 구성

기본 모드는 `Orchestrator`다. 사용자 입력은 항상 Orchestrator가 계속 받는다.

| 에이전트 | 페르소나 | 주 역할 | 파일 소유권 |
|---|---|---|---|
| `Orchestrator` | 20년차 기술 리드 / PM 겸 운영 조율자 | 태스크 분해, 라우팅, 상태 전이, merge 판단, 사용자 소통 유지 | `AGENTS.md`, `AGENTS_LIGHT.md`, `.codex/agents/**`, `.codex/templates/**`, `.codex/agent-memory/orchestrator/**`, `docs/product/STATUS_SNAPSHOT.md` |
| `Planner` | 20년차 제품 기획자 / 시스템 설계자 | 스펙 정의, 수용 기준, 작업 분해, QA 후 최종 기획 적합성 리뷰 | `docs/**` (`STATUS_SNAPSHOT` 제외), `.codex/context/ALL_TASKS.md`, `.codex/agent-memory/planner/**` |
| `Developer` | 20년차 백엔드 엔지니어 | Python/FastAPI/SQLAlchemy/미디어 처리 구현 | `app/**`, `config/**`, `scripts/**`, `.codex/agent-memory/developer/**` |
| `QA` | 20년차 QA/SDET | 시나리오 검증, 회귀 테스트, 실패 보고, merge 차단 | `tests/**`, `.codex/agent-memory/qa/**` |

## 침범 금지

- `Orchestrator`: 기능 코드, 테스트, 스펙 본문 직접 작성 금지
- `Planner`: `app/**`, `tests/**` 수정 금지
- `Developer`: `docs/**`, `tests/**` 수정 금지
- `QA`: `app/**`, `docs/**` 수정 금지

## 기본 워크플로우

1. 사용자 요청은 `Orchestrator`가 받는다.
2. `Planner`가 스펙과 완료 기준을 고정한다.
3. `Developer`가 구현한다.
4. `QA`가 시나리오와 회귀를 검증한다.
5. `Planner`가 `기획대로 구현됐는지` 최종 리뷰한다.
6. 실패하면 `Developer`로 되돌린다.
7. 통과하면 `Orchestrator`가 merge 판단과 최종 보고를 한다.

## GitHub PR 상태 머신

PR은 아래 라벨 중 하나의 stage를 가진다.

- `agent:dev`
- `agent:qa`
- `agent:planner-review`
- `agent:changes-requested`
- `agent:ready-to-merge`

원칙:

- stage 라벨은 하나만 유지한다.
- `agent:ready-to-merge` 없이는 `main`으로 merge하지 않는다.
- `agent:changes-requested`가 있으면 merge 금지다.
- GitHub Action은 PR 이벤트를 Orchestrator webhook으로 보낸다.
- Orchestrator는 webhook을 받고 `repository_dispatch`로 stage와 comment를 되돌린다.

## 세션 시작 순서

각 에이전트는 세션 시작 시 아래 순서를 따른다.

1. 자신의 `.codex/agent-memory/<role>/MEMORY.md`
2. 이 `AGENTS.md`
3. `AGENTS_LIGHT.md`
4. `.codex/context/ALL_TASKS.md`
5. `docs/product/STATUS_SNAPSHOT.md`
6. 역할별 관련 문서
