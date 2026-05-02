# AGENTS_LIGHT

정책 우선순위 요약과 런타임 치트시트. 전체 규정은 `AGENTS.md`에 있다.

## Policy Precedence

상충 시 위가 이긴다.

1. `AGENTS.md`
2. `AGENTS_LIGHT.md`
3. `.codex/context/ALL_TASKS.md`
4. `docs/product/STATUS_SNAPSHOT.md`
5. user prompt

## Objective Priority

1. correctness and reliability
2. task-bounded exploration
3. token efficiency

## Execution Rules

- active task는 항상 하나만 둔다.
- active owner도 항상 하나만 둔다.
- task 범위는 `ALL_TASKS.md` 기준으로 강제한다.
- `allowed_scope` 밖 파일이 필요하면 경로와 이유를 먼저 적는다.
- 구현 중 repo-wide scan 금지.
- Developer는 `app/**`만, QA는 `tests/**`만, Planner는 `docs/**`만 다룬다.
- 모든 개발/리뷰는 두 배포 경로를 확인한다:
  - `photome-base`: local AI pack 없이 기본 scan/gallery/status/search가 동작
  - `photome-local-ai-pack`: 모델 캐시 기반 CLIP/semantic 검색이 오프라인에서도 동작
- base runtime에서 PyTorch/open_clip/모델 weight가 필수 import가 되면 안 된다.
- 모델/provider 변경은 embedding version, concept/alias 변경은 auto tag version, search document 변경은 search version 검토 대상이다.

## Model Profile

- `Orchestrator`: `gpt-5.4`, reasoning `high`
- `Planner`: `gpt-5.4`, reasoning `medium`
- `Developer`: `gpt-5.4`, reasoning `medium`
- `QA`: `gpt-5.4`, reasoning `medium`

## GitHub Stage

- `agent:dev`
- `agent:qa`
- `agent:planner-review`
- `agent:changes-requested`
- `agent:ready-to-merge`
