# GitHub Agent Webhooks

## Goal

Use GitHub PR events as the handoff bus between Orchestrator, Planner, Developer, and QA.

## PR Stage Labels

Create these labels:

- `agent:dev`
- `agent:qa`
- `agent:planner-review`
- `agent:changes-requested`
- `agent:ready-to-merge`

## Repository Secrets

- `AGENT_WEBHOOK_URL`: external Orchestrator endpoint
- `AGENT_WEBHOOK_SECRET`: HMAC secret for webhook signature verification

## GitHub -> Orchestrator

Workflow: `.github/workflows/agent-pr-webhook.yml`

Events sent out:

- `pull_request`
- `pull_request_review`
- `issue_comment` on PRs

Headers:

- `X-Agent-Event`
- `X-Agent-Signature: sha256=<hmac>`

Payload shape:

```json
{
  "event_name": "pull_request",
  "delivery_id": "github-run-id",
  "repository": "dongeui/photomine",
  "raw_event": {}
}
```

## Orchestrator -> GitHub

Workflow: `.github/workflows/agent-transition-dispatch.yml`

The external orchestrator sends `repository_dispatch` with type `agent-transition`.

Expected `client_payload`:

```json
{
  "pr_number": 12,
  "replace_stage": "agent:qa",
  "add_labels": ["agent:qa"],
  "remove_labels": ["agent:dev"],
  "comment_body": "QA stage로 이동합니다."
}
```

## Merge Gate

Workflow: `.github/workflows/agent-pr-gate.yml`

Rules:

- draft PR은 gate를 통과시켜도 merge 대상이 아니다
- `agent:changes-requested`가 있으면 fail
- `agent:ready-to-merge`가 없으면 fail
- branch protection에서 `Agent PR Gate`를 required check로 건다

## Dispatch Example

External Orchestrator can advance a PR with GitHub API:

```bash
curl -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer <GITHUB_TOKEN>" \
  https://api.github.com/repos/dongeui/photomine/dispatches \
  -d '{
    "event_type": "agent-transition",
    "client_payload": {
      "pr_number": 12,
      "replace_stage": "agent:planner-review",
      "comment_body": "QA 통과. Planner final review로 이동합니다."
    }
  }'
```

## Initial Repo Setup

1. `scripts/setup_github_labels.sh` 실행
2. repo secrets에 `AGENT_WEBHOOK_URL`, `AGENT_WEBHOOK_SECRET` 추가
3. branch protection에서 `main`에 `Agent PR Gate` 필수화
4. 외부 Orchestrator 서비스에서 HMAC 검증 구현
5. Orchestrator가 stage 전이 시 `repository_dispatch` 호출

## Recommended Review Loop

1. Orchestrator opens or routes PR with `agent:dev`
2. Developer pushes implementation
3. Orchestrator or webhook moves PR to `agent:qa`
4. QA validates and either:
   - sets `agent:changes-requested`
   - or advances to `agent:planner-review`
5. Planner checks spec fit
6. Planner either:
   - sends back `agent:changes-requested`
   - or advances to `agent:ready-to-merge`
