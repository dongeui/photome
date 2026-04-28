# photome

Integrated local-first personal photo home server.

This workspace merges the production-shaped `photomine` backend with the working OCR + CLIP + hybrid search capabilities from `photomem`.

Start with [docs/INTEGRATION_PLAN.md](docs/INTEGRATION_PLAN.md) for the current merge direction.

This repository is set up for a multi-agent delivery loop:

- `Orchestrator` keeps the user conversation open and routes work.
- `Planner` writes and reviews spec/acceptance criteria.
- `Developer` implements code.
- `QA` validates scenarios and blocks regressions.

Original `photomine` collaboration docs are still present while integration is in progress:

- [AGENTS.md](AGENTS.md)
- [docs/README.md](docs/README.md)
- [docs/integrations/GITHUB_AGENT_WEBHOOKS.md](docs/integrations/GITHUB_AGENT_WEBHOOKS.md)
