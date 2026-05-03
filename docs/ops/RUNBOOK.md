# Runbook

## If GitHub Webhook Delivery Fails

- check `AGENT_WEBHOOK_URL` and `AGENT_WEBHOOK_SECRET`
- inspect workflow run `Agent PR Webhook`
- verify Orchestrator endpoint HMAC validation logs
- do not advance PR stage manually without leaving a comment

## If PR Is Stuck In Wrong Stage

- remove stale stage label
- add the single correct stage label
- leave a comment explaining who owns the next action
- update `STATUS_SNAPSHOT` only if this changes project state, not just one PR state

## If QA Rejects

- set `agent:changes-requested`
- comment with reproduction steps and expected vs actual
- return ownership to Developer

## If Planner Rejects

- set `agent:changes-requested`
- comment with spec mismatch, not implementation advice only
- keep PR open; do not squash the history until fixed

## If NAS Is Offline In Runtime

- do not delete records
- keep existing metadata readable
- record scan failure and retry on next polling cycle

## If Local AI Pack Is Missing

- keep the base app running
- keep Phase 1 scan, gallery, OCR/tag/date/place search, and dashboard usable
- show local AI image search as `not installed` or `model missing`
- do not attempt model download when `PHOTOME_OFFLINE_MODE=1`
- skip CLIP embedding work in Phase 2 and record the skip reason
- install/repair the optional local AI pack only through the online preparation flow
- inspect current state without loading the model:
  - `.venv/bin/python scripts/local_ai_pack.py status`

## If Local AI Pack Is Installed

- set model cache paths explicitly:
  - `HF_HOME=<photome-data>/models/hf`
  - `TORCH_HOME=<photome-data>/models/torch`
- verify offline load before enabling scheduled Phase 2 CLIP work
- online preparation:
  - `.venv/bin/python scripts/local_ai_pack.py --cache-root <photome-data>/models prepare`
- offline verification:
  - `.venv/bin/python scripts/local_ai_pack.py --cache-root <photome-data>/models verify-offline`
- run with:
  - `PHOTOME_CLIP_ENABLED=1`
  - `PHOTOME_OFFLINE_MODE=1`
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`
- if the model/provider changes, bump embedding version and rebuild embeddings
