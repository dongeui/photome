from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import PendingRollbackError
from sqlalchemy import func, select, text

from app.core.settings import load_settings
from app.db.bootstrap import build_database_state
from app.main import create_app
from app.models.job import ProcessingJob
from app.models.media import MediaFile
from app.models.runtime import SchedulerRuntimeConfig
from app.core.contracts import MediaTagInput
from app.models.semantic import MediaAnalysisSignal, MediaEmbedding, MediaOCR, SearchDocument, SearchEvent
from app.models.tag import Tag
from app.services.caption.registry import get_caption_provider
from app.services.processing.incremental import IncrementalScanSummary


SCAN_DELAY_SECONDS = 1.1


@pytest.fixture
def source_root(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source_root: Path) -> Iterator[TestClient]:
    data_root = tmp_path / "data"
    derived_root = tmp_path / "derived"
    database_path = data_root / "photome.sqlite3"

    monkeypatch.setenv("PHOTOME_SOURCE_ROOTS", str(source_root))
    monkeypatch.setenv("PHOTOME_DATA_ROOT", str(data_root))
    monkeypatch.setenv("PHOTOME_DERIVED_ROOT", str(derived_root))
    monkeypatch.setenv("PHOTOME_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("PHOTOME_STABILITY_WINDOW_SECONDS", "1")
    monkeypatch.setenv("PHOTOME_SCHEDULER_ENABLED", "0")
    monkeypatch.setenv("PHOTOME_FACE_ANALYSIS_ENABLED", "0")
    monkeypatch.setenv("PHOTOME_CLIP_ENABLED", "0")
    monkeypatch.setenv("PHOTOME_LOG_LEVEL", "ERROR")

    app = create_app(load_settings())
    with TestClient(app) as test_client:
        yield test_client


def test_search_finds_scanned_media_by_filename_and_semantic_rows_exist(
    client: TestClient,
    source_root: Path,
) -> None:
    create_image(source_root / "vacation-receipt.jpg")
    scan_twice(client)

    response = client.get("/search", params={"q": "receipt"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["security"]["runtime_mode"] in {"standard", "offline-local-only"}
    assert payload["items"][0]["filename"] == "vacation-receipt.jpg"

    file_id = payload["items"][0]["file_id"]
    with client.app.state.database.session_factory() as session:
        assert session.get(MediaOCR, file_id) is not None
        assert session.get(MediaAnalysisSignal, file_id) is not None
        search_document = session.get(SearchDocument, file_id)
        assert search_document is not None
        assert "vacation-receipt.jpg" in search_document.search_text
        assert search_document.version == client.app.state.settings.semantic_search_version
        indexed = session.execute(
            text("SELECT file_id FROM search_documents_fts WHERE search_documents_fts MATCH 'receipt'")
        ).all()
        assert indexed == [(file_id,)]


def test_search_tolerates_event_commit_failure(
    client: TestClient,
    source_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_image(source_root / "commit-failure-receipt.jpg")
    scan_twice(client)

    original_commit = client.app.state.database.session_factory.class_.commit

    def flaky_commit(session):  # type: ignore[no-untyped-def]
        raise PendingRollbackError("locked session")

    monkeypatch.setattr(client.app.state.database.session_factory.class_, "commit", flaky_commit)
    try:
        response = client.get("/search", params={"q": "receipt"})
    finally:
        monkeypatch.setattr(client.app.state.database.session_factory.class_, "commit", original_commit)

    assert response.status_code == 200
    assert response.json()["total"] >= 1


def test_offline_mode_disables_outbound_features(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    data_root = tmp_path / "data"
    derived_root = tmp_path / "derived"
    database_path = data_root / "photome.sqlite3"
    source_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("PHOTOME_SOURCE_ROOTS", str(source_root))
    monkeypatch.setenv("PHOTOME_DATA_ROOT", str(data_root))
    monkeypatch.setenv("PHOTOME_DERIVED_ROOT", str(derived_root))
    monkeypatch.setenv("PHOTOME_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("PHOTOME_OFFLINE_MODE", "1")
    monkeypatch.setenv("PHOTOME_CAPTION_PROVIDER", "moondream")
    monkeypatch.setenv("PHOTOME_FACE_ANALYSIS_ENABLED", "1")
    monkeypatch.setenv("PHOTOME_LOG_LEVEL", "ERROR")

    settings = load_settings()
    assert settings.offline_mode is True
    assert get_caption_provider() is None

    app = create_app(settings)
    with TestClient(app) as test_client:
        response = test_client.get("/status")
        assert response.status_code == 200
        payload = response.json()
        assert payload["security"]["offline_mode"] is True
        assert payload["security"]["outbound_network_enabled"] is False
        assert payload["security"]["runtime_mode"] == "offline-local-only"
        assert "Reverse geocoding is blocked." in payload["security"]["disabled_features"]
        assert "Caption generation is disabled." in payload["security"]["disabled_features"]
        states = {item["name"]: item["state"] for item in payload["security"]["local_dependencies"]}
        assert "ffmpeg" in states
        assert "ffprobe" in states
        assert states["CLIP semantic embedding"] == "disabled"
        assert states["Caption provider"] == "disabled"


def test_clip_status_degrades_when_local_ai_pack_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    data_root = tmp_path / "data"
    derived_root = tmp_path / "derived"
    database_path = data_root / "photome.sqlite3"
    source_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("PHOTOME_SOURCE_ROOTS", str(source_root))
    monkeypatch.setenv("PHOTOME_DATA_ROOT", str(data_root))
    monkeypatch.setenv("PHOTOME_DERIVED_ROOT", str(derived_root))
    monkeypatch.setenv("PHOTOME_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("PHOTOME_CLIP_ENABLED", "1")
    monkeypatch.setenv("PHOTOME_LOG_LEVEL", "ERROR")

    from app.services.embedding import clip as clip_embedding

    monkeypatch.setattr(
        clip_embedding,
        "dependency_status",
        lambda: {"open_clip_torch": "missing", "torch": "missing", "torchvision": "missing"},
    )

    app = create_app(load_settings())
    with TestClient(app) as test_client:
        response = test_client.get("/status")

    assert response.status_code == 200
    states = {item["name"]: item for item in response.json()["security"]["local_dependencies"]}
    clip_state = states["CLIP semantic embedding"]
    assert clip_state["state"] == "missing-local-ai-pack"
    assert clip_state["dependencies"]["open_clip_torch"] == "missing"


def test_semantic_maintenance_only_builds_missing_search_documents(
    client: TestClient,
    source_root: Path,
) -> None:
    create_image(source_root / "cycle-receipt.jpg")
    scan_twice(client)
    item = client.get("/search", params={"q": "receipt"}).json()["items"][0]

    no_op = client.app.state.pipeline.run_semantic_maintenance()
    assert no_op["pending"] == 0
    assert no_op["succeeded"] == 0

    with client.app.state.database.session_factory() as session:
        document = session.get(SearchDocument, item["file_id"])
        assert document is not None
        session.delete(document)
        session.commit()

    rebuilt = client.app.state.pipeline.run_semantic_maintenance()
    assert rebuilt["pending"] == 1
    assert rebuilt["succeeded"] == 1


def test_status_reports_phase2_coverage(
    client: TestClient,
    source_root: Path,
) -> None:
    create_image(source_root / "coverage-receipt.jpg")
    scan_twice(client)

    payload = client.get("/status").json()
    coverage = payload["semantic"]["coverage"]

    assert coverage["eligible_media"] >= 1
    assert coverage["search_current"] >= 1
    assert coverage["remaining_for_search"] >= 0
    assert "clip_embeddings_current" in coverage
    assert "semantic_job_errors" in coverage


def test_semantic_maintenance_fills_missing_clip_embeddings_when_enabled(
    client: TestClient,
    source_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_image(source_root / "plain-photo.jpg")
    scan_twice(client)

    pipeline = client.app.state.pipeline
    pipeline._semantic_clip_enabled = True

    def fake_embedding(media_file: MediaFile) -> dict:
        return {
            "model_name": "ViT-B-32/openai",
            "version": pipeline._semantic_embedding_version,
            "embedding_ref": f"embeddings/clip/{pipeline._semantic_embedding_version}/aa/{media_file.file_id}.npy",
            "dimensions": 3,
            "checksum": None,
        }

    monkeypatch.setattr(pipeline, "_materialize_clip_embedding", fake_embedding)
    from app.services.analysis import auto_tags

    monkeypatch.setattr(
        auto_tags,
        "tags_from_embedding_file",
        lambda *_args, **_kwargs: [MediaTagInput(tag_type="auto", tag_value="바다")],
    )

    result = pipeline.run_semantic_maintenance(batch_size=10)

    assert result["succeeded"] >= 1
    with client.app.state.database.session_factory() as session:
        media_file = session.scalar(select(MediaFile).where(MediaFile.filename == "plain-photo.jpg"))
        assert media_file is not None
        assert session.scalar(select(MediaEmbedding).where(MediaEmbedding.file_id == media_file.file_id)) is not None
        assert session.scalar(
            select(func.count())
            .select_from(Tag)
            .where(Tag.file_id == media_file.file_id, Tag.tag_type == "auto", Tag.tag_value == "바다")
        ) == 1
        search_document = session.get(SearchDocument, media_file.file_id)
        assert search_document is not None
        assert "바다" in search_document.search_text

    second_no_op = client.post("/scan/semantic-maintenance").json()
    assert second_no_op["pending"] == 0


def test_clip_embedding_reuse_requires_matching_model_name(
    client: TestClient,
    source_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_image(source_root / "model-change.jpg")
    scan_twice(client)

    pipeline = client.app.state.pipeline
    pipeline._semantic_clip_enabled = True
    calls = {"count": 0}

    with client.app.state.database.session_factory() as session:
        media_file = session.scalar(select(MediaFile).where(MediaFile.filename == "model-change.jpg"))
        assert media_file is not None
        session.add(
            MediaEmbedding(
                file_id=media_file.file_id,
                model_name="old-model/old-pretrained",
                version=pipeline._semantic_embedding_version,
                embedding_ref=f"embeddings/clip/{pipeline._semantic_embedding_version}/old/{media_file.file_id}.npy",
                dimensions=3,
            )
        )
        session.commit()

    monkeypatch.setenv("PHOTOME_CLIP_MODEL_NAME", "new-model")
    monkeypatch.setenv("PHOTOME_CLIP_PRETRAINED", "new-pretrained")

    def fake_embedding(media_file: MediaFile) -> dict:
        calls["count"] += 1
        return {
            "model_name": pipeline._clip_model_identifier(),
            "version": pipeline._semantic_embedding_version,
            "embedding_ref": f"embeddings/clip/{pipeline._semantic_embedding_version}/new/{media_file.file_id}.npy",
            "dimensions": 3,
            "checksum": None,
        }

    monkeypatch.setattr(pipeline, "_materialize_clip_embedding", fake_embedding)

    with client.app.state.database.session_factory() as session:
        media_file = session.scalar(select(MediaFile).where(MediaFile.filename == "model-change.jpg"))
        assert media_file is not None
        from app.services.processing.registry import MediaCatalog
        from app.services.semantic import SemanticCatalog

        result = pipeline._ensure_clip_embedding(
            session,
            media_file,
            MediaCatalog(session),
            SemanticCatalog(session),
        )
        session.commit()

    assert calls["count"] == 1
    assert result is not None
    assert result["model_name"] == "new-model/new-pretrained"
    with client.app.state.database.session_factory() as session:
        rows = session.scalars(select(MediaEmbedding).where(MediaEmbedding.file_id == media_file.file_id)).all()
        assert {row.model_name for row in rows} == {"old-model/old-pretrained", "new-model/new-pretrained"}


def test_embedding_pending_uses_current_model_and_version(
    client: TestClient,
    source_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_image(source_root / "pending-model-change.jpg")
    scan_twice(client)

    pipeline = client.app.state.pipeline
    pipeline._semantic_clip_enabled = True

    with client.app.state.database.session_factory() as session:
        media_file = session.scalar(select(MediaFile).where(MediaFile.filename == "pending-model-change.jpg"))
        assert media_file is not None
        session.add(
            MediaEmbedding(
                file_id=media_file.file_id,
                model_name="old-model/old-pretrained",
                version=pipeline._semantic_embedding_version,
                embedding_ref=f"embeddings/clip/{pipeline._semantic_embedding_version}/old/{media_file.file_id}.npy",
                dimensions=3,
            )
        )
        session.commit()

    monkeypatch.setenv("PHOTOME_CLIP_MODEL_NAME", "new-model")
    monkeypatch.setenv("PHOTOME_CLIP_PRETRAINED", "new-pretrained")
    calls = {"count": 0}

    def fake_embedding(media_file: MediaFile) -> dict:
        calls["count"] += 1
        return {
            "model_name": pipeline._clip_model_identifier(),
            "version": pipeline._semantic_embedding_version,
            "embedding_ref": f"embeddings/clip/{pipeline._semantic_embedding_version}/new/{media_file.file_id}.npy",
            "dimensions": 3,
            "checksum": None,
        }

    monkeypatch.setattr(pipeline, "_materialize_clip_embedding", fake_embedding)

    result = pipeline.run_semantic_maintenance(batch_size=10)

    assert result["succeeded"] >= 1
    assert calls["count"] == 1


def test_search_event_is_persisted_after_search(
    client: TestClient,
    source_root: Path,
) -> None:
    create_image(source_root / "event-receipt.jpg")
    scan_twice(client)

    response = client.get("/search", params={"q": "receipt"})
    cached_response = client.get("/search", params={"q": "receipt"})

    assert response.status_code == 200
    assert cached_response.status_code == 200
    with client.app.state.database.session_factory() as session:
        count = session.scalar(select(func.count()).select_from(SearchEvent))
        assert count == 2


def test_search_event_is_skipped_while_library_job_is_active(
    client: TestClient,
    source_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_image(source_root / "active-job-receipt.jpg")
    scan_twice(client)

    monkeypatch.setattr(client.app.state.pipeline, "has_active_library_job", lambda: True)
    response = client.get("/search", params={"q": "receipt"})

    assert response.status_code == 200
    with client.app.state.database.session_factory() as session:
        count = session.scalar(select(func.count()).select_from(SearchEvent))
        assert count == 0


def test_date_fallback_does_not_recurse_on_zero_results(client: TestClient) -> None:
    response = client.get("/search", params={"q": "2099년 12월 31일 qqqzzznotfound"})

    assert response.status_code == 200
    assert response.json()["items"] == []


def test_clip_disabled_search_does_not_load_clip_model(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.embedding import clip as clip_embedding

    called = False

    def mark_called() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(clip_embedding, "ensure_models", mark_called)

    response = client.get("/search", params={"q": "얼굴"})

    assert response.status_code == 200
    assert called is False


def test_feedback_invalidates_cached_search_results(
    client: TestClient,
    source_root: Path,
) -> None:
    create_image(source_root / "cached-receipt.jpg")
    scan_twice(client)

    first = client.get("/search", params={"q": "receipt"}).json()
    assert first["total"] == 1
    file_id = first["items"][0]["file_id"]

    feedback = client.post(
        "/search/feedback",
        json={"file_id": file_id, "action": "hide"},
    )
    assert feedback.status_code == 201

    second = client.get("/search", params={"q": "receipt"}).json()
    assert second["total"] == 0


def test_weight_profile_rejects_invalid_values(client: TestClient) -> None:
    negative = client.put(
        "/search/weights/hybrid/fallback",
        json={"w_ocr": -1, "w_clip": 0.5, "w_shadow": 0.5},
    )
    assert negative.status_code == 422

    zero_total = client.put(
        "/search/weights/hybrid/fallback",
        json={"w_ocr": 0, "w_clip": 0, "w_shadow": 0},
    )
    assert zero_total.status_code == 422


def test_media_annotation_updates_display_name_description_and_custom_tags(
    client: TestClient,
    source_root: Path,
) -> None:
    create_image(source_root / "vacation-receipt.jpg")
    scan_twice(client)
    item = client.get("/media", params={"q": "receipt"}).json()["items"][0]

    response = client.post(
        f"/media/{item['file_id']}/annotation",
        data={
            "title": "Trip receipt",
            "description": "Dinner receipt from the family trip.",
            "tags": "receipt, family, receipt",
            "next": "/gallery?q=receipt",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    detail = client.get(f"/media/{item['file_id']}").json()
    assert detail["annotation"] == {
        "title": "Trip receipt",
        "description": "Dinner receipt from the family trip.",
    }
    assert {
        (tag["tag_type"], tag["tag_value"])
        for tag in detail["tags"]
        if tag["tag_type"] == "custom"
    } == {("custom", "receipt"), ("custom", "family")}

    gallery = client.get("/gallery", params={"q": "receipt"}).text
    assert "Trip receipt" in gallery
    assert "Dinner receipt from the family trip." in gallery
    assert "no tags" not in gallery

    title_search = client.get("/search", params={"q": "Trip"})
    assert title_search.status_code == 200
    assert title_search.json()["items"][0]["file_id"] == item["file_id"]

    gallery_title_search = client.get("/gallery", params={"q": "Trip"}).text
    assert "Trip receipt" in gallery_title_search
    assert "Dinner receipt from the family trip." in gallery_title_search


def test_scan_accepts_source_roots_query_override(client: TestClient, tmp_path: Path) -> None:
    selected_root = tmp_path / "selected-source"
    selected_root.mkdir()
    create_image(selected_root / "manual-path-receipt.jpg")

    client.post("/scan", params={"source_roots": str(selected_root)})
    time.sleep(SCAN_DELAY_SECONDS)
    response = client.post("/scan", params={"source_roots": str(selected_root)})

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["payload"]["source_roots"] == [str(selected_root.resolve())]
    assert job["result"]["source_roots"] == [str(selected_root.resolve())]
    assert job["result"]["summary"]["created"] == 1

    search = client.get("/search", params={"q": "receipt"})
    assert search.status_code == 200
    assert search.json()["total"] == 1


def test_async_scan_starts_job_and_exposes_status(client: TestClient, tmp_path: Path) -> None:
    selected_root = tmp_path / "async-source"
    selected_root.mkdir()
    create_image(selected_root / "async-receipt.jpg")
    client.post("/scan", params={"source_roots": str(selected_root)})
    time.sleep(SCAN_DELAY_SECONDS)

    response = client.post("/scan/async", params={"source_roots": str(selected_root)})

    assert response.status_code == 202
    job = response.json()["job"]
    assert job["status"] in {"queued", "succeeded"}
    status_response = client.get(f"/scan/jobs/{job['job_id']}")
    assert status_response.status_code == 200
    status_job = status_response.json()["job"]
    assert status_job["job_id"] == job["job_id"]
    assert status_job["status"] == "succeeded"
    assert status_job["result"]["summary"]["created"] == 1


def test_full_scan_imports_old_archive_files_on_first_pass(client: TestClient, tmp_path: Path) -> None:
    nested_root = tmp_path / "archive-source"
    nested_file = nested_root / "album-a" / "album-b" / "archive-receipt.jpg"
    nested_file.parent.mkdir(parents=True, exist_ok=True)
    create_image(nested_file)
    old_timestamp = time.time() - 600
    os.utime(nested_file, (old_timestamp, old_timestamp))

    response = client.post(
        "/scan",
        params={"full_scan": "true", "source_roots": str(nested_root)},
    )

    assert response.status_code == 200
    summary = response.json()["job"]["result"]["summary"]
    assert summary["scanned"] == 1
    assert summary["created"] == 1

    search = client.get("/search", params={"q": "archive"})
    assert search.status_code == 200
    assert search.json()["total"] == 1


def test_video_assets_degrade_when_ffmpeg_is_missing(
    client: TestClient,
    source_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = source_root / "offline-video.mp4"
    video_path.write_bytes(b"fake mp4 bytes")
    monkeypatch.setattr("app.services.processing.pipeline.which", lambda name: None if name == "ffmpeg" else None)

    _, second = scan_twice(client)

    assert second["job"]["status"] == "succeeded"
    assert second["job"]["result"]["processed"]["failed"] == 0
    with client.app.state.database.session_factory() as session:
        media_file = session.scalar(select(MediaFile).where(MediaFile.filename == "offline-video.mp4"))
        assert media_file is not None
        assert media_file.status == "analysis_done"
        assert media_file.error_stage is None


def test_async_semantic_maintenance_job_exposes_status(client: TestClient, source_root: Path) -> None:
    create_image(source_root / "semantic-job-receipt.jpg")
    scan_twice(client)

    response = client.post("/scan/semantic-maintenance/async", params={"batch_size": 10})

    assert response.status_code == 202
    job = response.json()["job"]
    status_response = client.get(f"/scan/jobs/{job['job_id']}")
    assert status_response.status_code == 200
    status_job = status_response.json()["job"]
    assert status_job["job_id"] == job["job_id"]
    assert status_job["status"] == "succeeded"
    assert "pending" in status_job["result"]


def test_async_job_dashboard_restores_phase_cards_from_local_storage(client: TestClient) -> None:
    response = client.get("/dashboard")

    assert response.status_code == 200
    html = response.text
    assert "Outbound Network" in html
    assert 'const phase1StorageKey = "photome.dashboard.phase1.job";' in html
    assert 'const phase2StorageKey = "photome.dashboard.phase2.job";' in html
    assert 'const phase1SourceRootsStorageKey = "photome.dashboard.phase1.source_roots";' in html
    assert "let activeLibraryJob =" in html
    assert "function updateLibraryJobGuards()" in html
    assert "setInterval(refreshDashboardStatus, 3000);" in html
    assert 'id="phase1-schedule-button"' in html
    assert 'id="phase2-schedule-button"' in html
    assert "Offline Security" in html
    assert 'id="phase1-full-scan"' not in html
    assert 'params.set("full_scan", "true");' in html
    assert "function formatElapsed(startedAt, finishedAt)" in html
    assert "async function pollJob(jobId, resultNode, render)" in html
    assert "if (progress.message) lines.push(progress.message);" in html
    assert "resumeJob(phase1StorageKey, scanCard, scanButton, scanResult, renderScanJob);" in html
    assert "resumeJob(phase2StorageKey, semanticCard, semanticButton, semanticResult, renderSemanticJob);" in html
    assert "sourceRootsField.value = rememberedSourceRoots;" in html


def test_async_semantic_job_returns_conflict_when_catalog_is_locked(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_locked(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise OperationalError("INSERT INTO processing_jobs ...", {}, RuntimeError("database is locked"))

    monkeypatch.setattr(client.app.state.pipeline, "submit_semantic_maintenance_job", raise_locked)

    response = client.post("/scan/semantic-maintenance/async", params={"batch_size": 10})

    assert response.status_code == 409
    assert "Another library job is still writing to the catalog" in response.json()["detail"]


def test_phase2_async_is_blocked_while_phase1_job_is_active(client: TestClient) -> None:
    with client.app.state.database.session_factory() as session:
        session.add(
            ProcessingJob(
                job_kind="scan",
                status="running",
                payload_json={"trigger": "test"},
                attempts=1,
            )
        )
        session.commit()

    response = client.post("/scan/semantic-maintenance/async", params={"batch_size": 10})

    assert response.status_code == 409
    assert "Phase 1 scan is already active" in response.json()["detail"]


def test_phase1_async_is_blocked_while_phase2_job_is_active(client: TestClient) -> None:
    with client.app.state.database.session_factory() as session:
        session.add(
            ProcessingJob(
                job_kind="semantic_maintenance",
                status="running",
                payload_json={"trigger": "test"},
                attempts=1,
            )
        )
        session.commit()

    response = client.post("/scan/async")

    assert response.status_code == 409
    assert "Phase 2 semantic work is already active" in response.json()["detail"]


def test_startup_recovers_interrupted_library_jobs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    data_root = tmp_path / "data"
    derived_root = tmp_path / "derived"
    database_path = data_root / "photome.sqlite3"

    monkeypatch.setenv("PHOTOME_SOURCE_ROOTS", str(source_root))
    monkeypatch.setenv("PHOTOME_DATA_ROOT", str(data_root))
    monkeypatch.setenv("PHOTOME_DERIVED_ROOT", str(derived_root))
    monkeypatch.setenv("PHOTOME_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("PHOTOME_STABILITY_WINDOW_SECONDS", "1")
    monkeypatch.setenv("PHOTOME_SCHEDULER_ENABLED", "0")
    monkeypatch.setenv("PHOTOME_FACE_ANALYSIS_ENABLED", "0")
    monkeypatch.setenv("PHOTOME_CLIP_ENABLED", "0")
    monkeypatch.setenv("PHOTOME_LOG_LEVEL", "ERROR")

    settings = load_settings()
    database = build_database_state(settings)
    with database.session_factory() as session:
        job = ProcessingJob(
            job_kind="scan",
            status="running",
            payload_json={"source_roots": [str(source_root)]},
            attempts=1,
        )
        session.add(job)
        session.commit()
        job_id = job.id

    app = create_app(settings)
    with TestClient(app) as test_client:
        status_payload = test_client.get("/status").json()
        assert status_payload["jobs"]["active_library_job"] is None
        with test_client.app.state.database.session_factory() as session:
            recovered = session.get(ProcessingJob, job_id)
            assert recovered is not None
            assert recovered.status == "canceled"
            assert recovered.error_stage == "interrupted"
            assert recovered.result_json["progress"]["resume_supported"] is True


def test_cycle_scheduler_phase_updates_runtime_schedule(client: TestClient) -> None:
    client.app.state.scheduler.stop()
    first = client.post("/scheduler/cycle/phase1")
    second = client.post("/scheduler/cycle/phase1")
    phase2 = client.post("/scheduler/cycle/phase2")

    assert first.status_code == 200
    assert first.json()["scheduler"]["phase1_interval_hours"] == 6
    assert second.status_code == 200
    assert second.json()["scheduler"]["phase1_interval_hours"] == 12
    assert phase2.status_code == 200
    assert phase2.json()["scheduler"]["phase2_interval_hours"] == 6
    assert first.json()["scheduler"]["next_full_scan_at"] is not None
    assert phase2.json()["scheduler"]["next_semantic_maintenance_at"] is not None

    with client.app.state.database.session_factory() as session:
        runtime_config = session.get(SchedulerRuntimeConfig, 1)
        assert runtime_config is not None
        assert runtime_config.last_phase1_run_at is not None
        assert runtime_config.last_phase2_run_at is not None


def test_run_scan_job_persists_running_state_before_long_work(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = client.app.state.pipeline
    database = client.app.state.database
    source_root = tmp_path / "source"
    source_root.mkdir(exist_ok=True)
    image_path = source_root / "sample.jpg"
    image_path.write_bytes(b"fake")

    summary = pipeline.submit_scan_job(
        full_scan=True,
        run_now=False,
        trigger="test",
        source_roots=(source_root,),
    )

    def fake_run(self, session):  # type: ignore[no-untyped-def]
        with database.session_factory() as verify_session:
            job = verify_session.get(ProcessingJob, summary.job_id)
            assert job is not None
            assert job.status == "running"
            assert job.started_at is not None
            assert (job.result_json or {}).get("progress", {}).get("stage") == "scanning"
        return IncrementalScanSummary(scanned=1, created=0, updated=0, moved=0, missing=0, failed=0)

    monkeypatch.setattr("app.services.processing.pipeline.IncrementalScanService.run", fake_run)
    monkeypatch.setattr(pipeline, "_process_pending_media", lambda *args, **kwargs: {"pending": 0, "succeeded": 0, "failed": 0})

    result = pipeline.run_scan_job(summary.job_id)

    assert result.status == "succeeded"


def test_run_semantic_job_persists_running_state_before_long_work(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = client.app.state.pipeline
    database = client.app.state.database

    summary = pipeline.submit_semantic_maintenance_job(
        batch_size=10,
        run_now=False,
        trigger="test",
    )

    def fake_maintenance(*, batch_size: int, progress_callback=None):  # type: ignore[no-untyped-def]
        with database.session_factory() as verify_session:
            job = verify_session.get(ProcessingJob, summary.job_id)
            assert job is not None
            assert job.status == "running"
            assert job.started_at is not None
            assert (job.result_json or {}).get("progress", {}).get("stage") == "collecting"
        if progress_callback is not None:
            progress_callback({"mode": "maintenance", "pending": 0, "current": 0, "succeeded": 0, "failed": 0})
        return {"skipped": False, "pending": 0, "succeeded": 0, "failed": 0, "has_more": False}

    monkeypatch.setattr(pipeline, "run_semantic_maintenance", fake_maintenance)

    result = pipeline.run_semantic_job(summary.job_id)

    assert result.status == "succeeded"


def test_async_semantic_job_runs_chunks_until_exhausted(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = client.app.state.pipeline
    summary = pipeline.submit_semantic_maintenance_job(
        batch_size=10,
        run_now=False,
        trigger="test",
    )
    chunks = [
        {
            "skipped": False,
            "pending": 10,
            "succeeded": 10,
            "failed": 0,
            "has_more": True,
            "embeddings_created": 10,
            "auto_tag_files": 4,
            "auto_tag_values": 20,
            "search_documents_updated": 10,
        },
        {
            "skipped": False,
            "pending": 3,
            "succeeded": 3,
            "failed": 0,
            "has_more": False,
            "embeddings_created": 3,
            "auto_tag_files": 2,
            "auto_tag_values": 8,
            "search_documents_updated": 3,
        },
    ]
    calls = {"count": 0}

    def fake_maintenance(*, batch_size: int, progress_callback=None):  # type: ignore[no-untyped-def]
        result = chunks[calls["count"]]
        calls["count"] += 1
        if progress_callback is not None:
            progress_callback({
                "mode": "maintenance",
                "pending": result["pending"],
                "current": result["pending"],
                "succeeded": result["succeeded"],
                "failed": result["failed"],
                "batch_size": batch_size,
                "embeddings_created": result["embeddings_created"],
                "auto_tag_files": result["auto_tag_files"],
                "auto_tag_values": result["auto_tag_values"],
                "search_documents_updated": result["search_documents_updated"],
            })
        return result

    monkeypatch.setattr(pipeline, "run_semantic_maintenance", fake_maintenance)

    result = pipeline.run_semantic_job(summary.job_id)

    assert calls["count"] == 2
    assert result.status == "succeeded"
    assert result.result is not None
    assert result.result["full_run"] is True
    assert result.result["chunks"] == 2
    assert result.result["succeeded"] == 13
    assert result.result["embeddings_created"] == 13
    assert result.result["auto_tag_values"] == 28
    assert result.result["search_documents_updated"] == 13


def test_scan_rejects_missing_source_root(client: TestClient, tmp_path: Path) -> None:
    response = client.post("/scan", params={"source_roots": str(tmp_path / "missing")})

    assert response.status_code == 400
    assert "Source root does not exist" in response.json()["detail"]


def scan_twice(client: TestClient) -> tuple[dict, dict]:
    first = client.post("/scan").json()
    time.sleep(SCAN_DELAY_SECONDS)
    second = client.post("/scan").json()
    return first, second


def create_image(path: Path) -> None:
    image = Image.new("RGB", (80, 60), color=(40, 80, 120))
    image.save(path, format="JPEG")
