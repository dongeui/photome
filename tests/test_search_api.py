from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.exc import OperationalError
from sqlalchemy import func, select, text

from app.core.settings import load_settings
from app.main import create_app
from app.models.semantic import MediaAnalysisSignal, MediaOCR, SearchDocument, SearchEvent


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

    second_no_op = client.post("/scan/semantic-maintenance").json()
    assert second_no_op["pending"] == 0


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
    assert 'const phase1StorageKey = "photome.dashboard.phase1.job";' in html
    assert 'const phase2StorageKey = "photome.dashboard.phase2.job";' in html
    assert 'const phase1SourceRootsStorageKey = "photome.dashboard.phase1.source_roots";' in html
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
