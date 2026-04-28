from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.settings import load_settings
from app.main import create_app
from app.models.semantic import MediaAnalysisSignal, MediaOCR


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


def scan_twice(client: TestClient) -> tuple[dict, dict]:
    first = client.post("/scan").json()
    time.sleep(SCAN_DELAY_SECONDS)
    second = client.post("/scan").json()
    return first, second


def create_image(path: Path) -> None:
    image = Image.new("RGB", (80, 60), color=(40, 80, 120))
    image.save(path, format="JPEG")
