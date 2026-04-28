from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Iterator

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select

from app.core.contracts import MediaFaceInput, MediaTagInput
from app.core.settings import load_settings
from app.main import create_app
from app.models.person import Person
from app.services.analysis import FaceAnalysis, FaceBoundingBox, ImageFaceAnalysisResult, OpenCVFaceModelPaths
from app.services.processing.registry import MediaCatalog


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
    database_path = data_root / "photomine.sqlite3"

    monkeypatch.setenv("PHOTOMINE_SOURCE_ROOTS", str(source_root))
    monkeypatch.setenv("PHOTOMINE_DATA_ROOT", str(data_root))
    monkeypatch.setenv("PHOTOMINE_DERIVED_ROOT", str(derived_root))
    monkeypatch.setenv("PHOTOMINE_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("PHOTOMINE_STABILITY_WINDOW_SECONDS", "1")
    monkeypatch.setenv("PHOTOMINE_SCHEDULER_ENABLED", "0")
    monkeypatch.setenv("PHOTOMINE_LOG_LEVEL", "ERROR")

    app = create_app(load_settings())
    with TestClient(app) as test_client:
        yield test_client


def test_image_exif_gps_is_persisted_as_place_tag_after_processing(
    client: TestClient,
    source_root: Path,
) -> None:
    image_path = source_root / "gps-image.jpg"
    create_image_with_gps_exif(image_path, latitude=37.55, longitude=127.0)

    scan_twice(client)
    item = get_media_item(client, filename="gps-image.jpg")

    metadata_json = item["metadata_json"] or {}
    gps_payload = metadata_json["gps"]
    persist_tags(
        client,
        item["file_id"],
        [MediaTagInput(tag_type="place", tag_value=format_place_tag(gps_payload))],
    )

    detail = client.get(f"/media/{item['file_id']}").json()

    assert gps_payload["latitude"] == pytest.approx(37.55)
    assert gps_payload["longitude"] == pytest.approx(127.0)
    assert {(tag["tag_type"], tag["tag_value"]) for tag in detail["tags"]} == {
        ("place", "37.5500,127.0000"),
    }


def test_multi_face_analysis_persists_multiple_person_tags_and_faces(
    client: TestClient,
    source_root: Path,
) -> None:
    image_path = source_root / "group-photo.jpg"
    create_image(image_path)

    scan_twice(client)
    item = get_media_item(client, filename="group-photo.jpg")
    analyzer = FakeFaceAnalyzer(
        build_face_analysis_result(
            image_path,
            [
                {"name": "Alice Kim", "bbox": (6, 8, 18, 18), "confidence": 0.99, "embedding": (0.1, 0.2, 0.3)},
                {"name": "Bob Lee", "bbox": (30, 10, 20, 20), "confidence": 0.97, "embedding": (0.4, 0.5, 0.6)},
            ],
        )
    )

    persist_fake_face_analysis(client, item["file_id"], analyzer)

    detail = client.get(f"/media/{item['file_id']}").json()

    with client.app.state.database.session_factory() as session:
        people = session.scalars(select(Person).order_by(Person.display_name.asc())).all()

    assert analyzer.calls == [image_path.resolve()]
    assert {(tag["tag_type"], tag["tag_value"]) for tag in detail["tags"]} == {
        ("person", "Alice Kim"),
        ("person", "Bob Lee"),
    }
    assert len(detail["faces"]) == 2
    assert all(face["person_id"] is not None for face in detail["faces"])
    assert {person.display_name for person in people} == {"Alice Kim", "Bob Lee"}


def test_media_filtering_by_tag_and_face_count_still_works(
    client: TestClient,
    source_root: Path,
) -> None:
    create_image(source_root / "seoul-family.jpg", color=(40, 80, 120))
    create_image(source_root / "busan-portrait.jpg", color=(120, 40, 80))
    create_image(source_root / "untagged.jpg", color=(80, 120, 40))

    scan_twice(client)
    items = {item["filename"]: item for item in client.get("/media").json()["items"]}

    persist_fake_face_analysis(
        client,
        items["seoul-family.jpg"]["file_id"],
        FakeFaceAnalyzer(
            build_face_analysis_result(
                source_root / "seoul-family.jpg",
                [
                    {"name": "Alice Kim", "bbox": (4, 6, 16, 16), "confidence": 0.98, "embedding": (0.1, 0.0, 0.2)},
                    {"name": "Bob Lee", "bbox": (26, 8, 18, 18), "confidence": 0.96, "embedding": (0.3, 0.2, 0.1)},
                ],
            )
        ),
        extra_tags=[MediaTagInput(tag_type="place", tag_value="Seoul")],
    )
    persist_fake_face_analysis(
        client,
        items["busan-portrait.jpg"]["file_id"],
        FakeFaceAnalyzer(
            build_face_analysis_result(
                source_root / "busan-portrait.jpg",
                [
                    {"name": "Carol Park", "bbox": (10, 10, 22, 22), "confidence": 0.97, "embedding": (0.4, 0.3, 0.2)},
                ],
            )
        ),
        extra_tags=[MediaTagInput(tag_type="place", tag_value="Busan")],
    )

    tagged = client.get("/media/filter", params={"tag": "Busan", "tag_type": "place"}).json()
    crowded = client.get(
        "/media/filter",
        params={"tag": "Seoul", "tag_type": "place", "face_count_min": 2, "face_count_max": 2},
    ).json()
    face_free = client.get("/media/filter", params={"face_count_max": 0}).json()

    assert {item["filename"] for item in tagged["items"]} == {"busan-portrait.jpg"}
    assert {item["filename"] for item in crowded["items"]} == {"seoul-family.jpg"}
    assert {item["filename"] for item in face_free["items"]} == {"untagged.jpg"}


class FakeFaceAnalyzer:
    def __init__(self, result: ImageFaceAnalysisResult) -> None:
        self._result = result
        self.calls: list[Path] = []

    def analyze_image_file(self, image_path: Path | str) -> ImageFaceAnalysisResult:
        resolved = Path(image_path).expanduser().resolve()
        self.calls.append(resolved)
        return self._result


def persist_tags(client: TestClient, file_id: str, tags: Iterable[MediaTagInput]) -> None:
    with client.app.state.database.session_factory() as session:
        catalog = MediaCatalog(session)
        catalog.upsert_tags(file_id, list(tags))
        session.commit()


def persist_fake_face_analysis(
    client: TestClient,
    file_id: str,
    analyzer: FakeFaceAnalyzer,
    *,
    extra_tags: Iterable[MediaTagInput] = (),
) -> None:
    with client.app.state.database.session_factory() as session:
        catalog = MediaCatalog(session)
        media_file = catalog.get_media(file_id)
        assert media_file is not None

        result = analyzer.analyze_image_file(media_file.current_path)
        face_inputs = [
            MediaFaceInput(
                bbox={
                    "x": face.bbox.x,
                    "y": face.bbox.y,
                    "width": face.bbox.width,
                    "height": face.bbox.height,
                    "confidence": face.bbox.confidence,
                    "landmarks": [list(point) for point in face.bbox.landmarks],
                },
                embedding_ref=f"fake://{file_id}/{face.face_index}",
                person_display_name=face.person_label_suggestion,
            )
            for face in result.faces
        ]
        tags = list(extra_tags) + [
            MediaTagInput(tag_type="person", tag_value=face.person_label_suggestion)
            for face in result.faces
        ]

        catalog.upsert_tags(file_id, tags)
        catalog.upsert_faces(file_id, face_inputs, resolve_people_by_name=True)
        session.commit()


def build_face_analysis_result(image_path: Path, faces: list[dict[str, object]]) -> ImageFaceAnalysisResult:
    analyzed_faces = tuple(
        FaceAnalysis(
            face_index=index,
            person_label_suggestion=str(face["name"]),
            bbox=FaceBoundingBox(
                x=int(face["bbox"][0]),
                y=int(face["bbox"][1]),
                width=int(face["bbox"][2]),
                height=int(face["bbox"][3]),
                confidence=float(face["confidence"]),
                landmarks=((0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)),
            ),
            embedding=tuple(float(value) for value in face["embedding"]),
        )
        for index, face in enumerate(faces)
    )
    return ImageFaceAnalysisResult(
        image_path=image_path.resolve(),
        image_width=64,
        image_height=48,
        model_paths=OpenCVFaceModelPaths(
            root=image_path.parent,
            detector_path=image_path.parent / "fake-detector.onnx",
            recognizer_path=image_path.parent / "fake-recognizer.onnx",
        ),
        faces=analyzed_faces,
    )


def get_media_item(client: TestClient, *, filename: str) -> dict[str, object]:
    items = client.get("/media").json()["items"]
    for item in items:
        if item["filename"] == filename:
            return item
    raise AssertionError(f"media item not found for {filename}")


def scan_twice(client: TestClient) -> tuple[dict, dict]:
    first = client.post("/scan").json()
    time.sleep(SCAN_DELAY_SECONDS)
    second = client.post("/scan").json()
    return first, second


def create_image(path: Path, *, color: tuple[int, int, int] = (40, 80, 120)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 48), color=color)
    image.save(path, format="JPEG")


def create_image_with_gps_exif(path: Path, *, latitude: float, longitude: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 48), color=(60, 90, 130))
    exif = Image.Exif()
    exif[34853] = {
        1: "N" if latitude >= 0 else "S",
        2: to_dms(abs(latitude)),
        3: "E" if longitude >= 0 else "W",
        4: to_dms(abs(longitude)),
        29: "2026:04:23",
        7: (1.0, 2.0, 3.0),
    }
    image.save(path, format="JPEG", exif=exif)


def format_place_tag(gps_payload: dict[str, object]) -> str:
    latitude = float(gps_payload["latitude"])
    longitude = float(gps_payload["longitude"])
    return f"{latitude:.4f},{longitude:.4f}"


def to_dms(value: float) -> tuple[float, float, float]:
    degrees = int(value)
    minutes_float = (value - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60
    return (float(degrees), float(minutes), float(seconds))
