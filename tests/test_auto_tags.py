from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.contracts import MediaTagInput
from app.models.base import Base
from app.models.media import MediaFile
from app.models.semantic import MediaAutoTagState
from app.services.analysis import auto_tags
from app.services.semantic import SemanticCatalog


def test_auto_tags_from_signals_are_conservative_and_deduped() -> None:
    tags = auto_tags.tags_from_signals(
        {
            "is_screenshot_like": True,
            "is_document_like": True,
            "is_text_heavy": True,
        },
        "카드 승인 합계 12000",
    )

    assert [(tag.tag_type, tag.tag_value) for tag in tags] == [
        ("auto", "screenshot"),
        ("auto", "document"),
        ("auto", "text"),
        ("auto", "receipt"),
    ]


def test_auto_tag_state_records_version(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'auto-tags.sqlite3'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    with session_factory() as session:
        now = datetime.utcnow()
        session.add(
            MediaFile(
                file_id="sample-file-id",
                current_path="/tmp/sample.jpg",
                filename="sample.jpg",
                source_root="/tmp",
                relative_path="sample.jpg",
                media_kind="image",
                status="thumb_done",
                size_bytes=1,
                mtime_ns=1,
                partial_hash="hash",
                first_seen_at=now,
                last_seen_at=now,
                updated_at=now,
            )
        )
        state = SemanticCatalog(session).upsert_auto_tag_state(
            "sample-file-id",
            tags=[MediaTagInput(tag_type="auto", tag_value="screenshot")],
            version="auto-v2",
        )
        session.commit()

        reloaded = session.get(MediaAutoTagState, "sample-file-id")
        assert reloaded is not None
        assert reloaded.version == "auto-v2"
        assert reloaded.tags_json == [{"type": "auto", "value": "screenshot"}]
        assert state.file_id == "sample-file-id"
