"""Database bootstrap and runtime state."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.settings import AppSettings
from app.db.session import create_engine_for_settings, create_session_factory
from app.models.base import Base
from app.models import annotation, asset, face, job, media, observation, person, semantic, tag  # noqa: F401  ensure models are registered


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatabaseState:
    settings: AppSettings
    engine: Engine
    session_factory: sessionmaker[Session]

    @property
    def configured(self) -> bool:
        return True

    @property
    def database_url(self) -> str:
        return self.settings.database_url


def _ensure_runtime_directories(settings: AppSettings) -> None:
    settings.data_root.mkdir(parents=True, exist_ok=True)
    settings.derived_root.mkdir(parents=True, exist_ok=True)
    settings.thumbnail_root.mkdir(parents=True, exist_ok=True)
    settings.preview_root.mkdir(parents=True, exist_ok=True)
    settings.keyframe_root.mkdir(parents=True, exist_ok=True)
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)


def build_database_state(settings: AppSettings) -> DatabaseState:
    _ensure_runtime_directories(settings)
    engine = create_engine_for_settings(settings)
    Base.metadata.create_all(engine)
    _ensure_search_document_fts(engine)
    session_factory = create_session_factory(engine)
    logger.info("database bootstrapped", extra={"database_url": settings.database_url})
    return DatabaseState(settings=settings, engine=engine, session_factory=session_factory)


def _ensure_search_document_fts(engine: Engine) -> None:
    if engine.dialect.name != "sqlite":
        return
    with engine.begin() as connection:
        # Primary FTS — unicode61 (word-boundary, good for English)
        try:
            connection.execute(
                text(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS search_documents_fts
                    USING fts5(
                        file_id UNINDEXED,
                        search_text,
                        keyword_text,
                        semantic_text,
                        tokenize='unicode61'
                    )
                    """
                )
            )
        except Exception as exc:
            logger.warning("search document FTS unavailable", extra={"error": str(exc)})

        # Trigram FTS — character n-gram, better for Korean/CJK substring search.
        # Requires SQLite 3.34.0+ (2020). Falls back silently if unavailable.
        try:
            connection.execute(
                text(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS search_documents_fts_ko
                    USING fts5(
                        file_id UNINDEXED,
                        search_text,
                        keyword_text,
                        semantic_text,
                        tokenize='trigram'
                    )
                    """
                )
            )
        except Exception as exc:
            logger.info(
                "trigram FTS unavailable (SQLite < 3.34); Korean substring search will use n-gram fallback",
                extra={"error": str(exc)},
            )
