"""Vector search backend abstractions for image embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.media import MediaFile
from app.models.semantic import MediaEmbedding
from app.models.tag import Tag
from app.services.embedding import clip as clip_embedding


@dataclass(frozen=True)
class VectorSearchHit:
    media_file: MediaFile
    distance: float
    embedding_ref: str
    model_name: str
    version: str


class VectorIndexBackend(Protocol):
    def search(
        self,
        query_embedding: bytes,
        *,
        limit: int,
        place_filter: str | None = None,
        date_from: Any | None = None,
        date_to: Any | None = None,
    ) -> list[VectorSearchHit]: ...


class LocalNumpyVectorIndex:
    """Simple exact vector search for local-first libraries.

    This is intentionally small and replaceable. The next scale-up path is an
    adapter with the same interface backed by FAISS, LanceDB, or Qdrant.
    """

    def __init__(self, session: Session, *, embeddings_root: Path) -> None:
        self._session = session
        self._embeddings_root = embeddings_root

    def search(
        self,
        query_embedding: bytes,
        *,
        limit: int,
        place_filter: str | None = None,
        date_from: Any | None = None,
        date_to: Any | None = None,
    ) -> list[VectorSearchHit]:
        try:
            import numpy as np

            query_vector = clip_embedding.embedding_from_bytes(query_embedding)
            query_norm = float(np.linalg.norm(query_vector)) or 1.0
        except Exception:
            return []

        statement = (
            select(MediaEmbedding, MediaFile)
            .join(MediaFile, MediaFile.file_id == MediaEmbedding.file_id)
            .where(MediaFile.status != "missing")
        )
        if place_filter:
            statement = statement.join(Tag, Tag.file_id == MediaFile.file_id).where(Tag.tag_value == place_filter)
        if isinstance(date_from, datetime):
            statement = statement.where(MediaFile.exif_datetime >= date_from)
        if isinstance(date_to, datetime):
            statement = statement.where(MediaFile.exif_datetime <= date_to)

        scored: list[VectorSearchHit] = []
        for embedding, media_file in self._session.execute(statement):
            vector = self._load_embedding_vector(embedding.embedding_ref)
            if vector is None or vector.size != query_vector.size:
                continue
            denominator = (float(np.linalg.norm(vector)) or 1.0) * query_norm
            similarity = float(np.dot(query_vector, vector) / denominator)
            scored.append(
                VectorSearchHit(
                    media_file=media_file,
                    distance=1.0 - similarity,
                    embedding_ref=embedding.embedding_ref,
                    model_name=embedding.model_name,
                    version=embedding.version,
                )
            )

        scored.sort(key=lambda item: item.distance)
        return scored[:limit]

    def _load_embedding_vector(self, embedding_ref: str):
        try:
            import numpy as np

            path = Path(embedding_ref)
            if path.is_absolute():
                absolute_path = path
            else:
                try:
                    absolute_path = self._embeddings_root / path.relative_to("embeddings")
                except ValueError:
                    absolute_path = self._embeddings_root / path
            if not absolute_path.is_file():
                return None
            return np.load(absolute_path).astype("float32")
        except Exception:
            return None
