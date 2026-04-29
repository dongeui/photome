"""Vector search backend abstractions for image embeddings."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.media import MediaFile
from app.models.semantic import MediaEmbedding
from app.models.tag import Tag
from app.services.embedding import clip as clip_embedding

logger = logging.getLogger(__name__)

# Process-level FAISS index instance (shared across requests).
# Set by build_vector_index() when FAISS is active so that
# pipeline can call invalidate_global_vector_index() after maintenance.
_global_faiss_index: "FaissVectorIndex | None" = None


def invalidate_global_vector_index() -> bool:
    """Invalidate the process-level FAISS index so it rebuilds on next search.

    Returns True if an index was invalidated, False if FAISS is not active.
    """
    if _global_faiss_index is not None:
        _global_faiss_index.invalidate()
        return True
    return False


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


class FaissVectorIndex:
    """ANN vector search backed by FAISS IndexFlatIP.

    Falls back gracefully to returning [] if faiss is not installed.

    The index is built lazily on first search and cached in memory.
    Call invalidate() after bulk embedding changes to trigger a rebuild.

    Install: pip install photome[faiss]
    """

    def __init__(self, session: Session, *, embeddings_root: Path) -> None:
        self._session = session
        self._embeddings_root = embeddings_root
        self._lock = threading.Lock()
        self._index: Any | None = None          # faiss.IndexFlatIP
        self._id_map: list[str] = []            # position → file_id
        self._meta_map: dict[str, dict] = {}    # file_id → {model, version, ref, media_file}

    def invalidate(self) -> None:
        """Force index rebuild on next search."""
        with self._lock:
            self._index = None
            self._id_map = []
            self._meta_map = {}

    def update_session(self, session: Session) -> None:
        """Thread-safely replace the SQLAlchemy session reference.

        Must be called under the lock to avoid a data race with _build_index()
        reading self._session on the first search after the index was invalidated.
        """
        with self._lock:
            self._session = session

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
            import faiss
            import numpy as np
        except ImportError:
            logger.debug("faiss not installed; FaissVectorIndex returning empty results")
            return []

        try:
            query_vector = clip_embedding.embedding_from_bytes(query_embedding).astype("float32")
        except Exception:
            return []

        index, id_map, meta_map = self._get_or_build_index(faiss)
        if index is None or index.ntotal == 0:
            return []

        # Normalise query for inner-product cosine similarity
        import numpy as np
        norm = float(np.linalg.norm(query_vector)) or 1.0
        query_norm = (query_vector / norm).reshape(1, -1)

        # Fetch more candidates when filters will reduce the result set
        fetch_k = limit * 8 if (place_filter or date_from or date_to) else limit * 2
        fetch_k = min(fetch_k, index.ntotal)

        distances, indices = index.search(query_norm, fetch_k)

        hits: list[VectorSearchHit] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            file_id = id_map[idx]
            meta = meta_map.get(file_id)
            if meta is None:
                continue
            media_file: MediaFile = meta["media_file"]

            # Apply filters
            if place_filter:
                tags = {t.tag_value for t in (media_file.tags or [])}
                if place_filter not in tags:
                    continue
            if isinstance(date_from, datetime) and media_file.exif_datetime and media_file.exif_datetime < date_from:
                continue
            if isinstance(date_to, datetime) and media_file.exif_datetime and media_file.exif_datetime > date_to:
                continue

            hits.append(
                VectorSearchHit(
                    media_file=media_file,
                    distance=max(0.0, 1.0 - float(dist)),
                    embedding_ref=meta["ref"],
                    model_name=meta["model"],
                    version=meta["version"],
                )
            )
            if len(hits) >= limit:
                break

        return hits

    def _get_or_build_index(self, faiss: Any) -> tuple[Any | None, list[str], dict]:
        with self._lock:
            if self._index is not None:
                return self._index, self._id_map, self._meta_map
            self._index, self._id_map, self._meta_map = self._build_index(faiss)
            return self._index, self._id_map, self._meta_map

    def _build_index(self, faiss: Any) -> tuple[Any, list[str], dict]:
        import numpy as np

        statement = (
            select(MediaEmbedding, MediaFile)
            .join(MediaFile, MediaFile.file_id == MediaEmbedding.file_id)
            .where(MediaFile.status != "missing")
        )
        rows = self._session.execute(statement).all()
        if not rows:
            dim = 512  # default CLIP dim; index will be empty
            index = faiss.IndexFlatIP(dim)
            return index, [], {}

        vectors: list[Any] = []
        id_map: list[str] = []
        meta_map: dict[str, dict] = {}

        for embedding, media_file in rows:
            vec = self._load_embedding_vector(embedding.embedding_ref)
            if vec is None:
                continue
            norm = float(np.linalg.norm(vec)) or 1.0
            vectors.append(vec / norm)
            id_map.append(str(media_file.file_id))
            meta_map[str(media_file.file_id)] = {
                "ref": embedding.embedding_ref,
                "model": embedding.model_name,
                "version": embedding.version,
                "media_file": media_file,
            }

        if not vectors:
            index = faiss.IndexFlatIP(512)
            return index, [], {}

        matrix = np.vstack(vectors).astype("float32")
        dim = matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
        logger.info("FaissVectorIndex built: %d vectors, dim=%d", index.ntotal, dim)
        return index, id_map, meta_map

    def _load_embedding_vector(self, embedding_ref: str):
        try:
            import numpy as np

            path = Path(embedding_ref)
            absolute_path = path if path.is_absolute() else self._embeddings_root / (
                path.relative_to("embeddings") if path.parts and path.parts[0] == "embeddings" else path
            )
            if not absolute_path.is_file():
                return None
            return np.load(absolute_path).astype("float32")
        except Exception:
            return None


def build_vector_index(session: Session, *, embeddings_root: Path, backend: str = "auto") -> VectorIndexBackend:
    """Factory that returns the best available vector index backend.

    backend='auto': use FaissVectorIndex if faiss is installed, else LocalNumpyVectorIndex
    backend='faiss': always try FaissVectorIndex (logs warning if faiss missing)
    backend='numpy': always use LocalNumpyVectorIndex

    When FAISS is selected the instance is stored as a process-level singleton
    so that invalidate_global_vector_index() can trigger a rebuild after
    semantic maintenance without needing a direct reference.
    """
    global _global_faiss_index

    if backend == "numpy":
        return LocalNumpyVectorIndex(session, embeddings_root=embeddings_root)
    try:
        import faiss  # noqa: F401
        if _global_faiss_index is None:
            _global_faiss_index = FaissVectorIndex(session, embeddings_root=embeddings_root)
        else:
            # Update session reference thread-safely (avoids race with _build_index)
            _global_faiss_index.update_session(session)
        return _global_faiss_index
    except ImportError:
        if backend == "faiss":
            logger.warning("faiss not installed; falling back to LocalNumpyVectorIndex")
        return LocalNumpyVectorIndex(session, embeddings_root=embeddings_root)
