"""SQLAlchemy backend for hybrid search."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.models.face import Face
from app.models.annotation import MediaAnnotation
from app.models.media import MediaFile
from app.models.semantic import MediaAnalysisSignal, MediaEmbedding, MediaOCR
from app.models.tag import Tag
from app.services.embedding import clip as clip_embedding

FACE_HINTS = {
    "face", "faces", "person", "people", "portrait", "selfie",
    "woman", "women", "female", "girl", "baby", "infant", "toddler", "child", "kid",
    "얼굴", "사람", "인물", "셀카", "남자", "여자", "여성", "아기", "애기", "아이", "어린이",
}
TEXT_HINTS = {"text", "ocr", "document", "receipt", "screen", "텍스트", "글씨", "문서", "영수증", "화면", "스크린샷"}


class SqlAlchemyHybridSearchBackend:
    def __init__(self, session: Session, *, embeddings_root: Path) -> None:
        self._session = session
        self._embeddings_root = embeddings_root

    def search_by_ocr(self, query: str, *, limit: int) -> list[dict]:
        pattern = _like_pattern(query)
        statement = (
            select(MediaFile, MediaOCR, MediaAnalysisSignal)
            .join(MediaOCR, MediaOCR.file_id == MediaFile.file_id)
            .outerjoin(MediaAnalysisSignal, MediaAnalysisSignal.file_id == MediaFile.file_id)
            .where(MediaFile.status != "missing")
            .where(func.lower(MediaOCR.text_content).like(pattern))
            .order_by(MediaFile.exif_datetime.desc().nullslast(), MediaFile.updated_at.desc())
            .limit(max(1, limit * 4))
        )
        rows = []
        for media_file, ocr, analysis in self._session.execute(statement):
            rows.append(
                self._result_dict(
                    media_file,
                    ocr=ocr,
                    analysis=analysis,
                    match_reason="ocr",
                    ocr_match_kind=_ocr_match_kind(query, ocr.text_content),
                )
            )
        return rows[:limit]

    def search_by_shadow_doc(self, query: str, *, limit: int) -> list[dict]:
        hinted = self._hinted_shadow_results(query, limit=limit)
        if hinted:
            return hinted

        pattern = _like_pattern(query)
        statement = (
            select(MediaFile)
            .outerjoin(MediaAnnotation, MediaAnnotation.file_id == MediaFile.file_id)
            .outerjoin(Tag, Tag.file_id == MediaFile.file_id)
            .outerjoin(MediaOCR, MediaOCR.file_id == MediaFile.file_id)
            .where(MediaFile.status != "missing")
            .where(
                or_(
                    func.lower(MediaFile.filename).like(pattern),
                    func.lower(MediaFile.current_path).like(pattern),
                    func.lower(MediaFile.relative_path).like(pattern),
                    func.lower(MediaAnnotation.title).like(pattern),
                    func.lower(MediaAnnotation.description).like(pattern),
                    func.lower(Tag.tag_value).like(pattern),
                    func.lower(MediaOCR.text_content).like(pattern),
                )
            )
            .group_by(MediaFile.file_id)
            .order_by(MediaFile.exif_datetime.desc().nullslast(), MediaFile.updated_at.desc())
            .limit(max(1, limit * 4))
        )
        results = []
        for media_file in self._session.scalars(statement):
            ocr = self._session.get(MediaOCR, media_file.file_id)
            analysis = self._session.get(MediaAnalysisSignal, media_file.file_id)
            results.append(self._result_dict(media_file, ocr=ocr, analysis=analysis, match_reason="shadow"))
        return results[:limit]

    def _hinted_shadow_results(self, query: str, *, limit: int) -> list[dict]:
        lowered = query.casefold().strip()
        wants_faces = any(hint in lowered for hint in FACE_HINTS)
        wants_text = any(hint in lowered for hint in TEXT_HINTS)
        if not wants_faces and not wants_text:
            return []

        statement = (
            select(MediaFile, MediaOCR, MediaAnalysisSignal, func.count(Face.id).label("face_count"))
            .outerjoin(MediaOCR, MediaOCR.file_id == MediaFile.file_id)
            .outerjoin(MediaAnalysisSignal, MediaAnalysisSignal.file_id == MediaFile.file_id)
            .outerjoin(Face, Face.file_id == MediaFile.file_id)
            .where(MediaFile.status != "missing")
            .group_by(MediaFile.file_id)
        )
        if wants_text:
            statement = statement.where(
                or_(
                    MediaAnalysisSignal.is_text_heavy.is_(True),
                    MediaAnalysisSignal.is_document_like.is_(True),
                    MediaAnalysisSignal.is_screenshot_like.is_(True),
                    MediaOCR.text_content != "",
                )
            )
        statement = statement.order_by(
            func.count(Face.id).desc(),
            MediaAnalysisSignal.is_screenshot_like.desc().nullslast(),
            MediaAnalysisSignal.is_document_like.desc().nullslast(),
            MediaFile.exif_datetime.desc().nullslast(),
            MediaFile.updated_at.desc(),
        ).limit(max(1, limit))

        results = []
        for media_file, ocr, analysis, face_count in self._session.execute(statement):
            result = self._result_dict(media_file, ocr=ocr, analysis=analysis, match_reason="shadow")
            result["face_count"] = int(face_count or 0)
            result["hint_match"] = "face" if wants_faces else "text"
            results.append(result)
        return results

    def search_by_embedding(
        self,
        query_embedding: bytes,
        *,
        limit: int,
        place_filter: str | None = None,
        date_from: Any | None = None,
        date_to: Any | None = None,
    ) -> list[dict]:
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

        scored: list[tuple[float, MediaFile]] = []
        for embedding, media_file in self._session.execute(statement):
            vector = self._load_embedding_vector(embedding.embedding_ref)
            if vector is None or vector.size != query_vector.size:
                continue
            denominator = (float(np.linalg.norm(vector)) or 1.0) * query_norm
            similarity = float(np.dot(query_vector, vector) / denominator)
            distance = 1.0 - similarity
            scored.append((distance, media_file))

        scored.sort(key=lambda item: item[0])
        results = []
        for distance, media_file in scored[:limit]:
            ocr = self._session.get(MediaOCR, media_file.file_id)
            analysis = self._session.get(MediaAnalysisSignal, media_file.file_id)
            result = self._result_dict(media_file, ocr=ocr, analysis=analysis, match_reason="clip")
            result["distance"] = distance
            results.append(result)
        return results

    def encode_text(self, query: str) -> bytes:
        try:
            clip_embedding.ensure_models()
            return clip_embedding.encode_text(query)
        except Exception:
            return b""

    def _load_embedding_vector(self, embedding_ref: str):
        try:
            import numpy as np

            path = Path(embedding_ref)
            absolute_path = path if path.is_absolute() else self._embeddings_root / path.relative_to("embeddings")
            if not absolute_path.is_file():
                return None
            return np.load(absolute_path).astype("float32")
        except Exception:
            return None

    def _result_dict(
        self,
        media_file: MediaFile,
        *,
        ocr: MediaOCR | None = None,
        analysis: MediaAnalysisSignal | None = None,
        match_reason: str,
        ocr_match_kind: str | None = None,
    ) -> dict:
        tags = [
            {"type": tag_type, "value": tag_value}
            for tag_type, tag_value in self._session.execute(
                select(Tag.tag_type, Tag.tag_value)
                .where(Tag.file_id == media_file.file_id)
                .order_by(Tag.tag_type.asc(), Tag.tag_value.asc())
            )
        ]
        face_count = int(
            self._session.scalar(select(func.count()).select_from(Face).where(Face.file_id == media_file.file_id)) or 0
        )
        payload = {
            "file_id": media_file.file_id,
            "filename": media_file.filename,
            "current_path": media_file.current_path,
            "relative_path": media_file.relative_path,
            "media_kind": media_file.media_kind,
            "status": media_file.status,
            "captured_at": media_file.exif_datetime,
            "updated_at": media_file.updated_at,
            "tags": tags,
            "ocr_text": ocr.text_content if ocr is not None else "",
            "ocr_engine": ocr.engine if ocr is not None else None,
            "ocr_match_kind": ocr_match_kind,
            "face_count": face_count,
            "match_reason": match_reason,
        }
        if analysis is not None:
            payload.update(
                {
                    "text_char_count": analysis.text_char_count,
                    "text_line_count": analysis.text_line_count,
                    "edge_density": analysis.edge_density,
                    "brightness": analysis.brightness,
                    "is_text_heavy": analysis.is_text_heavy,
                    "is_document_like": analysis.is_document_like,
                    "is_screenshot_like": analysis.is_screenshot_like,
                }
            )
        return payload


def _like_pattern(query: str) -> str:
    return f"%{query.casefold().strip()}%"


def _ocr_match_kind(query: str, text: str) -> str | None:
    lowered_query = query.casefold().strip()
    lowered_text = text.casefold()
    if not lowered_query:
        return None
    if lowered_query in lowered_text:
        return "phrase" if " " in lowered_query else "word"
    tokens = [token for token in lowered_query.split() if token]
    if tokens and all(token in lowered_text for token in tokens):
        return "word"
    return None
