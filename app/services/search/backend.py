"""SQLAlchemy backend for hybrid search."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.models.face import Face
from app.models.annotation import MediaAnnotation
from app.models.media import MediaFile
from app.models.semantic import MediaAnalysisSignal, MediaEmbedding, MediaOCR, MediaOCRGram
from app.models.tag import Tag
from app.services.embedding import clip as clip_embedding

FACE_HINTS = {
    "face", "faces", "person", "people", "portrait", "selfie",
    "woman", "women", "female", "girl", "baby", "infant", "toddler", "child", "kid",
    "얼굴", "사람", "인물", "셀카", "남자", "여자", "여성", "아기", "애기", "아이", "어린이",
    "엄마", "아빠", "가족", "친구", "커플",
}
TEXT_HINTS = {"text", "ocr", "document", "receipt", "screen", "텍스트", "글씨", "문서", "영수증", "화면", "스크린샷"}

# Korean ↔ English tag synonyms: searching either side will also match the other
TAG_SYNONYMS: dict[str, set[str]] = {
    "baby": {"아기", "애기", "infant", "newborn", "toddler"},
    "아기": {"baby", "infant", "애기", "toddler"},
    "person": {"사람", "얼굴", "인물", "portrait"},
    "얼굴": {"face", "person", "portrait", "인물"},
    "receipt": {"영수증", "document"},
    "영수증": {"receipt", "document"},
    "screenshot": {"스크린샷", "screen", "화면"},
    "스크린샷": {"screenshot", "screen", "화면"},
    "food": {"음식", "meal"},
    "음식": {"food", "meal"},
    "outdoor": {"야외", "자연", "nature"},
    "야외": {"outdoor", "nature"},
    "beach": {"바다", "해변", "sea", "ocean"},
    "바다": {"beach", "sea", "ocean"},
    "travel": {"여행", "trip", "vacation"},
    "여행": {"travel", "trip", "vacation"},
    "wedding": {"결혼식", "결혼"},
    "결혼식": {"wedding", "결혼"},
    "birthday": {"생일", "celebration"},
    "생일": {"birthday", "celebration"},
    "night": {"야경", "야간"},
    "야경": {"night", "야간"},
    "celebration": {"파티", "생일", "party", "birthday"},
    "파티": {"party", "celebration", "birthday"},
    "dog": {"강아지", "puppy", "멍멍이"},
    "강아지": {"dog", "puppy"},
    "cat": {"고양이", "kitten"},
    "고양이": {"cat", "kitten"},
    "mountain": {"산", "등산", "hiking"},
    "산": {"mountain", "hiking", "등산"},
    "cake": {"케이크", "birthday"},
    "케이크": {"cake", "birthday"},
    "sunset": {"일몰", "노을"},
    "일몰": {"sunset", "노을"},
    "vehicle": {"차", "자동차", "car"},
    "car": {"차", "자동차", "vehicle"},
}


class SqlAlchemyHybridSearchBackend:
    def __init__(self, session: Session, *, embeddings_root: Path) -> None:
        self._session = session
        self._embeddings_root = embeddings_root

    def search_by_ocr(self, query: str, *, limit: int) -> list[dict]:
        pattern = _like_pattern(query)
        main_statement = (
            select(MediaFile, MediaOCR, MediaAnalysisSignal)
            .join(MediaOCR, MediaOCR.file_id == MediaFile.file_id)
            .outerjoin(MediaAnalysisSignal, MediaAnalysisSignal.file_id == MediaFile.file_id)
            .where(MediaFile.status != "missing")
            .where(func.lower(MediaOCR.text_content).like(pattern))
            .order_by(MediaFile.exif_datetime.desc().nullslast(), MediaFile.updated_at.desc())
            .limit(max(1, limit * 4))
        )
        rows = []
        seen_ids: set[str] = set()
        for media_file, ocr, analysis in self._session.execute(main_statement):
            seen_ids.add(media_file.file_id)
            rows.append(
                self._result_dict(
                    media_file,
                    ocr=ocr,
                    analysis=analysis,
                    match_reason="ocr",
                    ocr_match_kind=_ocr_match_kind(query, ocr.text_content),
                )
            )

        # Supplement with n-gram index results for Korean text when main results are few
        grams = _korean_2grams(query)
        if grams and len(rows) < limit:
            gram_hits = self._ngram_scored_ids(grams, exclude=seen_ids, limit=limit * 2)
            if gram_hits:
                max_hits = max(count for _, count in gram_hits) or 1
                hit_counts = {fid: count for fid, count in gram_hits}
                ngram_statement = (
                    select(MediaFile, MediaOCR, MediaAnalysisSignal)
                    .join(MediaOCR, MediaOCR.file_id == MediaFile.file_id)
                    .outerjoin(MediaAnalysisSignal, MediaAnalysisSignal.file_id == MediaFile.file_id)
                    .where(MediaFile.file_id.in_([fid for fid, _ in gram_hits]))
                    .where(MediaFile.status != "missing")
                )
                for media_file, ocr, analysis in self._session.execute(ngram_statement):
                    row = self._result_dict(
                        media_file,
                        ocr=ocr,
                        analysis=analysis,
                        match_reason="ocr",
                        ocr_match_kind="ngram",
                    )
                    # Encode gram coverage as a normalised pre-score (used by RRF)
                    row["ngram_score"] = hit_counts.get(media_file.file_id, 0) / max_hits
                    rows.append(row)

        return rows[:limit]

    def _ngram_scored_ids(
        self, grams: list[str], *, exclude: set[str], limit: int
    ) -> list[tuple[str, int]]:
        statement = (
            select(MediaOCRGram.file_id, func.count().label("hit_count"))
            .where(MediaOCRGram.gram.in_(grams))
            .group_by(MediaOCRGram.file_id)
            .order_by(func.count().desc())
            .limit(limit)
        )
        return [
            (file_id, int(hit_count))
            for file_id, hit_count in self._session.execute(statement)
            if file_id not in exclude
        ]

    def search_by_shadow_doc(self, query: str, *, limit: int) -> list[dict]:
        tagged = self._tagged_shadow_results(query, limit=limit)
        hinted = self._hinted_shadow_results(query, limit=limit, exclude_file_ids={str(item["file_id"]) for item in tagged})
        if hinted:
            return (tagged + hinted)[:limit]
        if tagged:
            return tagged

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

    def _tagged_shadow_results(self, query: str, *, limit: int) -> list[dict]:
        lowered = query.casefold().strip()
        if not lowered:
            return []

        # Build the set of tag values to search: exact match + synonyms
        search_values = {lowered} | TAG_SYNONYMS.get(lowered, set())
        # Also check each token of a multi-word query
        for token in lowered.split():
            search_values |= TAG_SYNONYMS.get(token, set())

        exact_statement = (
            select(MediaFile, Tag.tag_value)
            .join(Tag, Tag.file_id == MediaFile.file_id)
            .where(MediaFile.status != "missing")
            .where(func.lower(Tag.tag_value).in_(list(search_values)))
            .order_by(
                (func.lower(Tag.tag_value) == lowered).desc(),
                (Tag.tag_type == "auto").desc(),
                (Tag.tag_type == "custom").desc(),
                MediaFile.exif_datetime.desc().nullslast(),
                MediaFile.updated_at.desc(),
            )
            .limit(max(1, limit))
        )
        results = []
        seen: set[str] = set()
        for media_file, matched_tag_value in self._session.execute(exact_statement):
            if media_file.file_id in seen:
                continue
            seen.add(media_file.file_id)
            ocr = self._session.get(MediaOCR, media_file.file_id)
            analysis = self._session.get(MediaAnalysisSignal, media_file.file_id)
            result = self._result_dict(media_file, ocr=ocr, analysis=analysis, match_reason="tag")
            result["tag_exact_match"] = matched_tag_value.casefold() == lowered
            result["matched_tag"] = matched_tag_value
            results.append(result)
        return results

    def _hinted_shadow_results(self, query: str, *, limit: int, exclude_file_ids: set[str] | None = None) -> list[dict]:
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
        if exclude_file_ids:
            statement = statement.where(MediaFile.file_id.not_in(exclude_file_ids))
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

    def suggest_related_tags(self, query: str, *, limit: int = 8) -> list[str]:
        """Return existing tag values that partially match the query.

        Used to populate "did you mean?" suggestions when a search returns
        no results.
        """
        if not query.strip():
            return []
        lowered = query.casefold().strip()
        # Collect synonyms as candidate values to look for
        candidates = {lowered} | TAG_SYNONYMS.get(lowered, set())
        for token in lowered.split():
            candidates |= TAG_SYNONYMS.get(token, set())

        # Find tags that exist in the DB matching any candidate or partial
        pattern = f"%{lowered}%"
        statement = (
            select(Tag.tag_value, func.count(Tag.file_id).label("freq"))
            .where(
                or_(
                    func.lower(Tag.tag_value).in_(list(candidates)),
                    func.lower(Tag.tag_value).like(pattern),
                )
            )
            .group_by(Tag.tag_value)
            .order_by(func.count(Tag.file_id).desc())
            .limit(limit)
        )
        return [value for value, _ in self._session.execute(statement)]

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


def _korean_2grams(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", text)
    chars = [ch for ch in compact if "가" <= ch <= "힣"]
    return list(dict.fromkeys(
        "".join(chars[i: i + 2]) for i in range(max(0, len(chars) - 1))
    ))
