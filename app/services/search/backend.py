"""SQLAlchemy backend for hybrid search."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sqlalchemy import func, or_, select, text
from sqlalchemy.orm import Session

import os

from app.models.annotation import MediaAnnotation
from app.models.face import Face
from app.models.media import MediaFile
from app.models.person import Person
from app.models.semantic import MediaAnalysisSignal, MediaOCR, MediaOCRGram, SearchDocument, SearchFeedback, SearchWeightProfile
from app.services.search.vocab import TagVocabularyCache
from app.models.tag import Tag
from app.services.embedding import clip as clip_embedding
from app.services.search.vector import build_vector_index, VectorIndexBackend

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
    def __init__(
        self,
        session: Session,
        *,
        embeddings_root: Path,
        vector_index: VectorIndexBackend | None = None,
    ) -> None:
        self._session = session
        self._embeddings_root = embeddings_root
        _backend_setting = os.environ.get("PHOTOME_VECTOR_BACKEND", "auto")
        self._vector_index = vector_index or build_vector_index(
            session, embeddings_root=embeddings_root, backend=_backend_setting
        )
        self._tag_vocab_cache = TagVocabularyCache(session)

    def get_tag_vocabulary(self):
        """Return current TagVocabulary snapshot (TTL-cached, from DB)."""
        return self._tag_vocab_cache.get()

    def load_persisted_weights(self, intent: str, reason: str) -> dict[str, float] | None:
        """Return persisted weights for intent+reason, or None if not customised."""
        row = self._session.execute(
            select(SearchWeightProfile).where(
                SearchWeightProfile.intent == intent,
                SearchWeightProfile.reason == reason,
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        return {"ocr": row.w_ocr, "clip": row.w_clip, "shadow": row.w_shadow}

    def load_feedback_sets(self) -> tuple[set[str], set[str]]:
        """Return (hidden_file_ids, promoted_file_ids) from SearchFeedback.

        Only global feedback (query_hint='') is considered for the hidden set
        so that query-scoped hides don't bleed into unrelated queries.
        Promoted feedback is applied globally regardless of query_hint.
        """
        rows = self._session.execute(
            select(SearchFeedback.file_id, SearchFeedback.action, SearchFeedback.query_hint)
        ).all()
        hidden: set[str] = set()
        promoted: set[str] = set()
        for file_id, action, query_hint in rows:
            if action == "hide" and not query_hint:
                hidden.add(str(file_id))
            elif action == "promote":
                promoted.add(str(file_id))
        return hidden, promoted

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

        results = self._search_by_normalized_document(query, limit=limit)
        if results:
            return results[:limit]

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

    def _search_by_normalized_document(self, query: str, *, limit: int) -> list[dict]:
        fts_results = self._search_by_fts_document(query, limit=limit)
        if fts_results:
            return fts_results

        pattern = _like_pattern(query)
        statement = (
            select(MediaFile, SearchDocument)
            .join(SearchDocument, SearchDocument.file_id == MediaFile.file_id)
            .where(MediaFile.status != "missing")
            .where(
                or_(
                    func.lower(SearchDocument.search_text).like(pattern),
                    func.lower(SearchDocument.keyword_text).like(pattern),
                    func.lower(SearchDocument.semantic_text).like(pattern),
                )
            )
            .order_by(MediaFile.exif_datetime.desc().nullslast(), MediaFile.updated_at.desc())
            .limit(max(1, limit * 4))
        )
        results = []
        for media_file, document in self._session.execute(statement):
            ocr = self._session.get(MediaOCR, media_file.file_id)
            analysis = self._session.get(MediaAnalysisSignal, media_file.file_id)
            result = self._result_dict(media_file, ocr=ocr, analysis=analysis, match_reason="shadow")
            result["search_document_version"] = document.version
            results.append(result)
        return results[:limit]

    def _search_by_fts_document(self, query: str, *, limit: int) -> list[dict]:
        if self._session.bind is None or self._session.bind.dialect.name != "sqlite":
            return []
        fts_query = _fts_query(query)
        if not fts_query:
            return []

        fetch = max(1, limit * 4)
        # Primary FTS (unicode61 — word boundary, reliable for English)
        rows: list[tuple[str, float]] = []
        try:
            primary_rows = self._session.execute(
                text(
                    "SELECT file_id, bm25(search_documents_fts) AS score "
                    "FROM search_documents_fts "
                    "WHERE search_documents_fts MATCH :query "
                    "ORDER BY score ASC LIMIT :limit"
                ),
                {"query": fts_query, "limit": fetch},
            ).all()
            rows = [(str(r[0]), float(r[1] or 0.0)) for r in primary_rows]
        except Exception:
            pass

        # Trigram FTS (character n-gram — Korean substring search)
        # Uses raw LIKE-safe query since trigram doesn't support FTS5 prefix syntax
        trigram_query = query.strip()
        if trigram_query:
            try:
                trigram_rows = self._session.execute(
                    text(
                        "SELECT file_id, bm25(search_documents_fts_ko) AS score "
                        "FROM search_documents_fts_ko "
                        "WHERE search_documents_fts_ko MATCH :query "
                        "ORDER BY score ASC LIMIT :limit"
                    ),
                    {"query": trigram_query, "limit": fetch},
                ).all()
                # Merge: prefer lower (better) BM25 score per file_id
                existing: dict[str, float] = {fid: score for fid, score in rows}
                for r in trigram_rows:
                    fid = str(r[0])
                    s = float(r[1] or 0.0)
                    if fid not in existing or s < existing[fid]:
                        existing[fid] = s
                rows = list(existing.items())
            except Exception:
                pass  # trigram table may not exist on older SQLite

        if not rows:
            return []

        scored_ids: dict[str, float] = {}
        for file_id, score in rows:
            scored_ids.setdefault(str(file_id), float(score or 0.0))
        if not scored_ids:
            return []

        statement = select(MediaFile, SearchDocument).join(
            SearchDocument,
            SearchDocument.file_id == MediaFile.file_id,
        ).where(
            MediaFile.file_id.in_(list(scored_ids)),
            MediaFile.status != "missing",
        )
        by_id: dict[str, tuple[MediaFile, SearchDocument]] = {
            media_file.file_id: (media_file, document)
            for media_file, document in self._session.execute(statement)
        }
        results = []
        for file_id, score in sorted(scored_ids.items(), key=lambda item: item[1]):
            row = by_id.get(file_id)
            if row is None:
                continue
            media_file, document = row
            ocr = self._session.get(MediaOCR, media_file.file_id)
            analysis = self._session.get(MediaAnalysisSignal, media_file.file_id)
            result = self._result_dict(media_file, ocr=ocr, analysis=analysis, match_reason="shadow")
            result["search_document_version"] = document.version
            result["fts_score"] = score
            results.append(result)
            if len(results) >= limit:
                break
        return results

    def _tagged_shadow_results(self, query: str, *, limit: int) -> list[dict]:
        lowered = query.casefold().strip()
        if not lowered:
            return []

        # Build the set of tag values to search: exact match + synonyms
        search_values = {lowered} | TAG_SYNONYMS.get(lowered, set())
        # Also check each token of a multi-word query
        for token in lowered.split():
            search_values |= TAG_SYNONYMS.get(token, set())

        # Resolve named persons: if any token matches a Person.display_name,
        # add their person-XXXXXX tag so the face cluster is found
        person_tag_ids = self._resolve_person_tag_ids(query)
        search_values.update(person_tag_ids)

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
        results = []
        for hit in self._vector_index.search(
            query_embedding,
            limit=limit,
            place_filter=place_filter,
            date_from=date_from,
            date_to=date_to,
        ):
            media_file = hit.media_file
            ocr = self._session.get(MediaOCR, media_file.file_id)
            analysis = self._session.get(MediaAnalysisSignal, media_file.file_id)
            result = self._result_dict(media_file, ocr=ocr, analysis=analysis, match_reason="clip")
            result["distance"] = hit.distance
            result["embedding_model"] = hit.model_name
            result["embedding_version"] = hit.version
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

    def _resolve_person_tag_ids(self, query: str) -> set[str]:
        """Return person tag values (e.g. 'person-abc123') for any Person whose
        display_name matches a token in the query."""
        lowered = query.casefold()
        tokens = [t for t in lowered.split() if len(t) >= 2]
        if not tokens:
            return set()
        persons = self._session.scalars(select(Person)).all()
        matched_ids: set[str] = set()
        for person in persons:
            name = person.display_name.casefold()
            if any(token in name or name in token for token in tokens):
                # person tags are stored as "person-{person_id}" style values
                matched_ids.add(f"person-{person.id:06d}")
                matched_ids.add(person.display_name.casefold())
        return matched_ids

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


def _fts_query(query: str) -> str:
    tokens = re.findall(r"[0-9A-Za-z가-힣_]+", query.casefold())
    if not tokens:
        return ""
    deduped = list(dict.fromkeys(token for token in tokens if token))
    return " OR ".join(f'"{token}"' for token in deduped)


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
