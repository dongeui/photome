"""SQLAlchemy backend for hybrid search."""

from __future__ import annotations

from datetime import datetime, time
import logging
import re
from pathlib import Path
from typing import Any

from sqlalchemy import func, or_, select, text
from sqlalchemy.orm import Session

import os

logger = logging.getLogger(__name__)

# CLIP model has a 77-token limit; long queries degrade embedding quality.
# Truncating at ~200 chars keeps the most relevant content within token budget.
_CLIP_MAX_CHARS = 200

# Warn once per process if FTS tables are missing
_fts_warning_emitted = False

from app.models.annotation import MediaAnnotation
from app.models.face import Face
from app.models.media import MediaFile
from app.models.person import Person
from app.models.semantic import MediaAnalysisSignal, MediaOCR, MediaOCRGram, SearchDocument, SearchEvent, SearchFeedback, SearchWeightProfile
from app.services.analysis.clip_lexicon import load_concept_aliases
from app.services.search.vocab import TagVocabularyCache
from app.models.tag import Tag
from app.services.embedding import clip as clip_embedding
from app.services.search.planner import QueryPlan
from app.services.search.vector import build_vector_index, VectorIndexBackend
from app.services.search.hybrid import FACE_HINTS, TEXT_HINTS
from app.services.search.synonyms import load_tag_synonyms


class SqlAlchemyHybridSearchBackend:
    def __init__(
        self,
        session: Session,
        *,
        embeddings_root: Path,
        vector_index: VectorIndexBackend | None = None,
        clip_enabled: bool = True,
        log_events: bool = True,
    ) -> None:
        self._session = session
        self._embeddings_root = embeddings_root
        self._clip_enabled = clip_enabled
        self._log_events = log_events
        _backend_setting = os.environ.get("PHOTOME_VECTOR_BACKEND", "auto")
        self._vector_index = vector_index or build_vector_index(
            session, embeddings_root=embeddings_root, backend=_backend_setting
        )
        self._tag_vocab_cache = TagVocabularyCache(session)

    def get_tag_vocabulary(self):
        """Return current TagVocabulary snapshot (TTL-cached, from DB)."""
        return self._tag_vocab_cache.get()

    def supports_parallel_channels(self) -> bool:
        """SQLAlchemy sessions are not safe to share across worker threads."""
        return False

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

    def search_by_ocr(self, query: str, *, limit: int, plan: QueryPlan | None = None) -> list[dict]:
        pattern = _like_pattern(query)
        main_statement = (
            select(MediaFile, MediaOCR, MediaAnalysisSignal)
            .join(MediaOCR, MediaOCR.file_id == MediaFile.file_id)
            .outerjoin(MediaAnalysisSignal, MediaAnalysisSignal.file_id == MediaFile.file_id)
            .where(MediaFile.status != "missing")
            .where(func.lower(MediaOCR.text_content).like(pattern, escape=_LIKE_ESCAPE))
            .order_by(MediaFile.exif_datetime.desc().nullslast(), MediaFile.updated_at.desc())
            .limit(max(1, limit * 4))
        )
        main_rows = list(self._session.execute(main_statement))
        seen_ids = {mf.file_id for mf, _, _ in main_rows}

        # Supplement with n-gram index results for Korean text when main results are few
        ngram_rows: list[tuple] = []
        hit_counts: dict[str, float] = {}
        grams = _korean_2grams(query)
        if grams and len(main_rows) < limit:
            gram_hits = self._ngram_scored_ids(grams, exclude=seen_ids, limit=limit * 2)
            if gram_hits:
                max_hits = max(count for _, count in gram_hits) or 1
                hit_counts = {fid: count / max_hits for fid, count in gram_hits}
                ngram_statement = (
                    select(MediaFile, MediaOCR, MediaAnalysisSignal)
                    .join(MediaOCR, MediaOCR.file_id == MediaFile.file_id)
                    .outerjoin(MediaAnalysisSignal, MediaAnalysisSignal.file_id == MediaFile.file_id)
                    .where(MediaFile.file_id.in_([fid for fid, _ in gram_hits]))
                    .where(MediaFile.status != "missing")
                )
                ngram_rows = list(self._session.execute(ngram_statement))

        all_file_ids = [mf.file_id for mf, _, _ in main_rows + ngram_rows]
        tags_by_file, face_counts, _, _ = self._batch_load_supplements(all_file_ids)

        rows: list[dict] = []
        for media_file, ocr, analysis in main_rows:
            fid = media_file.file_id
            rows.append(self._build_result_dict(
                media_file, ocr=ocr, analysis=analysis,
                match_reason="ocr",
                ocr_match_kind=_ocr_match_kind(query, ocr.text_content),
                tags=tags_by_file.get(fid, []),
                face_count=face_counts.get(fid, 0),
            ))
        for media_file, ocr, analysis in ngram_rows:
            fid = media_file.file_id
            row = self._build_result_dict(
                media_file, ocr=ocr, analysis=analysis,
                match_reason="ocr", ocr_match_kind="ngram",
                tags=tags_by_file.get(fid, []),
                face_count=face_counts.get(fid, 0),
            )
            row["ngram_score"] = hit_counts.get(fid, 0.0)
            rows.append(row)

        return self._filter_results(rows, plan, limit=limit)

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

    def search_by_shadow_doc(self, query: str, *, limit: int, plan: QueryPlan | None = None) -> list[dict]:
        tagged = self._tagged_shadow_results(query, limit=limit)
        hinted = self._hinted_shadow_results(query, limit=limit, exclude_file_ids={str(item["file_id"]) for item in tagged})
        if hinted:
            return self._filter_results(tagged + hinted, plan, limit=limit)
        if tagged:
            return self._filter_results(tagged, plan, limit=limit)

        results = self._search_by_normalized_document(query, limit=limit)
        if results:
            return self._filter_results(results, plan, limit=limit)

        pattern = _like_pattern(query)
        statement = (
            select(MediaFile)
            .outerjoin(MediaAnnotation, MediaAnnotation.file_id == MediaFile.file_id)
            .outerjoin(Tag, Tag.file_id == MediaFile.file_id)
            .outerjoin(MediaOCR, MediaOCR.file_id == MediaFile.file_id)
            .where(MediaFile.status != "missing")
            .where(
                or_(
                    func.lower(MediaFile.filename).like(pattern, escape=_LIKE_ESCAPE),
                    func.lower(MediaFile.current_path).like(pattern, escape=_LIKE_ESCAPE),
                    func.lower(MediaFile.relative_path).like(pattern, escape=_LIKE_ESCAPE),
                    func.lower(MediaAnnotation.title).like(pattern, escape=_LIKE_ESCAPE),
                    func.lower(MediaAnnotation.description).like(pattern, escape=_LIKE_ESCAPE),
                    func.lower(Tag.tag_value).like(pattern, escape=_LIKE_ESCAPE),
                    func.lower(MediaOCR.text_content).like(pattern, escape=_LIKE_ESCAPE),
                )
            )
            .group_by(MediaFile.file_id)
            .order_by(MediaFile.exif_datetime.desc().nullslast(), MediaFile.updated_at.desc())
            .limit(max(1, limit * 4))
        )
        fallback_files = list(self._session.scalars(statement))
        file_ids = [mf.file_id for mf in fallback_files]
        tags_by_file, face_counts, ocr_by_file, analysis_by_file = self._batch_load_supplements(file_ids)
        results = []
        for media_file in fallback_files:
            fid = media_file.file_id
            results.append(self._build_result_dict(
                media_file,
                ocr=ocr_by_file.get(fid),
                analysis=analysis_by_file.get(fid),
                match_reason="shadow",
                tags=tags_by_file.get(fid, []),
                face_count=face_counts.get(fid, 0),
            ))
        return self._filter_results(results, plan, limit=limit)

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
                    func.lower(SearchDocument.search_text).like(pattern, escape=_LIKE_ESCAPE),
                    func.lower(SearchDocument.keyword_text).like(pattern, escape=_LIKE_ESCAPE),
                    func.lower(SearchDocument.semantic_text).like(pattern, escape=_LIKE_ESCAPE),
                )
            )
            .order_by(MediaFile.exif_datetime.desc().nullslast(), MediaFile.updated_at.desc())
            .limit(max(1, limit * 4))
        )
        doc_rows = list(self._session.execute(statement))
        file_ids = [mf.file_id for mf, _ in doc_rows]
        tags_by_file, face_counts, ocr_by_file, analysis_by_file = self._batch_load_supplements(file_ids)
        results = []
        for media_file, document in doc_rows:
            fid = media_file.file_id
            result = self._build_result_dict(
                media_file,
                ocr=ocr_by_file.get(fid),
                analysis=analysis_by_file.get(fid),
                match_reason="shadow",
                tags=tags_by_file.get(fid, []),
                face_count=face_counts.get(fid, 0),
            )
            result["search_document_version"] = document.version
            results.append(result)
        return results[:limit]

    def _search_by_fts_document(self, query: str, *, limit: int) -> list[dict]:
        fts_query = _fts_query(query)
        if not fts_query:
            return []

        fetch = max(1, limit * 4)
        global _fts_warning_emitted

        # Primary FTS (unicode61 — word boundary, reliable for English)
        # BM25 returns negative values in FTS5: more negative = better match.
        # We sort ASC (most negative first) and pass raw scores upstream;
        # callers treat lower (more negative) as higher relevance.
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
        except Exception as exc:
            if not _fts_warning_emitted:
                logger.warning(
                    "FTS search_documents_fts unavailable (run DB migration?): %s", exc
                )
                _fts_warning_emitted = True

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
        all_file_ids = list(scored_ids.keys())
        tags_by_file, face_counts, ocr_by_file, analysis_by_file = self._batch_load_supplements(all_file_ids)
        results = []
        for file_id, score in sorted(scored_ids.items(), key=lambda item: item[1]):
            row = by_id.get(file_id)
            if row is None:
                continue
            media_file, document = row
            result = self._build_result_dict(
                media_file,
                ocr=ocr_by_file.get(file_id),
                analysis=analysis_by_file.get(file_id),
                match_reason="shadow",
                tags=tags_by_file.get(file_id, []),
                face_count=face_counts.get(file_id, 0),
            )
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
        search_values = {lowered} | load_tag_synonyms().get(lowered, set())
        # Also check each token of a multi-word query
        for token in lowered.split():
            search_values |= load_tag_synonyms().get(token, set())

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
        tag_result_rows = list(self._session.execute(exact_statement))
        seen_ids: set[str] = set()
        deduped_rows = []
        for media_file, matched_tag_value in tag_result_rows:
            if media_file.file_id not in seen_ids:
                seen_ids.add(media_file.file_id)
                deduped_rows.append((media_file, matched_tag_value))

        file_ids = [mf.file_id for mf, _ in deduped_rows]
        tags_by_file, face_counts, ocr_by_file, analysis_by_file = self._batch_load_supplements(file_ids)

        results = []
        for media_file, matched_tag_value in deduped_rows:
            fid = media_file.file_id
            result = self._build_result_dict(
                media_file,
                ocr=ocr_by_file.get(fid),
                analysis=analysis_by_file.get(fid),
                match_reason="tag",
                tags=tags_by_file.get(fid, []),
                face_count=face_counts.get(fid, 0),
            )
            result["tag_exact_match"] = matched_tag_value.casefold() == lowered
            result["matched_tag"] = matched_tag_value
            results.append(result)
        return results

    def _hinted_shadow_results(self, query: str, *, limit: int, exclude_file_ids: set[str] | None = None) -> list[dict]:
        lowered = query.casefold().strip()
        tokens = set(lowered.split())
        # Extend face detection with DB person tags so user-named people trigger it
        try:
            person_tags = self._tag_vocab_cache.get().person_tags
        except Exception:
            person_tags = frozenset()
        wants_faces = any(hint in lowered for hint in FACE_HINTS) or bool(tokens & person_tags)
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

        hint_rows = list(self._session.execute(statement))
        file_ids = [mf.file_id for mf, _, _, _ in hint_rows]
        tags_by_file, face_counts_batch, _, _ = self._batch_load_supplements(file_ids)

        results = []
        for media_file, ocr, analysis, face_count_val in hint_rows:
            fid = media_file.file_id
            result = self._build_result_dict(
                media_file, ocr=ocr, analysis=analysis, match_reason="shadow",
                tags=tags_by_file.get(fid, []),
                face_count=face_counts_batch.get(fid, int(face_count_val or 0)),
            )
            result["face_count"] = face_counts_batch.get(fid, int(face_count_val or 0))
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
        plan: QueryPlan | None = None,
    ) -> list[dict]:
        hits = self._vector_index.search(
            query_embedding,
            limit=limit,
            place_filter=place_filter,
            date_from=date_from,
            date_to=date_to,
        )
        if not hits:
            return []

        file_ids = [str(hit.media_file.file_id) for hit in hits]
        tags_by_file, face_counts, ocr_by_file, analysis_by_file = self._batch_load_supplements(file_ids)

        results = []
        for hit in hits:
            media_file = hit.media_file
            fid = str(media_file.file_id)
            result = self._build_result_dict(
                media_file,
                ocr=ocr_by_file.get(fid),
                analysis=analysis_by_file.get(fid),
                match_reason="clip",
                tags=tags_by_file.get(fid, []),
                face_count=face_counts.get(fid, 0),
            )
            result["distance"] = hit.distance
            result["embedding_model"] = hit.model_name
            result["embedding_version"] = hit.version
            results.append(result)
        return self._filter_results(results, plan, limit=limit)

    def encode_text(self, query: str) -> bytes:
        if not self._clip_enabled:
            return b""
        try:
            clip_embedding.ensure_models()
            # Truncate to CLIP's effective token budget (~77 tokens ≈ 200 chars)
            truncated = query[:_CLIP_MAX_CHARS] if len(query) > _CLIP_MAX_CHARS else query
            return clip_embedding.encode_text(truncated)
        except Exception:
            return b""

    def log_search_event(
        self,
        query: str,
        *,
        effective_mode: str,
        intent: str,
        result_count: int,
        fallback: str | None = None,
    ) -> None:
        """Record a search event for implicit feedback analysis.

        Silently skips if the DB write fails — logging must never break search.
        """
        if not self._log_events:
            return
        try:
            self._session.add(SearchEvent(
                query=query[:512],
                effective_mode=effective_mode,
                intent=intent,
                result_count=result_count,
                fallback=fallback,
            ))
            self._session.flush()
        except Exception as exc:
            logger.debug("Failed to log search event: %s", exc)

    def suggest_related_tags(self, query: str, *, limit: int = 8) -> list[str]:
        """Return existing tag values that partially match the query.

        Used to populate "did you mean?" suggestions when a search returns
        no results.
        """
        lowered = query.casefold().strip()
        if len(lowered) < 2:
            return []
        # Collect synonyms as candidate values to look for
        candidates = {lowered} | load_tag_synonyms().get(lowered, set())
        for token in lowered.split():
            candidates |= load_tag_synonyms().get(token, set())

        # Find tags that exist in the DB matching any candidate or partial
        pattern = f"%{lowered}%"
        statement = (
            select(Tag.tag_value, func.count(Tag.file_id).label("freq"))
            .where(
                or_(
                    func.lower(Tag.tag_value).in_(list(candidates)),
                    func.lower(Tag.tag_value).like(pattern, escape=_LIKE_ESCAPE),
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

    def _batch_load_supplements(
        self, file_ids: list[str]
    ) -> tuple[dict[str, list[dict]], dict[str, int], dict[str, "MediaOCR"], dict[str, "MediaAnalysisSignal"]]:
        """Batch-load tags, face counts, OCR, and analysis for a list of file IDs.

        Eliminates N+1 query patterns when building result dicts for a list of
        media files.  Returns four dicts keyed by file_id.
        """
        if not file_ids:
            return {}, {}, {}, {}

        tag_rows = self._session.execute(
            select(Tag.file_id, Tag.tag_type, Tag.tag_value)
            .where(Tag.file_id.in_(file_ids))
            .order_by(Tag.tag_type.asc(), Tag.tag_value.asc())
        ).all()
        tags_by_file: dict[str, list[dict]] = {fid: [] for fid in file_ids}
        for fid, tag_type, tag_value in tag_rows:
            if fid in tags_by_file:
                tags_by_file[fid].append({"type": tag_type, "value": tag_value})

        face_rows = self._session.execute(
            select(Face.file_id, func.count(Face.id).label("cnt"))
            .where(Face.file_id.in_(file_ids))
            .group_by(Face.file_id)
        ).all()
        face_counts: dict[str, int] = {str(fid): int(cnt) for fid, cnt in face_rows}

        ocr_rows = self._session.scalars(
            select(MediaOCR).where(MediaOCR.file_id.in_(file_ids))
        ).all()
        ocr_by_file: dict[str, MediaOCR] = {row.file_id: row for row in ocr_rows}

        analysis_rows = self._session.scalars(
            select(MediaAnalysisSignal).where(MediaAnalysisSignal.file_id.in_(file_ids))
        ).all()
        analysis_by_file: dict[str, MediaAnalysisSignal] = {row.file_id: row for row in analysis_rows}

        return tags_by_file, face_counts, ocr_by_file, analysis_by_file

    def _build_result_dict(
        self,
        media_file: MediaFile,
        *,
        ocr: "MediaOCR | None" = None,
        analysis: "MediaAnalysisSignal | None" = None,
        match_reason: str,
        ocr_match_kind: str | None = None,
        tags: list[dict] | None = None,
        face_count: int = 0,
    ) -> dict:
        """Build result dict from pre-loaded data — no extra DB queries."""
        payload = {
            "file_id": media_file.file_id,
            "filename": media_file.filename,
            "current_path": media_file.current_path,
            "relative_path": media_file.relative_path,
            "media_kind": media_file.media_kind,
            "status": media_file.status,
            "captured_at": media_file.exif_datetime,
            "updated_at": media_file.updated_at,
            "tags": tags if tags is not None else [],
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

    def _filter_results(self, results: list[dict], plan: QueryPlan | None, *, limit: int) -> list[dict]:
        if plan is None or not plan.has_hard_filters():
            return results[:limit]
        filtered = [result for result in results if self._matches_query_plan_result(result, plan)]
        return filtered[:limit]

    def _matches_query_plan_result(self, result: dict, plan: QueryPlan) -> bool:
        if not _matches_face_count(result, plan):
            return False
        if not _matches_date_range(result, plan):
            return False
        if not _matches_time_constraints(result, plan):
            return False
        if not _matches_place_terms(result, plan):
            return False
        if not self._matches_person_terms(result, plan):
            return False
        if not _matches_excluded_terms(result, plan):
            return False
        return True

    def _matches_person_terms(self, result: dict, plan: QueryPlan) -> bool:
        if not plan.requires_person_match():
            return True
        allowed_terms = self._expanded_person_terms(plan.person_terms)
        result_person_terms = {
            str(tag.get("value", "")).casefold()
            for tag in (result.get("tags") or [])
            if tag.get("type") in {"person", "people", "face", "auto_person"}
        }
        if not (allowed_terms & result_person_terms):
            return False
        if not plan.person_exclusive:
            return True
        informative_auto_terms = {
            str(tag.get("value", "")).casefold()
            for tag in (result.get("tags") or [])
            if tag.get("type") == "auto_person"
        } - self._generic_auto_person_terms()
        if informative_auto_terms and not informative_auto_terms.issubset(allowed_terms):
            return False
        return True

    def _expanded_person_terms(self, person_terms: list[str]) -> set[str]:
        lowered_terms = {term.casefold() for term in person_terms if term}
        expanded = set(lowered_terms)
        for canonical, aliases in load_concept_aliases().items():
            cluster = {canonical.casefold(), *[alias.casefold() for alias in aliases]}
            if cluster & lowered_terms:
                expanded |= cluster
        return expanded

    def _generic_auto_person_terms(self) -> set[str]:
        aliases = load_concept_aliases()
        generic: set[str] = {
            "face", "faces", "portrait", "selfie", "human", "people", "person", "group",
        }
        for canonical in ("person", "group"):
            generic.add(canonical.casefold())
            generic.update(alias.casefold() for alias in aliases.get(canonical, ()))
        return generic


_LIKE_ESCAPE = "\\"


def _matches_face_count(result: dict, plan: QueryPlan) -> bool:
    count = int(result.get("face_count") or 0)
    if plan.face_count_exact is not None and count != plan.face_count_exact:
        return False
    if plan.face_count_min is not None and count < plan.face_count_min:
        return False
    if plan.face_count_max is not None and count > plan.face_count_max:
        return False
    return True


def _matches_date_range(result: dict, plan: QueryPlan) -> bool:
    if not plan.require_date_match or plan.date_from is None:
        return True
    captured = result.get("captured_at")
    if captured is None:
        return False
    try:
        captured_dt = datetime.fromisoformat(captured) if isinstance(captured, str) else captured
        date_from = datetime.combine(plan.date_from, time.min)
        date_to = datetime.combine(plan.date_to, time.max) if plan.date_to else None
        return captured_dt >= date_from and (date_to is None or captured_dt <= date_to)
    except Exception:
        return False


def _matches_place_terms(result: dict, plan: QueryPlan) -> bool:
    if not plan.require_place_match or not plan.place_terms:
        return True
    place_set = {term.casefold() for term in plan.place_terms}
    tag_values = {str(tag.get("value", "")).casefold() for tag in (result.get("tags") or [])}
    return bool(place_set & tag_values)


_DAYPART_HOURS: dict[str, tuple[int, int]] = {
    "dawn": (4, 7),
    "morning": (5, 11),
    "noon": (11, 14),
    "afternoon": (12, 17),
    "evening": (17, 21),
    "night": (21, 24),
}

_GENERIC_ABSENT_PERSON_TERMS = {"face", "faces", "person", "people", "human", "얼굴", "사람", "인물"}


def _matches_time_constraints(result: dict, plan: QueryPlan) -> bool:
    if plan.daypart is None and not plan.allowed_weekdays:
        return True
    captured = result.get("captured_at")
    if captured is None:
        return False
    try:
        captured_dt = datetime.fromisoformat(captured) if isinstance(captured, str) else captured
    except Exception:
        return False

    if plan.allowed_weekdays and captured_dt.weekday() not in set(plan.allowed_weekdays):
        return False

    if plan.daypart is not None:
        hour = captured_dt.hour
        start, end = _DAYPART_HOURS.get(plan.daypart, (0, 24))
        if plan.daypart == "night":
            if hour < start and hour >= 4:
                return False
        elif hour < start or hour >= end:
            return False

    return True


def _matches_excluded_terms(result: dict, plan: QueryPlan) -> bool:
    excluded_terms = {term.casefold() for term in (plan.excluded_terms or []) if term}
    if not excluded_terms:
        return True
    tag_values = {str(tag.get("value", "")).casefold() for tag in (result.get("tags") or [])}
    ocr_text = str(result.get("ocr_text") or "").casefold()
    expanded_excluded = _expanded_filter_terms(excluded_terms)

    if expanded_excluded & _GENERIC_ABSENT_PERSON_TERMS:
        if int(result.get("face_count") or 0) > 0:
            return False
        expanded_excluded -= _GENERIC_ABSENT_PERSON_TERMS

    if expanded_excluded & tag_values:
        return False

    return not any(term and len(term) >= 2 and term in ocr_text for term in expanded_excluded)


def _expanded_filter_terms(terms: set[str]) -> set[str]:
    expanded = set(terms)
    for canonical, aliases in load_concept_aliases().items():
        cluster = {canonical.casefold(), *[alias.casefold() for alias in aliases]}
        if cluster & terms:
            expanded |= cluster
    return expanded


def _like_pattern(query: str) -> str:
    cleaned = query.casefold().strip()
    escaped = cleaned.replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2).replace("%", r"\%").replace("_", r"\_")
    return f"%{escaped}%"


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
