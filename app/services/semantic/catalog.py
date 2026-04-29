"""Catalog operations for OCR, image signals, and embeddings."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime
import re
from typing import Iterable

from sqlalchemy import delete, func, or_, select, text
from sqlalchemy.orm import Session

from app.core.contracts import MediaTagInput
from app.models.annotation import MediaAnnotation
from app.models.face import Face
from app.models.media import MediaFile
from app.models.semantic import (
    MediaAnalysisSignal,
    MediaAutoTagState,
    MediaCaption,
    MediaEmbedding,
    MediaOCR,
    MediaOCRBlock,
    MediaOCRGram,
    SearchDocument,
)
from app.models.tag import Tag
from app.services.caption import CaptionResult
from app.services.ocr import OCRBlock, OCRResult


class SemanticCatalog:
    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert_ocr(self, file_id: str, result: OCRResult, *, version: str = "ocr-v1") -> MediaOCR:
        row = self._session.get(MediaOCR, file_id)
        if row is None:
            row = MediaOCR(file_id=file_id, text_content=result.text, engine=result.engine, version=version)
            self._session.add(row)
        else:
            row.text_content = result.text
            row.engine = result.engine
            row.version = version

        self._replace_ocr_blocks(file_id, result.blocks)
        self._replace_ocr_grams(file_id, result.text)
        self._session.flush()
        return row

    def upsert_analysis(self, file_id: str, payload: dict, *, version: str = "analysis-v1") -> MediaAnalysisSignal:
        row = self._session.get(MediaAnalysisSignal, file_id)
        values = {
            "text_char_count": int(payload.get("text_char_count") or 0),
            "text_line_count": int(payload.get("text_line_count") or 0),
            "edge_density": float(payload.get("edge_density") or 0.0),
            "brightness": float(payload.get("brightness") or 0.0),
            "is_text_heavy": bool(payload.get("is_text_heavy")),
            "is_document_like": bool(payload.get("is_document_like")),
            "is_screenshot_like": bool(payload.get("is_screenshot_like")),
            "version": version,
        }
        if row is None:
            row = MediaAnalysisSignal(file_id=file_id, **values)
            self._session.add(row)
        else:
            for key, value in values.items():
                setattr(row, key, value)
        self._session.flush()
        return row

    def register_embedding(
        self,
        file_id: str,
        *,
        model_name: str,
        version: str,
        embedding_ref: str,
        dimensions: int,
        checksum: str | None = None,
    ) -> MediaEmbedding:
        row = self._session.execute(
            select(MediaEmbedding).where(
                MediaEmbedding.file_id == file_id,
                MediaEmbedding.model_name == model_name,
                MediaEmbedding.version == version,
            )
        ).scalar_one_or_none()
        if row is None:
            row = MediaEmbedding(
                file_id=file_id,
                model_name=model_name,
                version=version,
                embedding_ref=embedding_ref,
                dimensions=dimensions,
                checksum=checksum,
            )
            self._session.add(row)
        else:
            row.embedding_ref = embedding_ref
            row.dimensions = dimensions
            row.checksum = checksum
        self._session.flush()
        return row

    def upsert_caption(
        self,
        file_id: str,
        result: CaptionResult,
        *,
        version: str = "caption-v1",
    ) -> MediaCaption:
        row = self._session.get(MediaCaption, file_id)
        values = {
            "short_caption": result.short_caption,
            "objects_json": result.objects,
            "activities_json": result.activities,
            "setting": result.setting,
            "provider": result.provider,
            "version": version,
        }
        if row is None:
            row = MediaCaption(file_id=file_id, **values)
            self._session.add(row)
        else:
            for key, value in values.items():
                setattr(row, key, value)
        self._session.flush()
        return row

    def upsert_auto_tag_state(
        self,
        file_id: str,
        *,
        tags: list[MediaTagInput],
        version: str,
        source: str = "thumb+clip",
    ) -> MediaAutoTagState:
        payload = [{"type": tag.tag_type, "value": tag.tag_value} for tag in tags]
        row = self._session.get(MediaAutoTagState, file_id)
        if row is None:
            row = MediaAutoTagState(file_id=file_id, version=version, source=source, tags_json=payload)
            self._session.add(row)
        else:
            row.version = version
            row.source = source
            row.tags_json = payload
        self._session.flush()
        return row

    def upsert_search_document(self, media_file: MediaFile, *, version: str) -> SearchDocument:
        annotation = self._session.get(MediaAnnotation, media_file.file_id)
        ocr = self._session.get(MediaOCR, media_file.file_id)
        analysis = self._session.get(MediaAnalysisSignal, media_file.file_id)
        caption = self._session.get(MediaCaption, media_file.file_id)
        tags = list(
            self._session.scalars(
                select(Tag)
                .where(Tag.file_id == media_file.file_id)
                .order_by(Tag.tag_type.asc(), Tag.tag_value.asc())
            )
        )
        embeddings = list(
            self._session.scalars(
                select(MediaEmbedding)
                .where(MediaEmbedding.file_id == media_file.file_id)
                .order_by(MediaEmbedding.model_name.asc(), MediaEmbedding.version.asc())
            )
        )
        people = sorted({tag.tag_value for tag in tags if tag.tag_type in {"person", "people", "face"}})
        places = sorted({tag.tag_value for tag in tags if tag.tag_type in {"place", "place_detail", "location"}})
        tag_payload = [{"type": tag.tag_type, "value": tag.tag_value} for tag in tags]
        embedding_payload = [
            {
                "model_name": embedding.model_name,
                "version": embedding.version,
                "embedding_ref": embedding.embedding_ref,
                "dimensions": embedding.dimensions,
            }
            for embedding in embeddings
        ]
        face_count = int(
            self._session.scalar(select(func.count()).select_from(Face).where(Face.file_id == media_file.file_id)) or 0
        )
        signals = _analysis_payload(analysis, face_count=face_count)
        keyword_parts = [
            media_file.filename,
            media_file.relative_path,
            annotation.title if annotation else None,
            annotation.description if annotation else None,
            ocr.text_content if ocr else None,
            *[tag.tag_value for tag in tags],
        ]
        caption_parts = _caption_terms(caption)
        semantic_parts = [
            media_file.filename,
            annotation.title if annotation else None,
            annotation.description if annotation else None,
            *[tag.tag_value for tag in tags],
            *_signal_terms(signals),
            *caption_parts,
        ]
        search_text = _join_text([*keyword_parts, *semantic_parts])
        source_updated_at = _max_datetime(
            media_file.updated_at,
            annotation.updated_at if annotation else None,
            ocr.updated_at if ocr else None,
            analysis.updated_at if analysis else None,
            caption.updated_at if caption else None,
            *(embedding.updated_at for embedding in embeddings),
        )

        row = self._session.get(SearchDocument, media_file.file_id)
        values = {
            "version": version,
            "source_updated_at": source_updated_at,
            "search_text": search_text,
            "keyword_text": _join_text(keyword_parts),
            "semantic_text": _join_text(semantic_parts),
            "tags_json": tag_payload,
            "people_json": people,
            "places_json": places,
            "signals_json": signals,
            "embedding_refs_json": embedding_payload,
        }
        if row is None:
            row = SearchDocument(file_id=media_file.file_id, **values)
            self._session.add(row)
        else:
            for key, value in values.items():
                setattr(row, key, value)
        self._session.flush()
        self._upsert_search_document_fts(row)
        return row

    def list_media_needing_search_document(self, *, version: str, limit: int = 100) -> list[MediaFile]:
        statement = (
            select(MediaFile)
            .outerjoin(SearchDocument, SearchDocument.file_id == MediaFile.file_id)
            .where(
                MediaFile.status.in_(("thumb_done", "analysis_done")),
                MediaFile.media_kind.in_(("image", "video")),
                or_(
                    SearchDocument.file_id.is_(None),
                    SearchDocument.version != version,
                    SearchDocument.source_updated_at < MediaFile.updated_at,
                ),
            )
            .order_by(MediaFile.updated_at.asc(), MediaFile.file_id.asc())
            .limit(max(1, limit))
        )
        return list(self._session.scalars(statement))

    def _upsert_search_document_fts(self, row: SearchDocument) -> None:
        if self._session.bind is None or self._session.bind.dialect.name != "sqlite":
            return
        params = {
            "file_id": row.file_id,
            "search_text": row.search_text,
            "keyword_text": row.keyword_text,
            "semantic_text": row.semantic_text,
        }
        # Primary FTS (unicode61 — word boundary, good for English)
        try:
            self._session.execute(
                text("DELETE FROM search_documents_fts WHERE file_id = :file_id"),
                {"file_id": row.file_id},
            )
            self._session.execute(
                text(
                    "INSERT INTO search_documents_fts(file_id, search_text, keyword_text, semantic_text) "
                    "VALUES (:file_id, :search_text, :keyword_text, :semantic_text)"
                ),
                params,
            )
        except Exception:
            # FTS is an acceleration layer; search_documents remains canonical.
            pass

        # Trigram FTS (character n-gram — Korean/CJK substring search)
        try:
            self._session.execute(
                text("DELETE FROM search_documents_fts_ko WHERE file_id = :file_id"),
                {"file_id": row.file_id},
            )
            self._session.execute(
                text(
                    "INSERT INTO search_documents_fts_ko(file_id, search_text, keyword_text, semantic_text) "
                    "VALUES (:file_id, :search_text, :keyword_text, :semantic_text)"
                ),
                params,
            )
        except Exception:
            # Trigram FTS is optional; silently skip if table doesn't exist.
            pass

    def _replace_ocr_blocks(self, file_id: str, blocks: Iterable[OCRBlock]) -> None:
        self._session.execute(delete(MediaOCRBlock).where(MediaOCRBlock.file_id == file_id))
        for block in blocks:
            payload = asdict(block)
            self._session.add(MediaOCRBlock(file_id=file_id, **payload))

    def _replace_ocr_grams(self, file_id: str, text: str) -> None:
        self._session.execute(delete(MediaOCRGram).where(MediaOCRGram.file_id == file_id))
        for gram, count in Counter(_korean_2grams(text)).items():
            self._session.add(MediaOCRGram(file_id=file_id, gram=gram, count=count))


def _korean_2grams(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", text)
    chars = [char for char in compact if "\uac00" <= char <= "\ud7a3"]
    return ["".join(chars[index : index + 2]) for index in range(max(0, len(chars) - 1))]


def _join_text(parts: Iterable[object | None]) -> str:
    values = [str(part).strip() for part in parts if part is not None and str(part).strip()]
    return "\n".join(dict.fromkeys(values))


def _analysis_payload(analysis: MediaAnalysisSignal | None, *, face_count: int) -> dict:
    if analysis is None:
        return {"face_count": face_count}
    return {
        "text_char_count": analysis.text_char_count,
        "text_line_count": analysis.text_line_count,
        "edge_density": analysis.edge_density,
        "brightness": analysis.brightness,
        "is_text_heavy": analysis.is_text_heavy,
        "is_document_like": analysis.is_document_like,
        "is_screenshot_like": analysis.is_screenshot_like,
        "face_count": face_count,
    }


def _caption_terms(caption: MediaCaption | None) -> list[str]:
    if caption is None:
        return []
    terms: list[str] = []
    if caption.short_caption:
        terms.append(caption.short_caption)
    terms.extend(caption.objects_json or [])
    terms.extend(caption.activities_json or [])
    if caption.setting:
        terms.append(caption.setting)
    return terms


def _signal_terms(signals: dict) -> list[str]:
    terms: list[str] = []
    if signals.get("is_text_heavy"):
        terms.extend(["text", "document", "텍스트", "문서"])
    if signals.get("is_document_like"):
        terms.extend(["document", "receipt", "문서", "영수증"])
    if signals.get("is_screenshot_like"):
        terms.extend(["screenshot", "screen", "스크린샷", "화면"])
    if int(signals.get("face_count") or 0) > 0:
        terms.extend(["face", "person", "people", "얼굴", "사람", "인물"])
    return terms


def _max_datetime(*values: datetime | None) -> datetime:
    available = [value for value in values if value is not None]
    if not available:
        return datetime.utcnow()
    return max(available)
