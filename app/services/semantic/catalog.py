"""Catalog operations for OCR, image signals, and embeddings."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import re
from typing import Iterable

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models.semantic import MediaAnalysisSignal, MediaEmbedding, MediaOCR, MediaOCRBlock, MediaOCRGram
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
