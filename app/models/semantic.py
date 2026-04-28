"""Semantic enrichment models for OCR, image signals, and CLIP embeddings."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class MediaOCR(Base):
    __tablename__ = "media_ocr"

    file_id: Mapped[str] = mapped_column(ForeignKey("media_files.file_id", ondelete="CASCADE"), primary_key=True)
    text_content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    engine: Mapped[str] = mapped_column(String(64), nullable=False, default="tesseract")
    version: Mapped[str] = mapped_column(String(32), nullable=False, default="ocr-v1")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    media_file = relationship("MediaFile")


class MediaOCRBlock(Base):
    __tablename__ = "media_ocr_blocks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    file_id: Mapped[str] = mapped_column(ForeignKey("media_files.file_id", ondelete="CASCADE"), index=True)
    level: Mapped[str] = mapped_column(String(32), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    left: Mapped[int] = mapped_column(nullable=False, default=0)
    top: Mapped[int] = mapped_column(nullable=False, default=0)
    width: Mapped[int] = mapped_column(nullable=False, default=0)
    height: Mapped[int] = mapped_column(nullable=False, default=0)


class MediaOCRGram(Base):
    __tablename__ = "media_ocr_grams"
    __table_args__ = (UniqueConstraint("file_id", "gram", name="uq_media_ocr_gram"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    file_id: Mapped[str] = mapped_column(ForeignKey("media_files.file_id", ondelete="CASCADE"), index=True)
    gram: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    count: Mapped[int] = mapped_column(nullable=False, default=1)


class MediaAnalysisSignal(Base):
    __tablename__ = "media_analysis_signals"

    file_id: Mapped[str] = mapped_column(ForeignKey("media_files.file_id", ondelete="CASCADE"), primary_key=True)
    text_char_count: Mapped[int] = mapped_column(nullable=False, default=0)
    text_line_count: Mapped[int] = mapped_column(nullable=False, default=0)
    edge_density: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    brightness: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_text_heavy: Mapped[bool] = mapped_column(nullable=False, default=False)
    is_document_like: Mapped[bool] = mapped_column(nullable=False, default=False)
    is_screenshot_like: Mapped[bool] = mapped_column(nullable=False, default=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False, default="analysis-v1")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    media_file = relationship("MediaFile")


class MediaEmbedding(Base):
    __tablename__ = "media_embeddings"
    __table_args__ = (UniqueConstraint("file_id", "model_name", "version", name="uq_media_embedding_version"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    file_id: Mapped[str] = mapped_column(ForeignKey("media_files.file_id", ondelete="CASCADE"), index=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_ref: Mapped[str] = mapped_column(String(1024), nullable=False)
    dimensions: Mapped[int] = mapped_column(nullable=False)
    checksum: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
