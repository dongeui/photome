"""Backfill automatic visual tags from existing analysis and CLIP embeddings."""

from __future__ import annotations

from app.core.settings import load_settings
from app.db.bootstrap import build_database_state
from app.models.media import MediaFile
from app.models.semantic import MediaAnalysisSignal, MediaEmbedding, MediaOCR
from app.services.analysis import auto_tags
from app.services.processing.registry import MediaCatalog


def main() -> None:
    settings = load_settings()
    database = build_database_state(settings)
    updated = 0
    tagged = 0

    with database.session_factory() as session:
        catalog = MediaCatalog(session)
        items = session.query(MediaFile).filter(MediaFile.status != "missing").all()
        for media_file in items:
            analysis = session.get(MediaAnalysisSignal, media_file.file_id)
            ocr = session.get(MediaOCR, media_file.file_id)
            embedding = (
                session.query(MediaEmbedding)
                .filter(MediaEmbedding.file_id == media_file.file_id)
                .order_by(MediaEmbedding.updated_at.desc())
                .first()
            )

            signal_payload = {
                "is_screenshot_like": analysis.is_screenshot_like if analysis else False,
                "is_document_like": analysis.is_document_like if analysis else False,
                "is_text_heavy": analysis.is_text_heavy if analysis else False,
            }
            signal_tags = auto_tags.tags_from_signals(signal_payload, ocr.text_content if ocr else "")
            embedding_tags = (
                auto_tags.tags_from_embedding_file(embedding.embedding_ref, settings.embeddings_root)
                if embedding is not None
                else []
            )
            generated_tags = auto_tags.merge_auto_tags(signal_tags, embedding_tags)
            catalog.upsert_tags_for_types(media_file.file_id, ["auto"], generated_tags)
            updated += 1
            if generated_tags:
                tagged += 1

        session.commit()

    print(f"processed={updated} tagged={tagged}")


if __name__ == "__main__":
    main()
