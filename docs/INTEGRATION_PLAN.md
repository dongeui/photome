# photome Integration Plan

`photome` is the integration workspace for the two source projects:

- `photomine`: production-shaped local-first media catalog, pipeline, jobs, scheduler, gallery, and dashboard
- `photomem`: working OCR + CLIP + hybrid search experiment

## Direction

Use `photomine` as the base application and migrate `photomem` search capabilities into it.

The stable identity key is `MediaFile.file_id`; `photomem`'s old integer `photo_id` schema should not be carried forward.

## Imported So Far

- OCR extraction service: `app/services/ocr`
- CLIP image/text encoder wrapper: `app/services/embedding`
- Korean-first CLIP query expansion: `app/services/search/query_translate.py`
- Hybrid OCR/semantic rank fusion shell: `app/services/search/hybrid.py`
- Image text/screenshot/document signal extraction: `app/services/analysis/image_signals.py`
- Semantic persistence models: `MediaOCR`, `MediaOCRBlock`, `MediaOCRGram`, `MediaAnalysisSignal`, `MediaEmbedding`

## Next Steps

1. Wire OCR and image-signal generation into the processing pipeline after thumbnail generation.
2. Add CLIP embedding generation as a versioned derived asset under `embeddings/clip/<version>`.
3. Implement a SQLAlchemy search backend for `HybridSearchService`.
4. Add `/search` JSON API.
5. Fold search controls into the gallery UI after the API is stable.
6. Port `photomem` search tests to `file_id`-based fixtures.

## Dependency Notes

`photome` keeps `photomine`'s modular structure but adds `photomem`'s heavier semantic dependencies:

- `open_clip_torch`
- `torch`
- `torchvision`
- `pytesseract`

For deployment, install PyTorch CPU wheels deliberately to avoid accidentally pulling CUDA builds on small local servers.
