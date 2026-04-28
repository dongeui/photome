"""Automatic visual tags derived from analysis signals and CLIP embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.core.contracts import MediaTagInput
from app.services.embedding import clip as clip_embedding


@dataclass(frozen=True)
class ClipConcept:
    tag: str
    prompts: tuple[str, ...]
    threshold: float


CLIP_CONCEPTS = (
    ClipConcept("person", ("a photo of a person", "a portrait photo", "a human face"), 0.235),
    ClipConcept("baby", ("a photo of a baby", "an infant lying down"), 0.260),
    ClipConcept("receipt", ("a photo of a receipt", "a purchase receipt with text"), 0.255),
    ClipConcept("screenshot", ("a smartphone screenshot", "a screenshot of an app"), 0.250),
    ClipConcept("document", ("a document with text", "a page full of text"), 0.255),
    ClipConcept("outdoor", ("an outdoor photo", "a street or park scene"), 0.245),
    ClipConcept("food", ("a photo of food", "a meal on a table"), 0.252),
    ClipConcept("vehicle", ("a car or vehicle", "a photo of transportation"), 0.232),
    ClipConcept("night", ("a night photo", "a dark evening scene"), 0.245),
)


def tags_from_signals(analysis: dict, ocr_text: str = "") -> list[MediaTagInput]:
    tags: list[str] = []
    text = ocr_text.strip()
    if analysis.get("is_screenshot_like"):
        tags.append("screenshot")
    if analysis.get("is_document_like"):
        tags.append("document")
    if analysis.get("is_text_heavy") or text:
        tags.append("text")
    if _looks_like_receipt(text):
        tags.append("receipt")
    return _to_auto_tags(tags)


def tags_from_embedding_file(embedding_ref: str, embeddings_root: Path) -> list[MediaTagInput]:
    vector = _load_embedding_vector(embedding_ref, embeddings_root)
    if vector is None:
        return []
    return tags_from_embedding_vector(vector)


def tags_from_embedding_vector(vector: np.ndarray) -> list[MediaTagInput]:
    if vector.size == 0:
        return []
    normalized = _normalize(vector.astype("float32"))
    hits: list[tuple[float, str]] = []
    for concept in CLIP_CONCEPTS:
        score = float(normalized.dot(_concept_vector(concept.tag)))
        if score >= concept.threshold:
            hits.append((score, concept.tag))

    hits.sort(reverse=True)
    return _to_auto_tags([tag for _, tag in hits[:3]])


def merge_auto_tags(*tag_groups: list[MediaTagInput]) -> list[MediaTagInput]:
    seen: set[str] = set()
    merged: list[MediaTagInput] = []
    for group in tag_groups:
        for tag in group:
            key = tag.tag_value.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(tag)
    return merged


@lru_cache(maxsize=1)
def _concept_vectors() -> dict[str, np.ndarray]:
    clip_embedding.ensure_models()
    vectors: dict[str, np.ndarray] = {}
    for concept in CLIP_CONCEPTS:
        prompt_vectors = [
            clip_embedding.embedding_from_bytes(clip_embedding.encode_text(prompt))
            for prompt in concept.prompts
        ]
        vectors[concept.tag] = _normalize(np.mean(prompt_vectors, axis=0).astype("float32"))
    return vectors


def _concept_vector(tag: str) -> np.ndarray:
    return _concept_vectors()[tag]


def _load_embedding_vector(embedding_ref: str, embeddings_root: Path) -> np.ndarray | None:
    try:
        path = Path(embedding_ref)
        absolute_path = path if path.is_absolute() else embeddings_root / path.relative_to("embeddings")
        if not absolute_path.is_file():
            return None
        return np.load(absolute_path).astype("float32")
    except Exception:
        return None


def _looks_like_receipt(text: str) -> bool:
    lowered = text.casefold()
    hints = ("total", "subtotal", "receipt", "결제", "합계", "영수증", "카드", "승인")
    return any(hint in lowered for hint in hints)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector)) or 1.0
    return vector / norm


def _to_auto_tags(values: list[str]) -> list[MediaTagInput]:
    return [MediaTagInput(tag_type="auto", tag_value=value) for value in dict.fromkeys(values)]
