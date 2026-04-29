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
    # People
    ClipConcept("person", ("a photo of a person", "a portrait photo", "a human face"), 0.235),
    ClipConcept("baby", ("a photo of a baby", "an infant lying down", "a newborn baby"), 0.255),
    ClipConcept("group", ("a group photo of people", "friends posing for a photo", "people gathered together"), 0.248),
    # Documents / screens
    ClipConcept("receipt", ("a photo of a receipt", "a purchase receipt with text", "a cash register receipt"), 0.252),
    ClipConcept("screenshot", ("a smartphone screenshot", "a screenshot of an app", "a mobile app interface"), 0.248),
    ClipConcept("document", ("a document with text", "a page full of text", "a printed document"), 0.252),
    # Outdoors / nature
    ClipConcept("outdoor", ("an outdoor photo", "a street or park scene", "outside in natural light"), 0.242),
    ClipConcept("beach", ("a beach photo", "the sea and sand", "ocean waves on a beach"), 0.250),
    ClipConcept("mountain", ("a mountain landscape", "a hiking trail on a mountain", "a mountain view"), 0.248),
    ClipConcept("nature", ("a nature photo", "trees and greenery", "a forest or countryside scene"), 0.242),
    ClipConcept("sky", ("a blue sky with clouds", "a sunset sky", "a dramatic sky landscape"), 0.240),
    # Food & drink
    ClipConcept("food", ("a photo of food", "a meal on a table", "a dish at a restaurant"), 0.248),
    ClipConcept("cake", ("a birthday cake", "a decorated cake with candles", "a slice of cake"), 0.255),
    ClipConcept("coffee", ("a cup of coffee", "a latte or espresso at a cafe", "a coffee drink"), 0.252),
    # Life events
    ClipConcept("celebration", ("a party or celebration", "people celebrating with drinks", "birthday party balloons"), 0.250),
    ClipConcept("wedding", ("a wedding ceremony", "bride and groom together", "wedding flowers and dress"), 0.258),
    ClipConcept("travel", ("a travel photo", "a tourist at a famous landmark", "sightseeing abroad"), 0.245),
    # Lighting / time of day
    ClipConcept("night", ("a night photo", "a dark evening scene", "city lights at night"), 0.242),
    ClipConcept("sunset", ("a sunset photo", "golden hour sky", "sun setting over the horizon"), 0.248),
    # Vehicles
    ClipConcept("vehicle", ("a car or vehicle", "a photo of transportation", "a car on the road"), 0.232),
    # Animals
    ClipConcept("animal", ("a pet or animal", "a dog or cat", "an animal close-up portrait"), 0.240),
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
    return _to_auto_tags([tag for _, tag in hits[:5]])


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
