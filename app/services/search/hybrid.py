"""Hybrid semantic/OCR search ranking.

This is the first photomem search transplant. It is intentionally backend
agnostic so the next step can wire it to photome's SQLAlchemy catalog instead
of photomem's old raw SQLite schema.
"""

from __future__ import annotations

from typing import Protocol

from app.services.search import query_translate


RRF_K = 60.0

FACE_HINTS = {
    "face", "faces", "person", "people", "portrait", "selfie",
    "woman", "women", "female", "girl", "baby", "infant", "toddler", "child", "kid",
    "얼굴", "사람", "인물", "셀카", "남자", "여자", "여성", "아기", "애기", "아이", "어린이",
}
TEXT_HINTS = {
    "text", "ocr", "document", "receipt", "error", "dialog", "message", "screen",
    "텍스트", "글씨", "문서", "영수증", "오류", "대화", "메시지", "화면",
}
SCREEN_HINTS = {"screenshot", "screen", "ui", "chat", "popup", "스크린샷", "화면", "앱", "대화창", "팝업"}


class HybridSearchBackend(Protocol):
    def search_by_ocr(self, query: str, *, limit: int) -> list[dict]: ...

    def search_by_embedding(
        self,
        query_embedding: bytes,
        *,
        limit: int,
        place_filter: str | None = None,
        date_from: object | None = None,
        date_to: object | None = None,
    ) -> list[dict]: ...

    def search_by_shadow_doc(self, query: str, *, limit: int) -> list[dict]: ...

    def encode_text(self, query: str) -> bytes: ...


class HybridSearchService:
    def __init__(self, backend: HybridSearchBackend) -> None:
        self._backend = backend

    def search_with_meta(
        self,
        query: str,
        *,
        limit: int = 20,
        place_filter: str | None = None,
        date_from: object | None = None,
        date_to: object | None = None,
        mode: str = "hybrid",
    ) -> tuple[list[dict], dict]:
        if not query.strip():
            return [], {"effective_mode": mode, "intent_reason": "empty"}

        cleaned = query_translate.normalize_query(query)
        normalized_mode = mode if mode in {"hybrid", "ocr", "semantic"} else "hybrid"
        ocr_results = self._backend.search_by_ocr(cleaned, limit=limit) if normalized_mode in {"hybrid", "ocr"} else []
        effective_mode, intent_reason = resolve_effective_mode(cleaned, normalized_mode, ocr_results)

        shadow_results = (
            self._backend.search_by_shadow_doc(cleaned, limit=limit)
            if effective_mode in {"hybrid", "ocr", "semantic"}
            else []
        )
        clip_results = (
            self._search_clip_variants(cleaned, limit, place_filter, date_from, date_to)
            if effective_mode in {"hybrid", "semantic"}
            else []
        )
        if effective_mode == "semantic" and not clip_results:
            shadow_results = self._backend.search_by_shadow_doc(cleaned, limit=limit)

        merged = fuse_ranked_results(
            effective_mode,
            intent_reason,
            ocr_results if effective_mode in {"hybrid", "ocr"} else [],
            clip_results,
            shadow_results,
        )
        apply_exact_ocr_boost(cleaned, merged)
        apply_exact_tag_boost(merged)
        set_match_explanations(merged)
        merged.sort(key=search_sort_key, reverse=True)
        return merged[:limit], {"effective_mode": effective_mode, "intent_reason": intent_reason}

    def _search_clip_variants(
        self,
        query: str,
        limit: int,
        place_filter: str | None,
        date_from: object | None,
        date_to: object | None,
    ) -> list[dict]:
        merged: dict[str, dict] = {}
        for variant in query_translate.expand_for_clip(query):
            query_bytes = self._backend.encode_text(variant)
            results = self._backend.search_by_embedding(
                query_bytes,
                limit=limit,
                place_filter=place_filter,
                date_from=date_from,
                date_to=date_to,
            )
            for rank, result in enumerate(results, start=1):
                file_id = str(result["file_id"])
                current = merged.get(file_id)
                if current is None or float(result.get("distance", 99.0)) < float(current.get("distance", 99.0)):
                    result["semantic_query"] = variant
                    result["semantic_variant_rank"] = rank
                    merged[file_id] = result

        values = list(merged.values())
        values.sort(key=lambda item: float(item.get("distance", 99.0)))
        return values[:limit]


def fuse_ranked_results(
    effective_mode: str,
    intent_reason: str,
    ocr_results: list[dict],
    clip_results: list[dict],
    shadow_results: list[dict],
) -> list[dict]:
    weights = intent_weights(effective_mode, intent_reason)
    candidates: dict[str, dict] = {}
    channel_hits: dict[str, set[str]] = {}

    def merge_result(result: dict, channel: str, rank: int) -> None:
        file_id = str(result["file_id"])
        existing = candidates.setdefault(file_id, dict(result))
        for key, value in result.items():
            if key not in existing or existing[key] in (None, ""):
                existing[key] = value
        existing["effective_mode"] = effective_mode
        existing[f"{channel}_rank"] = rank
        existing[f"rrf_{channel}"] = weights[channel] / (RRF_K + rank)
        channel_hits.setdefault(file_id, set()).add(channel)

    for rank, result in enumerate(ocr_results, start=1):
        merge_result(result, "ocr", rank)
    for rank, result in enumerate(clip_results, start=1):
        merge_result(result, "clip", rank)
    for rank, result in enumerate(shadow_results, start=1):
        merge_result(result, "shadow", rank)

    fused = []
    for file_id, result in candidates.items():
        hits = channel_hits.get(file_id, set())
        result["match_reason"] = combined_match_reason(hits)
        result["rrf_score"] = (
            float(result.get("rrf_ocr") or 0.0)
            + float(result.get("rrf_clip") or 0.0)
            + float(result.get("rrf_shadow") or 0.0)
        )
        fused.append(result)

    if not fused:
        return []

    max_score = max(float(item.get("rrf_score") or 0.0) for item in fused) or 1.0
    for result in fused:
        result["rank_score"] = max(0.0, min(1.0, float(result.get("rrf_score") or 0.0) / max_score))
    return fused


def resolve_effective_mode(query: str, requested_mode: str, ocr_results: list[dict]) -> tuple[str, str]:
    if requested_mode != "hybrid":
        return requested_mode, "manual"

    lowered = query.casefold()
    has_face_hint = any(hint in lowered for hint in FACE_HINTS)
    has_text_hint = any(hint in lowered for hint in TEXT_HINTS)
    has_screen_hint = any(hint in lowered for hint in SCREEN_HINTS)
    word_hits = [result for result in ocr_results if result.get("ocr_match_kind") == "word"]
    phrase_hits = [result for result in ocr_results if result.get("ocr_match_kind") == "phrase"]
    has_code_like_text = any(ch.isdigit() for ch in query) or any(ch in query for ch in "-_:/[]()#")
    is_short_query = len(query.strip()) <= 12

    if has_face_hint and (has_text_hint or has_screen_hint or has_code_like_text):
        return "hybrid", "auto-mixed"
    if has_face_hint:
        return "semantic", "auto-face"
    if has_text_hint and not ocr_results:
        return "ocr", "auto-text-hint"
    if has_screen_hint and (word_hits or phrase_hits or is_short_query):
        return "ocr", "auto-screen-text"
    if has_code_like_text:
        return "ocr", "auto-code"
    if word_hits and (is_short_query or len(word_hits) >= 2 or has_code_like_text):
        return "ocr", "auto-word-match"
    if phrase_hits and has_code_like_text:
        return "ocr", "auto-phrase-code"
    return "hybrid", "fallback"


def intent_weights(effective_mode: str, intent_reason: str) -> dict[str, float]:
    if effective_mode == "ocr":
        return {"ocr": 0.62, "clip": 0.04, "shadow": 0.22}
    if effective_mode == "semantic":
        return {"ocr": 0.03, "clip": 0.70, "shadow": 0.18}
    if intent_reason == "auto-mixed":
        return {"ocr": 0.36, "clip": 0.34, "shadow": 0.18}
    return {"ocr": 0.35, "clip": 0.36, "shadow": 0.17}


def combined_match_reason(hits: set[str]) -> str:
    if "ocr" in hits and "clip" in hits:
        return "ocr+clip"
    if "clip" in hits and "shadow" in hits:
        return "clip+shadow"
    if "ocr" in hits:
        return "ocr"
    if "clip" in hits:
        return "clip"
    if "shadow" in hits:
        return "shadow"
    return "analysis"


def apply_exact_ocr_boost(query: str, results: list[dict]) -> None:
    lowered = query.casefold()
    tokens = [token for token in lowered.split() if token]
    for result in results:
        ocr_text = str(result.get("ocr_text") or "")
        if not ocr_text:
            continue
        ocr_lower = ocr_text.casefold()
        if lowered in ocr_lower:
            result["rank_score"] = min(1.0, float(result.get("rank_score", 0.0)) + 0.22)
            result["ocr_exact_match"] = True
        elif tokens and all(token in ocr_lower for token in tokens):
            result["rank_score"] = min(1.0, float(result.get("rank_score", 0.0)) + 0.12)


def apply_exact_tag_boost(results: list[dict]) -> None:
    for result in results:
        if result.get("tag_exact_match"):
            result["rank_score"] = min(1.0, float(result.get("rank_score", 0.0)) + 0.9)


def search_sort_key(item: dict) -> tuple[bool, bool, float]:
    return (
        bool(item.get("tag_exact_match")),
        bool(item.get("ocr_exact_match")),
        float(item.get("rank_score") or 0.0),
    )


def set_match_explanations(results: list[dict]) -> None:
    for result in results:
        if result.get("ocr_exact_match"):
            result["match_explanation"] = "exact OCR text"
        elif result.get("tag_exact_match"):
            result["match_explanation"] = "exact tag match"
        elif result.get("match_reason") == "ocr+clip":
            result["match_explanation"] = "OCR and visual match"
        elif result.get("match_reason") == "clip+shadow":
            result["match_explanation"] = "visual and tag match"
        elif result.get("match_reason") == "clip":
            result["match_explanation"] = "visual semantic match"
        elif result.get("match_reason") == "ocr":
            result["match_explanation"] = "OCR text match"
        else:
            result["match_explanation"] = result.get("match_reason", "match")
