"""Hybrid semantic/OCR search ranking.

Backend-agnostic so it can be wired to any HybridSearchBackend implementation.
Natural language date expressions (작년, 여름, etc.) are extracted here and
propagated to the embedding backend as date filters.
"""

from __future__ import annotations

import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time
from typing import Protocol

from app.services.search.planner import QueryPlan, plan_query


RRF_K = 60.0

# ---------------------------------------------------------------------------
# Simple TTL query result cache (in-memory, per-process)
# ---------------------------------------------------------------------------
_CACHE_TTL_SECONDS = 60
_query_cache: dict[str, tuple[float, list[dict], dict]] = {}  # key → (ts, results, meta)


def _cache_key(query: str, limit: int, mode: str, place_filter: str | None) -> str:
    return f"{query}|{limit}|{mode}|{place_filter or ''}"


def _cache_get(key: str) -> tuple[list[dict], dict] | None:
    entry = _query_cache.get(key)
    if entry is None:
        return None
    ts, results, meta = entry
    if _time.monotonic() - ts > _CACHE_TTL_SECONDS:
        del _query_cache[key]
        return None
    return results, meta


def _cache_set(key: str, results: list[dict], meta: dict) -> None:
    # Evict oldest entries when cache grows beyond 256 keys
    if len(_query_cache) >= 256:
        oldest_key = min(_query_cache, key=lambda k: _query_cache[k][0])
        del _query_cache[oldest_key]
    _query_cache[key] = (_time.monotonic(), results, meta)

FACE_HINTS = {
    "face", "faces", "person", "people", "portrait", "selfie",
    "woman", "women", "female", "girl", "baby", "infant", "toddler", "child", "kid",
    "얼굴", "사람", "인물", "셀카", "남자", "남성", "여자", "여성", "아기", "애기", "아이", "어린이",
    "엄마", "아빠", "할머니", "할아버지", "가족", "친구", "커플",
}
TEXT_HINTS = {
    "text", "ocr", "document", "receipt", "error", "dialog", "message", "screen",
    "텍스트", "글씨", "문서", "영수증", "오류", "대화", "메시지", "화면",
}
SCREEN_HINTS = {"screenshot", "screen", "ui", "chat", "popup", "스크린샷", "화면", "앱", "대화창", "팝업"}
TRAVEL_HINTS = {
    "travel", "trip", "vacation", "abroad", "tour",
    "여행", "휴가", "해외", "관광", "제주", "해변", "바다",
}
CELEBRATION_HINTS = {
    "birthday", "party", "wedding", "graduation", "celebration",
    "생일", "파티", "결혼식", "졸업", "졸업식", "축하", "크리스마스", "새해",
}


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

    def suggest_related_tags(self, query: str, *, limit: int = 8) -> list[str]: ...

    def load_persisted_weights(self, intent: str, reason: str) -> dict[str, float] | None: ...


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
        debug: bool = False,
        weight_overrides: dict[str, float] | None = None,
    ) -> tuple[list[dict], dict]:
        if not query.strip():
            return [], {"effective_mode": mode, "intent_reason": "empty"}

        # Cache only non-debug requests without caller-provided date filters
        use_cache = (
            not debug
            and not weight_overrides
            and date_from is None
            and date_to is None
        )
        cache_key = _cache_key(query, limit, mode, place_filter) if use_cache else ""
        if use_cache:
            cached = _cache_get(cache_key)
            if cached is not None:
                results, meta = cached
                meta = dict(meta)
                meta["cache_hit"] = True
                return results, meta

        plan = plan_query(query)
        cleaned = plan.normalized_query
        normalized_mode = mode if mode in {"hybrid", "ocr", "semantic"} else "hybrid"

        # Auto-extract date range from natural language when caller didn't specify one
        if date_from is None and date_to is None and plan.date_from is not None:
            date_from = datetime.combine(plan.date_from, time.min)
            date_to = datetime.combine(plan.date_to, time.max) if plan.date_to else None

        keyword_query = plan.keyword_query or cleaned

        # OCR must run first — its results drive effective_mode resolution
        ocr_results = self._backend.search_by_ocr(keyword_query, limit=limit) if normalized_mode in {"hybrid", "ocr"} else []
        effective_mode, intent_reason = resolve_effective_mode(cleaned, normalized_mode, ocr_results)

        # Shadow and CLIP are independent → run in parallel
        need_shadow = effective_mode in {"hybrid", "ocr", "semantic"}
        need_clip = effective_mode in {"hybrid", "semantic"}

        shadow_results: list[dict] = []
        clip_results: list[dict] = []

        if need_shadow and need_clip:
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_shadow = pool.submit(self._backend.search_by_shadow_doc, keyword_query, limit=limit)
                fut_clip = pool.submit(self._search_clip_variants, plan, limit, place_filter, date_from, date_to)
                shadow_results = fut_shadow.result()
                clip_results = fut_clip.result()
        elif need_shadow:
            shadow_results = self._backend.search_by_shadow_doc(keyword_query, limit=limit)
        elif need_clip:
            clip_results = self._search_clip_variants(plan, limit, place_filter, date_from, date_to)

        if effective_mode == "semantic" and not clip_results:
            shadow_results = self._backend.search_by_shadow_doc(keyword_query, limit=limit)

        # Persisted DB weights take precedence over built-in defaults,
        # but explicit per-request overrides take the highest priority
        persisted: dict[str, float] | None = None
        if not weight_overrides and hasattr(self._backend, "load_persisted_weights"):
            persisted = self._backend.load_persisted_weights(effective_mode, intent_reason)

        weights = resolved_intent_weights(
            effective_mode, intent_reason,
            overrides=weight_overrides or persisted,
        )
        merged = fuse_ranked_results(
            effective_mode,
            intent_reason,
            ocr_results if effective_mode in {"hybrid", "ocr"} else [],
            clip_results,
            shadow_results,
            weights=weights,
        )
        debug_candidates = [dict(item) for item in merged] if debug else None
        apply_exact_ocr_boost(cleaned, merged)
        apply_exact_tag_boost(merged)
        set_match_explanations(merged)
        merged.sort(key=search_sort_key, reverse=True)
        final = merged[:limit]
        meta: dict = {
            "effective_mode": effective_mode,
            "intent_reason": intent_reason,
            "query_plan": plan.to_meta(),
            "weight_overrides": weight_overrides or {},
        }
        if debug:
            fused_for_debug = debug_candidates or merged
            meta["debug"] = {
                "requested_mode": normalized_mode,
                "weights": weights,
                "applied_filters": {
                    "place_filter": place_filter,
                    "date_from": _isoformat_or_none(date_from),
                    "date_to": _isoformat_or_none(date_to),
                    "planner_place_terms": plan.place_terms,
                    "planner_person_terms": plan.person_terms,
                    "planner_ocr_terms": plan.ocr_terms,
                    "planner_visual_terms": plan.visual_terms,
                },
                "channel_stats": {
                    "ocr": len(ocr_results),
                    "clip": len(clip_results),
                    "shadow": len(shadow_results),
                    "fused": len(fused_for_debug),
                    "final": len(final),
                },
                "channel_overlap": _channel_overlap(fused_for_debug),
                "channels": {
                    "ocr": _preview_results(ocr_results),
                    "clip": _preview_results(clip_results),
                    "shadow": _preview_results(shadow_results),
                },
                "fused": _preview_results(fused_for_debug),
                "final": _preview_results(final),
            }
        if not final and hasattr(self._backend, "suggest_related_tags"):
            suggestions = self._backend.suggest_related_tags(cleaned, limit=8)
            if suggestions:
                meta["suggestions"] = suggestions

        if use_cache:
            _cache_set(cache_key, final, meta)

        return final, meta

    def _search_clip_variants(
        self,
        plan: QueryPlan,
        limit: int,
        place_filter: str | None,
        date_from: object | None,
        date_to: object | None,
    ) -> list[dict]:
        merged: dict[str, dict] = {}
        for variant in plan.visual_queries:
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
    *,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    weights = weights or intent_weights(effective_mode, intent_reason)
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
        existing.setdefault("score_breakdown", [])
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
        result["score_breakdown"] = [
            {
                "stage": "rrf",
                "rrf_ocr": float(result.get("rrf_ocr") or 0.0),
                "rrf_clip": float(result.get("rrf_clip") or 0.0),
                "rrf_shadow": float(result.get("rrf_shadow") or 0.0),
                "rrf_total": float(result.get("rrf_score") or 0.0),
            }
        ]
        fused.append(result)

    if not fused:
        return []

    max_score = max(float(item.get("rrf_score") or 0.0) for item in fused) or 1.0
    for result in fused:
        result["rank_score"] = max(0.0, min(1.0, float(result.get("rrf_score") or 0.0) / max_score))
        result.setdefault("score_breakdown", []).append(
            {
                "stage": "normalize",
                "rank_score": float(result.get("rank_score") or 0.0),
                "max_rrf_score": max_score,
            }
        )
    return fused


def resolve_effective_mode(query: str, requested_mode: str, ocr_results: list[dict]) -> tuple[str, str]:
    if requested_mode != "hybrid":
        return requested_mode, "manual"

    lowered = query.casefold()
    has_face_hint = any(hint in lowered for hint in FACE_HINTS)
    has_text_hint = any(hint in lowered for hint in TEXT_HINTS)
    has_screen_hint = any(hint in lowered for hint in SCREEN_HINTS)
    has_travel_hint = any(hint in lowered for hint in TRAVEL_HINTS)
    has_celebration_hint = any(hint in lowered for hint in CELEBRATION_HINTS)
    word_hits = [result for result in ocr_results if result.get("ocr_match_kind") == "word"]
    phrase_hits = [result for result in ocr_results if result.get("ocr_match_kind") == "phrase"]
    has_code_like_text = any(ch.isdigit() for ch in query) or any(ch in query for ch in "-_:/[]()#")
    is_short_query = len(query.strip()) <= 12

    if has_face_hint and (has_text_hint or has_screen_hint or has_code_like_text):
        return "hybrid", "auto-mixed"
    if has_face_hint:
        return "semantic", "auto-face"
    # Travel and celebration photos are strongly visual → prefer semantic
    if has_travel_hint and not has_text_hint:
        return "semantic", "auto-travel"
    if has_celebration_hint and not has_text_hint:
        return "semantic", "auto-celebration"
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


def resolved_intent_weights(
    effective_mode: str,
    intent_reason: str,
    *,
    overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    weights = dict(intent_weights(effective_mode, intent_reason))
    if not overrides:
        return weights
    for key in ("ocr", "clip", "shadow"):
        if key in overrides:
            weights[key] = max(0.0, float(overrides[key]))
    return weights


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
            # Still apply ngram score bonus when OCR text is absent
            ngram = float(result.get("ngram_score") or 0.0)
            if ngram > 0:
                bonus = ngram * 0.10
                result["rank_score"] = min(1.0, float(result.get("rank_score", 0.0)) + bonus)
                result.setdefault("score_breakdown", []).append(
                    {"stage": "ocr_ngram_bonus", "delta": bonus, "rank_score": float(result.get("rank_score") or 0.0)}
                )
            continue
        ocr_lower = ocr_text.casefold()
        ngram_bonus = float(result.get("ngram_score") or 0.0) * 0.08
        if lowered in ocr_lower:
            bonus = 0.22 + ngram_bonus
            result["rank_score"] = min(1.0, float(result.get("rank_score", 0.0)) + bonus)
            result["ocr_exact_match"] = True
            result.setdefault("score_breakdown", []).append(
                {"stage": "ocr_exact_bonus", "delta": bonus, "rank_score": float(result.get("rank_score") or 0.0)}
            )
        elif tokens and all(token in ocr_lower for token in tokens):
            bonus = 0.12 + ngram_bonus
            result["rank_score"] = min(1.0, float(result.get("rank_score", 0.0)) + bonus)
            result.setdefault("score_breakdown", []).append(
                {"stage": "ocr_token_bonus", "delta": bonus, "rank_score": float(result.get("rank_score") or 0.0)}
            )
        elif ngram_bonus > 0:
            result["rank_score"] = min(1.0, float(result.get("rank_score", 0.0)) + ngram_bonus)
            result.setdefault("score_breakdown", []).append(
                {"stage": "ocr_ngram_bonus", "delta": ngram_bonus, "rank_score": float(result.get("rank_score") or 0.0)}
            )


def apply_exact_tag_boost(results: list[dict]) -> None:
    for result in results:
        if result.get("tag_exact_match"):
            bonus = 0.9
            result["rank_score"] = min(1.0, float(result.get("rank_score", 0.0)) + bonus)
            result.setdefault("score_breakdown", []).append(
                {"stage": "tag_exact_bonus", "delta": bonus, "rank_score": float(result.get("rank_score") or 0.0)}
            )


def search_sort_key(item: dict) -> tuple[bool, bool, float]:
    return (
        bool(item.get("tag_exact_match")),
        bool(item.get("ocr_exact_match")),
        float(item.get("rank_score") or 0.0),
    )


def set_match_explanations(results: list[dict]) -> None:
    for result in results:
        parts: list[str] = []

        # Primary match signal
        if result.get("ocr_exact_match"):
            parts.append("OCR 텍스트 일치")
        elif result.get("tag_exact_match"):
            matched = result.get("matched_tag")
            parts.append(f"태그 일치: {matched}" if matched else "태그 일치")
        else:
            reason = result.get("match_reason", "")
            if reason == "ocr+clip":
                parts.append("OCR + 시각 의미 일치")
            elif reason == "clip+shadow":
                parts.append("시각 의미 + 태그 일치")
            elif reason == "clip":
                parts.append("시각 의미 일치")
            elif reason == "ocr":
                parts.append("OCR 텍스트 일치")
            elif reason == "shadow":
                parts.append("태그/문서 일치")
            elif reason:
                parts.append(reason)

        # Contextual enrichments
        face_count = int((result.get("signals") or {}).get("face_count") or 0)
        if face_count > 0:
            parts.append(f"얼굴 {face_count}명")

        place_tags = [
            tag["value"]
            for tag in (result.get("tags") or [])
            if tag.get("type") in ("place",) and not _is_coordinate_tag(str(tag.get("value", "")))
        ]
        if place_tags:
            parts.append(f"장소: {place_tags[0]}")

        exif_dt = result.get("exif_datetime")
        if exif_dt:
            try:
                from datetime import datetime as _dt
                dt = _dt.fromisoformat(str(exif_dt)) if isinstance(exif_dt, str) else exif_dt
                parts.append(f"{dt.year}년 {dt.month}월")
            except Exception:
                pass

        result["match_explanation"] = " · ".join(parts) if parts else "일치"


def _is_coordinate_tag(value: str) -> bool:
    """Return True if the tag value looks like a raw GPS coordinate."""
    return bool(__import__("re").match(r"^-?\d+\.\d+,-?\d+\.\d+$", value))


def _preview_results(results: list[dict], *, limit: int = 8) -> list[dict]:
    preview: list[dict] = []
    for item in results[:limit]:
        preview.append(
            {
                "file_id": item.get("file_id"),
                "filename": item.get("filename"),
                "match_reason": item.get("match_reason"),
                "match_explanation": item.get("match_explanation"),
                "ocr_match_kind": item.get("ocr_match_kind"),
                "matched_tag": item.get("matched_tag"),
                "semantic_query": item.get("semantic_query"),
                "distance": item.get("distance"),
                "tag_exact_match": item.get("tag_exact_match"),
                "ocr_exact_match": item.get("ocr_exact_match"),
                "rrf_score": item.get("rrf_score"),
                "rank_score": item.get("rank_score"),
                "rrf_ocr": item.get("rrf_ocr"),
                "rrf_clip": item.get("rrf_clip"),
                "rrf_shadow": item.get("rrf_shadow"),
                "score_breakdown": item.get("score_breakdown"),
            }
        )
    return preview


def _channel_overlap(results: list[dict]) -> dict[str, int]:
    overlap = {
        "ocr_only": 0,
        "clip_only": 0,
        "shadow_only": 0,
        "ocr_clip": 0,
        "ocr_shadow": 0,
        "clip_shadow": 0,
        "all_three": 0,
    }
    for item in results:
        hits = {
            channel
            for channel in ("ocr", "clip", "shadow")
            if item.get(f"rrf_{channel}") not in (None, 0, 0.0)
        }
        if hits == {"ocr"}:
            overlap["ocr_only"] += 1
        elif hits == {"clip"}:
            overlap["clip_only"] += 1
        elif hits == {"shadow"}:
            overlap["shadow_only"] += 1
        elif hits == {"ocr", "clip"}:
            overlap["ocr_clip"] += 1
        elif hits == {"ocr", "shadow"}:
            overlap["ocr_shadow"] += 1
        elif hits == {"clip", "shadow"}:
            overlap["clip_shadow"] += 1
        elif hits == {"ocr", "clip", "shadow"}:
            overlap["all_three"] += 1
    return overlap


def _isoformat_or_none(value: object | None) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)
