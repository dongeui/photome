"""Hybrid semantic/OCR search ranking.

Backend-agnostic so it can be wired to any HybridSearchBackend implementation.
Natural language date expressions (작년, 여름, etc.) are extracted here and
propagated to the embedding backend as date filters.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time, timedelta
from typing import Protocol

from app.services.search.planner import QueryPlan, plan_query
from app.services.search.seed import seed_list

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants — override via environment variables without code change
# ---------------------------------------------------------------------------
RRF_K: float = float(os.environ.get("PHOTOME_RRF_K", "60.0"))

# Scoring bonuses (0.0–1.0 scale)
BOOST_OCR_EXACT: float     = float(os.environ.get("PHOTOME_BOOST_OCR_EXACT",     "0.22"))
BOOST_OCR_TOKEN: float     = float(os.environ.get("PHOTOME_BOOST_OCR_TOKEN",     "0.12"))
BOOST_TAG_EXACT: float     = float(os.environ.get("PHOTOME_BOOST_TAG_EXACT",     "0.9"))
BOOST_PROMOTED: float      = float(os.environ.get("PHOTOME_BOOST_PROMOTED",      "0.15"))
BOOST_PLACE_MATCH: float   = float(os.environ.get("PHOTOME_BOOST_PLACE_MATCH",   "0.08"))
BOOST_PERSON_MATCH: float  = float(os.environ.get("PHOTOME_BOOST_PERSON_MATCH",  "0.06"))
BOOST_DATE_IN_RANGE: float = float(os.environ.get("PHOTOME_BOOST_DATE_IN_RANGE", "0.08"))

# Multi-channel agreement multipliers
CHANNEL_BONUS_2: float = float(os.environ.get("PHOTOME_CHANNEL_BONUS_2", "1.15"))
CHANNEL_BONUS_3: float = float(os.environ.get("PHOTOME_CHANNEL_BONUS_3", "1.30"))

# Result diversity & dedup
DIVERSITY_MAX_PER_DAY: int = int(os.environ.get("PHOTOME_DIVERSITY_MAX_PER_DAY", "5"))
BURST_DEDUP_SECONDS: int   = int(os.environ.get("PHOTOME_BURST_DEDUP_SECONDS",   "3"))

# Fuzzy correction
FUZZY_SIMILARITY_THRESHOLD: float = float(os.environ.get("PHOTOME_FUZZY_SIMILARITY", "0.5"))
FUZZY_MAX_TAGS: int                = int(os.environ.get("PHOTOME_FUZZY_MAX_TAGS",     "2000"))

# NGram scoring multipliers (OCR n-gram bonus)
BOOST_NGRAM_NO_TEXT: float = float(os.environ.get("PHOTOME_BOOST_NGRAM_NO_TEXT", "0.10"))
BOOST_NGRAM_FACTOR: float  = float(os.environ.get("PHOTOME_BOOST_NGRAM_FACTOR",  "0.08"))

# ---------------------------------------------------------------------------
# Simple TTL query result cache (in-memory, per-process)
# ---------------------------------------------------------------------------
_CACHE_TTL_SECONDS: int  = int(os.environ.get("PHOTOME_SEARCH_CACHE_TTL",      "60"))
_CACHE_MAX_SIZE: int     = int(os.environ.get("PHOTOME_SEARCH_CACHE_MAX_SIZE",  "256"))
_query_cache: dict[str, tuple[float, list[dict], dict]] = {}  # key → (ts, results, meta)
_cache_lock = threading.Lock()  # protects _query_cache against concurrent writers


def _cache_key(query: str, limit: int, mode: str, place_filter: str | None) -> str:
    return f"{query}|{limit}|{mode}|{place_filter or ''}"


def _cache_get(key: str) -> tuple[list[dict], dict] | None:
    with _cache_lock:
        entry = _query_cache.get(key)
        if entry is None:
            return None
        ts, results, meta = entry
        if _time.monotonic() - ts > _CACHE_TTL_SECONDS:
            del _query_cache[key]
            return None
        return results, meta


def _cache_set(key: str, results: list[dict], meta: dict) -> None:
    with _cache_lock:
        # Evict oldest entries when cache grows beyond 256 keys
        if len(_query_cache) >= _CACHE_MAX_SIZE:
            oldest_key = min(_query_cache, key=lambda k: _query_cache[k][0])
            del _query_cache[oldest_key]
        _query_cache[key] = (_time.monotonic(), results, meta)


def clear_query_cache() -> int:
    """Invalidate all cached search results.

    Call this after semantic maintenance completes so that newly indexed
    content is visible to subsequent queries without waiting for TTL expiry.
    Returns the number of entries cleared.
    """
    with _cache_lock:
        count = len(_query_cache)
        _query_cache.clear()
    return count

FACE_HINTS: frozenset[str] = frozenset(seed_list("face_hints"))
TEXT_HINTS: frozenset[str] = frozenset(seed_list("text_hints"))
SCREEN_HINTS: frozenset[str] = frozenset(seed_list("screen_hints"))
TRAVEL_HINTS: frozenset[str] = frozenset(seed_list("travel_hints"))
CELEBRATION_HINTS: frozenset[str] = frozenset(seed_list("celebration_hints"))


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

    def load_feedback_sets(self) -> tuple[set[str], set[str]]:
        """Returns (hidden_file_ids, promoted_file_ids) from SearchFeedback table."""
        ...


class RerankerProtocol(Protocol):
    """Pluggable reranker interface.

    A reranker receives the fused result list and the parsed query plan, and
    returns a re-sorted list.  The default pass-through implementation lets the
    existing RRF + boost pipeline run unchanged; a learned reranker can replace
    it without touching search_with_meta.

    To wire a custom reranker:
        service = HybridSearchService(backend, reranker=MyReranker())
    """

    def rerank(self, results: list[dict], plan: "QueryPlan") -> list[dict]: ...


class PassThroughReranker:
    """Default no-op reranker: returns results unchanged."""

    def rerank(self, results: list[dict], plan: "QueryPlan") -> list[dict]:  # noqa: ARG002
        return results


class HybridSearchService:
    def __init__(
        self,
        backend: HybridSearchBackend,
        reranker: RerankerProtocol | None = None,
    ) -> None:
        self._backend = backend
        self._reranker: RerankerProtocol = reranker or PassThroughReranker()

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
        allow_date_fallback: bool = True,
        allow_condition_fallback: bool = True,
        use_planner_dates: bool = True,
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
                self._log_search_event(
                    query,
                    effective_mode=str(meta.get("effective_mode") or mode),
                    intent=str((meta.get("query_plan") or {}).get("intent") or ""),
                    result_count=len(results),
                    fallback=meta.get("fallback"),
                )
                return results, meta

        # Load tag vocabulary from DB (TTL-cached) so planner recognises
        # user-created tags like "여수밤바다" without any hardcoded dictionary.
        tag_vocab = None
        if hasattr(self._backend, "get_tag_vocabulary"):
            try:
                tag_vocab = self._backend.get_tag_vocabulary()
            except Exception as exc:
                logger.warning("Failed to load tag vocabulary — search quality may be reduced: %s", exc)
        plan = plan_query(query, tag_vocab=tag_vocab)
        cleaned = plan.normalized_query
        # Guard punctuation/noise while allowing meaningful one-character visual
        # queries such as "꽃" when they are present in the packaged vocabulary.
        if not re.search(r"[A-Za-z가-힣0-9]{2,}", query) and not (
            plan.person_terms or plan.place_terms or plan.ocr_terms or plan.visual_terms
        ):
            return [], {"effective_mode": mode, "intent_reason": "degenerate"}
        normalized_mode = mode if mode in {"hybrid", "ocr", "semantic"} else "hybrid"

        # Auto-extract date range from natural language when caller didn't specify one
        if use_planner_dates and date_from is None and date_to is None and plan.date_from is not None:
            date_from = datetime.combine(plan.date_from, time.min)
            date_to = datetime.combine(plan.date_to, time.max) if plan.date_to else None

        keyword_query = plan.keyword_query or cleaned

        # OCR must run first — its results drive effective_mode resolution
        ocr_results = self._backend.search_by_ocr(keyword_query, limit=limit) if normalized_mode in {"hybrid", "ocr"} else []
        effective_mode, intent_reason = resolve_effective_mode(
            cleaned, normalized_mode, ocr_results, planner_intent=plan.intent, tag_vocab=tag_vocab
        )

        # Shadow and CLIP are independent → run in parallel
        need_shadow = effective_mode in {"hybrid", "ocr", "semantic"}
        need_clip = effective_mode in {"hybrid", "semantic"}

        shadow_results: list[dict] = []
        clip_results: list[dict] = []

        supports_parallel = False
        if hasattr(self._backend, "supports_parallel_channels"):
            try:
                supports_parallel = bool(self._backend.supports_parallel_channels())
            except Exception:
                supports_parallel = False

        if need_shadow and need_clip and supports_parallel:
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_shadow = pool.submit(self._backend.search_by_shadow_doc, keyword_query, limit=limit)
                fut_clip = pool.submit(self._search_clip_variants, plan, limit, place_filter, date_from, date_to)
                shadow_results = fut_shadow.result()
                clip_results = fut_clip.result()
        elif need_shadow and need_clip:
            shadow_results = self._backend.search_by_shadow_doc(keyword_query, limit=limit)
            clip_results = self._search_clip_variants(plan, limit, place_filter, date_from, date_to)
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
        # Adaptive weight redistribution: if CLIP channel yielded nothing,
        # shift its budget to shadow so scoring isn't artificially deflated.
        # This handles both "CLIP not installed" and "no embeddings yet" gracefully.
        if not clip_results and weights.get("clip", 0.0) > 0:
            weights = dict(weights)
            weights["shadow"] = weights.get("shadow", 0.0) + weights["clip"]
            weights["clip"] = 0.0

        merged = fuse_ranked_results(
            effective_mode,
            intent_reason,
            ocr_results if effective_mode in {"hybrid", "ocr"} else [],
            clip_results,
            shadow_results,
            weights=weights,
        )
        # Load feedback signals (hidden + promoted file IDs)
        hidden_ids: set[str] = set()
        promoted_ids: set[str] = set()
        if hasattr(self._backend, "load_feedback_sets"):
            try:
                hidden_ids, promoted_ids = self._backend.load_feedback_sets()
            except Exception as exc:
                logger.warning("Failed to load search feedback sets: %s", exc)

        # Filter out hidden files before scoring
        if hidden_ids:
            merged = [r for r in merged if str(r.get("file_id", "")) not in hidden_ids]

        debug_candidates = [dict(item) for item in merged] if debug else None
        apply_exact_ocr_boost(cleaned, merged)
        apply_exact_tag_boost(merged)
        apply_context_filter_boost(merged, plan)
        apply_date_soft_scoring(merged, plan)
        apply_feedback_boost(merged, promoted_ids)
        merged.sort(key=search_sort_key, reverse=True)
        merged = remove_near_duplicates(merged)
        merged = apply_diversity_cap(merged)
        merged = self._reranker.rerank(merged, plan)
        set_match_explanations(merged)
        final = merged[:limit]

        meta: dict = {
            "effective_mode": effective_mode,
            "intent_reason": intent_reason,
            "query_plan": plan.to_meta(),
            "weight_overrides": weight_overrides or {},
        }

        # Zero-result fallback 1: loosen date filter and retry once
        if allow_date_fallback and not final and (plan.date_from is not None) and not debug:
            loosened = self._loosened_date_fallback(
                query=query,
                limit=limit,
                place_filter=place_filter,
                mode=mode,
                weight_overrides=weight_overrides,
            )
            if loosened:
                final = loosened
                meta["fallback"] = "date_relaxed"

        # Zero-result fallback 2: fuzzy-correct query tokens via DB tag vocabulary
        if not final and not debug and tag_vocab is not None:
            corrected = fuzzy_correct_query(cleaned, tag_vocab)
            if corrected and corrected != cleaned:
                corrected_results, _ = self.search_with_meta(
                    corrected,
                    limit=limit,
                    place_filter=place_filter,
                    mode=mode,
                    debug=False,
                    weight_overrides=weight_overrides,
                )
                if corrected_results:
                    final = corrected_results
                    meta["fallback"] = "fuzzy_corrected"
                    meta["fuzzy_corrected_query"] = corrected
        # Zero-result fallback 3: progressively relax place/person conditions
        # e.g. "작년 바다에서 가족이랑" → no match → try "바다" alone → try "가족" alone
        if (
            not final
            and not debug
            and allow_condition_fallback
            and (plan.place_terms or plan.person_terms)
        ):
            relaxed, relaxed_label = self._loosened_condition_fallback(
                plan=plan,
                limit=limit,
                mode=mode,
                weight_overrides=weight_overrides,
            )
            if relaxed:
                final = relaxed
                meta["fallback"] = relaxed_label

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

        # Implicit feedback: log every search event for future analysis/tuning
        self._log_search_event(
            query,
            effective_mode=effective_mode,
            intent=plan.intent,
            result_count=len(final),
            fallback=meta.get("fallback"),
        )

        logger.debug(
            "search query=%r mode=%s intent=%s channels=ocr:%d clip:%d shadow:%d final=%d",
            query, effective_mode, plan.intent,
            len(ocr_results), len(clip_results), len(shadow_results), len(final),
        )
        return final, meta

    def _log_search_event(
        self,
        query: str,
        *,
        effective_mode: str,
        intent: str,
        result_count: int,
        fallback: str | None = None,
    ) -> None:
        if not hasattr(self._backend, "log_search_event"):
            return
        try:
            self._backend.log_search_event(
                query,
                effective_mode=effective_mode,
                intent=intent,
                result_count=result_count,
                fallback=fallback,
            )
        except Exception:
            pass

    def _loosened_date_fallback(
        self,
        *,
        query: str,
        limit: int,
        place_filter: str | None,
        mode: str,
        weight_overrides: dict[str, float] | None,
    ) -> list[dict]:
        """Retry the search without date filters when the original returns nothing.

        Passes date_from=None so that semantic/shadow channels are not restricted,
        letting the user discover relevant photos even if the planner's date
        extraction was slightly off.
        """
        results, _ = self.search_with_meta(
            query,
            limit=limit,
            place_filter=place_filter,
            date_from=None,
            date_to=None,
            mode=mode,
            debug=False,
            weight_overrides=weight_overrides,
            allow_date_fallback=False,
            use_planner_dates=False,
        )
        return results

    def _loosened_condition_fallback(
        self,
        *,
        plan: "QueryPlan",
        limit: int,
        mode: str,
        weight_overrides: dict[str, float] | None,
    ) -> tuple[list[dict], str]:
        """Retry with progressively fewer conditions when the combined query returns nothing.

        Order: visual terms only → each place term → each person term.
        Each sub-search disables further fallbacks to prevent recursion.
        """
        common: dict = dict(
            limit=limit,
            mode=mode,
            debug=False,
            weight_overrides=weight_overrides,
            allow_date_fallback=False,
            allow_condition_fallback=False,
        )
        # Step 1: visual terms stripped of place/person context
        if plan.visual_terms:
            results, _ = self.search_with_meta(" ".join(plan.visual_terms[:3]), **common)
            if results:
                return results, "condition_visual_only"
        # Step 2: each place term individually
        for term in plan.place_terms[:2]:
            results, _ = self.search_with_meta(term, **common)
            if results:
                return results, "condition_place_only"
        # Step 3: each person term individually
        for term in plan.person_terms[:2]:
            results, _ = self.search_with_meta(term, **common)
            if results:
                return results, "condition_person_only"
        return [], ""

    def _search_clip_variants(
        self,
        plan: QueryPlan,
        limit: int,
        place_filter: str | None,
        date_from: object | None,
        date_to: object | None,
    ) -> list[dict]:
        # Determine effective place filters:
        # - explicit caller-provided place_filter takes precedence (single value)
        # - otherwise use place_terms from the query plan (OR semantics across terms)
        if place_filter is not None:
            effective_place_filters: list[str | None] = [place_filter]
        elif plan.place_terms:
            effective_place_filters = list(plan.place_terms)
        else:
            effective_place_filters = [None]

        merged: dict[str, dict] = {}
        multi_place = len(effective_place_filters) > 1

        for variant in plan.visual_queries:
            query_bytes = self._backend.encode_text(variant)
            if not query_bytes:
                # CLIP model not available or encoding failed — skip this variant
                continue

            if multi_place:
                # Fetch once without place_filter (larger limit to compensate for
                # selectivity), then filter by tag values in Python.
                # This avoids N FAISS searches + N _batch_load_supplements calls.
                pf_set = {pf.casefold() for pf in effective_place_filters}
                raw = self._backend.search_by_embedding(
                    query_bytes,
                    limit=limit * len(effective_place_filters) * 4,
                    place_filter=None,
                    date_from=date_from,
                    date_to=date_to,
                )
                rank = 0
                for result in raw:
                    tag_values = {
                        str(t.get("value", "")).casefold()
                        for t in (result.get("tags") or [])
                    }
                    if not (pf_set & tag_values):
                        continue
                    rank += 1
                    file_id = str(result["file_id"])
                    if file_id not in merged or float(result.get("distance", 99.0)) < float(merged[file_id].get("distance", 99.0)):
                        result["semantic_query"] = variant
                        result["semantic_variant_rank"] = rank
                        merged[file_id] = result
            else:
                results = self._backend.search_by_embedding(
                    query_bytes,
                    limit=limit,
                    place_filter=effective_place_filters[0],
                    date_from=date_from,
                    date_to=date_to,
                )
                for rank, result in enumerate(results, start=1):
                    file_id = str(result["file_id"])
                    if file_id not in merged or float(result.get("distance", 99.0)) < float(merged[file_id].get("distance", 99.0)):
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
        rrf_base = (
            float(result.get("rrf_ocr") or 0.0)
            + float(result.get("rrf_clip") or 0.0)
            + float(result.get("rrf_shadow") or 0.0)
        )
        # Multi-channel agreement bonus: more channels = higher confidence
        n_channels = len(hits)
        if n_channels >= 3:
            channel_multiplier = CHANNEL_BONUS_3
        elif n_channels == 2:
            channel_multiplier = CHANNEL_BONUS_2
        else:
            channel_multiplier = 1.0
        result["rrf_score"] = rrf_base * channel_multiplier
        result["score_breakdown"] = [
            {
                "stage": "rrf",
                "rrf_ocr": float(result.get("rrf_ocr") or 0.0),
                "rrf_clip": float(result.get("rrf_clip") or 0.0),
                "rrf_shadow": float(result.get("rrf_shadow") or 0.0),
                "rrf_base": rrf_base,
                "channel_multiplier": channel_multiplier,
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


def resolve_effective_mode(
    query: str,
    requested_mode: str,
    ocr_results: list[dict],
    *,
    planner_intent: str | None = None,
    tag_vocab=None,
) -> tuple[str, str]:
    if requested_mode != "hybrid":
        return requested_mode, "manual"

    lowered = query.casefold()
    tokens = set(lowered.split())

    # Extend seed hint sets with actual DB-sourced tag vocabulary so user-created
    # tags (e.g. specific person names or place names) trigger the right mode.
    person_tags: frozenset[str] = tag_vocab.person_tags if tag_vocab is not None else frozenset()
    place_tags: frozenset[str] = tag_vocab.place_tags if tag_vocab is not None else frozenset()

    has_face_hint = any(hint in lowered for hint in FACE_HINTS) or bool(tokens & person_tags)
    has_text_hint = any(hint in lowered for hint in TEXT_HINTS)
    has_screen_hint = any(hint in lowered for hint in SCREEN_HINTS)
    has_travel_hint = any(hint in lowered for hint in TRAVEL_HINTS) or bool(tokens & place_tags)
    has_celebration_hint = any(hint in lowered for hint in CELEBRATION_HINTS)
    word_hits = [result for result in ocr_results if result.get("ocr_match_kind") == "word"]
    phrase_hits = [result for result in ocr_results if result.get("ocr_match_kind") == "phrase"]
    has_code_like_text = any(ch.isdigit() for ch in query) or any(ch in query for ch in "-_:/[]()#")
    is_short_query = len(query.strip()) <= 12

    # planner_intent='ocr' signals a clear text-search query — defer to it
    # before running the hint-based heuristics below.
    if planner_intent == "ocr" and not has_face_hint:
        return "ocr", "planner-ocr"

    if has_face_hint and (has_text_hint or has_screen_hint or has_code_like_text):
        return "hybrid", "auto-mixed"
    if has_face_hint:
        return "semantic", "auto-face"
    # Travel and celebration photos are strongly visual → prefer semantic
    if has_travel_hint and not has_text_hint:
        return "semantic", "auto-travel"
    if has_celebration_hint and not has_text_hint:
        return "semantic", "auto-celebration"

    # planner detected a visual intent (date + place/person, pure visual)
    # and there are no strong OCR signals → go semantic
    if planner_intent == "visual" and not has_text_hint and not word_hits and not phrase_hits:
        return "semantic", "planner-visual"

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
        raw = {"ocr": 0.62, "clip": 0.04, "shadow": 0.22}
    elif effective_mode == "semantic":
        raw = {"ocr": 0.03, "clip": 0.70, "shadow": 0.18}
    elif intent_reason == "auto-mixed":
        raw = {"ocr": 0.36, "clip": 0.34, "shadow": 0.18}
    else:
        raw = {"ocr": 0.35, "clip": 0.36, "shadow": 0.17}
    # Normalise so channels always sum to 1.0 — prevents max_score distortion
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


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


def apply_date_soft_scoring(results: list[dict], plan: "QueryPlan") -> None:
    """Soft-score OCR/shadow results by date proximity when planner extracted a date range.

    CLIP results are already hard-filtered by date in search_by_embedding.
    OCR and shadow results have no date filter — this function rewards results
    that fall within the queried date range and gently discounts those outside.
    """
    if plan.date_from is None:
        return

    from datetime import datetime as _dt

    date_from = _dt.combine(plan.date_from, time.min)
    date_to = _dt.combine(plan.date_to, time.max) if plan.date_to else None

    for result in results:
        # Skip pure CLIP hits — they're already date-filtered
        if result.get("match_reason") == "clip":
            continue
        captured = result.get("captured_at")
        if captured is None:
            continue
        try:
            if isinstance(captured, str):
                captured = _dt.fromisoformat(captured)
            in_range = captured >= date_from and (date_to is None or captured <= date_to)
        except Exception:
            continue

        if in_range:
            bonus = BOOST_DATE_IN_RANGE
            result["rank_score"] = min(1.0, float(result.get("rank_score", 0.0)) + bonus)
            result.setdefault("score_breakdown", []).append(
                {"stage": "date_in_range_bonus", "delta": bonus,
                 "rank_score": float(result.get("rank_score") or 0.0)}
            )


def apply_context_filter_boost(results: list[dict], plan: "QueryPlan") -> None:
    """Boost results that match place/person terms from the query plan.

    This rewards multi-condition matches (e.g. "제주 + 가족") so they rank
    above results matching only one condition.
    """
    if not plan.place_terms and not plan.person_terms:
        return

    place_set = {t.casefold() for t in plan.place_terms}
    person_set = {t.casefold() for t in plan.person_terms}

    for result in results:
        tags = result.get("tags") or []
        tag_values = {str(tag.get("value", "")).casefold() for tag in tags}
        bonus = 0.0

        matched_places = place_set & tag_values
        if matched_places:
            bonus += BOOST_PLACE_MATCH * len(matched_places)

        matched_persons = person_set & tag_values
        if matched_persons:
            bonus += BOOST_PERSON_MATCH * len(matched_persons)

        if bonus > 0:
            result["rank_score"] = min(1.0, float(result.get("rank_score", 0.0)) + bonus)
            result.setdefault("score_breakdown", []).append(
                {"stage": "context_filter_bonus", "delta": bonus,
                 "rank_score": float(result.get("rank_score") or 0.0)}
            )


def apply_feedback_boost(results: list[dict], promoted_ids: set[str]) -> None:
    """Apply a rank bonus to files the user explicitly promoted.

    Promoted files receive a +0.15 boost so they consistently appear near the
    top without hard-overriding relevance scores.
    """
    if not promoted_ids:
        return
    for result in results:
        if str(result.get("file_id", "")) in promoted_ids:
            bonus = BOOST_PROMOTED
            result["rank_score"] = min(1.0, float(result.get("rank_score", 0.0)) + bonus)
            result.setdefault("score_breakdown", []).append(
                {"stage": "user_promoted", "delta": bonus, "rank_score": float(result.get("rank_score") or 0.0)}
            )


def apply_exact_ocr_boost(query: str, results: list[dict]) -> None:
    lowered = query.casefold()
    tokens = [token for token in lowered.split() if token]
    for result in results:
        ocr_text = str(result.get("ocr_text") or "")
        if not ocr_text:
            # Still apply ngram score bonus when OCR text is absent
            ngram = float(result.get("ngram_score") or 0.0)
            if ngram > 0:
                bonus = ngram * BOOST_NGRAM_NO_TEXT
                result["rank_score"] = min(1.0, float(result.get("rank_score", 0.0)) + bonus)
                result.setdefault("score_breakdown", []).append(
                    {"stage": "ocr_ngram_bonus", "delta": bonus, "rank_score": float(result.get("rank_score") or 0.0)}
                )
            continue
        ocr_lower = ocr_text.casefold()
        ngram_bonus = float(result.get("ngram_score") or 0.0) * BOOST_NGRAM_FACTOR
        if lowered in ocr_lower:
            bonus = BOOST_OCR_EXACT + ngram_bonus
            result["rank_score"] = min(1.0, float(result.get("rank_score", 0.0)) + bonus)
            result["ocr_exact_match"] = True
            result.setdefault("score_breakdown", []).append(
                {"stage": "ocr_exact_bonus", "delta": bonus, "rank_score": float(result.get("rank_score") or 0.0)}
            )
        elif tokens and all(token in ocr_lower for token in tokens):
            bonus = BOOST_OCR_TOKEN + ngram_bonus
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
            bonus = BOOST_TAG_EXACT
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
        face_count = int(result.get("face_count") or 0)
        if face_count > 0:
            parts.append(f"얼굴 {face_count}명")

        place_tags = [
            tag["value"]
            for tag in (result.get("tags") or [])
            if tag.get("type") in ("place",) and not _is_coordinate_tag(str(tag.get("value", "")))
        ]
        if place_tags:
            parts.append(f"장소: {place_tags[0]}")

        exif_dt = result.get("captured_at")
        if exif_dt:
            try:
                from datetime import datetime as _dt
                dt = _dt.fromisoformat(str(exif_dt)) if isinstance(exif_dt, str) else exif_dt
                parts.append(f"{dt.year}년 {dt.month}월")
            except Exception:
                pass

        result["match_explanation"] = " · ".join(parts) if parts else "일치"


def apply_diversity_cap(
    results: list[dict],
    *,
    max_per_day: int = DIVERSITY_MAX_PER_DAY,
) -> list[dict]:
    """Limit how many results come from a single calendar day.

    Prevents a single event/album from dominating results when many photos
    share the same capture date.  Results without a date are always kept.
    The cap is applied in score order so the highest-scoring photos per day
    are kept.
    """
    day_counts: dict[str, int] = {}
    kept: list[dict] = []
    for result in results:
        captured = result.get("captured_at")
        if captured is None:
            kept.append(result)
            continue
        try:
            if isinstance(captured, str):
                from datetime import datetime as _dt
                captured = _dt.fromisoformat(captured)
            day_key = captured.strftime("%Y-%m-%d")
        except Exception:
            kept.append(result)
            continue

        if day_counts.get(day_key, 0) < max_per_day:
            day_counts[day_key] = day_counts.get(day_key, 0) + 1
            kept.append(result)
        # else: day is saturated — drop this result
    return kept


def remove_near_duplicates(
    results: list[dict],
    *,
    burst_seconds: int = BURST_DEDUP_SECONDS,
) -> list[dict]:
    """Remove burst duplicates: keep only the highest-scoring photo per burst window.

    Photos captured within `burst_seconds` of each other are treated as a burst.
    Within each burst, only the top-scored result is kept.  Results without a
    timestamp are always kept (conservative approach).
    """
    kept: list[dict] = []
    # bucket_key → (best_score, index_in_kept) — O(1) replacement via stored index
    seen_windows: dict[str, tuple[float, int]] = {}

    for result in results:
        captured = result.get("captured_at")
        if captured is None:
            kept.append(result)
            continue

        # Normalise to UTC timestamp integer, bucketed to burst_seconds
        try:
            if isinstance(captured, str):
                from datetime import datetime as _dt
                captured = _dt.fromisoformat(captured)
            ts = int(captured.timestamp())
        except Exception:
            kept.append(result)
            continue

        bucket = ts // burst_seconds
        score = float(result.get("rank_score") or 0.0)
        key = str(bucket)

        if key not in seen_windows:
            seen_windows[key] = (score, len(kept))
            kept.append(result)
        elif score > seen_windows[key][0]:
            _, idx = seen_windows[key]
            seen_windows[key] = (score, idx)
            kept[idx] = result
        # else: a better result for this burst is already kept — skip

    return kept


def fuzzy_correct_query(query: str, tag_vocab: "object") -> str | None:
    """Attempt to correct query tokens using character bigram Jaccard against DB tags.

    Dynamically reads the user's own TagVocabulary so no hardcoded dictionary is needed.
    Returns a corrected query string if any token was improved (similarity >= 0.5),
    otherwise None.

    Skipped when:
    - tag_vocab has no tags (library not yet indexed)
    - all tokens already match known tags exactly
    - tag vocabulary is very large (>2000 tags) to avoid O(n*m) slowness
    """
    from app.services.search.tokenizer import korean_nouns

    all_tags = getattr(tag_vocab, "all_tags", frozenset())
    if not all_tags or len(all_tags) > FUZZY_MAX_TAGS:
        return None

    tokens = korean_nouns(query)
    if not tokens:
        return None

    corrected: list[str] = []
    changed = False

    for token in tokens:
        if len(token) < 2 or token in all_tags:
            corrected.append(token)
            continue

        best_tag, best_sim = _best_bigram_match(token, all_tags)
        if best_tag and best_sim >= FUZZY_SIMILARITY_THRESHOLD:
            corrected.append(best_tag)
            changed = True
        else:
            corrected.append(token)

    return " ".join(corrected) if changed else None


def _best_bigram_match(token: str, tags: frozenset) -> tuple[str | None, float]:
    """Return (best_tag, similarity) using character bigram Jaccard index."""
    tok_bi = {token[i:i + 2] for i in range(len(token) - 1)}
    if not tok_bi:
        return None, 0.0

    best_tag: str | None = None
    best_sim = 0.0
    max_len_diff = max(2, len(token))  # ignore tags whose length differs too much

    for tag in tags:
        if abs(len(tag) - len(token)) > max_len_diff:
            continue
        tag_bi = {tag[i:i + 2] for i in range(len(tag) - 1)}
        if not tag_bi:
            continue
        union = len(tok_bi | tag_bi)
        if union == 0:
            continue
        sim = len(tok_bi & tag_bi) / union
        if sim > best_sim:
            best_sim = sim
            best_tag = tag

    return best_tag, best_sim


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
