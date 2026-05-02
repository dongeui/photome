from __future__ import annotations

from app.services.search.hybrid import HybridSearchService, resolve_effective_mode
from app.services.search.planner import plan_query
from app.services.search.query_translate import expand_for_clip, normalize_query
from app.services.search.tokenizer import korean_nouns


class FakeBackend:
    def search_by_ocr(self, query: str, *, limit: int) -> list[dict]:
        if query == "동의":
            return [
                {
                    "file_id": "file-a",
                    "ocr_text": "동의 필요",
                    "ocr_match_kind": "word",
                }
            ]
        return []

    def search_by_embedding(self, query_embedding: bytes, **_kwargs) -> list[dict]:
        if query_embedding == b"face":
            return [{"file_id": "face-file", "distance": 0.1}]
        return [
            {"file_id": "file-a", "distance": 0.2},
            {"file_id": "file-b", "distance": 0.4},
        ]

    def search_by_shadow_doc(self, query: str, *, limit: int) -> list[dict]:
        if query == "baby":
            return [{"file_id": "auto-baby", "tag_exact_match": True, "tags": [{"type": "auto", "value": "baby"}]}]
        return []

    def encode_text(self, query: str) -> bytes:
        if "face" in query:
            return b"face"
        return b"embedding"

    def suggest_related_tags(self, query: str, *, limit: int = 8) -> list[str]:
        return []


class ReverseReranker:
    def rerank(self, results: list[dict], plan) -> list[dict]:  # noqa: ANN001
        return list(reversed(results))


def test_korean_query_expands_for_clip() -> None:
    variants = expand_for_clip("자전거")

    assert variants[0] == "자전거"
    assert any("bicycle" in variant for variant in variants)


def test_typo_query_normalizes_before_expansion() -> None:
    assert normalize_query("어르굴") == "얼굴"
    variants = expand_for_clip("어르굴")

    assert variants[0] == "얼굴"
    assert any("face" in variant for variant in variants)


def test_baby_and_woman_queries_expand_for_clip() -> None:
    assert any("baby" in variant for variant in expand_for_clip("아기"))
    assert any("woman" in variant for variant in expand_for_clip("여자"))
    assert any("baby" in variant for variant in expand_for_clip("baby"))
    assert any("woman" in variant for variant in expand_for_clip("woman"))


def test_effective_mode_routes_short_ocr_hits_to_ocr() -> None:
    mode, reason = resolve_effective_mode(
        "동의",
        "hybrid",
        [{"file_id": "file-a", "ocr_match_kind": "word"}],
    )

    assert mode == "ocr"
    assert reason == "auto-word-match"


def test_query_planner_extracts_image_search_intents() -> None:
    plan = plan_query("작년 여름 바다에서 가족이랑 찍은 사진")

    assert plan.intent == "visual"
    assert plan.date_from is not None
    assert "바다" in plan.place_terms
    assert "가족" in plan.person_terms
    assert any("sea" in variant or "ocean" in variant for variant in plan.visual_queries)


def test_seaside_natural_language_query_expands_to_visual_search() -> None:
    plan = plan_query("바닷가에서 찍은 사진")

    assert plan.intent == "visual"
    assert "바닷가" in plan.place_terms
    assert any("seaside" in variant or "ocean" in variant for variant in plan.visual_queries)


def test_hybrid_search_prefers_cross_channel_agreement() -> None:
    service = HybridSearchService(FakeBackend())

    results, meta = service.search_with_meta("동의", limit=5, mode="hybrid")

    assert meta["effective_mode"] == "ocr"
    assert meta["query_plan"]["intent"] == "hybrid"
    assert results[0]["file_id"] == "file-a"
    assert results[0]["match_reason"] == "ocr"
    assert results[0]["rank_score"] == 1.0


def test_face_query_routes_to_semantic() -> None:
    service = HybridSearchService(FakeBackend())

    results, meta = service.search_with_meta("남자 얼굴", limit=5, mode="hybrid")

    assert meta["effective_mode"] == "semantic"
    assert results[0]["file_id"] == "face-file"
    assert results[0]["match_reason"] == "clip"


def test_semantic_query_uses_auto_tags_as_ranking_signal() -> None:
    service = HybridSearchService(FakeBackend())

    results, meta = service.search_with_meta("baby", limit=5, mode="hybrid")

    assert meta["effective_mode"] == "semantic"
    assert results[0]["file_id"] == "auto-baby"
    assert "태그 일치" in results[0]["match_explanation"]


def test_custom_reranker_order_is_preserved() -> None:
    service = HybridSearchService(FakeBackend(), reranker=ReverseReranker())

    results, meta = service.search_with_meta("random visual query", limit=5, mode="hybrid")

    assert meta["effective_mode"] == "hybrid"
    assert [item["file_id"] for item in results[:2]] == ["file-b", "file-a"]


def test_condition_fallback_relaxes_to_place_term() -> None:
    """When a compound query finds nothing, fallback tries each place term alone."""

    class NoResultBackend(FakeBackend):
        def search_by_ocr(self, query: str, *, limit: int) -> list[dict]:
            return []

        def search_by_embedding(self, query_embedding: bytes, **_kwargs) -> list[dict]:
            return []

        def search_by_shadow_doc(self, query: str, *, limit: int) -> list[dict]:
            # Only returns results when queried with a single known place term
            if query in ("바다", "sea", "beach"):
                return [{"file_id": "beach-file", "distance": 0.3}]
            return []

    service = HybridSearchService(NoResultBackend())

    # Complex query with place+person+date that finds nothing combined
    results, meta = service.search_with_meta(
        "작년 여름 바다에서 가족이랑 찍은 사진", limit=5, mode="hybrid"
    )

    assert results, "expected condition fallback to find results"
    assert meta.get("fallback") in (
        "condition_visual_only", "condition_place_only", "date_relaxed"
    )


def test_heuristic_splits_inner_joiner_compound() -> None:
    """가족이랑바다여행 should tokenize to [가족, 바다, 여행] without KoNLPy."""
    tokens = korean_nouns("가족이랑바다여행")
    assert "가족" in tokens
    assert "바다" in tokens
    assert "여행" in tokens


def test_heuristic_splits_particle_attached_token() -> None:
    """바다에서 (4 chars) should be split to [바다] with the lowered threshold."""
    tokens = korean_nouns("바다에서")
    assert "바다" in tokens


def test_heuristic_splits_rang_joiner() -> None:
    """엄마랑카페 should split to [엄마, 카페]."""
    tokens = korean_nouns("엄마랑카페")
    assert "엄마" in tokens
    assert "카페" in tokens
