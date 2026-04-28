from __future__ import annotations

from app.services.search.hybrid import HybridSearchService, resolve_effective_mode
from app.services.search.query_translate import expand_for_clip, normalize_query


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
        return []

    def encode_text(self, query: str) -> bytes:
        if "face" in query:
            return b"face"
        return b"embedding"


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


def test_hybrid_search_prefers_cross_channel_agreement() -> None:
    service = HybridSearchService(FakeBackend())

    results, meta = service.search_with_meta("동의", limit=5, mode="hybrid")

    assert meta["effective_mode"] == "ocr"
    assert results[0]["file_id"] == "file-a"
    assert results[0]["match_reason"] == "ocr"
    assert results[0]["rank_score"] == 1.0


def test_face_query_routes_to_semantic() -> None:
    service = HybridSearchService(FakeBackend())

    results, meta = service.search_with_meta("남자 얼굴", limit=5, mode="hybrid")

    assert meta["effective_mode"] == "semantic"
    assert results[0]["file_id"] == "face-file"
    assert results[0]["match_reason"] == "clip"
