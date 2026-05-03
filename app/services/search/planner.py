"""Rule-based query planning for image-first natural language search."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging
import re
from typing import TYPE_CHECKING

from app.services.search import query_translate
from app.services.search.seed import seed_dict, seed_list

if TYPE_CHECKING:
    from app.services.search.vocab import TagVocabulary

logger = logging.getLogger(__name__)
PERSON_TERMS: set[str] = set(seed_list("person_terms"))
OCR_TERMS: set[str] = set(seed_list("ocr_terms"))
PLACE_TERMS: set[str] = set(seed_list("place_terms"))
VISUAL_TERMS: set[str] = set(seed_list("visual_terms"))
PLACE_ALIASES: dict[str, str] = seed_dict("place_aliases")

DATE_STOP_TERMS = {
    "작년", "지난해", "재작년", "올해", "이번해",
    "지난달", "저번달", "이번달", "이번", "달",
    "이번주", "지난주", "저번주", "이번주말", "지난주말", "저번주말", "주",
    "봄", "여름", "가을", "겨울", "spring", "summer", "fall", "autumn", "winter",
    "오늘", "어제", "그제",
    "설날", "추석", "크리스마스", "성탄절",
    "전", "전날", "며칠전",
}


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    normalized_query: str
    keyword_query: str
    visual_queries: list[str]
    date_from: date | None
    date_to: date | None
    person_terms: list[str]
    place_terms: list[str]
    ocr_terms: list[str]
    visual_terms: list[str]
    intent: str

    def to_meta(self) -> dict:
        return {
            "normalized_query": self.normalized_query,
            "keyword_query": self.keyword_query,
            "visual_queries": self.visual_queries,
            "date_from": self.date_from.isoformat() if self.date_from else None,
            "date_to": self.date_to.isoformat() if self.date_to else None,
            "person_terms": self.person_terms,
            "place_terms": self.place_terms,
            "ocr_terms": self.ocr_terms,
            "visual_terms": self.visual_terms,
            "intent": self.intent,
        }


def plan_query(query: str, *, tag_vocab: "TagVocabulary | None" = None) -> QueryPlan:
    """Parse a natural-language query into a structured QueryPlan.

    tag_vocab: optional TagVocabulary loaded from the user's DB.  When provided,
    place/person matching is extended with every tag the user has ever created —
    so long-tail names like "여수밤바다" or "에버랜드" are recognised without
    any hardcoded dictionary entry.
    """
    normalized = query_translate.normalize_query(query)
    tokens = _tokens(normalized)
    date_from, date_to = query_translate.extract_date_range(normalized)
    person_terms = _matching_terms_with_vocab(tokens, normalized, PERSON_TERMS,
                                              tag_vocab.person_tags if tag_vocab else None)
    place_terms = _matching_terms_with_vocab(tokens, normalized, PLACE_TERMS,
                                             tag_vocab.place_tags if tag_vocab else None)
    ocr_terms = _matching_terms(tokens, normalized, OCR_TERMS)
    visual_terms = _matching_terms(tokens, normalized, VISUAL_TERMS)
    keyword_tokens = [
        token
        for token in tokens
        if token not in DATE_STOP_TERMS and not _is_year_token(token)
    ]
    keyword_query = " ".join(keyword_tokens) if keyword_tokens else normalized
    visual_queries = query_translate.expand_for_clip(normalized)
    intent = _intent(
        person_terms=person_terms,
        place_terms=place_terms,
        ocr_terms=ocr_terms,
        visual_terms=visual_terms,
        date_from=date_from,
        keyword_query=keyword_query,
        normalized_query=normalized,
    )
    return QueryPlan(
        original_query=query,
        normalized_query=normalized,
        keyword_query=keyword_query,
        visual_queries=visual_queries,
        date_from=date_from,
        date_to=date_to,
        person_terms=person_terms,
        place_terms=place_terms,
        ocr_terms=ocr_terms,
        visual_terms=visual_terms,
        intent=intent,
    )


# 흔한 한국어 어미/조사 — 붙어있는 복합 토큰을 분해할 때 제거
_KO_TRAILING_ENDINGS = (
    "에서의", "으로의", "로부터",
    "갔던", "찍었던", "찍은", "촬영한", "찍힌",
    "에서", "으로", "에게", "한테", "부터", "까지",
    "이랑", "하고", "와", "과",
    "이라는", "라는", "이라", "이고", "이며",
    "하는", "하던", "했던", "된", "되는", "되던",
    "있는", "있던", "없는",
    "사진", "영상", "이미지", "그림", "컷",
    "에의", "의", "을", "를", "이", "가", "은", "는", "에", "로", "와", "과",
)


def _split_compound_token(token: str) -> list[str]:
    """공백 없이 붙어있는 복합 한글 토큰을 어미/조사 기준으로 분해.

    예: "제주도갔던사진" → ["제주도", "갔던", "사진"]
        "엄마랑" → ["엄마"]
        "서울에서" → ["서울"]
    의미 없는 잔여 토큰(1글자, 순수 어미)은 제거.
    """
    result: list[str] = []
    remaining = token
    # 최대 5회 반복으로 순차적으로 어미 분리
    for _ in range(5):
        if not remaining or not re.search(r"[가-힣]", remaining):
            break
        split_here: str | None = None
        for ending in _KO_TRAILING_ENDINGS:
            if remaining.endswith(ending) and len(remaining) > len(ending):
                core = remaining[: -len(ending)]
                if len(core) >= 2:
                    split_here = core
                    tail = ending
                    break
        if split_here:
            # 잘린 어미 자체도 의미 있으면 보존 (예: "사진", "영상")
            if tail in {"사진", "영상", "이미지", "그림", "컷"} and len(tail) >= 2:
                result.append(tail)
            remaining = split_here
        else:
            break
    if remaining and len(remaining) >= 2:
        result.insert(0, remaining)
    return result if result else [token]


def _tokens(query: str) -> list[str]:
    """Tokenize query into noun-level tokens.

    Uses the morphological tokenizer (KoNLPy Okt/Mecab) when available,
    falling back to the heuristic compound-token splitter.  Both paths
    are imported from tokenizer.py so this file has no direct NLP dependency.
    """
    from app.services.search.tokenizer import korean_nouns
    return korean_nouns(query)


def _matching_terms(tokens: list[str], normalized: str, terms: set[str]) -> list[str]:
    return _matching_terms_with_vocab(tokens, normalized, terms, None)


def _matching_terms_with_vocab(
    tokens: list[str],
    normalized: str,
    static_terms: set[str],
    dynamic_tags: "frozenset[str] | None",
) -> list[str]:
    """Match tokens/normalized query against static_terms + dynamic_tags from DB.

    dynamic_tags (from TagVocabularyCache) extends matching beyond the hardcoded
    static_terms so that user-created tags like "여수밤바다" or "에버랜드" are
    automatically detected without any dictionary entry.
    """
    lowered = normalized.casefold()
    hits: set[str] = set()

    # 1. Static vocabulary match (token exact or substring in normalized query)
    for term in static_terms:
        if term in tokens or term in lowered:
            hits.add(term)

    # 2. PLACE_ALIASES expansion for static vocab
    for token in tokens:
        canonical = PLACE_ALIASES.get(token)
        if canonical and canonical in static_terms:
            hits.add(canonical)
    for alias, canonical in PLACE_ALIASES.items():
        if alias in lowered and canonical in static_terms:
            hits.add(canonical)

    # 3. Dynamic tag vocabulary from DB — substring match within query tokens
    # Each DB tag value is checked against each query token (both directions):
    #   "제주" tag  + "제주도갔던" token → "제주" in token  → hit
    #   "에버랜드" tag + "에버랜드" token  → exact match      → hit
    if dynamic_tags:
        for tag in dynamic_tags:
            if not tag:
                continue
            # Exact token match or tag is a substring of a token (handles compound tokens)
            if tag in tokens:
                hits.add(tag)
            elif tag in lowered:
                hits.add(tag)
            else:
                # Check if any query token contains or is contained by the tag value
                for token in tokens:
                    if tag in token or (len(tag) >= 3 and token in tag):
                        hits.add(tag)
                        break

    return sorted(hits, key=lambda item: (len(item), item))


def _is_year_token(token: str) -> bool:
    return bool(re.fullmatch(r"(20\d{2}|[2-9]\d)년?", token))


def _intent(
    *,
    person_terms: list[str],
    place_terms: list[str],
    ocr_terms: list[str],
    visual_terms: list[str],
    date_from: date | None,
    keyword_query: str,
    normalized_query: str,
) -> str:
    has_text = bool(ocr_terms)
    has_person = bool(person_terms)
    has_place = bool(place_terms)
    has_visual = bool(visual_terms or has_person or has_place or date_from)

    # OCR + visual signals together → mixed (e.g. "엄마 카톡 오류")
    if has_text and has_visual:
        return "mixed"
    if has_text:
        return "ocr"

    # Person + place together → visual (e.g. "엄마랑 카페")
    if has_person and has_place:
        return "visual"
    # Date + person or place → visual (e.g. "작년 제주 가족")
    if date_from and (has_person or has_place):
        return "visual"

    if has_visual:
        return "visual"
    if keyword_query != normalized_query:
        return "keyword"
    return "hybrid"
