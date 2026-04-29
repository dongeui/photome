"""Rule-based query planning for image-first natural language search."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re

from app.services.search import query_translate


PERSON_TERMS = {
    "face", "faces", "person", "people", "portrait", "selfie", "family", "friend", "friends",
    "얼굴", "사람", "인물", "셀카", "가족", "친구", "커플", "엄마", "아빠", "할머니", "할아버지",
    "남자", "남성", "여자", "여성", "아기", "애기", "아이", "어린이",
    "형", "오빠", "누나", "언니", "동생", "남동생", "여동생", "조카", "남편", "아내",
    "선생님", "친척", "이모", "삼촌", "외삼촌", "고모", "사촌",
}
OCR_TERMS = {
    "text", "ocr", "document", "receipt", "screenshot", "screen", "error", "message", "chat",
    "텍스트", "글씨", "문서", "영수증", "스크린샷", "화면", "오류", "메시지", "대화", "카톡",
    "카카오톡", "캡처", "캡쳐", "갈무리", "인스타", "인스타그램", "앱화면",
    "공지", "알림", "알림창", "팝업", "대화창", "채팅",
}
PLACE_TERMS = {
    "서울", "제주", "부산", "공항", "카페", "식당", "학교", "집", "공원", "해변", "바다", "산",
    "seoul", "jeju", "busan", "airport", "cafe", "restaurant", "school", "home", "park", "beach",
    "sea", "ocean", "mountain",
    "강남", "홍대", "명동", "경복궁", "인사동", "이태원", "한강", "속초", "강릉", "경주",
    "전주", "대전", "광주", "인천", "수원", "대구", "울산",
    "해수욕장", "박물관", "동물원", "수족관", "놀이공원", "놀이터", "체육관", "도서관",
    "지하철", "버스", "기차", "ktx", "고속도로",
}
VISUAL_TERMS = {
    "baby", "food", "beach", "travel", "wedding", "birthday", "dog", "cat", "mountain", "sunset",
    "아기", "음식", "바다", "여행", "결혼식", "생일", "강아지", "고양이", "산", "일몰",
    "꽃", "하늘", "공원", "캠핑", "케이크", "파티",
    "소풍", "나들이", "드라이브", "산책", "등산", "운동", "수영", "자전거",
    "야경", "일출", "노을", "설경", "단풍", "벚꽃",
    "졸업식", "입학식", "돌잔치", "환갑", "칠순",
}
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


def plan_query(query: str) -> QueryPlan:
    normalized = query_translate.normalize_query(query)
    tokens = _tokens(normalized)
    date_from, date_to = query_translate.extract_date_range(normalized)
    person_terms = _matching_terms(tokens, normalized, PERSON_TERMS)
    place_terms = _matching_terms(tokens, normalized, PLACE_TERMS)
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


def _tokens(query: str) -> list[str]:
    return re.findall(r"[0-9A-Za-z가-힣_]+", query.casefold())


def _matching_terms(tokens: list[str], normalized: str, terms: set[str]) -> list[str]:
    lowered = normalized.casefold()
    hits = [term for term in terms if term in tokens or term in lowered]
    return sorted(set(hits), key=lambda item: (len(item), item))


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
