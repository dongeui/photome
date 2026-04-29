"""Query expansion helpers for Korean-first CLIP search."""
from __future__ import annotations

import os
import re
import time as _time
from datetime import date, timedelta
from functools import lru_cache

# ---------------------------------------------------------------------------
# Core lexicon: Korean phrase → English CLIP prompts
# ---------------------------------------------------------------------------
LEXICON = {
    # People
    "자전거": "bicycle bike cycling",
    "자동차": "car automobile vehicle",
    "차": "car vehicle",
    "사람": "person people human",
    "남자": "man male person",
    "남성": "man male person",
    "여자": "woman female girl lady person portrait",
    "여성": "woman female girl lady person portrait",
    "얼굴": "face portrait person close-up selfie",
    "셀카": "selfie portrait face person",
    "아기": "baby infant toddler child kid newborn",
    "애기": "baby infant toddler child kid newborn",
    "아이": "child kid baby toddler",
    "어린이": "child kid toddler",
    "엄마": "mother mom woman family portrait",
    "아빠": "father dad man family portrait",
    "할머니": "grandmother elderly woman family portrait",
    "할아버지": "grandfather elderly man family portrait",
    "가족": "family group portrait people",
    "친구": "friends group people smiling",
    "커플": "couple love romantic portrait",
    # Animals / nature
    "강아지": "dog puppy cute animal",
    "고양이": "cat kitten cute animal",
    "꽃": "flower floral bloom garden",
    "나무": "tree forest nature green",
    "하늘": "sky clouds blue",
    "바다": "sea ocean beach water waves",
    "산": "mountain hiking outdoor landscape",
    "강": "river stream water nature",
    "공원": "park outdoor green nature",
    "해변": "beach sand sea ocean",
    "숲": "forest trees nature green",
    # Food & drinks
    "음식": "food meal dish restaurant",
    "밥": "rice meal korean food",
    "케이크": "cake dessert birthday celebration sweet",
    "커피": "coffee cafe drink",
    "초밥": "sushi japanese food meal",
    "치킨": "chicken fried food meal",
    "피자": "pizza food meal",
    "술": "alcohol drink beer wine glass",
    # Life events
    "생일": "birthday cake celebration party happy",
    "생일파티": "birthday party celebration cake candles",
    "결혼식": "wedding ceremony couple formal",
    "졸업": "graduation ceremony diploma academic",
    "졸업식": "graduation ceremony diploma academic",
    "파티": "party celebration people happy",
    "여행": "travel trip landscape outdoor",
    "여행사진": "travel trip landscape outdoor",
    "휴가": "vacation travel leisure outdoor",
    "운동": "exercise sports outdoor fitness",
    "등산": "hiking mountain outdoor nature trail",
    "수영": "swimming pool water sport",
    "캠핑": "camping outdoor tent nature",
    # Places
    "카페": "cafe coffee shop interior cozy",
    "식당": "restaurant food dining",
    "집": "home house indoor interior",
    "학교": "school classroom building",
    "공항": "airport travel terminal",
    "해외": "abroad overseas travel landmark",
    "서울": "Seoul Korea cityscape urban",
    "제주": "Jeju island beach Korea outdoor",
    # Documents & screens
    "영수증": "receipt document text paper purchase",
    "문서": "document paper page text",
    "화면": "screen screenshot app interface",
    "스크린샷": "screenshot screen capture app",
    "오류": "error failure warning message screen",
    "전송": "send transfer submit button",
    "실패": "failure failed error warning",
    "동의": "agree consent accept button",
    "확인": "confirm ok check button",
    "취소": "cancel button",
    "버튼": "button interface screen",
    "대화": "chat conversation message screen",
    "메시지": "message chat text screen",
    # Holidays
    "크리스마스": "Christmas holiday decoration tree",
    "새해": "new year celebration fireworks",
    "추석": "chuseok Korean holiday family traditional",
    "설날": "seollal Korean new year traditional",
    # Seasons (used for CLIP expansion, date extraction handled separately)
    "봄": "spring flowers cherry blossom",
    "여름": "summer beach outdoor sunny hot",
    "가을": "autumn fall foliage orange leaves",
    "겨울": "winter snow cold ice",
    # Mood / style
    "귀여운": "cute adorable",
    "웃는": "smiling happy laughing",
    "잠자는": "sleeping resting",
    "야경": "night city lights cityscape",
    "일출": "sunrise dawn morning",
    "일몰": "sunset dusk evening",
    # Family relations
    "형": "brother male family portrait",
    "오빠": "brother male family portrait",
    "누나": "sister female family portrait",
    "언니": "sister female family portrait",
    "동생": "sibling family portrait",
    "남동생": "younger brother male family portrait",
    "여동생": "younger sister female family portrait",
    "조카": "nephew niece child family portrait",
    "남편": "husband couple family portrait",
    "아내": "wife couple family portrait",
    # Additional places
    "강남": "Gangnam Seoul Korea urban cityscape",
    "홍대": "Hongdae Seoul Korea street",
    "명동": "Myeongdong Seoul Korea street shopping",
    "경복궁": "Gyeongbokgung palace Korea traditional",
    "인사동": "Insadong Seoul Korea traditional street",
    "이태원": "Itaewon Seoul Korea street night",
    "한강": "Han River Seoul Korea park outdoor",
    "속초": "Sokcho Korea beach ocean",
    "강릉": "Gangneung Korea beach coffee",
    "경주": "Gyeongju Korea historic cultural",
    "전주": "Jeonju Korea traditional hanok",
    "해수욕장": "beach swimming sea ocean summer",
    "놀이공원": "amusement park ride fun",
    "동물원": "zoo animal outdoor",
    # Additional moods / activities
    "소풍": "picnic outdoor park nature",
    "나들이": "outing outdoor nature",
    "드라이브": "drive road trip car scenic",
    "산책": "walk stroll outdoor park",
    "조깅": "jogging running outdoor fitness",
    "독서": "reading book indoor",
    "공연": "performance concert stage",
    "전시": "exhibition gallery indoor art",
}

# ---------------------------------------------------------------------------
# Typo / informal form corrections
# ---------------------------------------------------------------------------
TYPO_CORRECTIONS = {
    "어르굴": "얼굴",
    "얼구": "얼굴",
    "얼굴ㄹ": "얼굴",
    "얼굴사진": "얼굴 사진",
    "여ㅈ": "여자",
    "여잔": "여자",
    "아긔": "아기",
    "아가": "아기",
    "애긔": "아기",
    "베이비": "아기",
    "멍멍이": "강아지",
    "냥이": "고양이",
    "야옹이": "고양이",
}

# ---------------------------------------------------------------------------
# Filler words to strip from Korean natural language queries
# ---------------------------------------------------------------------------
_KO_FILLER = re.compile(
    r"(?:찍은|찍었던|찍힌|촬영한|촬영된|에서|에서의|에서 찍은|에서 찍|에서 본|에서 본)\s*사진|"
    r"찍은\s*사진|찍은\s*거|찍은\s*것|사진\s*좀|의\s*사진|이\s*사진|인\s*사진",
    re.UNICODE,
)

# Korean particles / endings to strip token-by-token
# Order matters: longer suffixes first to avoid partial matches
_KO_PARTICLES = (
    "에서의", "으로의", "로부터", "에서", "으로", "에게", "한테", "부터",
    "까지", "마다", "이나", "이랑", "와", "과", "이랑", "하고",
    "에서", "에서", "에게", "에서", "에서", "에서",
    "이라", "이라는", "라는", "이는", "이고", "이며",
    "에서", "에서", "에서", "에서", "에서",
    "에서", "에서",
    "에서", "에의", "에의",
    "의", "을", "를", "이", "가", "은", "는",
    "에", "로", "와", "과",
)
# Build a unique ordered tuple (de-dup while preserving order)
_seen: set[str] = set()
_KO_PARTICLES_UNIQUE: tuple[str, ...] = tuple(
    p for p in (
        "에서의", "으로의", "로부터", "에서", "으로", "에게", "한테",
        "부터", "까지", "마다", "이나", "이랑", "하고",
        "이라는", "라는", "이라", "이고", "이며",
        "에의", "의", "을", "를", "이", "가", "은", "는", "에", "로", "와", "과",
    )
    if p not in _seen and not _seen.add(p)  # type: ignore[func-returns-value]
)
del _seen

# ---------------------------------------------------------------------------
# English expansions for common photo terms
# ---------------------------------------------------------------------------
ENGLISH_EXPANSIONS = {
    "face": "face portrait person close-up selfie",
    "faces": "face portrait people person close-up selfie",
    "portrait": "portrait face person",
    "selfie": "selfie portrait face person",
    "woman": "woman female girl lady person portrait",
    "women": "woman female girl lady person portrait",
    "female": "woman female girl lady person portrait",
    "girl": "girl woman female person portrait",
    "baby": "baby infant toddler child kid newborn",
    "infant": "baby infant toddler child newborn",
    "toddler": "baby toddler child kid",
    "child": "child kid baby toddler",
    "kid": "child kid baby toddler",
    "dog": "dog puppy animal",
    "cat": "cat kitten animal",
    "food": "food meal dish restaurant",
    "beach": "beach sea ocean sand waves",
    "mountain": "mountain hiking outdoor landscape",
    "travel": "travel trip landmark outdoor",
    "party": "party celebration people happy",
    "birthday": "birthday cake celebration party",
    "wedding": "wedding ceremony couple formal",
    "graduation": "graduation ceremony diploma academic",
    "family": "family group portrait people",
    "friends": "friends group people smiling",
    "sunset": "sunset dusk evening orange sky",
    "sunrise": "sunrise dawn morning sky",
    "snow": "snow winter cold white",
    "rain": "rain wet weather outdoor",
    "flower": "flower floral bloom garden",
    "sky": "sky clouds blue outdoor",
}


def expand_for_clip(query: str) -> list[str]:
    """Return original query plus optional English variants for CLIP."""
    cleaned = normalize_query(query)
    if not cleaned:
        return []

    # Strip filler phrases so CLIP gets the semantic core
    semantic_core = _KO_FILLER.sub("", cleaned).strip()
    if not semantic_core:
        semantic_core = cleaned

    variants = [semantic_core]
    translated = translate_to_english(semantic_core)
    if translated and translated.casefold() != semantic_core.casefold():
        variants.append(translated)
    english = _expand_english_terms(semantic_core)
    if english:
        variants.append(english)
    # Include original cleaned query too if different from semantic_core
    if cleaned != semantic_core:
        variants.append(cleaned)
    return _dedupe(variants)


def extract_date_range(query: str) -> tuple[date | None, date | None]:
    """Extract an implicit date range from natural language Korean/English queries.

    Returns (date_from, date_to) or (None, None) when no time expression found.
    Recognized patterns:
    - 작년/올해/재작년, N년전
    - 이번달/지난달, N달전
    - 이번주/지난주/이번주말/지난주말, N주전
    - 봄/여름/가을/겨울 + year modifier
    - 어제/오늘
    - 1월~12월 specific month
    - 설날/추석 approximate window
    """
    today = date.today()
    year = today.year
    lowered = query.casefold().strip()

    # ── absolute: 오늘 / 어제 ──
    if "오늘" in lowered:
        return today, today
    if "어제" in lowered:
        yesterday = today - timedelta(days=1)
        return yesterday, yesterday

    # ── N년전 ──
    m = re.search(r"([1-9]\d?)년\s*전", lowered)
    if m:
        n = int(m.group(1))
        y = year - n
        return date(y, 1, 1), date(y, 12, 31)

    # ── Explicit year mentions: "2023년", "23년" ──
    explicit = re.search(r"(20\d{2}|[2-9]\d)년", query)
    ref_year: int | None = None
    if explicit:
        raw = explicit.group(1)
        ref_year = int(raw) if len(raw) == 4 else 2000 + int(raw)

    # Year modifiers
    if "재작년" in lowered:
        ref_year = year - 2
    elif "작년" in lowered or "지난해" in lowered:
        ref_year = year - 1
    elif "올해" in lowered or "이번해" in lowered:
        ref_year = year

    # ── N달전 ──
    m = re.search(r"([1-9]\d?)달\s*전", lowered)
    if m:
        n = int(m.group(1))
        target = today.replace(day=1)
        for _ in range(n):
            target = (target - timedelta(days=1)).replace(day=1)
        last_day = (target.replace(month=target.month % 12 + 1, day=1) - timedelta(days=1)) if target.month < 12 else date(target.year, 12, 31)
        return target, last_day

    # ── N주전 ──
    m = re.search(r"([1-9]\d?)주\s*전", lowered)
    if m:
        n = int(m.group(1))
        week_start = today - timedelta(days=today.weekday() + 7 * n)
        week_end = week_start + timedelta(days=6)
        return week_start, min(week_end, today)

    # ── Month modifiers ──
    if "지난달" in lowered or "저번달" in lowered:
        first_this = today.replace(day=1)
        last_month_end = first_this - timedelta(days=1)
        return last_month_end.replace(day=1), last_month_end
    if "이번달" in lowered or "이번 달" in lowered:
        return today.replace(day=1), today

    # ── Week modifiers ──
    if "이번주말" in lowered or "이번 주말" in lowered:
        sat = today + timedelta(days=(5 - today.weekday()) % 7)
        sun = sat + timedelta(days=1)
        return sat, sun
    if "지난주말" in lowered or "저번주말" in lowered:
        last_sat = today - timedelta(days=today.weekday() + 2)
        last_sun = last_sat + timedelta(days=1)
        return last_sat, last_sun
    if "지난주" in lowered or "저번주" in lowered:
        this_monday = today - timedelta(days=today.weekday())
        last_monday = this_monday - timedelta(weeks=1)
        last_sunday = this_monday - timedelta(days=1)
        return last_monday, last_sunday
    if "이번주" in lowered or "이번 주" in lowered:
        start = today - timedelta(days=today.weekday())
        return start, today

    # ── Specific month: "3월", "12월" ──
    m = re.search(r"(1[0-2]|[1-9])월", lowered)
    if m:
        mo = int(m.group(1))
        y = ref_year or year
        import calendar
        last_day_num = calendar.monthrange(y, mo)[1]
        return date(y, mo, 1), date(y, mo, last_day_num)

    # ── Korean holidays (approximate windows) ──
    # 설날: late Jan ~ late Feb (lunar new year, varies)
    if "설날" in lowered or "설연휴" in lowered:
        y = ref_year or year
        return date(y, 1, 15), date(y, 2, 28)
    # 추석: mid Sep ~ mid Oct (chuseok, varies)
    if "추석" in lowered or "추석연휴" in lowered:
        y = ref_year or year
        return date(y, 9, 10), date(y, 10, 15)
    # 크리스마스: Dec 20~31
    if "크리스마스" in lowered or "성탄절" in lowered:
        y = ref_year or year
        return date(y, 12, 20), date(y, 12, 31)

    # ── Season mapping ──
    season_ranges: dict[str, tuple[int, int, int, int]] = {
        "봄": (3, 1, 5, 31),
        "spring": (3, 1, 5, 31),
        "여름": (6, 1, 8, 31),
        "summer": (6, 1, 8, 31),
        "가을": (9, 1, 11, 30),
        "fall": (9, 1, 11, 30),
        "autumn": (9, 1, 11, 30),
        "겨울": (12, 1, 2, 28),
        "winter": (12, 1, 2, 28),
    }
    for season_key, (sm, sd, em, ed) in season_ranges.items():
        if season_key in lowered:
            y = ref_year or year
            if sm > em:  # winter wraps year
                date_from = date(y - 1 if ref_year is None else y, sm, sd)
                date_to = date(y, em, ed)
            else:
                date_from = date(y, sm, sd)
                date_to = date(y, em, min(ed, 30 if em in (4, 6, 9, 11) else 31))
            return date_from, date_to

    if ref_year is not None:
        return date(ref_year, 1, 1), date(ref_year, 12, 31)

    return None, None


def normalize_query(query: str) -> str:
    cleaned = re.sub(r"\s+", " ", query.strip())
    if not cleaned:
        return ""
    for typo, correction in TYPO_CORRECTIONS.items():
        cleaned = cleaned.replace(typo, correction)
    cleaned = strip_korean_particles(cleaned)
    return cleaned.strip()


def strip_korean_particles(text: str) -> str:
    """Remove common Korean postpositional particles from each token.

    Only strips if the token contains Hangul and at least 2 characters
    remain after stripping, to avoid accidentally removing word endings
    that are part of the stem (e.g. "동의" should not become "동").
    """
    tokens = text.split()
    result = []
    for token in tokens:
        if _has_hangul(token):
            for particle in _KO_PARTICLES_UNIQUE:
                candidate = token[: -len(particle)]
                if token.endswith(particle) and len(candidate) >= 2:
                    token = candidate
                    break
        result.append(token)
    return " ".join(result)


def _has_hangul(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))


def translate_to_english(query: str) -> str | None:
    """Translate Korean query to English when possible.

    The default path is a small deterministic lexicon, so the app stays fast and
    offline even before optional translation models are installed. Set
    PHOTOME_TRANSLATOR=opus to use a local HuggingFace MarianMT model when the
    dependencies and model cache are available.
    """
    if os.environ.get("PHOTOME_TRANSLATOR", "lexicon").casefold() == "opus":
        translated = _translate_with_opus(query)
        if translated:
            return translated
    return _translate_with_lexicon(query)


def _translate_with_lexicon(query: str) -> str | None:
    hits = [english for korean, english in LEXICON.items() if korean in query]
    if not hits:
        return None
    return " ".join(hits)


def _expand_english_terms(query: str) -> str | None:
    lowered = query.casefold()
    hits = [
        expanded
        for token, expanded in ENGLISH_EXPANSIONS.items()
        if re.search(rf"\b{re.escape(token)}\b", lowered)
    ]
    if not hits:
        return None
    return " ".join(hits)


@lru_cache(maxsize=1)
def _opus_pipeline():
    from transformers import MarianMTModel, MarianTokenizer

    model_name = os.environ.get("PHOTOME_TRANSLATOR_MODEL", "Helsinki-NLP/opus-mt-ko-en")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _translate_with_opus(query: str) -> str | None:
    if not _has_hangul(query):
        return None
    try:
        import torch

        tokenizer, model = _opus_pipeline()
        tokens = tokenizer([query], return_tensors="pt", padding=True)
        with torch.no_grad():
            output = model.generate(**tokens, max_new_tokens=48)
        translated = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return translated or None
    except Exception:
        return None


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
