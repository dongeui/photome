"""Query expansion helpers for Korean-first CLIP search."""
from __future__ import annotations

import os
import re
from functools import lru_cache

LEXICON = {
    "자전거": "bicycle bike cycling",
    "자동차": "car automobile vehicle",
    "차": "car vehicle",
    "사람": "person people human",
    "남자": "man male person",
    "여자": "woman female girl lady person portrait",
    "여성": "woman female girl lady person portrait",
    "얼굴": "face portrait person close-up selfie",
    "셀카": "selfie portrait face person",
    "아기": "baby infant toddler child kid newborn",
    "애기": "baby infant toddler child kid newborn",
    "아이": "child kid baby toddler",
    "어린이": "child kid toddler",
    "강아지": "dog puppy",
    "고양이": "cat",
    "음식": "food meal",
    "영수증": "receipt document",
    "문서": "document paper page",
    "화면": "screen screenshot",
    "오류": "error failure warning",
    "전송": "send transfer submit",
    "실패": "failure failed error",
    "동의": "agree consent accept",
    "확인": "confirm ok check",
    "취소": "cancel",
    "버튼": "button",
    "대화": "chat conversation message",
    "메시지": "message chat",
}

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
}

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
}


def expand_for_clip(query: str) -> list[str]:
    """Return original query plus optional English variants for CLIP."""
    cleaned = normalize_query(query)
    if not cleaned:
        return []

    variants = [cleaned]
    translated = translate_to_english(cleaned)
    if translated and translated.casefold() != cleaned.casefold():
        variants.append(translated)
    english = _expand_english_terms(cleaned)
    if english:
        variants.append(english)
    return _dedupe(variants)


def normalize_query(query: str) -> str:
    cleaned = re.sub(r"\s+", " ", query.strip())
    if not cleaned:
        return ""
    for typo, correction in TYPO_CORRECTIONS.items():
        cleaned = cleaned.replace(typo, correction)
    return cleaned


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


def _has_hangul(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))


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
