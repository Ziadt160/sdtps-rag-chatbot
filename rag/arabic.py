"""Arabic text normalization and tokenization.

Used to make lexical matching (BM25), service-name fuzzy matching, and label
detection robust to diacritics, tatweel, and the alef/ya/ta-marbuta variants
that plague Arabic PDF extraction.
"""
from __future__ import annotations

import re

# Combining marks: harakat/tanwin/shadda/sukun (064B-0652), superscript alef (0670)
_TASHKEEL = re.compile(r"[ً-ْٰـ]")  # incl. tatweel 0640
_NON_TOKEN = re.compile(r"[^؀-ۿ0-9A-Za-z]+")
_ARABIC = re.compile(r"[؀-ۿ]")

_TRANSLATE = {
    ord("أ"): "ا", ord("إ"): "ا", ord("آ"): "ا", ord("ٱ"): "ا",
    ord("ى"): "ي", ord("ئ"): "ي",
    ord("ة"): "ه",
    ord("ؤ"): "و",
    # Arabic-Indic digits -> ASCII
    ord("٠"): "0", ord("١"): "1", ord("٢"): "2", ord("٣"): "3", ord("٤"): "4",
    ord("٥"): "5", ord("٦"): "6", ord("٧"): "7", ord("٨"): "8", ord("٩"): "9",
}


def normalize(text: str) -> str:
    """Aggressive normalization for matching/search (not for display)."""
    if not text:
        return ""
    text = _TASHKEEL.sub("", text)
    text = text.translate(_TRANSLATE)
    text = text.replace("ء", "")  # drop bare hamza (extraction noise)
    text = _NON_TOKEN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# Light stemming for lexical (BM25) matching: strip the definite article and a
# few frequent suffixes so morphological variants collide
# (موقعاً≈موقع، الطابقي≈طابقي). Conservative length guards avoid over-stemming.
_SUFFIXES = ("ات", "ون", "ين", "ها", "ة", "ه", "ي", "ا")


def stem(token: str) -> str:
    if token.startswith("ال") and len(token) > 4:
        token = token[2:]
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    return token


def tokenize(text: str) -> list[str]:
    """Normalized + lightly-stemmed token list for BM25."""
    norm = normalize(text)
    return [stem(t) for t in norm.split(" ") if t]


def has_arabic(text: str) -> bool:
    return bool(_ARABIC.search(text))
