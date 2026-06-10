"""RTL-correct PDF text extraction using PyMuPDF word boxes.

`UnstructuredPDFLoader` scrambles Arabic. Instead we read word bounding boxes
and reconstruct reading order ourselves: group words into lines by their y
position, order words right-to-left within a line, and fix embedded
left-to-right runs (phone numbers, times, "A2", ...) so they read correctly.
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from . import arabic

# Words whose y-centers fall within this fraction of line height are one line.
_LINE_TOL = 3.0


def _reconstruct_line(words: list[tuple]) -> str:
    """words: (x0, y0, x1, y1, text). Return logical reading-order line."""
    rtl = sorted(words, key=lambda w: -w[0])  # right-to-left
    toks = [w[4] for w in rtl]
    out: list[str] = []
    i, n = 0, len(toks)
    while i < n:
        if arabic.has_arabic(toks[i]):
            out.append(toks[i])
            i += 1
        else:
            # maximal non-Arabic (LTR/neutral) run -> flip back to left-to-right
            j = i
            while j < n and not arabic.has_arabic(toks[j]):
                j += 1
            out.extend(reversed(toks[i:j]))
            i = j
    return " ".join(out)


def extract_page(page: fitz.Page) -> str:
    words = page.get_text("words")  # (x0,y0,x1,y1,word,block,line,word_no)
    if not words:
        return ""
    buckets: dict[int, list[tuple]] = {}
    for w in words:
        key = round(w[1] / _LINE_TOL)
        buckets.setdefault(key, []).append(w)
    lines = [_reconstruct_line(buckets[k]) for k in sorted(buckets)]
    return "\n".join(line for line in lines if line.strip())


def extract_pages(pdf_path: str | Path) -> list[str]:
    """Return one logical-order text string per page."""
    doc = fitz.open(str(pdf_path))
    try:
        return [extract_page(page) for page in doc]
    finally:
        doc.close()


if __name__ == "__main__":  # quick manual check
    import sys

    from .config import PDF_PATH

    path = sys.argv[1] if len(sys.argv) > 1 else PDF_PATH
    pages = extract_pages(path)
    print(f"pages: {len(pages)}")
    print("=== page 1 ===")
    print(pages[0])
