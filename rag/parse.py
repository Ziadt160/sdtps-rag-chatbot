"""Parse extracted page text into structured ServiceRecords.

Each page describes one Sharjah town-planning service with consistent Arabic
field labels. We detect those labels (fuzzily, to survive extraction noise like
the lam-alef reordering) and slice the values between them. Bulleted document
lines are collected globally because the form layout sometimes floats a bullet
above its label.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rapidfuzz import fuzz

from . import arabic
from .extract import extract_pages

# field -> canonical Arabic label (order matters only for readability)
LABELS: dict[str, str] = {
    "service_name": "اسم الخدمة",
    "description": "وصف الخدمة",
    "documents": "الوثائق المطلوبة",
    "fees": "الرسوم المطلوبة",
    "duration": "المدة المتوقعة",
    "phone": "رقم الهاتف",
    "hours": "ساعات العمل",
    "form": "نموذج التقديم",
    "department": "الإدارة",
}
_NORM_LABELS = {fieldname: arabic.normalize(lab) for fieldname, lab in LABELS.items()}
_LABEL_THRESHOLD = 82.0
_BULLETS = ("●", "•", "-", "*", "▪", "◦")


@dataclass
class ServiceRecord:
    service_id: str
    page: int
    service_name: str = ""
    description: str = ""
    documents: list[str] = field(default_factory=list)
    fees: str = ""
    duration: str = ""
    phone: str = ""
    hours: str = ""
    form: str = ""
    department: str = ""
    raw_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _match_label(line: str) -> tuple[str | None, str]:
    """If `line` starts with a known label, return (field, remaining_value)."""
    norm_tokens = arabic.normalize(line).split(" ")
    raw_tokens = line.split(" ")
    best: tuple[float, str, int] = (0.0, "", 0)
    for fieldname, norm_label in _NORM_LABELS.items():
        n = len(norm_label.split(" "))
        head = " ".join(norm_tokens[:n])
        score = fuzz.ratio(norm_label, head)
        if score > best[0]:
            best = (score, fieldname, n)
    score, fieldname, n = best
    if score >= _LABEL_THRESHOLD:
        return fieldname, " ".join(raw_tokens[n:]).strip()
    return None, ""


def parse_page(text: str, page_idx: int) -> ServiceRecord:
    rec = ServiceRecord(service_id=f"svc-{page_idx:02d}", page=page_idx, raw_text=text)
    buffers: dict[str, list[str]] = {f: [] for f in LABELS}
    current: str | None = None

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        stripped = line.lstrip("".join(_BULLETS) + " ").strip()
        if line[0] in _BULLETS:  # document bullet (position-independent)
            if stripped:
                rec.documents.append(stripped)
            continue
        fieldname, value = _match_label(line)
        if fieldname is not None:
            current = fieldname
            if value:
                buffers[fieldname].append(value)
            continue
        if current is not None:
            buffers[current].append(line)

    rec.service_name = " ".join(buffers["service_name"]).strip()
    rec.description = " ".join(buffers["description"]).strip()
    rec.fees = " ".join(buffers["fees"]).strip()
    rec.duration = " ".join(buffers["duration"]).strip()
    rec.phone = " ".join(buffers["phone"]).strip()
    rec.hours = " ".join(buffers["hours"]).strip()
    rec.form = " ".join(buffers["form"]).strip()
    rec.department = " ".join(buffers["department"]).strip()
    # fall back to the page title (first line) when the service-name label is absent
    if not rec.service_name:
        first = text.split("\n", 1)[0].strip()
        rec.service_name = first
    return rec


def parse_pdf(pdf_path: str | Path) -> list[ServiceRecord]:
    pages = extract_pages(pdf_path)
    return [parse_page(text, i + 1) for i, text in enumerate(pages)]


def save_records(records: list[ServiceRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")


def load_records(path: str | Path) -> list[ServiceRecord]:
    records: list[ServiceRecord] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(ServiceRecord(**json.loads(line)))
    return records


if __name__ == "__main__":
    from .config import PDF_PATH, SERVICES_PATH

    recs = parse_pdf(PDF_PATH)
    save_records(recs, SERVICES_PATH)
    print(f"parsed {len(recs)} services -> {SERVICES_PATH}")
    for r in recs[:3]:
        print("-" * 60)
        print("id:", r.service_id, "| name:", r.service_name)
        print("phone:", r.phone, "| hours:", r.hours, "| duration:", r.duration)
        print("docs:", len(r.documents), "| dept:", r.department)
