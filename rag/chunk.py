"""Custom, table-aware chunking.

The PDF is a table/form of services, so we deliberately keep **one chunk per
service** — never splitting a record mid-row. Each chunk carries the service's
fields as metadata so retrieval can filter/boost by service name. `text` is a
clean templated passage (embedded + shown to the LLM); `search_text` is the full
raw page, used for BM25 lexical recall.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .parse import ServiceRecord


@dataclass
class Chunk:
    id: str
    text: str
    search_text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _template(rec: ServiceRecord) -> str:
    parts = [f"اسم الخدمة: {rec.service_name}"]
    if rec.description:
        parts.append(f"وصف الخدمة: {rec.description}")
    if rec.documents:
        docs = "\n".join(f"- {d}" for d in rec.documents)
        parts.append(f"الوثائق المطلوبة:\n{docs}")
    if rec.fees:
        parts.append(f"الرسوم المطلوبة: {rec.fees}")
    if rec.duration:
        parts.append(f"المدة المتوقعة لإنجاز المعاملة: {rec.duration}")
    if rec.phone:
        parts.append(f"رقم الهاتف: {rec.phone}")
    if rec.hours:
        parts.append(f"ساعات العمل: {rec.hours}")
    return "\n".join(parts)


def record_to_chunk(rec: ServiceRecord) -> Chunk:
    return Chunk(
        id=rec.service_id,
        text=_template(rec),
        search_text=rec.raw_text or _template(rec),
        metadata={
            "service_id": rec.service_id,
            "service_name": rec.service_name,
            "department": rec.department,
            "phone": rec.phone,
            "hours": rec.hours,
            "fees": rec.fees,
            "duration": rec.duration,
            "page": rec.page,
            "n_documents": len(rec.documents),
        },
    )


def records_to_chunks(records: list[ServiceRecord]) -> list[Chunk]:
    return [record_to_chunk(r) for r in records]


def save_chunks(chunks: list[Chunk], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(path: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunks.append(Chunk(**json.loads(line)))
    return chunks
