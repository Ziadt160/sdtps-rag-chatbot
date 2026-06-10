from rag.chunk import record_to_chunk
from rag.parse import ServiceRecord


def test_record_to_chunk_carries_metadata_and_templated_text():
    rec = ServiceRecord(
        service_id="svc-01",
        page=1,
        service_name="خدمة تجريبية",
        description="وصف الخدمة",
        documents=["وثيقة أولى"],
        fees="مجاناً",
        duration="يوم عمل",
        phone="06-0000000",
        hours="8:00 am - 2:30 pm",
        department="إدارة",
    )
    chunk = record_to_chunk(rec)
    assert chunk.id == "svc-01"
    assert chunk.metadata["service_name"] == "خدمة تجريبية"
    assert chunk.metadata["page"] == 1
    assert chunk.metadata["n_documents"] == 1
    assert "اسم الخدمة: خدمة تجريبية" in chunk.text
    assert "وثيقة أولى" in chunk.text
