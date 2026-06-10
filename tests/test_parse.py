from rag.parse import parse_page

PAGE = """طلب بيان التخطيط العمراني
اسم الخدمة طلب بيان التخطيط العمراني
وصف الخدمة خدمة تتيح الحصول على بيان التخطيط
● صورة من بطاقة الهوية
● صورة من خارطة الأرض
الوثائق المطلوبة
الرسوم المطلوبة 100 درهم
المدة المتوقعة 3 أيام عمل
رقم الهاتف 06-5289999
ساعات العمل 8:00 am - 2:30 pm
"""


def test_parse_page_extracts_structured_fields():
    rec = parse_page(PAGE, 7)
    assert rec.service_id == "svc-07"
    assert rec.page == 7
    assert "بيان التخطيط العمراني" in rec.service_name
    assert rec.phone == "06-5289999"
    assert "100" in rec.fees
    assert len(rec.documents) == 2
    assert any("الهوية" in d for d in rec.documents)
