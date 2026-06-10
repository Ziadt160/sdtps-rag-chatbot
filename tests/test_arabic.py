from rag import arabic


def test_normalize_unifies_alef_and_strips_diacritics():
    assert arabic.normalize("أحمد") == "احمد"
    assert arabic.normalize("إدارة") == "اداره"        # hamza-alef -> alef, ta-marbuta -> ha
    assert arabic.normalize("مَدْرَسَة") == "مدرسه"      # diacritics removed
    assert arabic.normalize("كــتــاب") == "كتاب"       # tatweel removed


def test_arabic_indic_digits_become_ascii():
    assert arabic.normalize("٢٠٢٦") == "2026"


def test_stem_strips_definite_article_and_suffixes():
    assert arabic.stem("الطابقي") == "طابق"
    assert arabic.stem("طابقي") == "طابق"
    assert arabic.stem("موقعا") == "موقع"
    assert arabic.stem("في") == "في"                    # too short to over-stem


def test_tokenize_keeps_digits():
    toks = arabic.tokenize("ساعات العمل 8:00")
    assert "8" in toks
    assert all(toks)
