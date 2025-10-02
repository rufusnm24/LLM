from ats.keyword_extract import extract_keywords


def test_extract_keywords_prioritises_relevant_terms():
    text = "Python analytics automation python dashboard analytics"
    keywords = extract_keywords(text, max_keywords=5)
    tokens = [keyword.token for keyword in keywords]
    assert tokens[0] == "python"
    assert "analytics" in tokens
    assert len(tokens) <= 5
