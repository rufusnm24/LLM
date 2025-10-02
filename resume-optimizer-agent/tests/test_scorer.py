from ats.keyword_extract import KeywordCandidate
from ats.scorer import ATSScorer


def test_scorer_increases_with_more_keywords():
    scorer = ATSScorer()
    resume_text = "Led python analytics projects with tableau dashboards."
    bullet_texts = ["Led python analytics projects with tableau dashboards."]
    keywords = [
        KeywordCandidate(token="python", score=1.0, synonyms=[]),
        KeywordCandidate(token="analytics", score=1.0, synonyms=[]),
        KeywordCandidate(token="dashboard", score=1.0, synonyms=["tableau"]),
    ]
    sections = ["experience", "education", "skills"]
    before = scorer.score(
        resume_text="Led projects.",
        bullet_texts=["Led projects."],
        keywords=keywords,
        sections_present=sections,
        page_estimate=1,
    )
    after = scorer.score(
        resume_text=resume_text,
        bullet_texts=bullet_texts,
        keywords=keywords,
        sections_present=sections,
        page_estimate=1,
    )
    assert after.total > before.total
