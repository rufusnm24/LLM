from ats.keyword_extract import KeywordCandidate
from latex.rewriter import optimize_resume


def test_optimize_resume_replaces_synonyms_without_large_changes():
    tex = """\\begin{itemize}\n  \\item Led analytics initiatives with pandas tooling.\n  \\item Drove reporting automation for stakeholders.\n\\end{itemize}"""
    keywords = [
        KeywordCandidate(token="python", score=2.0, synonyms=["pandas"]),
        KeywordCandidate(token="automation", score=1.0, synonyms=[]),
    ]
    result = optimize_resume(tex, keywords, strict=True)
    assert "python" in result.optimized_tex.lower()
    # Strict mode ensures small diffs.
    for original, updated in zip(tex.splitlines(), result.optimized_tex.splitlines()):
        if original.strip().startswith("\\item"):
            assert abs(len(original) - len(updated)) <= 10
