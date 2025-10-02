"""Keyword extraction utilities for ATS alignment."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "will",
    "your",
    "have",
    "work",
    "team",
    "role",
    "skills",
    "required",
    "responsibilities",
    "experience",
}

# Lightweight synonym map intentionally curated for data / product / engineering roles.
SYNONYMS = {
    "sql": {"postgres", "mysql", "redshift"},
    "python": {"pandas", "numpy"},
    "dashboard": {"tableau", "looker", "powerbi"},
    "analysis": {"analytics", "analytical"},
    "manage": {"lead", "led", "oversaw"},
    "project": {"program", "initiative"},
    "communication": {"stakeholder", "presentation"},
    "cloud": {"aws", "gcp", "azure"},
    "automation": {"workflow", "orchestration"},
    "testing": {"qa", "quality"},
}


@dataclass
class KeywordCandidate:
    token: str
    score: float
    synonyms: List[str]


TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9+.#-]{2,}")


def normalise(text: str) -> List[str]:
    tokens = [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]
    filtered = [tok for tok in tokens if tok not in STOPWORDS]
    return filtered


def _ngram(tokens: Sequence[str], n: int) -> Iterable[str]:
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


def extract_keywords(text: str, *, max_keywords: int = 20) -> List[KeywordCandidate]:
    """Return ranked keyword candidates.

    The extractor balances unigrams and bigrams then appends synonym suggestions to
    encourage distribution-aware scoring.
    """

    tokens = normalise(text)
    if not tokens:
        return []

    unigram_counts = Counter(tokens)
    bigram_tokens = list(_ngram(tokens, 2))
    bigram_counts = Counter(bigram_tokens)
    first_seen: dict[str, int] = {}
    for idx, token in enumerate(tokens):
        first_seen.setdefault(token, idx)
    offset = len(tokens)
    for idx, token in enumerate(bigram_tokens):
        first_seen.setdefault(token, offset + idx)

    candidates: List[KeywordCandidate] = []

    for token, count in unigram_counts.most_common():
        synonyms = sorted(SYNONYMS.get(token, []))
        candidates.append(KeywordCandidate(token=token, score=float(count), synonyms=synonyms))

    # Promote meaningful bigrams by boosting their score.
    for token, count in bigram_counts.most_common():
        if any(part in STOPWORDS for part in token.split()):
            continue
        synonyms = []
        candidates.append(KeywordCandidate(token=token, score=float(count) + 0.5, synonyms=synonyms))

    # Deduplicate while preserving order (higher score first).
    seen: set[str] = set()
    deduped: List[KeywordCandidate] = []
    for candidate in sorted(candidates, key=lambda cand: (-cand.score, first_seen.get(cand.token, 0))):
        if candidate.token in seen:
            continue
        seen.add(candidate.token)
        deduped.append(candidate)
        if len(deduped) >= max_keywords:
            break

    return deduped
