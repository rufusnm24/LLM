"""ATS scoring implementation with transparent metrics."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .keyword_extract import KeywordCandidate, SYNONYMS


ACTION_VERBS = {
    "accelerated",
    "achieved",
    "built",
    "coordinated",
    "delivered",
    "designed",
    "developed",
    "drove",
    "enabled",
    "improved",
    "implemented",
    "launched",
    "led",
    "managed",
    "optimized",
    "owned",
    "reduced",
    "scaled",
    "spearheaded",
    "streamlined",
}

REQUIRED_SECTIONS = {"experience", "education", "skills"}


@dataclass
class ScoreBreakdown:
    coverage: float
    section: float
    quality: float
    distribution: float
    total: float
    details: Dict[str, str]


class ATSScorer:
    """Compute ATS score across coverage, format, quality, and distribution."""

    def __init__(self, *, max_pages: int = 2) -> None:
        self.max_pages = max_pages

    def score(
        self,
        *,
        resume_text: str,
        bullet_texts: Sequence[str],
        keywords: Sequence[KeywordCandidate],
        sections_present: Iterable[str],
        page_estimate: int,
    ) -> ScoreBreakdown:
        coverage = self._coverage_score(resume_text, keywords)
        section = self._section_score(sections_present, page_estimate)
        quality = self._quality_score(bullet_texts)
        distribution = self._distribution_score(bullet_texts, keywords)
        total = round(coverage * 0.4 + section * 0.2 + quality * 0.2 + distribution * 0.2, 2)
        details = {
            "coverage": f"{coverage:.2f} / 100",
            "section": f"{section:.2f} / 100",
            "quality": f"{quality:.2f} / 100",
            "distribution": f"{distribution:.2f} / 100",
        }
        return ScoreBreakdown(
            coverage=coverage,
            section=section,
            quality=quality,
            distribution=distribution,
            total=total,
            details=details,
        )

    # Individual metrics ---------------------------------------------------------
    def _coverage_score(self, resume_text: str, keywords: Sequence[KeywordCandidate]) -> float:
        text = resume_text.lower()
        matched = 0
        for candidate in keywords:
            if candidate.token in text:
                matched += 1
                continue
            synonyms = SYNONYMS.get(candidate.token, set()) | set(candidate.synonyms)
            if any(syn in text for syn in synonyms):
                matched += 1
        if not keywords:
            return 100.0
        coverage_ratio = matched / len(keywords)
        return round(min(coverage_ratio, 1.0) * 100.0, 2)

    def _section_score(self, sections_present: Iterable[str], page_estimate: int) -> float:
        present = {section.lower() for section in sections_present}
        missing = REQUIRED_SECTIONS - present
        score = 100.0 - len(missing) * 20.0
        if page_estimate > self.max_pages:
            score -= 20.0
        score = max(score, 0.0)
        return round(score, 2)

    def _quality_score(self, bullet_texts: Sequence[str]) -> float:
        if not bullet_texts:
            return 50.0
        verb_hits = 0
        short_or_long_penalties = 0
        tense_inconsistencies = 0
        for bullet in bullet_texts:
            words = bullet.lower().split()
            if words and words[0] in ACTION_VERBS:
                verb_hits += 1
            if len(bullet) < 35 or len(bullet) > 220:
                short_or_long_penalties += 1
            if any(word.endswith("ed") for word in words) and any(word.endswith("ing") for word in words[:5]):
                tense_inconsistencies += 1
        verb_ratio = verb_hits / len(bullet_texts)
        score = 60.0 + verb_ratio * 30.0
        score -= short_or_long_penalties * 4.0
        score -= tense_inconsistencies * 4.0
        return round(max(score, 0.0), 2)

    def _distribution_score(self, bullet_texts: Sequence[str], keywords: Sequence[KeywordCandidate]) -> float:
        keyword_tokens = [candidate.token for candidate in keywords]
        if not keyword_tokens:
            return 100.0
        counts = Counter()
        per_bullet_counts: List[Counter] = []
        for bullet in bullet_texts:
            bullet_counter = Counter()
            lowered = bullet.lower()
            for keyword in keyword_tokens:
                occurrences = lowered.count(keyword)
                if occurrences:
                    bullet_counter[keyword] += occurrences
                    counts[keyword] += occurrences
            per_bullet_counts.append(bullet_counter)

        penalties = 0
        for keyword, count in counts.items():
            if count > 2:
                penalties += (count - 2) * 5
        for bullet_counter in per_bullet_counts:
            for keyword, count in bullet_counter.items():
                if count > 1:
                    penalties += (count - 1) * 5
        unique_coverage = sum(1 for keyword in keyword_tokens if counts[keyword] > 0)
        diversity_ratio = unique_coverage / len(keyword_tokens)
        base_score = 70.0 + diversity_ratio * 30.0
        score = max(base_score - penalties, 0.0)
        return round(min(score, 100.0), 2)


def summarise_keywords(keywords: Sequence[KeywordCandidate], bullet_texts: Sequence[str]) -> Dict[str, Dict[str, int]]:
    """Return keyword usage stats for reporting."""

    summary: Dict[str, Dict[str, int]] = {}
    for candidate in keywords:
        summary[candidate.token] = {"global": 0}
    for bullet in bullet_texts:
        lowered = bullet.lower()
        for candidate in keywords:
            count = lowered.count(candidate.token)
            if count:
                summary[candidate.token]["global"] += count
    return summary
