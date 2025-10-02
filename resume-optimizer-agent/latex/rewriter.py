"""LaTeX resume optimizer that respects formatting constraints."""
from __future__ import annotations

import difflib
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ats.keyword_extract import SYNONYMS, KeywordCandidate
from .ast_parser import parse_document


@dataclass
class RewriteResult:
    optimized_tex: str
    keyword_map: Dict[str, Dict[str, int]]
    diff: str


def _replace_synonym(text: str, synonym: str, keyword: str) -> Tuple[str, bool]:
    lowered = text.lower()
    idx = lowered.find(synonym)
    if idx == -1:
        return text, False
    # Ensure we are not touching LaTeX macros.
    if "\\" in text[max(0, idx - 2) : idx + len(synonym) + 2]:
        return text, False
    new_text = text[:idx] + keyword + text[idx + len(synonym) :]
    return new_text, True


def optimize_resume(tex_content: str, keywords: Sequence[KeywordCandidate], *, strict: bool = False) -> RewriteResult:
    document = parse_document(tex_content)
    usage = Counter()

    before_counts: Dict[str, int] = {candidate.token: tex_content.lower().count(candidate.token) for candidate in keywords}

    for idx, bullet in enumerate(document.bullets):
        original = bullet.content
        updated = original
        per_bullet = Counter({candidate.token: updated.lower().count(candidate.token) for candidate in keywords})
        for candidate in keywords:
            if usage[candidate.token] >= 2:
                continue
            if per_bullet[candidate.token] >= 1:
                usage[candidate.token] += per_bullet[candidate.token]
                continue
            synonyms = SYNONYMS.get(candidate.token, set()) | set(candidate.synonyms)
            replaced = False
            for synonym in synonyms:
                new_text, success = _replace_synonym(updated, synonym, candidate.token)
                if success:
                    if strict and abs(len(new_text) - len(original)) > 10:
                        continue
                    updated = new_text
                    usage[candidate.token] += 1
                    per_bullet[candidate.token] += 1
                    replaced = True
                    break
            if replaced:
                continue
            if not strict and usage[candidate.token] < 2:
                addition = f" ({candidate.token})"
                new_text = updated + addition
                updated = new_text
                usage[candidate.token] += 1
                per_bullet[candidate.token] += 1
        if strict and abs(len(updated) - len(original)) > 10:
            updated = original
        if updated != original:
            document.replace_bullet(idx, updated)

    optimized_tex = document.render()
    optimized_counts: Dict[str, int] = {candidate.token: optimized_tex.lower().count(candidate.token) for candidate in keywords}
    keyword_map = {
        candidate.token: {"before": before_counts[candidate.token], "after": optimized_counts[candidate.token]}
        for candidate in keywords
    }
    diff = "\n".join(
        difflib.unified_diff(
            tex_content.splitlines(),
            optimized_tex.splitlines(),
            fromfile="original.tex",
            tofile="main_optimized.tex",
            lineterm="",
        )
    )
    return RewriteResult(optimized_tex=optimized_tex, keyword_map=keyword_map, diff=diff)
