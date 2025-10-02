"""Minimal LaTeX AST for resume rewriting."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

SECTION_PATTERN = re.compile(r"\\section\{(?P<name>[^}]*)\}")
SUBSECTION_PATTERN = re.compile(r"\\subsection\{(?P<name>[^}]*)\}")
ITEM_PATTERN = re.compile(r"^(?P<lead>\s*)\\item(?:\[[^\]]*\])?\s*(?P<content>.*)$")
PAGEBREAK_PATTERN = re.compile(r"\\newpage|\\pagebreak")


@dataclass
class Bullet:
    line_index: int
    leading: str
    content: str


@dataclass
class Section:
    name: str
    line_index: int


@dataclass
class Document:
    lines: List[str]
    bullets: List[Bullet]
    sections: List[Section]

    def replace_bullet(self, index: int, new_content: str) -> None:
        bullet = self.bullets[index]
        self.lines[bullet.line_index] = f"{bullet.leading}\\item {new_content}"
        self.bullets[index] = Bullet(
            line_index=bullet.line_index,
            leading=bullet.leading,
            content=new_content,
        )

    def render(self) -> str:
        return "\n".join(self.lines)

    @property
    def bullet_texts(self) -> List[str]:
        return [bullet.content for bullet in self.bullets]

    @property
    def section_names(self) -> List[str]:
        return [section.name for section in self.sections]

    def page_estimate(self) -> int:
        page_breaks = sum(1 for line in self.lines if PAGEBREAK_PATTERN.search(line))
        approx_lines_per_page = 55
        pages_by_length = max(1, (len(self.lines) + approx_lines_per_page - 1) // approx_lines_per_page)
        return max(page_breaks + 1, pages_by_length)


def parse_document(text: str) -> Document:
    lines = text.splitlines()
    bullets: List[Bullet] = []
    sections: List[Section] = []

    for idx, line in enumerate(lines):
        section_match = SECTION_PATTERN.search(line)
        if section_match:
            sections.append(Section(name=section_match.group("name").strip().lower(), line_index=idx))
            continue
        subsection_match = SUBSECTION_PATTERN.search(line)
        if subsection_match:
            sections.append(Section(name=subsection_match.group("name").strip().lower(), line_index=idx))
            continue
        item_match = ITEM_PATTERN.match(line)
        if item_match:
            bullets.append(
                Bullet(
                    line_index=idx,
                    leading=item_match.group("lead"),
                    content=item_match.group("content").strip(),
                )
            )

    return Document(lines=lines, bullets=bullets, sections=sections)
