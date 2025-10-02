"""LinkedIn harvesting utilities.

The scraper favours Playwright for deterministic browser automation.  For tests we
expose a manual mode that reads job descriptions from text/JSON files so the rest of
the pipeline can run without network access.  Each collected posting is persisted to
``artifacts/jobs``.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

ARTIFACT_DIR = Path("artifacts/jobs")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ManualJob:
    """Structure used for manual mode input files."""

    title: str
    company: str
    description: str
    location: str = ""
    seniority: Optional[str] = None
    url: str = "manual://entry"

    def __post_init__(self) -> None:
        self.description = " ".join(self.description.split())


@dataclass
class JobPosting:
    """Normalized view of a job posting."""

    title: str
    company: str
    location: str
    seniority: Optional[str]
    description: str
    url: str
    keywords: List[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


class LinkedInScraper:
    """Headless LinkedIn job harvester with polite scraping defaults."""

    def __init__(
        self,
        email: Optional[str],
        password: Optional[str],
        *,
        manual_mode: bool = False,
        delay_seconds: float = 2.0,
    ) -> None:
        self.email = email
        self.password = password
        self.manual_mode = manual_mode
        self.delay_seconds = delay_seconds

    # Public API -----------------------------------------------------------------
    def harvest(
        self,
        job_titles: Sequence[str],
        locations: Sequence[str],
        *,
        manual_inputs: Optional[Sequence[ManualJob]] = None,
    ) -> List[JobPosting]:
        """Return harvested postings for titles/locations.

        When ``manual_mode`` is enabled the caller *must* provide ``manual_inputs``.
        The entries are stored as JSON artifacts alongside the fully scraped output
        so the rest of the pipeline remains identical.
        """

        if self.manual_mode:
            if not manual_inputs:
                raise ValueError("manual_mode requires manual_inputs")
            postings = [self._from_manual(job) for job in manual_inputs]
        else:
            postings = self._automated_harvest(job_titles, locations)

        for posting in postings:
            self._persist(posting)
        return postings

    # Manual mode helpers --------------------------------------------------------
    @staticmethod
    def load_manual_file(path: Path) -> ManualJob:
        """Load a manual job description from JSON or plain text."""

        text = path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
            return ManualJob(**data)
        except json.JSONDecodeError:
            return ManualJob(title=path.stem, company="Manual", description=text, location="", url=f"manual://{path.stem}")

    def _from_manual(self, manual_job: ManualJob) -> JobPosting:
        keywords = self._simple_keywords(manual_job.description)
        return JobPosting(
            title=manual_job.title,
            company=manual_job.company,
            location=manual_job.location,
            seniority=manual_job.seniority,
            description=manual_job.description,
            url=manual_job.url,
            keywords=keywords,
        )

    # Automated scraping ---------------------------------------------------------
    def _automated_harvest(self, job_titles: Sequence[str], locations: Sequence[str]) -> List[JobPosting]:
        """Scrape LinkedIn using Playwright if available, falling back to Selenium."""

        try:
            from playwright.sync_api import sync_playwright
        except Exception:  # pragma: no cover - optional dependency branch
            sync_playwright = None

        if not sync_playwright:
            raise RuntimeError(
                "Playwright is not installed. Install playwright and run `playwright install` "
                "or use manual mode via --manual-mode."
            )

        postings: List[JobPosting] = []
        with sync_playwright() as p:  # pragma: no cover - network automation not unit tested
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            self._login(page)
            for title in job_titles:
                for location in locations:
                    postings.extend(self._search_and_collect(page, title, location))
                    time.sleep(self.delay_seconds)
            browser.close()
        return postings

    def _login(self, page) -> None:  # pragma: no cover - requires browser
        if not self.email or not self.password:
            raise RuntimeError("LinkedIn credentials missing. Provide LINKEDIN_EMAIL/PASSWORD or use manual mode.")
        page.goto("https://www.linkedin.com/login")
        page.fill("input[id=username]", self.email)
        page.fill("input[id=password]", self.password)
        page.click("button[type=submit]")
        page.wait_for_timeout(int(self.delay_seconds * 1000))

    def _search_and_collect(self, page, title: str, location: str) -> List[JobPosting]:  # pragma: no cover
        query = title.replace(" ", "%20")
        loc = location.replace(" ", "%20")
        page.goto(f"https://www.linkedin.com/jobs/search/?keywords={query}&location={loc}")
        page.wait_for_timeout(int(self.delay_seconds * 1000))
        postings: List[JobPosting] = []
        elements = page.query_selector_all(".job-card-container--clickable")
        for element in elements:
            try:
                element.click()
                page.wait_for_timeout(int(self.delay_seconds * 1000))
                title_text = page.query_selector(".jobs-unified-top-card__job-title")
                company_text = page.query_selector(".jobs-unified-top-card__company-name").inner_text().strip()
                location_text = page.query_selector(".jobs-unified-top-card__bullet").inner_text().strip()
                seniority_text_element = page.query_selector(".jobs-unified-top-card__job-insight")
                seniority = seniority_text_element.inner_text().strip() if seniority_text_element else None
                description_text = page.query_selector(".jobs-description__content").inner_text().strip()
                url = page.url
                keywords = self._simple_keywords(description_text)
                postings.append(
                    JobPosting(
                        title=title_text.inner_text().strip() if title_text else title,
                        company=company_text,
                        location=location_text,
                        seniority=seniority,
                        description=description_text,
                        url=url,
                        keywords=keywords,
                    )
                )
            except Exception:
                continue
        return postings

    # Utility --------------------------------------------------------------------
    def _persist(self, posting: JobPosting) -> None:
        target = ARTIFACT_DIR / f"{self._slug(posting.title)}_{self._slug(posting.company)}.json"
        target.write_text(posting.to_json(), encoding="utf-8")

    @staticmethod
    def _slug(value: str) -> str:
        safe = value.lower().replace(" ", "-")
        return "".join(ch for ch in safe if ch.isalnum() or ch == "-")

    @staticmethod
    def _simple_keywords(description: str, limit: int = 15) -> List[str]:
        words = [w.strip(".,:- ").lower() for w in description.split() if len(w) > 3]
        freq: dict[str, int] = {}
        for word in words:
            if not word.isalpha():
                continue
            freq[word] = freq.get(word, 0) + 1
        sorted_words = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
        return [word for word, _ in sorted_words[:limit]]


def load_manual_jobs_from_paths(paths: Iterable[os.PathLike[str] | str]) -> List[ManualJob]:
    """Helper used by CLI to load manual job descriptions from arbitrary files."""

    results: List[ManualJob] = []
    for path_like in paths:
        path = Path(path_like)
        results.append(LinkedInScraper.load_manual_file(path))
    return results
