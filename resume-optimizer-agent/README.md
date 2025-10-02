# Resume Optimizer Agent

An end-to-end automation agent that harvests LinkedIn job descriptions, optimizes a LaTeX resume for ATS performance, and compiles updated artifacts.

## Features
- 🔍 Playwright-based LinkedIn harvester with optional manual mode for compliance.
- 🧠 Keyword extraction and ATS scoring across coverage, format, quality, and distribution.
- ✍️ LaTeX-aware rewriting that preserves macros and constrains bullet edits.
- 📄 PDF compilation via optional cloud API with local/minimal fallbacks.
- 📊 Transparent reports including diff, keyword usage map, and score breakdown.

## Getting Started
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install  # if you intend to run automated harvesting
cp .env.example .env
```
Populate `.env` with your LinkedIn credentials (or rely on manual mode).

## CLI Usage
### Full Pipeline
```bash
python -m cli.resume_opt pipeline \
  --job-titles "Data Analyst, Project Manager" \
  --locations "New York, Remote" \
  --resume-path "~/resumes/main.tex" \
  --output-dir "./artifacts" \
  --ats-threshold 82 --strict
```

### Manual Optimization
```bash
python -m cli.resume_opt optimize \
  --resume-path "~/resumes/main.tex" \
  --jd-file "./samples/jd_pm.txt" \
  --ats-threshold 85 --strict
```

Artifacts include `main_optimized.tex`, `diff.patch`, `keyword_map.json`, `report.md`, and `Resume_Optimized.pdf`.

## Testing
```bash
pytest
```

Static checks:
```bash
ruff check .
mypy .
```

## Compliance & Notes
- Credentials remain in `.env` and are never logged.
- Manual mode allows using local job descriptions for privacy-sensitive workflows.
- Polite scraping delays are built into the harvester.

## Project Structure
```
resume-optimizer-agent/
 ├── harvest/             # LinkedIn scraping + JD storage
 ├── ats/                 # Keyword extraction + ATS scoring
 ├── latex/               # Resume parsing + rewriting
 ├── compile/             # PDF compilation
 ├── cli/                 # Typer-based CLI orchestration
 ├── artifacts/           # Outputs (PDFs, diffs, reports)
 ├── tests/               # Unit + e2e tests
 ├── .env.example         # Environment variables template
 ├── requirements.txt
 ├── README.md
 └── setup.py
```
