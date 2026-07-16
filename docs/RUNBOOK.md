---
runbook: true
repo: watsonx-vision-toolkit
status: active
type: tool
updated: 2026-07-16
health: unknown
deploy: GitHub Pages documentation site configured by mkdocs.yml and .github/workflows/docs.yml
next: review recent Ollama max token and provider import fixes
---

# watsonx-vision-toolkit — Runbook

## Purpose

Reusable Python toolkit for vision-based document analysis using IBM Watsonx AI or Ollama. The README lists document classification, information extraction, fraud detection, cross-validation, and decision engines. It was extracted from the IBM Watsonx Loan Preprocessing Agents project.

## Stack

- Python package requiring Python 3.9 or higher.
- Build backend: Hatchling.
- Optional providers: IBM Watsonx AI via `langchain-ibm` and `ibm-watson-machine-learning`; Ollama via `langchain-ollama`.
- CLI entry point: `watsonx-vision`.
- Dev tools: `pytest`, `pytest-cov`, `pytest-asyncio`, `ruff`, `mypy`, `black`.
- Docs: MkDocs Material with mkdocstrings and pymdown extensions.
- CI: GitHub Actions tests Python 3.9 through 3.13, lint, format check, and typecheck.

## Where it runs

The toolkit runs wherever the Python package is installed. Provider endpoints are configured by environment or code:

- IBM Watsonx AI: URL and project ID are configured by the user.
- Ollama: docs show local Ollama at `http://localhost:11434`.
- Documentation is configured for GitHub Pages at `https://qvidal01.github.io/watsonx-vision-toolkit/`.
- Application hosting beyond package and docs hosting: unknown.

## Run / deploy

```bash
# Install from PyPI
pip install watsonx-vision-toolkit
pip install "watsonx-vision-toolkit[watsonx]"
pip install "watsonx-vision-toolkit[ollama]"
pip install "watsonx-vision-toolkit[all]"
```

```bash
# Install from source
git clone https://github.com/qvidal01/watsonx-vision-toolkit.git
cd watsonx-vision-toolkit
pip install -e ".[all,dev]"
```

```bash
# Configure providers
export WATSONX_APIKEY="your-api-key"
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
export WATSONX_PROJECT_ID="your-project-id"
export OLLAMA_URL="http://localhost:11434"
```

```bash
# Run CLI examples
watsonx-vision --help
watsonx-vision classify passport.png
watsonx-vision extract passport.jpg --output json
watsonx-vision validate license.png
watsonx-vision fraud doc1.png doc2.jpg
```

```bash
# Run checks
pytest
pytest --cov=watsonx_vision --cov-report=html
ruff check watsonx_vision/ tests/
ruff format --check watsonx_vision/ tests/
mypy watsonx_vision/ --ignore-missing-imports --no-error-summary
```

```bash
# Build docs
pip install "watsonx-vision-toolkit[docs]"
mkdocs build --strict
```

Docs deploy is handled by `.github/workflows/docs.yml` on pushes to `main` that touch docs or `mkdocs.yml`, and by manual `workflow_dispatch`, using GitHub Pages.

## Health & recovery

Health status is unknown. CI checks include tests with coverage, ruff linting, ruff format check, and mypy. Documentation build health is checked with `mkdocs build --strict`.

```bash
# Known recovery steps from docs and config
pip install -e ".[all,dev]"
pytest tests/ -v --cov=watsonx_vision --cov-report=term-missing
ruff check watsonx_vision/ tests/
ruff format --check watsonx_vision/ tests/
mkdocs build --strict
```

```bash
# Provider-specific checks
watsonx-vision config
watsonx-vision classify doc.png --provider ollama --url http://localhost:11434
```

## Current status

Active. The latest commit in the repo is from 2026-07-14, within 30 days of this runbook date. The most recent work fixes Ollama `num_predict` handling from `config.max_tokens`, exports `LLMProvider`, ungates `langchain_core` imports from the Watsonx-only extra, and adds CodeRabbit, Claude workflow, and docs deployment changes. The changelog documents 0.2.0 with CLI, caching, environment config, async support, retry logic, exceptions, logging, examples, CI/CD, and docs.

## Links

- Repository: `https://github.com/qvidal01/watsonx-vision-toolkit`
- Documentation: `https://qvidal01.github.io/watsonx-vision-toolkit/`
- PyPI: `https://pypi.org/project/watsonx-vision-toolkit/`
- Issues: `https://github.com/qvidal01/watsonx-vision-toolkit/issues`
- Source project named by README: IBM Watsonx Loan Preprocessing Agents project
