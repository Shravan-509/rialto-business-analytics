# Contributing

Thank you for your interest in improving the Rialto Decision Intelligence Platform.

This repository is a Master's Business Analytics capstone project. Contributions should preserve the existing architecture and avoid changing validated business logic unless the change is clearly documented and tested.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the dataset at:

```text
data/raw/Rialto Data.csv
```

Run the app:

```bash
streamlit run app/main.py
```

## Contribution Guidelines

- Keep changes focused and easy to review.
- Do not commit API keys, `.env` files, or `.streamlit/secrets.toml`.
- Preserve dynamic calculations from `Rialto Data.csv`; do not hardcode business metrics.
- Keep GenAI as an explanation layer, not a replacement for analytics.
- Add or update documentation when behavior changes.
- Prefer small, well-named functions over large page-level logic.

## Quality Checks

Before submitting changes, run:

```bash
pytest
python -m py_compile app/main.py src/*.py app/pages/*.py
```

## Pull Request Checklist

- The app starts locally with `streamlit run app/main.py`.
- Existing tests pass.
- No secrets or local cache files are included.
- Documentation is updated where needed.
- Screenshots are added or refreshed if the UI changes.

## Reporting Issues

When reporting an issue, include:

- Steps to reproduce
- Expected behavior
- Actual behavior
- Python version
- Relevant error message or screenshot
