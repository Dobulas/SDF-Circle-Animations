# AGENTS

This repository holds Python animations for PNF 404 Radio. The code uses PyQtGraph with Matplotlib.

## Best Practices

- Target Python 3.10 or newer.
- Use `black` for formatting with the default line length (88) and `flake8` for linting.
- Include type hints and docstrings for all public functions.
- Keep dependencies in `requirements.txt`.
- New animations should live in the `pnf404_radio` package and be callable from `run.py`.
- Run `pytest` to execute lightweight unit tests for helper functions.
- For GUI checks, run `python run.py` with the environment variable `QT_QPA_PLATFORM=offscreen` to avoid display issues.

## Pull Request Guidelines

Provide a short bullet list of changes in the PR body. Mention any new dependencies or setup steps.
