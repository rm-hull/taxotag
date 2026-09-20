# AGENTS.md

## Project overview

This repo contains `taxotag`, a Python package that wraps the Desert Ant Labs Gist inference pipeline.

- Package layout uses a `src/` layout under `src/taxotag`.
- Runtime dependencies are managed with `uv`.
- Python requirement is 3.10 or newer.
- Model assets are not bundled in the distribution and are downloaded from the pinned Hugging Face revision at runtime or via the local test fixture workflow.
- Releases are driven by Conventional Commits and `python-semantic-release`.

## Working conventions

- Prefer `uv` commands over direct `python -m pip` usage.
- Keep changes reproducible; do not hand-edit the generated lockfile unless the repo explicitly requires it.
- Keep documentation in sync with behavior, especially `README.md`, `CHANGELOG.md`, and any user-facing CLI changes.
- Do not add model weights or downloaded fixtures to git-tracked source files.
- Keep CI in a single workflow file at `.github/workflows/ci.yml` unless a specific need requires a new workflow.

## Local setup

```sh
uv venv
source .venv/bin/activate
uv sync --extra dev --extra litert
```

On PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Validation commands

Run the full test suite:

```sh
uv run pytest
```

Run lint and format checks:

```sh
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
```

Run type checking:

```sh
uv run mypy src/
```

Build the package and validate the distribution:

```sh
uv run python -m build
uv run twine check dist/*
```

Preview the next semantic-release without changing files:

```sh
uv run semantic-release --noop version
```

## CI and release rules

- CI runs on pushes to `main`, pull requests, a weekly schedule, and manual dispatch.
- The main workflow is `.github/workflows/ci.yml`.
- The `checks` job tests Python 3.10 through 3.14.
- Model-backed tests are gated by markers such as `@pytest.mark.slow` and skip gracefully when the runtime or pretrained assets are missing.
- Releases happen only after the `checks` job succeeds.
- `publish` depends on the release job and only runs when a release is produced.
- PyPI publication uses Trusted Publishing; no long-lived PyPI token should be stored in GitHub secrets.

## Commit and release expectations

Use Conventional Commits:

- `fix:` → patch release
- `feat:` → minor release
- `BREAKING CHANGE:` or `!` → major release

Examples:

```sh
git commit -m "fix: improve model loading error"
git commit -m "feat: add english-only variant"
git commit -m "feat!: replace CLI options"
```

## Model fixtures and test assets

The repo includes a local asset fetch script for CI/dev use:

```sh
uv run python scripts/fetch_test_assets.py
```

The expected local asset directory is:

```text
tests/fixtures/model/
```

This directory is for fetched test assets and should not be committed unless the project explicitly requires it.

## Notes for future agents

- Keep the implementation aligned with the repository plan in `docs/PLAN.md`.
- Validate changes with the smallest relevant command set before concluding work.
- If a change affects packaging, release automation, or model-runtime behavior, re-run the relevant validation commands and inspect the workflow file for expected dependency ordering.
- Prefer minimal, surgical edits over broad rewrites.
