# taxotag

Python port of Desert Ant Labs' Gist on-device topic tagger.

The package is under active development. The inference pipeline and model
asset downloads are implemented in the package; remaining work follows the
project plan.

## Development setup

Requirements:

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/)

Create an isolated virtual environment and install the package with its
development tools:

```sh
uv venv
source .venv/bin/activate
uv sync --extra dev
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

`uv sync` keeps the environment and `uv.lock` reproducible. The commands
below also work without activating the environment by using `uv run`.

## Build, test, and check

Run the test suite:

```sh
uv run pytest
```

Run linting and formatting checks:

```sh
uv run ruff check src/
uv run ruff format --check src/
```

Run the type checker:

```sh
uv run mypy src/
```

Build and validate the distribution:

```sh
uv run python -m build
uv run twine check dist/*
```

## Release

The release workflow builds and publishes both the wheel and source archive
when a `v*` tag is pushed. It uses PyPI Trusted Publishing, so no PyPI token is
stored in GitHub Actions. Configure the PyPI project publisher for this
repository, workflow, and the `pypi` environment before publishing.

To prepare a release locally:

```sh
uv run python -m build
uv run twine check dist/*
```

After updating the version in `pyproject.toml`, create and push a matching tag:

```sh
git tag v0.1.0
git push origin v0.1.0
```

## Continuous integration

GitHub Actions checks Python 3.10 through 3.14 with Ruff, mypy, and pytest.
The pinned model assets are cached by revision so model-backed tests do not
redownload them on every run. Scheduled and manual workflows also run the
explicit `slow` model-integration group.

## Run

The command-line interface is exposed as `taxotag`:

```sh
uv run taxotag "How to start a podcast with your iPhone" \
	--directory tests/fixtures/model
```

The inference pipeline is available when the local model assets and an
optional TFLite runtime are installed. The command uses the multilingual model
by default and downloads missing assets from the pinned Hugging Face revision.
Use `--json` for machine-readable output, `--top-k` to limit results, and
`--threshold` to override the configured cutoff. Use `--variant english` for
the English-only model build.

### Choose an inference runtime

Use **LiteRT** for new installations. It is the official successor to
`tflite-runtime` and is the lighter, purpose-built option for running the
bundled TFLite head:

```sh
uv sync --extra dev --extra litert
```

Use **TensorFlow** instead when your project already depends on TensorFlow or
you need its broader compatibility with an existing TensorFlow toolchain:

```sh
uv sync --extra dev --extra tensorflow
```

The head tries LiteRT first and falls back to TensorFlow. Install only one
runtime when possible; if both are installed, LiteRT is used.

## Reference assets

Phase 1 reference fixtures are stored under `tests/fixtures/` and the pinned
Swift implementation is under `tests/reference/swift/`. Download the local
model assets from Hugging Face revision `v2.2.0` with:

```sh
uv run python scripts/fetch_test_assets.py
```

The model files are ignored by git and are not included in distributions. The
script downloads them into `tests/fixtures/model/` for local development.

## License

The `taxotag` wrapper code is available under the MIT license. Model weights
and taxonomy assets are not bundled with the package. Runtime downloads from
the `desert-ant-labs/gist` Hugging Face repository at revision `v2.2.0` are
subject to the Desert Ant Labs Source-Available License. See [NOTICE.md](NOTICE.md)
before using those assets commercially.

## AI-generated code disclaimer

This codebase was generated with assistance from **GitHub Copilot**. The
exact underlying model identifier used for this session is not exposed by the
available session metadata, so no more specific model name is claimed here.
Review and test all generated code before using it in production.
