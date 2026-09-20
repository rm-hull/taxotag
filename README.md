# taxotag

Python port of Desert Ant Labs' Gist on-device topic tagger.

The package is under active development. The inference pipeline and model
asset downloads are implemented in the package; remaining work follows the
project plan.

## Use as a library

Install `taxotag` with the LiteRT runtime for the smallest on-device setup:

```sh
pip install "taxotag[litert]"
```

If your application already uses TensorFlow, install the TensorFlow extra
instead:

```sh
pip install "taxotag[tensorflow]"
```

The package downloads the pinned model assets from the
[Desert Ant Labs Gist model repository on Hugging Face](https://huggingface.co/desert-ant-labs/gist)
when the model is first constructed. To use assets that you downloaded or
packaged yourself, pass their directory explicitly. The directory must contain
the tokenizer, embedding table and metadata, config, taxonomy, and
`gist.tflite` files.

### Classify text

Create one `Gist` instance and reuse it for multiple inputs. `classify` returns
ranked `Topic` objects. Each topic has a `slug`, display `name`, and model
`score` between 0 and 1:

```python
from taxotag import Gist

gist = Gist()
topics = gist.classify("How to start a podcast with just your iPhone")

for topic in topics:
    print(topic.slug, topic.name, topic.score)
# technology Technology & Software 0.93
```

The default result contains at most three topics. Set `top_k` to change the
maximum, and set `threshold` to override the model's configured cutoff. The
highest-scoring topic is retained even when it is below the threshold:

```python
topics = gist.classify(
    "A short article about software development",
    top_k=5,
    threshold=0.7,
)
```

### Get all topic scores

Use `scores` when you need the complete multi-label distribution, such as for
storing scores or aggregating results across a collection. It returns a
dictionary containing all 36 topic slugs:

```python
scores = gist.scores("A new camera app uses machine learning")
print(scores["technology"])
```

Scores are independent probabilities and do not sum to 1. Empty or
whitespace-only input returns `{}` from `scores` and `[]` from `classify`.

### Choose a model variant

The default `multilingual` variant covers 101 languages. Use `english` only
when input is reliably English or Latin-script text; it is smaller but does
not cover scripts such as Arabic, Cyrillic, Chinese, Japanese, or Devanagari:

```python
gist = Gist(variant="english")
```

### Use local model assets

For offline or controlled deployments, point `directory` at a complete local
asset directory. This skips network access when all required files are
present:

```python
from pathlib import Path
from taxotag import Gist

gist = Gist(directory=Path("/opt/models/gist"))
topics = gist.classify("A guide to personal finance")
```

The model is a 36-topic, multi-label classifier. It is intended for short
content such as titles, posts, and title-plus-description text. Reuse the
same `Gist` instance because loading the tokenizer, embedding table, and
inference head is substantially more expensive than classifying one input.

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

Releases are driven by Conventional Commits on `main` using
`python-semantic-release`. It updates the version in `pyproject.toml`, keeps
`uv.lock` synchronized, updates [CHANGELOG.md](CHANGELOG.md), creates a `v*`
tag and GitHub release, and publishes the wheel and source archive to PyPI.

PyPI publishing uses Trusted Publishing, so no PyPI token is stored in GitHub
Actions. Configure the PyPI project publisher for this repository, workflow,
and the `pypi` environment before publishing.

Use commit prefixes to select the release level:

- `fix:` creates a patch release
- `feat:` creates a minor release
- `BREAKING CHANGE:` or a `!` after the commit type creates a major release

To preview the next release locally without changing files, run:

```sh
uv run semantic-release --noop version
```

Push a Conventional Commit to `main` to trigger the release workflow:

```sh
git commit -m "fix: improve model loading error"
git push origin main
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
