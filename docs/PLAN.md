# `taxotag` — Python port of Desert Ant Labs' Gist SDK

**Implementation plan for a coding agent.** Source of truth for behavior is the
Swift reference implementation at [`Desert-Ant-Labs/desert-ant-core`](https://github.com/Desert-Ant-Labs/desert-ant-core/tree/main), specifically
`Sources/Gist/*.swift` at tag/revision matching model revision `v2.2.0` /
SDK version `3.2.0`. Do not guess at behavior — port the Swift line-for-line
where this plan says "port," and validate every stage against the oracle
fixtures in `Tests/GistTests/Resources/`.

Model weights and taxonomy come from the Hugging Face repo
`desert-ant-labs/gist`, pinned at revision `v2.2.0`. **License note:** the
model weights are under the Desert Ant Labs Source-Available License (free
under a MAU threshold, commercial license required at scale) — this applies
to the weights, not to `taxotag`'s wrapper code. Put a clear notice in the
README and don't bundle the weights in the sdist/wheel; download them at
runtime or let the user point at a local copy.

---

## 0. Goal and non-goals

**Goal:** a pure-Python (+ numpy + a TFLite runtime) package that reproduces
`Gist.classify()` / `Gist.scores()` from the Swift/Kotlin/JS SDKs, to
numerical parity with the reference oracle fixtures, installable via
`pip install taxotag`.

**Non-goals for v1:**
- No training code, no fine-tuning.
- No `channelTopics` roll-up port (nice-to-have, see Phase 6 — do it last).
- No Core ML backend — LiteRT/TFLite only (Python has no first-class Core ML
  runtime outside macOS `coremltools`, and TFLite float32 is simpler and
  cross-platform).
- No English-only variant in v1 (ship multilingual first; the variant system
  is additive, see Phase 7).

---

## 1. Package layout

Use a `src/` layout, `pyproject.toml`-only (no `setup.py`), targeting
Python 3.9+.

```
taxotag/
  pyproject.toml
  README.md
  LICENSE.md                  # taxotag's own code license (MIT/Apache-2.0)
  NOTICE.md                   # points to Desert Ant Labs weight license
  src/
    taxotag/
      __init__.py             # exports: Gist, Topic, GistError
      _tokenizer.py           # binary parser + Viterbi unigram tokenizer
      _ngrams.py               # hashed n-gram featurizer
      _embedding.py            # potion/model2vec embedding pooling
      _head.py                 # TFLite head wrapper
      _assets.py                # HF Hub download/cache, file resolution
      _taxonomy.py              # taxonomy.json / gist_config.json parsing
      model.py                 # Gist class tying it all together
      cli.py                   # optional: `taxotag "some text"` CLI
      py.typed
  tests/
    conftest.py
    test_ngrams.py
    test_tokenizer.py
    test_embedding.py
    test_head.py
    test_integration.py
    fixtures/
      gist-feature-oracle.json   # copied from upstream repo (see Phase 1)
      gist-sdk-oracle.json
  .github/workflows/ci.yml
  .github/workflows/release.yml
```

---

## 2. Dependencies

Runtime:
- `numpy` — feature vectors, pooling, matmul fallback
- `huggingface_hub` — download + cache model assets from `desert-ant-labs/gist`
- `model2vec` — semantic embedding stream (confirmed API-compatible per the
  Swift `Embedding.swift` comment: *"Matches the Python model2vec `encode`
  for the pruned model"*). **Verify this claim empirically in Phase 3** —
  don't take the comment on faith, the pruned/quantized int8 table taxotag
  ships (`gist_embedding.i8`) may need to be loaded directly rather than
  through `model2vec.StaticModel.from_pretrained`, since that convenience
  loader expects a specific repo layout. Plan for **both** paths (see Phase 3).
- One TFLite runtime, try in this order and let the user override via extra:
  - `ai-edge-litert` (the current official successor to `tflite-runtime`)
  - fallback: `tensorflow` (heavier, but always available)
  - Expose as extras: `pip install taxotag[litert]` / `taxotag[tensorflow]`

Dev/test:
- `pytest`, `pytest-cov`
- `ruff` (lint + format)
- `mypy`
- `build`, `twine`

---

## 3. Phase 1 — Pull reference materials (do this first, don't skip)

1. Clone/download `Desert-Ant-Labs/desert-ant-core` at the commit tagged for
   SDK version `3.2.0` (check `Sources/Gist/Catalog.swift` — `sdkVersion` —
   to confirm you have the right revision).
2. Copy these files into your working notes (not into the package — they're
   reference only, GPL-none-of-that, just don't want to lose track):
   - `Sources/Gist/Tokenizer.swift`
   - `Sources/Gist/NGrams.swift`
   - `Sources/Gist/Embedding.swift`
   - `Sources/Gist/Model.swift`
   - `Sources/Gist/Catalog.swift`, `Variant.swift` (file names / manifest)
3. Copy `Tests/GistTests/Resources/gist-feature-oracle.json` and
   `gist-sdk-oracle.json` into `tests/fixtures/`. These are your ground truth:
   - `gist-sdk-oracle.json`: text → expected token ids (tokenizer oracle)
   - `gist-feature-oracle.json`: text → expected pooled embedding vector
     (embedding oracle)
4. Download the model assets themselves from
   `https://huggingface.co/desert-ant-labs/gist` at revision `v2.2.0`:
   - `gist_tokenizer.bin`
   - `gist_embedding.i8` + its `.json` sidecar (check exact sidecar filename
     in the repo — `Embedding.swift`'s `EmbeddingMeta` decodes
     `vocab_size`, `dim`, `scale`, `normalize`)
   - `gist_config.json` (`slugs`, `ngram_dim`, `threshold`)
   - `taxonomy.json`
   - `gist.tflite`
   - `taxonomy_crosswalk.json` (optional, for IAB/Apple category mapping —
     nice-to-have, not required for `classify`/`scores`)

   Stash these under `tests/fixtures/model/` for local dev (gitignored,
   large) and write a small `scripts/fetch_test_assets.py` that pulls them
   via `huggingface_hub.hf_hub_download` so CI can reproduce it.

**Exit criterion for Phase 1:** you have the two oracle JSON files and all
six model asset files sitting on disk, and you've confirmed the `.i8`
sidecar's actual filename and JSON schema by inspection (don't assume the
exact name — `Embedding.swift` reads `assets.embedding` +
`assets.embeddingMetaJSON` as separate files, confirm what the HF repo
actually calls them).

---

## 4. Phase 2 — N-gram featurizer (`_ngrams.py`)

Direct, mechanical port of `NGrams.swift`. This has no external dependencies
and should be gettable to exact parity quickly.

Algorithm (port faithfully):
1. Lowercase the input text.
2. Extract words: maximal runs of ASCII `[a-z0-9']` (non-ASCII characters act
   as delimiters — **not** a general Unicode word-boundary split; confirm
   this matches Swift's `isWordScalar` check, which only tests `a-z`, `0-9`,
   `'`).
3. Build the gram list:
   - all word unigrams
   - all adjacent-word bigrams, joined as `f"{w1}_{w2}"`
   - for each word, wrap as `f"^{word}$"` and extract all contiguous
     character 3-, 4-, and 5-grams (only when the wrapped word is long
     enough)
4. Hash each gram with **zlib-compatible CRC-32** of its UTF-8 bytes, bucket
   into `dim` slots via `crc32(gram) % dim`, incrementing a count vector.
   Python's `zlib.crc32` is the same algorithm/polynomial as the Swift
   implementation here (both are the standard CRC-32, poly `0xEDB88320`) —
   confirm with a few hand-computed values in a unit test rather than
   assuming.
5. L2-normalize the resulting vector; leave as all-zero if the input has no
   grams (matches Swift, which only divides `if norm > 0`).

```python
def features(text: str, dim: int) -> np.ndarray: ...
```

**Test:** since there's no dedicated n-gram oracle file, cross-check by
extracting the exact tail of `gist-feature-oracle.json`'s `emb` field — no,
wait, that's the *embedding* oracle, not n-grams. The n-gram half has no
upstream fixture. Options, in order of preference:
1. If the reference repo happens to also expose a `gist_config.json` sample
   pair of `(text, full 8448-dim feature vector)` anywhere (check
   `Tests/GistTests/GistTests.swift` — it may assemble features + run the
   head), reuse it.
2. Otherwise, write a hand-verified test: compute CRC-32 of `"technology"`
   etc. by hand/python `zlib.crc32(b"technology")` and confirm the bucket
   index arithmetic, then treat internal consistency (norm == 1, deterministic
   output, dimension matches config) as the acceptance bar for this module,
   and rely on **Phase 5's end-to-end oracle** (`gist-sdk-oracle.json` inputs
   run all the way through the head) to catch any real mismatch.

---

## 5. Phase 3 — Semantic embedding (`_embedding.py`)

Two-track plan — try the easy path first, fall back to the manual port if it
doesn't check out:

**Track A (preferred): `model2vec.StaticModel.from_pretrained`**
```python
from model2vec import StaticModel
model = StaticModel.from_pretrained("desert-ant-labs/gist")
vec = model.encode(["some text"])
```
Test this against `gist-feature-oracle.json`'s `emb` field for the same
input strings. If it matches within float tolerance (`np.allclose(..., atol=1e-5)`
or so, since Swift dequantizes int8→float32 and mean-pools in float32) —
ship this path, it's zero maintenance.

**Track B (fallback, if Track A's tokenization/pooling doesn't match exactly,
e.g. because `model2vec`'s own tokenizer isn't the same pruned-vocab unigram
tokenizer gist uses): manual port of `Embedding.swift`**
- Load `gist_embedding.i8` as a flat int8 array, reshape to
  `(vocab_size, dim)` per `EmbeddingMeta`.
- Given token ids from **your own tokenizer** (Phase 4 — this is why
  tokenizer parity matters, the embedding pooling depends on getting
  identical ids), gather rows, dequantize (`row.astype(np.float32) * scale`),
  mean-pool across the token dimension, L2-normalize if `meta.normalize`.
- This is a ~15-line numpy function; keep it as the guaranteed-correct
  fallback regardless of whether Track A ships, since Track B is what
  actually guarantees byte-for-byte parity with the Swift SDK.

**Decision point for the agent:** implement Track B regardless (it's cheap
and it's the real source of truth), and only wire in Track A as an
optional fast-path / convenience if it validates cleanly. Don't ship Track A
as the only implementation — if `model2vec`'s internal tokenizer ever drifts
from gist's pruned vocab, silent divergence is worse than an extra 15 lines
of code.

**Test:** run both tracks against every `(text, emb)` pair in
`gist-feature-oracle.json`, assert numerical closeness.

---

## 6. Phase 4 — Tokenizer (`_tokenizer.py`)

This is the most involved port. Structure it as two pieces:

### 6a. Binary format parser
Port the `init?(bytes:)` reader in `Tokenizer.swift` directly:
- Validate magic bytes `b"GSTK"` + 1 version byte (5-byte header).
- Read `unk_id`, `bos_id`, `eos_id`, `count` as little-endian int32 (Swift
  reads them via `readU32`/bitPattern-cast to `Int32` — use
  `struct.unpack("<i", ...)` in Python).
- Read `count` float32 scores (little-endian).
- Read `count` uint16 piece lengths (little-endian).
- Read `count` UTF-8 piece strings back-to-back per those lengths, building
  a `dict[str, int]` (piece → id).
- Validate `offset == len(bytes)` at the end (integrity check — port this,
  it catches truncated/corrupt files).
- Compute `max_len = min(max_piece_length_in_scalars, 32)` and
  `unk_penalty = min(scores) - 10.0`.

```python
class Tokenizer:
    def __init__(self, data: bytes) -> None: ...
    def tokenize(self, text: str) -> list[Token]: ...
    def encode(self, text: str) -> list[int]: ...
```

### 6b. Normalization + Viterbi segmentation
Port `nmtNormalize` and the tokenize loop:
1. **Control-char strip + whitespace normalization**: drop the specific
   control code points Swift lists (`0x01-0x08, 0x0B, 0x0E-0x1F, 0x7F, 0x8F,
   0x9F`), map the listed whitespace-ish code points to a plain space
   (`0x09,0x0A,0x0C,0x0D,0xA0,0x1680,0x2028,0x2029,0x202F,0x205F,0x2000-0x200F,
   0x2581,0x3000,0xFEFF,0xFFFD`).
2. **NFKC normalize**, with the fullwidth-tilde (`U+FF5E`) special case:
   split on that character, NFKC each segment independently, rejoin with the
   literal tilde (this works around an ICU/Unicode quirk in the training
   charsmap — port it exactly, don't "simplify" it away). Python's
   `unicodedata.normalize("NFKC", s)` matches Swift's `.nfkc` here.
3. **Collapse whitespace runs** to single spaces, trim, drop a single
   trailing space if that's what's left.
4. **Metaspace substitution**: prefix with `▁` (U+2581), replace remaining
   spaces with `▁`.
5. **Viterbi DP** over the unigram vocab: for each end position `i`, look
   back up to `max_len` characters, take the best-scoring substring that's a
   known piece; also always allow an "unknown char" transition from `i-1`
   at `unk_penalty` cost. Backtrack from `n` to build the token sequence,
   then reverse.

**Critical gotcha to flag for the agent:** Swift operates over
`unicodeScalars`, i.e. Unicode scalar values, not grapheme clusters and not
UTF-16 code units. In Python, iterate over the string's **code points**
directly (Python 3 strings already are sequences of code points — just
`for ch in s` — this is actually *simpler* in Python than the Swift version,
just be sure you're not accidentally doing byte-level indexing anywhere).

**Test:** run every `(text, ids)` pair in `gist-sdk-oracle.json` through
`Tokenizer.encode()` and assert exact list equality. This file already
covers Dutch, Spanish, French, German, English, tricky whitespace, and
casing — treat 100% pass on this fixture as the phase's hard exit criterion,
not just "close enough."

---

## 7. Phase 5 — Classifier head + end-to-end model (`_head.py`, `model.py`)

### 7a. Head wrapper
```python
class Head:
    def __init__(self, tflite_path: str | Path) -> None:
        # try ai_edge_litert.interpreter.Interpreter first,
        # fall back to tensorflow.lite.Interpreter
        ...
    def run(self, features: np.ndarray) -> np.ndarray:
        # features: shape (1, feature_dim) float32
        # returns: shape (36,) float32 (or (1,36) squeezed)
        ...
```
Confirm the exact input/output tensor names by inspecting
`interpreter.get_input_details()` / `get_output_details()` at load time
(the Swift side calls them `"features"` / `"topic_probs"` via the shared
`InferenceSession` abstraction — TFLite's own tensor names may differ from
these logical names, don't hardcode assuming they match).

### 7b. `Gist` class (public API, `model.py`)
Mirror the Swift/JS public surface:
```python
@dataclass(frozen=True)
class Topic:
    slug: str
    name: str
    score: float

class Gist:
    def __init__(self, *, directory: str | Path | None = None,
                 variant: str = "multilingual") -> None: ...
    def scores(self, text: str) -> dict[str, float]: ...
    def classify(self, text: str, top_k: int = 3,
                 threshold: float | None = None) -> list[Topic]: ...
```
- `scores`: tokenize → concat(embedding.pool(ids), ngrams.features(text)) →
  head.run → dict of slug→prob using `slugs` order from `gist_config.json`.
- `classify`: sort by `(score desc, slug asc)` (port the tie-break exactly —
  Swift/JS both break ties alphabetically by slug), take `top_k`, filter to
  `score >= threshold` **except always keep index 0** (the Swift/JS logic
  keeps the top topic even under threshold — port this exact "always keep
  rank 0" behavior, it's easy to drop by accident).
- Empty/whitespace-only input → `[]` for `classify`, `{}` for `scores`
  (matches the JS wrapper's explicit early return).

### 7c. Asset loading (`_assets.py`)
- Use `huggingface_hub.snapshot_download` or per-file
  `hf_hub_download(repo_id="desert-ant-labs/gist", revision="v2.2.0", filename=...)`
  to fetch and cache the six files, defaulting to HF's own cache dir.
- Support a `directory` override that, if it already contains the files,
  skips the network entirely (mirrors the Swift SDK's "ship it yourself"
  path) — check for file presence, don't just try/except blindly so error
  messages are clear about what's missing.

**Test (integration, the big one):** for every `(text, ids)` pair in
`gist-sdk-oracle.json`, run the **full** `Gist.scores()` pipeline and sanity
check outputs are well-formed probabilities (0–1, sums roughly sensible for
a multi-label sigmoid head — don't expect them to sum to 1). If you can get
even a handful of `(text, expected top-3 topics)` pairs from manual testing
against the JS/Swift SDK directly (run the same strings through
`npm i @desert-ant-labs/gist` in a scratch Node project), use those as a
final parity check — the repo's committed oracles stop at tokenizer/embedding,
they don't include head outputs, so this last mile is on you to verify by
running the reference SDK side by side.

---

## 8. Phase 6 — Polish

- `cli.py`: `taxotag "some text" [--top-k 3] [--json]` via
  `argparse`, entry point in `pyproject.toml`'s `[project.scripts]`.
- Optional: port `channelTopics` (`Channel.swift` — check that file, not
  covered above) as a pure function for roll-up use cases. Low priority,
  do only if requested.
- Optional: English-only variant (`Variant.swift` shows the `en/` subfolder
  pattern) — add a `variant="english"` parameter that changes which HF paths
  get downloaded.
- Docstrings + a `README.md` with the same style of quickstart as the other
  SDKs, explicit license note per this doc's header.

---

## 9. Testing & CI

- `pytest` with the fixtures above; target **100% pass on
  `gist-sdk-oracle.json` token ids** and **numerical parity (atol ~1e-4) on
  `gist-feature-oracle.json` embeddings** as hard gates.
- `.github/workflows/ci.yml`: matrix over Python 3.9–3.12, run lint
  (`ruff check`), type-check (`mypy src/`), tests. Cache the downloaded model
  assets between runs (they're large and unlikely to change).
- Don't run the TFLite head against real network-downloaded weights on every
  CI run if avoidable — cache aggressively, or mark the full integration
  test `@pytest.mark.slow` and run it on a schedule rather than every push.

---

## 10. Packaging & publishing to PyPI

`pyproject.toml` essentials:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "taxotag"
dynamic = ["version"]
description = "Python port of Desert Ant Labs' Gist on-device topic tagger"
readme = "README.md"
license = "MIT"                     # taxotag's own code, not the weights
requires-python = ">=3.9"
dependencies = [
  "numpy>=1.24",
  "huggingface_hub>=0.20",
  "model2vec>=0.3",
]

[project.optional-dependencies]
litert = ["ai-edge-litert"]
tensorflow = ["tensorflow"]

[project.scripts]
taxotag = "taxotag.cli:main"
```

Release flow:
1. `python -m build`
2. `twine check dist/*`
3. Tag `v0.1.0`, push tag → `.github/workflows/release.yml` builds and
   `twine upload`s using a trusted-publisher / API-token secret (prefer
   PyPI's Trusted Publishing via GitHub Actions OIDC over a long-lived token).
4. Register the name on PyPI early (even an empty `0.0.1` placeholder) once
   you've finalized it, since names are first-come-first-served.

---

## 11. Suggested execution order for the agent

1. Phase 1 (pull reference + oracles + assets) — cannot proceed without this.
2. Phase 2 (n-grams) — fast, self-contained, builds confidence.
3. Phase 4 (tokenizer) — do this before Phase 3, since Phase 3's Track B
   fallback depends on having correct token ids.
4. Phase 3 (embedding) — validate both tracks against the oracle.
5. Phase 5 (head + `Gist` class) — wire it all together, do end-to-end checks.
6. Phase 9 (CI) as you go, not bolted on at the end.
7. Phase 6 (polish) and Phase 10 (packaging) last.

Flag back to the user (don't silently guess) if:
- The `.i8` embedding sidecar filenames/schema on the HF repo don't match
  what `Embedding.swift`/`EmbeddingMeta` implies.
- `model2vec.StaticModel.from_pretrained("desert-ant-labs/gist")` doesn't
  reproduce the oracle embeddings — that's a real finding worth surfacing,
  not something to paper over.
- Any oracle fixture fails after a careful, exact port — that means a real
  bug in the port, not a tolerance issue to loosen.