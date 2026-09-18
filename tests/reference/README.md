# Phase 1 reference materials

The files in `swift/` are copied from `Desert-Ant-Labs/desert-ant-core` at
tag `v3.2.0`, commit `3a8d13cd1c4f027f095822956d9b08eb5ce6684d`.

The JSON files in `../fixtures/` are the upstream tokenizer and embedding
oracles from the same tag. They are test references only and are not packaged
with the library.

Model assets are downloaded separately into `../fixtures/model/` by:

```sh
uv run python scripts/fetch_test_assets.py
```

The model assets are pinned to Hugging Face revision `v2.2.0` and are ignored
by git because they are large and subject to the Desert Ant Labs
Source-Available License.
