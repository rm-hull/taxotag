import json
from pathlib import Path

import numpy as np

from taxotag._embedding import Embedding, EmbeddingMeta

ROOT = Path(__file__).parent
MODEL = ROOT / "fixtures" / "model"


def test_embedding_metadata_matches_pinned_asset() -> None:
    meta = EmbeddingMeta.from_json(MODEL / "gist_embedding.json")

    assert meta == EmbeddingMeta(
        vocab_size=261349,
        dim=256,
        scale=1.0,
        normalize=True,
    )


def test_pool_skips_invalid_ids_and_normalizes() -> None:
    rows = np.array([1, 0, 0, 0, 2, 0], dtype=np.int8)
    embedding = Embedding(rows, EmbeddingMeta(2, 3, 1.0, True))

    pooled = embedding.pool([0, 1, -1, 99])

    expected = np.array([0.5, 1, 0], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(pooled, expected)


def test_pool_matches_every_embedding_oracle() -> None:
    embedding = Embedding.from_files(
        MODEL / "gist_embedding.i8",
        MODEL / "gist_embedding.json",
    )
    feature_oracle = json.loads(
        (ROOT / "fixtures" / "gist-feature-oracle.json").read_text()
    )
    sdk_oracle = {
        item["text"]: item["ids"]
        for item in json.loads((ROOT / "fixtures" / "gist-sdk-oracle.json").read_text())
    }

    matched = 0
    for item in feature_oracle:
        if item["text"] not in sdk_oracle:
            continue
        actual = embedding.pool(sdk_oracle[item["text"]])
        np.testing.assert_allclose(actual, item["emb"], atol=1e-5)
        matched += 1

    assert matched == 6
