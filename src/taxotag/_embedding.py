"""Quantized potion embedding table and mean-pooling implementation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EmbeddingMeta:
    """Metadata describing the quantized embedding table."""

    vocab_size: int
    dim: int
    scale: float
    normalize: bool

    @classmethod
    def from_json(cls, path: str | Path) -> EmbeddingMeta:
        """Load embedding metadata from a JSON sidecar."""

        values = json.loads(Path(path).read_text())
        return cls(
            vocab_size=int(values["vocab_size"]),
            dim=int(values["dim"]),
            scale=float(values["scale"]),
            normalize=bool(values["normalize"]),
        )


class Embedding:
    """Pool rows from a quantized embedding table."""

    def __init__(self, rows: np.ndarray, meta: EmbeddingMeta) -> None:
        if rows.dtype != np.int8:
            raise TypeError("rows must have dtype int8")
        expected_size = meta.vocab_size * meta.dim
        if rows.size != expected_size:
            raise ValueError(
                f"embedding table has {rows.size} values; expected {expected_size}"
            )
        if meta.dim <= 0 or meta.vocab_size <= 0:
            raise ValueError("embedding metadata dimensions must be positive")

        self.dim = meta.dim
        self._rows = rows.reshape(meta.vocab_size, meta.dim)
        self._scale = np.float32(meta.scale)
        self._normalize = meta.normalize

    @classmethod
    def from_files(
        cls, embedding_path: str | Path, metadata_path: str | Path
    ) -> Embedding:
        """Load an int8 embedding table and its JSON metadata sidecar."""

        meta = EmbeddingMeta.from_json(metadata_path)
        rows = np.fromfile(embedding_path, dtype=np.int8)
        return cls(rows, meta)

    def pool(self, ids: Sequence[int]) -> np.ndarray:
        """Mean-pool valid token rows and optionally L2-normalize the result."""

        output = np.zeros(self.dim, dtype=np.float32)
        count = 0
        for token_id in ids:
            if token_id < 0 or token_id >= self._rows.shape[0]:
                continue
            output += self._rows[token_id].astype(np.float32) * self._scale
            count += 1

        if count:
            output /= np.float32(count)
        if self._normalize:
            norm = np.sqrt(np.sum(output * output, dtype=np.float32))
            if norm > 0:
                output /= norm
        return output
