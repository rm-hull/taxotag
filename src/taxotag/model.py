"""Public Gist model API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._assets import resolve_assets
from ._embedding import Embedding
from ._head import Head
from ._ngrams import features
from ._taxonomy import load_config, load_names
from ._tokenizer import Tokenizer


class GistError(Exception):
    """Base exception for taxotag errors."""


@dataclass(frozen=True)
class Topic:
    """A classified topic and its score."""

    slug: str
    name: str
    score: float


class Gist:
    """On-device multilingual topic tagger."""

    def __init__(
        self,
        *,
        directory: str | Path | None = None,
        variant: str = "multilingual",
    ) -> None:
        try:
            assets = resolve_assets(directory=directory, variant=variant)
            self._tokenizer = Tokenizer(assets.tokenizer.read_bytes())
            self._embedding = Embedding.from_files(
                assets.embedding, assets.embedding_meta
            )
            self._config = load_config(assets.config)
            self._names = load_names(assets.taxonomy)
            self._head = Head(assets.tflite)
        except GistError:
            raise
        except Exception as error:
            raise GistError("failed to load Gist model") from error

    def scores(self, text: str) -> dict[str, float]:
        """Return the full slug-to-probability distribution for ``text``."""

        if not text.strip():
            return {}
        token_ids = self._tokenizer.encode(text)
        semantic = self._embedding.pool(token_ids)
        lexical = features(text, self._config.ngram_dim)
        model_input = np.concatenate((semantic, lexical)).reshape(1, -1)
        probabilities = self._head.run(model_input)
        return {
            slug: float(probabilities[index])
            for index, slug in enumerate(self._config.slugs)
        }

    def classify(
        self,
        text: str,
        top_k: int = 3,
        threshold: float | None = None,
    ) -> list[Topic]:
        """Return ranked topics, retaining the top topic below the threshold."""

        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not text.strip() or top_k == 0:
            return []

        distribution = self.scores(text)
        ranked = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
        cutoff = self._config.threshold if threshold is None else threshold
        return [
            Topic(slug, self._names.get(slug, slug), score)
            for index, (slug, score) in enumerate(ranked[:top_k])
            if score >= cutoff or index == 0
        ]
