"""Load Gist classifier configuration and topic names."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Classifier dimensions and ranking configuration."""

    slugs: tuple[str, ...]
    ngram_dim: int
    threshold: float


def load_config(path: str | Path) -> ModelConfig:
    """Load the Gist configuration JSON."""

    values = json.loads(Path(path).read_text(encoding="utf-8"))
    return ModelConfig(
        slugs=tuple(values["slugs"]),
        ngram_dim=int(values["ngram_dim"]),
        threshold=float(values["threshold"]),
    )


def load_names(path: str | Path) -> dict[str, str]:
    """Return taxonomy slug-to-display-name mappings."""

    values = json.loads(Path(path).read_text(encoding="utf-8"))
    return {topic["slug"]: topic["name"] for topic in values["topics"]}
