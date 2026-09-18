"""Hashed lexical n-gram features used by the Gist classifier."""

from __future__ import annotations

import zlib

import numpy as np


def _words(text: str) -> list[str]:
    words: list[str] = []
    current: list[str] = []

    for character in text.lower():
        if "a" <= character <= "z" or "0" <= character <= "9" or character == "'":
            current.append(character)
        elif current:
            words.append("".join(current))
            current = []

    if current:
        words.append("".join(current))
    return words


def _grams(words: list[str]) -> list[str]:
    grams = list(words)
    grams.extend(f"{left}_{right}" for left, right in zip(words, words[1:]))

    for word in words:
        wrapped = f"^{word}$"
        grams.extend(
            wrapped[index : index + size]
            for size in (3, 4, 5)
            for index in range(len(wrapped) - size + 1)
        )
    return grams


def features(text: str, dim: int) -> np.ndarray:
    """Return normalized hashed word and character n-gram features."""

    if dim <= 0:
        raise ValueError("dim must be positive")

    vector = np.zeros(dim, dtype=np.float32)
    for gram in _grams(_words(text)):
        vector[zlib.crc32(gram.encode("utf-8")) % dim] += 1.0

    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector
