import zlib

import numpy as np
import pytest

from taxotag._ngrams import features


def test_single_word_hashes_word_and_character_grams() -> None:
    vector = features("technology", dim=8192)

    assert vector.shape == (8192,)
    assert vector.dtype == np.float32
    assert np.isclose(np.linalg.norm(vector), 1.0)
    assert np.count_nonzero(vector) == 28

    technology_bucket = zlib.crc32(b"technology") % 8192
    assert np.isclose(vector[technology_bucket], 1 / np.sqrt(28))


def test_words_are_ascii_only_and_lowercased() -> None:
    expected = features("caf technology", dim=1024)

    assert np.array_equal(expected, features("CAFÉ technology", dim=1024))
    assert not np.array_equal(expected, features("cafe-technology", dim=1024))


def test_adjacent_words_add_a_bigram() -> None:
    vector = features("red blue", dim=8192)
    bigram_bucket = zlib.crc32(b"red_blue") % 8192

    assert vector[bigram_bucket] > 0
    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_empty_input_returns_zero_vector() -> None:
    assert np.array_equal(features("... !!!", dim=8), np.zeros(8, dtype=np.float32))


def test_dimension_must_be_positive() -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        features("text", dim=0)
