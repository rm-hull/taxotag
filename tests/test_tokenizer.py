import json
from pathlib import Path

import pytest

from taxotag._tokenizer import Tokenizer, nmt_normalize

ROOT = Path(__file__).parent
TOKENIZER_PATH = ROOT / "fixtures" / "model" / "gist_tokenizer.bin"


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer(TOKENIZER_PATH.read_bytes())


def test_tokenizer_matches_every_upstream_oracle_record(
    tokenizer: Tokenizer,
) -> None:
    oracle = json.loads((ROOT / "fixtures" / "gist-sdk-oracle.json").read_text())

    for item in oracle:
        assert tokenizer.encode(item["text"]) == item["ids"]


def test_normalization_removes_controls_and_maps_whitespace() -> None:
    assert nmt_normalize("a\x01\u00a0b") == "a b"
    assert nmt_normalize("\uff21\uff5e\uff22") == "A\uff5eB"


def test_truncated_tokenizer_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid tokenizer header"):
        Tokenizer(b"GSTK\x01")
