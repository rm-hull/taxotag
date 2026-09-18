import json
from pathlib import Path

import numpy as np
import pytest

from taxotag import Gist

ROOT = Path(__file__).parent
MODEL = ROOT / "fixtures" / "model"
pytestmark = pytest.mark.skipif(
    not (MODEL / "gist.tflite").is_file(),
    reason="local Phase 1 model assets are not available",
)
pytestmark = [pytestmark, pytest.mark.slow]


def test_scores_cover_the_taxonomy_for_oracle_texts() -> None:
    gist = Gist(directory=MODEL)
    oracle = json.loads((ROOT / "fixtures" / "gist-sdk-oracle.json").read_text())

    for item in oracle:
        scores = gist.scores(item["text"])
        assert len(scores) == 36
        assert set(scores) == set(gist._config.slugs)
        assert np.all(
            (np.array(list(scores.values())) >= 0)
            & (np.array(list(scores.values())) <= 1)
        )


def test_empty_input_returns_empty_results() -> None:
    gist = Gist(directory=MODEL)

    assert gist.scores(" \t\n") == {}
    assert gist.classify(" \t\n") == []


def test_classify_keeps_top_topic_below_threshold() -> None:
    gist = Gist(directory=MODEL)

    topics = gist.classify("a rare unrelated string", threshold=1.0, top_k=3)

    assert len(topics) == 1
    assert topics[0].score < 1.0
