import importlib.util
from pathlib import Path

import numpy as np
import pytest

from taxotag._head import Head

MODEL = Path(__file__).parent / "fixtures" / "model"
pytestmark = [
    pytest.mark.skipif(
        not (MODEL / "gist.tflite").is_file(),
        reason="local Phase 1 model assets are not available",
    ),
    pytest.mark.skipif(
        not (
            importlib.util.find_spec("ai_edge_litert")
            or importlib.util.find_spec("tensorflow")
        ),
        reason="LiteRT or TensorFlow is not installed",
    ),
    pytest.mark.slow,
]


def test_head_returns_36_probabilities() -> None:
    head = Head(MODEL / "gist.tflite")

    output = head.run(np.zeros((1, 8448), dtype=np.float32))

    assert output.shape == (36,)
    assert output.dtype == np.float32
    assert np.all((output >= 0) & (output <= 1))


def test_head_rejects_wrong_feature_shape() -> None:
    head = Head(MODEL / "gist.tflite")

    with pytest.raises(ValueError, match=r"expected \(1, 8448\)"):
        head.run(np.zeros((1, 10), dtype=np.float32))
