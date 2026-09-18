"""TFLite/LiteRT classifier-head wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class HeadError(RuntimeError):
    """Raised when no supported TFLite runtime can load or run the head."""


def _interpreter_class() -> type:
    try:
        from ai_edge_litert.interpreter import (  # type: ignore[import-untyped]
            Interpreter,
        )

        return Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter  # type: ignore[import-untyped]

            return Interpreter
        except ImportError as error:
            raise HeadError(
                "Install taxotag[litert] or taxotag[tensorflow] to run inference"
            ) from error


class Head:
    """Run the 8,448-feature Gist TFLite classifier head."""

    def __init__(self, tflite_path: str | Path) -> None:
        interpreter_type = _interpreter_class()
        try:
            self._interpreter = interpreter_type(model_path=str(tflite_path))
            self._interpreter.allocate_tensors()
        except Exception as error:
            raise HeadError(f"failed to load TFLite model: {tflite_path}") from error

        inputs = self._interpreter.get_input_details()
        outputs = self._interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise HeadError("Gist head must have one input and one output tensor")
        self._input = inputs[0]
        self._output = outputs[0]

    def run(self, features: np.ndarray) -> np.ndarray:
        """Run the head and return a one-dimensional float32 probability vector."""

        values = np.asarray(features, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[0] != 1:
            raise ValueError("features must have shape (1, feature_dim)")

        expected_shape = tuple(int(value) for value in self._input["shape"])
        if expected_shape != tuple(values.shape):
            raise ValueError(
                f"features have shape {tuple(values.shape)}; expected {expected_shape}"
            )

        self._interpreter.set_tensor(self._input["index"], values)
        self._interpreter.invoke()
        output = np.asarray(self._interpreter.get_tensor(self._output["index"]))
        output = output.reshape(-1).astype(np.float32, copy=False)
        if output.size != 36:
            raise HeadError(f"Gist head returned {output.size} values; expected 36")
        return output
