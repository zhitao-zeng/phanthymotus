"""Inference backends used by SCRFD and face-recognition models."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class InferenceBackend(Protocol):
    output_names: list[str]

    def infer(self, array: np.ndarray) -> list[np.ndarray]: ...

    def close(self) -> None: ...


class TensorRTBackend:
    """Thin adapter over the repository's shared TensorRT runtime."""

    def __init__(self, path: str | Path, *, device_id: int = 0):
        from utils.tensorrt_runtime import TensorRTEngine

        self.engine = TensorRTEngine(path, device_id=device_id)
        self.output_names = list(self.engine.output_names)

    def infer(self, array: np.ndarray) -> list[np.ndarray]:
        return self.engine.infer(array)

    def close(self) -> None:
        self.engine.close()


class OnnxRuntimeBackend:
    """Host-development fallback; TensorRT remains the Jetson target."""

    def __init__(
        self,
        path: str | Path,
        *,
        providers: list[str] | None = None,
    ):
        import onnxruntime as ort

        model_path = str(path)
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"ONNX model does not exist: {model_path}")
        selected = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=selected)
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(
                f"face model must have exactly one input; got {len(inputs)}"
            )
        self.input_name = inputs[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def infer(self, array: np.ndarray) -> list[np.ndarray]:
        array = np.ascontiguousarray(array, dtype=np.float32)
        return self.session.run(self.output_names, {self.input_name: array})

    def close(self) -> None:
        # onnxruntime releases native state when the session is collected.
        self.session = None


def build_backend(
    backend: str,
    path: str | Path,
    *,
    device_id: int = 0,
    providers: list[str] | None = None,
) -> InferenceBackend:
    name = str(backend).strip().lower()
    if name == "tensorrt":
        return TensorRTBackend(path, device_id=device_id)
    if name in {"onnx", "onnxruntime"}:
        return OnnxRuntimeBackend(path, providers=providers)
    raise ValueError(f"unsupported face inference backend: {backend}")
