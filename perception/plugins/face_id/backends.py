"""Inference backends used by SCRFD and face-recognition models."""

from __future__ import annotations

import os
from pathlib import Path
import threading
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
        intra_op_threads: int | None = None,
        inter_op_threads: int | None = None,
    ):
        import onnxruntime as ort

        model_path = str(path)
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"ONNX model does not exist: {model_path}")
        selected = providers or ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        if intra_op_threads is not None:
            if intra_op_threads < 1:
                raise ValueError("ONNX intra-op threads must be at least 1")
            options.intra_op_num_threads = int(intra_op_threads)
        if inter_op_threads is not None:
            if inter_op_threads < 1:
                raise ValueError("ONNX inter-op threads must be at least 1")
            options.inter_op_num_threads = int(inter_op_threads)
        self.session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=selected,
        )
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


class OpenCVDNNBackend:
    """CPU-only ONNX inference using OpenCV already present in the image."""

    def __init__(self, path: str | Path):
        import cv2

        model_path = str(path)
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"ONNX model does not exist: {model_path}")
        threads = int(os.environ.get("FACE_OPENCV_THREADS", "1"))
        if threads < 1:
            raise ValueError("FACE_OPENCV_THREADS must be at least 1")
        cv2.setNumThreads(threads)
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.output_names = list(self.net.getUnconnectedOutLayersNames())
        if not self.output_names:
            raise ValueError(f"OpenCV DNN model has no outputs: {model_path}")
        self._lock = threading.Lock()

    def infer(self, array: np.ndarray) -> list[np.ndarray]:
        array = np.ascontiguousarray(array, dtype=np.float32)
        with self._lock:
            if self.net is None:
                raise RuntimeError("OpenCV DNN backend is closed")
            self.net.setInput(array)
            outputs = self.net.forward(self.output_names)
        if isinstance(outputs, np.ndarray):
            return [outputs]
        return [np.asarray(output) for output in outputs]

    def close(self) -> None:
        with self._lock:
            self.net = None


def build_backend(
    backend: str,
    path: str | Path,
    *,
    device_id: int = 0,
    providers: list[str] | None = None,
    intra_op_threads: int | None = None,
    inter_op_threads: int | None = None,
) -> InferenceBackend:
    name = str(backend).strip().lower()
    if name == "tensorrt":
        return TensorRTBackend(path, device_id=device_id)
    if name in {"onnx", "onnxruntime"}:
        return OnnxRuntimeBackend(
            path,
            providers=providers,
            intra_op_threads=intra_op_threads,
            inter_op_threads=inter_op_threads,
        )
    if name in {"opencv", "opencv-dnn", "cpu"}:
        return OpenCVDNNBackend(path)
    raise ValueError(f"unsupported face inference backend: {backend}")
