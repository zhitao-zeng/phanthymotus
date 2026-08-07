"""Pixel-content indoor/outdoor routing with the zero-shot Places365 labels."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .contracts import ErrorCode, ObstacleDistanceError, SceneDomain
from .hybrid_tensorrt_backends import _decode_image, _model_path


def _read_io_labels(path: str) -> np.ndarray:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        labels = np.asarray(
            [int(line.split()[-1]) - 1 for line in lines],
            dtype=np.uint8,
        )
    except Exception:
        raise ObstacleDistanceError(
            ErrorCode.MODEL_ERROR,
            "Places365 indoor/outdoor labels could not be loaded",
        ) from None
    if labels.shape != (365,) or not np.isin(labels, (0, 1)).all():
        raise ObstacleDistanceError(
            ErrorCode.MODEL_ERROR,
            "Places365 indoor/outdoor labels are incompatible",
        )
    return labels


def _prepare_places365_image(image: np.ndarray) -> np.ndarray:
    import cv2

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    scale = 256.0 / min(height, width)
    image = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_LINEAR,
    )
    height, width = image.shape[:2]
    row = (height - 224) // 2
    column = (width - 224) // 2
    crop = image[row : row + 224, column : column + 224].astype(np.float32) / 255.0
    crop = (crop - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    return np.ascontiguousarray(crop.transpose(2, 0, 1)[None])


class Places365SceneRouter:
    def __init__(
        self,
        model_path: str,
        io_labels_path: str,
        *,
        top_k: int = 5,
    ) -> None:
        if type(top_k) is not int or top_k <= 0 or top_k > 365 or top_k % 2 == 0:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "Places365 top_k must be a positive odd integer",
            )
        self._outdoor = _read_io_labels(io_labels_path)
        try:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                model_path,
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "Places365 ONNX model could not be loaded",
            ) from None
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "Places365 ONNX model has incompatible inputs or outputs",
            )
        self._input_name = inputs[0].name
        self._top_k = top_k

    def predict(self, image_bytes: bytes) -> SceneDomain:
        image = _decode_image(image_bytes)
        tensor = _prepare_places365_image(image)
        try:
            outputs = self._session.run(None, {self._input_name: tensor})
            logits = np.asarray(outputs[0]).squeeze()
        except Exception:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "Places365 inference failed",
            ) from None
        if logits.shape != (365,) or not np.isfinite(logits).all():
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "Places365 output is invalid",
            )
        top_indices = np.argpartition(-logits, self._top_k - 1)[: self._top_k]
        outdoor_vote = float(self._outdoor[top_indices].mean())
        return SceneDomain.VEHICLE if outdoor_vote >= 0.5 else SceneDomain.INDOOR


def create_scene_router(config: Mapping[str, object]) -> Places365SceneRouter:
    model_path = _model_path(config, "scene_router_model")
    labels_path = _model_path(config, "scene_router_io_labels")
    top_k = config.get("scene_router_top_k", 5)
    return Places365SceneRouter(model_path, labels_path, top_k=top_k)
