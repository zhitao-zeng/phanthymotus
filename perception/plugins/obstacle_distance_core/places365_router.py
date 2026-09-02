"""Pixel-content indoor/vehicle routing with Places365 labels."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .contracts import (
    ErrorCode,
    ObstacleDistanceError,
    SceneDomain,
    SceneRouteDecision,
)
from .runtime_utils import decode_image, model_path


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
        model_path_value: str,
        io_labels_path: str,
        *,
        top_k: int = 3,
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
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            self._session = ort.InferenceSession(
                model_path_value,
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

    def _decision_from_logits(self, logits: np.ndarray) -> SceneRouteDecision:
        if logits.shape != (365,) or not np.isfinite(logits).all():
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "Places365 output is invalid",
            )
        top_indices = np.argpartition(-logits, self._top_k - 1)[: self._top_k]
        top_indices = top_indices[np.argsort(-logits[top_indices])]
        outdoor_vote = float(self._outdoor[top_indices].mean())
        shifted = logits - float(logits.max())
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        top_two = np.argpartition(-probabilities, 1)[:2]
        top_two = top_two[np.argsort(-probabilities[top_two])]
        top1_index, top2_index = (int(value) for value in top_two)
        top1_probability = float(probabilities[top1_index])
        top2_probability = float(probabilities[top2_index])
        outdoor_probability = float(probabilities[self._outdoor == 1].sum())
        domain = (
            SceneDomain.VEHICLE
            if outdoor_vote >= 0.5
            else SceneDomain.INDOOR
        )
        confidence = (
            outdoor_probability
            if domain is SceneDomain.VEHICLE
            else 1.0 - outdoor_probability
        )
        return SceneRouteDecision(
            domain=domain,
            confidence=confidence,
            top1_index=top1_index,
            top1_probability=top1_probability,
            top2_index=top2_index,
            top2_probability=top2_probability,
            top1_top2_margin=top1_probability - top2_probability,
            outdoor_vote=outdoor_vote,
            outdoor_probability=outdoor_probability,
        )

    def predict_decision(self, image_bytes: bytes) -> SceneRouteDecision:
        image = decode_image(image_bytes)
        tensor = _prepare_places365_image(image)
        try:
            outputs = self._session.run(None, {self._input_name: tensor})
            logits = np.asarray(outputs[0]).squeeze()
        except Exception:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "Places365 inference failed",
            ) from None
        return self._decision_from_logits(logits)

    def predict(self, image_bytes: bytes) -> SceneDomain:
        return self.predict_decision(image_bytes).domain


def create_scene_router(config: Mapping[str, object]) -> Places365SceneRouter:
    router_model = model_path(config, "scene_router_model")
    labels_path = model_path(config, "scene_router_io_labels")
    top_k = config.get("scene_router_top_k", 3)
    return Places365SceneRouter(router_model, labels_path, top_k=top_k)
