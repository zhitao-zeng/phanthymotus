"""TensorRT backends for the mixed indoor/vehicle obstacle model.

Indoor depth uses a fixed-shape Depth Anything V2 Metric Small engine. Vehicle
depth and instance segmentation use YOLO26 TensorRT engines through the
Ultralytics post-processing runtime.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Mapping, Sequence

import numpy as np

from .contracts import (
    DepthBackend,
    DepthPrediction,
    ErrorCode,
    InstanceMask,
    InstanceSegmentationBackend,
    ObstacleDistanceError,
    SceneDomain,
)


log = logging.getLogger(__name__)

_DAV2_INPUT_HEIGHT = 518
_DAV2_INPUT_WIDTH = 686
_DAV2_MULTIPLE = 14
_SEG_INPUT_SIZE = 640
_SEG_CONF_FLOOR = 0.05
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _check_deadline(deadline_monotonic: float) -> None:
    if deadline_monotonic > 0 and time.monotonic() >= deadline_monotonic:
        raise ObstacleDistanceError(ErrorCode.TIMEOUT, "model inference timed out")


def _decode_image(image_bytes: bytes) -> np.ndarray:
    import cv2

    try:
        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
    except Exception:
        image = None
    if image is None or image.size == 0:
        raise ObstacleDistanceError(
            ErrorCode.INVALID_IMAGE,
            "image bytes could not be decoded",
        )
    return image


def _constrain_to_multiple(value: float, minimum: int) -> int:
    constrained = int(np.round(value / _DAV2_MULTIPLE) * _DAV2_MULTIPLE)
    if constrained < minimum:
        constrained = int(np.ceil(value / _DAV2_MULTIPLE) * _DAV2_MULTIPLE)
    return constrained


def _prepare_dav2_image(image: np.ndarray) -> np.ndarray:
    """Apply the official lower-bound DAv2 preprocessing."""
    import cv2

    height, width = image.shape[:2]
    scale = max(_DAV2_INPUT_HEIGHT / height, 518 / width)
    resized_height = _constrain_to_multiple(
        scale * height,
        _DAV2_INPUT_HEIGHT,
    )
    resized_width = _constrain_to_multiple(scale * width, 518)
    if (resized_height, resized_width) != (
        _DAV2_INPUT_HEIGHT,
        _DAV2_INPUT_WIDTH,
    ):
        raise ObstacleDistanceError(
            ErrorCode.MODEL_ERROR,
            "indoor image aspect ratio is incompatible with the fixed DAv2 engine",
        )
    resized = cv2.resize(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0,
        (resized_width, resized_height),
        interpolation=cv2.INTER_CUBIC,
    )
    normalized = (resized - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.transpose(normalized, (2, 0, 1))
    return np.ascontiguousarray(chw, dtype=np.float32)[None]


def _letterbox(
    image: np.ndarray,
    size: int,
) -> tuple[np.ndarray, float, int, int]:
    import cv2

    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    unpad_h = int(round(height * ratio))
    unpad_w = int(round(width * ratio))
    dw = (size - unpad_w) // 2
    dh = (size - unpad_h) // 2
    resized = cv2.resize(
        image,
        (unpad_w, unpad_h),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[dh : dh + unpad_h, dw : dw + unpad_w] = resized
    return canvas, ratio, dw, dh


def _model_path(config: Mapping, key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value or not os.path.isfile(value):
        raise ObstacleDistanceError(
            ErrorCode.MODEL_ERROR,
            f"{key} does not identify a model file",
        )
    return value


class _TensorRTEngine:
    def __init__(self, path: str) -> None:
        import tensorrt as trt
        import torch

        self._trt = trt
        self._torch = torch
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        try:
            with open(path, "rb") as handle:
                serialized = handle.read()
        except OSError:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "TensorRT engine could not be read",
            ) from None
        self._engine = self._runtime.deserialize_cuda_engine(serialized)
        if self._engine is None:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "TensorRT engine could not be deserialized",
            )
        self._context = self._engine.create_execution_context()
        self._names = [
            self._engine.get_tensor_name(index)
            for index in range(self._engine.num_io_tensors)
        ]
        self._input_name = next(
            name
            for name in self._names
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        )
        self._output_names = [
            name
            for name in self._names
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        if len(self._output_names) != 1:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "DAv2 TensorRT engine must have exactly one output",
            )
        self._stream = torch.cuda.Stream()
        self._lock = threading.Lock()

    def _torch_dtype(self, tensor_name: str):
        dtype = np.dtype(
            self._trt.nptype(self._engine.get_tensor_dtype(tensor_name))
        )
        mapping = {
            np.dtype(np.float16): self._torch.float16,
            np.dtype(np.float32): self._torch.float32,
            np.dtype(np.int8): self._torch.int8,
            np.dtype(np.int32): self._torch.int32,
            np.dtype(bool): self._torch.bool,
        }
        try:
            return mapping[dtype]
        except KeyError:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                f"unsupported TensorRT tensor dtype: {dtype}",
            ) from None

    def infer(self, images):
        with self._lock:
            images = images.to(
                device="cuda",
                dtype=self._torch_dtype(self._input_name),
            ).contiguous()
            if not self._context.set_input_shape(
                self._input_name,
                tuple(images.shape),
            ):
                raise ObstacleDistanceError(
                    ErrorCode.MODEL_ERROR,
                    "TensorRT rejected the DAv2 input shape",
                )
            output_name = self._output_names[0]
            output = self._torch.empty(
                tuple(self._context.get_tensor_shape(output_name)),
                device="cuda",
                dtype=self._torch_dtype(output_name),
            )
            current_stream = self._torch.cuda.current_stream()
            self._stream.wait_stream(current_stream)
            with self._torch.cuda.stream(self._stream):
                self._context.set_tensor_address(
                    self._input_name,
                    int(images.data_ptr()),
                )
                self._context.set_tensor_address(
                    output_name,
                    int(output.data_ptr()),
                )
                executed = self._context.execute_async_v3(
                    self._stream.cuda_stream
                )
            if not executed:
                raise ObstacleDistanceError(
                    ErrorCode.MODEL_ERROR,
                    "TensorRT DAv2 inference failed",
                )
            current_stream.wait_stream(self._stream)
            return output


class HybridTensorRTDepthBackend:
    def __init__(self, indoor_engine: str, vehicle_engine: str) -> None:
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "hybrid TensorRT depth backend requires CUDA",
            )
        self._torch = torch
        self._indoor = _TensorRTEngine(indoor_engine)
        self._vehicle = YOLO(vehicle_engine, task="depth")
        self._vehicle_lock = threading.Lock()
        log.info(
            "[obstacle] hybrid TensorRT depth loaded indoor=%s vehicle=%s",
            indoor_engine,
            vehicle_engine,
        )

    def predict_depth(
        self,
        image_bytes: bytes,
        domain: SceneDomain,
        deadline_monotonic: float,
    ) -> DepthPrediction:
        _check_deadline(deadline_monotonic)
        image = _decode_image(image_bytes)
        height, width = image.shape[:2]
        if domain is SceneDomain.INDOOR:
            tensor = self._torch.from_numpy(_prepare_dav2_image(image)).cuda()
            depth = self._indoor.infer(tensor)
            depth = self._torch.nn.functional.interpolate(
                depth[:, None],
                (height, width),
                mode="bilinear",
                align_corners=True,
            )[0, 0].float().cpu().numpy()
        elif domain is SceneDomain.VEHICLE:
            with self._vehicle_lock:
                result = self._vehicle.predict(
                    image,
                    imgsz=768,
                    device=0,
                    verbose=False,
                )[0]
            if result.depth is None:
                raise ObstacleDistanceError(
                    ErrorCode.INVALID_DEPTH,
                    "YOLO depth model returned no depth map",
                )
            depth = result.depth.data.detach().float().cpu().numpy()
        else:
            raise ObstacleDistanceError(
                ErrorCode.MISSING_SCENE,
                "scene domain is invalid",
            )
        _check_deadline(deadline_monotonic)
        return DepthPrediction(
            depth_m=depth,
            source_height=height,
            source_width=width,
        )


class YoloTensorRTSegBackend:
    def __init__(self, engine: str) -> None:
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "YOLO TensorRT segmentation backend requires CUDA",
            )
        self._model = YOLO(engine, task="segment")
        self._lock = threading.Lock()
        log.info("[obstacle] YOLO TensorRT segmentation loaded %s", engine)

    def predict_instances(
        self,
        image_bytes: bytes,
        deadline_monotonic: float,
    ) -> Sequence[InstanceMask]:
        import cv2

        _check_deadline(deadline_monotonic)
        image = _decode_image(image_bytes)
        orig_h, orig_w = image.shape[:2]
        letterboxed, ratio, dw, dh = _letterbox(image, _SEG_INPUT_SIZE)
        with self._lock:
            result = self._model.predict(
                letterboxed,
                imgsz=_SEG_INPUT_SIZE,
                conf=_SEG_CONF_FLOOR,
                device=0,
                verbose=False,
            )[0]
        _check_deadline(deadline_monotonic)
        if result.masks is None:
            return ()

        boxes = result.boxes
        masks = result.masks.data.cpu().numpy()
        unpad_h = min(
            int(round(orig_h * ratio)),
            letterboxed.shape[0] - dh,
        )
        unpad_w = min(
            int(round(orig_w * ratio)),
            letterboxed.shape[1] - dw,
        )
        instances = []
        for index in range(len(boxes)):
            class_id = int(boxes.cls[index].item())
            confidence = float(boxes.conf[index].item())
            mask_crop = masks[index, dh : dh + unpad_h, dw : dw + unpad_w]
            mask = cv2.resize(
                (mask_crop > 0.5).astype(np.uint8),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            instances.append(
                InstanceMask(
                    class_name=str(result.names[class_id]),
                    confidence=confidence,
                    mask=mask,
                )
            )
        return tuple(instances)


def create_backends(
    config: Mapping,
) -> tuple[DepthBackend, InstanceSegmentationBackend]:
    indoor_engine = _model_path(config, "indoor_depth_engine")
    vehicle_engine = _model_path(config, "vehicle_depth_engine")
    segmentation_engine = _model_path(config, "segmentation_engine")
    return (
        HybridTensorRTDepthBackend(indoor_engine, vehicle_engine),
        YoloTensorRTSegBackend(segmentation_engine),
    )
