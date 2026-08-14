"""Low-RSS TensorRT runtime using CUDA Runtime, NumPy and OpenCV only."""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .contracts import (
    ErrorCode,
    InstanceMask,
    ObstacleDistanceError,
)
from .runtime_utils import (
    SEG_CONF_FLOOR,
    SEG_INPUT_SIZE,
    check_deadline,
    decode_image,
    letterbox,
)


log = logging.getLogger(__name__)

_CUDA_MEMCPY_HOST_TO_DEVICE = 1
_CUDA_MEMCPY_DEVICE_TO_HOST = 2
_YOLO_DEPTH_INPUT_SIZE = 768


class _CudaRuntime:
    """Small checked wrapper around the CUDA Runtime C API."""

    def __init__(self) -> None:
        library = ctypes.util.find_library("cudart") or "libcudart.so"
        try:
            self._lib = ctypes.CDLL(library)
        except OSError:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "CUDA Runtime library could not be loaded",
            ) from None

        self._lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self._lib.cudaGetErrorString.restype = ctypes.c_char_p
        self._lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self._lib.cudaSetDevice.restype = ctypes.c_int
        self._lib.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self._lib.cudaMalloc.restype = ctypes.c_int
        self._lib.cudaFree.argtypes = [ctypes.c_void_p]
        self._lib.cudaFree.restype = ctypes.c_int
        self._lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._lib.cudaMemcpyAsync.restype = ctypes.c_int
        self._lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._lib.cudaStreamCreate.restype = ctypes.c_int
        self._lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self._lib.cudaStreamDestroy.restype = ctypes.c_int
        self._lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self._lib.cudaStreamSynchronize.restype = ctypes.c_int
        self._check(self._lib.cudaSetDevice(0), "cudaSetDevice")

    def _check(self, status: int, operation: str) -> None:
        if status == 0:
            return
        message = self._lib.cudaGetErrorString(status)
        detail = message.decode("utf-8", errors="replace") if message else str(status)
        raise ObstacleDistanceError(
            ErrorCode.MODEL_ERROR,
            f"{operation} failed: {detail}",
        )

    def malloc(self, size: int) -> int:
        pointer = ctypes.c_void_p()
        self._check(
            self._lib.cudaMalloc(ctypes.byref(pointer), size),
            "cudaMalloc",
        )
        return int(pointer.value)

    def free(self, pointer: int) -> None:
        if pointer:
            self._check(self._lib.cudaFree(ctypes.c_void_p(pointer)), "cudaFree")

    def create_stream(self) -> int:
        stream = ctypes.c_void_p()
        self._check(
            self._lib.cudaStreamCreate(ctypes.byref(stream)),
            "cudaStreamCreate",
        )
        return int(stream.value)

    def destroy_stream(self, stream: int) -> None:
        if stream:
            self._check(
                self._lib.cudaStreamDestroy(ctypes.c_void_p(stream)),
                "cudaStreamDestroy",
            )

    def copy_host_to_device(self, destination: int, source: np.ndarray, stream: int) -> None:
        self._check(
            self._lib.cudaMemcpyAsync(
                ctypes.c_void_p(destination),
                ctypes.c_void_p(source.ctypes.data),
                source.nbytes,
                _CUDA_MEMCPY_HOST_TO_DEVICE,
                ctypes.c_void_p(stream),
            ),
            "cudaMemcpyAsync(H2D)",
        )

    def copy_device_to_host(self, destination: np.ndarray, source: int, stream: int) -> None:
        self._check(
            self._lib.cudaMemcpyAsync(
                ctypes.c_void_p(destination.ctypes.data),
                ctypes.c_void_p(source),
                destination.nbytes,
                _CUDA_MEMCPY_DEVICE_TO_HOST,
                ctypes.c_void_p(stream),
            ),
            "cudaMemcpyAsync(D2H)",
        )

    def synchronize(self, stream: int) -> None:
        self._check(
            self._lib.cudaStreamSynchronize(ctypes.c_void_p(stream)),
            "cudaStreamSynchronize",
        )


def _read_engine(path: str) -> tuple[dict, bytes]:
    try:
        serialized = Path(path).read_bytes()
    except OSError:
        raise ObstacleDistanceError(
            ErrorCode.MODEL_ERROR,
            "TensorRT engine could not be read",
        ) from None
    if len(serialized) < 8:
        return {}, serialized
    metadata_size = int.from_bytes(serialized[:4], "little", signed=True)
    if metadata_size <= 0 or metadata_size > min(len(serialized) - 4, 1 << 20):
        return {}, serialized
    try:
        metadata = json.loads(serialized[4 : 4 + metadata_size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}, serialized
    if not isinstance(metadata, dict) or "task" not in metadata:
        return {}, serialized
    return metadata, serialized[4 + metadata_size :]


class _NativeTensorRTEngine:
    """Fixed-shape TensorRT execution context with reusable CUDA buffers."""

    def __init__(self, path: str, expected_task: str | None = None) -> None:
        import tensorrt as trt

        self._trt = trt
        self.metadata, serialized = _read_engine(path)
        if expected_task and self.metadata.get("task") != expected_task:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                f"TensorRT engine task must be {expected_task}",
            )
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(serialized)
        del serialized
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
        inputs = [
            name
            for name in self._names
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        if len(inputs) != 1:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "TensorRT engine must have exactly one input",
            )
        self.input_name = inputs[0]
        self.output_names = [
            name
            for name in self._names
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        self.input_shape = tuple(self._engine.get_tensor_shape(self.input_name))
        if any(dimension <= 0 for dimension in self.input_shape):
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "native TensorRT backend requires fixed input shapes",
            )
        if not self._context.set_input_shape(self.input_name, self.input_shape):
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "TensorRT rejected its static input shape",
            )
        self.input_dtype = np.dtype(
            trt.nptype(self._engine.get_tensor_dtype(self.input_name))
        )
        self.output_shapes = {
            name: tuple(self._context.get_tensor_shape(name))
            for name in self.output_names
        }
        self.output_dtypes = {
            name: np.dtype(trt.nptype(self._engine.get_tensor_dtype(name)))
            for name in self.output_names
        }
        if any(
            dimension <= 0
            for shape in self.output_shapes.values()
            for dimension in shape
        ):
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "native TensorRT backend requires fixed output shapes",
            )

        self._cuda = _CudaRuntime()
        self._stream = self._cuda.create_stream()
        input_nbytes = int(np.prod(self.input_shape)) * self.input_dtype.itemsize
        self._device_buffers = {
            self.input_name: self._cuda.malloc(input_nbytes),
        }
        self._host_outputs = {}
        try:
            for name in self.output_names:
                output = np.empty(
                    self.output_shapes[name],
                    dtype=self.output_dtypes[name],
                )
                self._host_outputs[name] = output
                self._device_buffers[name] = self._cuda.malloc(output.nbytes)
            for name, pointer in self._device_buffers.items():
                self._context.set_tensor_address(name, pointer)
        except Exception:
            self.close()
            raise
        self._lock = threading.Lock()

    def infer(self, image: np.ndarray) -> tuple[np.ndarray, ...]:
        image = np.ascontiguousarray(image, dtype=self.input_dtype)
        if tuple(image.shape) != self.input_shape:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                f"TensorRT input shape mismatch: {image.shape} != {self.input_shape}",
            )
        with self._lock:
            self._cuda.copy_host_to_device(
                self._device_buffers[self.input_name],
                image,
                self._stream,
            )
            executed = self._context.execute_async_v3(self._stream)
            if not executed:
                raise ObstacleDistanceError(
                    ErrorCode.MODEL_ERROR,
                    "TensorRT execute_async_v3 failed",
                )
            for name in self.output_names:
                self._cuda.copy_device_to_host(
                    self._host_outputs[name],
                    self._device_buffers[name],
                    self._stream,
                )
            self._cuda.synchronize(self._stream)
            return tuple(self._host_outputs[name].copy() for name in self.output_names)

    def close(self) -> None:
        buffers = getattr(self, "_device_buffers", {})
        cuda = getattr(self, "_cuda", None)
        if cuda is not None:
            for pointer in buffers.values():
                try:
                    cuda.free(pointer)
                except Exception:
                    pass
            stream = getattr(self, "_stream", 0)
            try:
                cuda.destroy_stream(stream)
            except Exception:
                pass
        self._device_buffers = {}
        self._stream = 0

    def __del__(self) -> None:
        self.close()


def _prepare_yolo_image(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    letterboxed, ratio, dw, dh = letterbox(image, size)
    rgb = letterboxed[:, :, ::-1]
    chw = np.transpose(rgb, (2, 0, 1))
    tensor = np.ascontiguousarray(chw, dtype=np.float32)[None] / 255.0
    return tensor, ratio, dw, dh


def _resize_align_corners(image: np.ndarray, height: int, width: int) -> np.ndarray:
    import cv2

    if image.shape == (height, width):
        return image.astype(np.float32, copy=False)
    source_height, source_width = image.shape
    x = np.linspace(0, source_width - 1, width, dtype=np.float32)
    y = np.linspace(0, source_height - 1, height, dtype=np.float32)
    map_x, map_y = np.meshgrid(x, y)
    return cv2.remap(
        image.astype(np.float32, copy=False),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _scale_depth_to_original(
    depth: np.ndarray,
    original_height: int,
    original_width: int,
) -> np.ndarray:
    import cv2

    input_height, input_width = depth.shape
    gain = min(input_height / original_height, input_width / original_width)
    pad_w = (input_width - round(original_width * gain)) / 2
    pad_h = (input_height - round(original_height * gain)) / 2
    top = round(pad_h - 0.1)
    left = round(pad_w - 0.1)
    bottom = input_height - round(pad_h + 0.1)
    right = input_width - round(pad_w + 0.1)
    cropped = depth[top:bottom, left:right]
    return cv2.resize(
        cropped.astype(np.float32, copy=False),
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )


def _process_yolo_masks(
    detections: np.ndarray,
    prototypes: np.ndarray,
    original_height: int,
    original_width: int,
    ratio: float,
    dw: int,
    dh: int,
    *,
    confidence_floor: float = SEG_CONF_FLOOR,
    allowed_class_ids: frozenset[int] | None = None,
) -> list[tuple[int, float, np.ndarray]]:
    import cv2

    detections = np.asarray(detections, dtype=np.float32)
    prototypes = np.asarray(prototypes, dtype=np.float32)
    if detections.ndim == 3:
        detections = detections[0]
    if prototypes.ndim == 4:
        prototypes = prototypes[0]
    selected_mask = detections[:, 4] > SEG_CONF_FLOOR
    selected_mask &= detections[:, 4] >= confidence_floor
    if allowed_class_ids is not None:
        selected_mask &= np.isin(
            detections[:, 5].astype(np.int64),
            tuple(allowed_class_ids),
        )
    selected = detections[selected_mask]
    if not len(selected):
        return []
    channels, mask_height, mask_width = prototypes.shape
    coefficients = selected[:, 6 : 6 + channels]
    logits = (coefficients @ prototypes.reshape(channels, -1)).reshape(
        -1,
        mask_height,
        mask_width,
    )
    unpad_height = min(
        int(round(original_height * ratio)),
        SEG_INPUT_SIZE - dh,
    )
    unpad_width = min(
        int(round(original_width * ratio)),
        SEG_INPUT_SIZE - dw,
    )
    rows = np.arange(SEG_INPUT_SIZE, dtype=np.float32)[:, None]
    columns = np.arange(SEG_INPUT_SIZE, dtype=np.float32)[None, :]
    results = []
    for detection, logit in zip(selected, logits):
        upsampled = cv2.resize(
            logit,
            (SEG_INPUT_SIZE, SEG_INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        x1, y1, x2, y2 = detection[:4]
        mask = upsampled > 0.0
        mask &= columns >= x1
        mask &= columns < x2
        mask &= rows >= y1
        mask &= rows < y2
        if not mask.any():
            continue
        mask = mask[dh : dh + unpad_height, dw : dw + unpad_width]
        mask = cv2.resize(
            mask.astype(np.uint8),
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        results.append((int(detection[5]), float(detection[4]), mask))
    return results


class NativeTensorRTSegBackend:
    def __init__(
        self,
        engine: str,
        *,
        allowed_classes: object = None,
        min_confidence: object = None,
    ) -> None:
        self._engine_path = engine
        metadata, _ = _read_engine(engine)
        if metadata.get("task") != "segment":
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "TensorRT engine task must be segment",
            )
        names = metadata.get("names", {})
        if not isinstance(names, dict):
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "YOLO segmentation metadata does not contain class names",
            )
        self._names = {int(key): str(value) for key, value in names.items()}
        self._allowed_class_ids = self._configured_class_ids(allowed_classes)
        self._confidence_floor = self._configured_confidence(min_confidence)
        # The engine itself is loaded lazily on the first vehicle frame so an
        # indoor-only judge run never pays for segmentation residency.
        self._model: _NativeTensorRTEngine | None = None
        self._engine_init_lock = threading.Lock()
        log.info("[obstacle] native TensorRT segmentation configured %s", engine)

    def _get_model(self) -> _NativeTensorRTEngine:
        model = self._model
        if model is None:
            with self._engine_init_lock:
                if self._model is None:
                    started = time.monotonic()
                    model = _NativeTensorRTEngine(
                        self._engine_path,
                        expected_task="segment",
                    )
                    if model.input_shape != (
                        1,
                        3,
                        SEG_INPUT_SIZE,
                        SEG_INPUT_SIZE,
                    ):
                        raise ObstacleDistanceError(
                            ErrorCode.MODEL_ERROR,
                            "YOLO segmentation TensorRT engine input shape is incompatible",
                        )
                    self._model = model
                    log.info(
                        "[obstacle] native TensorRT segmentation loaded %s "
                        "in %.1fms",
                        self._engine_path,
                        1000.0 * (time.monotonic() - started),
                    )
                model = self._model
        return model

    def _configured_class_ids(
        self,
        allowed_classes: object,
    ) -> frozenset[int] | None:
        if not isinstance(allowed_classes, (list, tuple, set, frozenset)):
            return None
        if not allowed_classes or any(
            not isinstance(value, str) for value in allowed_classes
        ):
            return None
        allowed = set(allowed_classes)
        return frozenset(
            class_id
            for class_id, class_name in self._names.items()
            if class_name in allowed
        )

    @staticmethod
    def _configured_confidence(min_confidence: object) -> float:
        if (
            isinstance(min_confidence, bool)
            or not isinstance(min_confidence, (int, float))
        ):
            return SEG_CONF_FLOOR
        confidence = float(min_confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            return SEG_CONF_FLOOR
        return max(SEG_CONF_FLOOR, confidence)

    def predict_instances(
        self,
        image_bytes: bytes,
        deadline_monotonic: float,
    ) -> Sequence[InstanceMask]:
        check_deadline(deadline_monotonic)
        image = decode_image(image_bytes)
        height, width = image.shape[:2]
        tensor, ratio, dw, dh = _prepare_yolo_image(image, SEG_INPUT_SIZE)
        outputs = self._get_model().infer(tensor)
        detection_outputs = [output for output in outputs if output.ndim == 3]
        prototype_outputs = [output for output in outputs if output.ndim == 4]
        if len(detection_outputs) != 1 or len(prototype_outputs) != 1:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                "YOLO segmentation engine outputs are incompatible",
            )
        processed = _process_yolo_masks(
            detection_outputs[0],
            prototype_outputs[0],
            height,
            width,
            ratio,
            dw,
            dh,
            confidence_floor=self._confidence_floor,
            allowed_class_ids=self._allowed_class_ids,
        )
        check_deadline(deadline_monotonic)
        return tuple(
            InstanceMask(
                class_name=self._names.get(class_id, str(class_id)),
                confidence=confidence,
                mask=mask,
            )
            for class_id, confidence, mask in processed
        )
