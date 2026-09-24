"""Fixed-camera DepthART Metric-S TensorRT backend for visual_depth."""

from __future__ import annotations

import ctypes
from pathlib import Path
import threading

import numpy as np


ENGINE = Path("/opt/vision-depth/depthart-metric-s-fp16.engine")
PLUGIN = Path("/opt/vision-depth/libdepthart_selective_scan_trt.so")
HEIGHT, WIDTH = 480, 864
MEAN = np.asarray((.485, .456, .406), dtype=np.float32)
STD = np.asarray((.229, .224, .225), dtype=np.float32)


def prepare_image(frame_bgr: np.ndarray, dtype) -> np.ndarray:
    """Match official DepthART RGB normalization and the fixed 480x864 graph."""
    import cv2

    height, width = frame_bgr.shape[:2]
    if (height, width) != (1080, 1920):
        raise ValueError(f"this engine expects 1920x1080, got {width}x{height}")
    image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    image = ((image - MEAN) / STD).transpose(2, 0, 1).copy()
    return np.ascontiguousarray(image[None], dtype=dtype)


def restore_align_corners(depth: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear output resize with align-corners coordinates, without PyTorch."""
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.shape != (HEIGHT, WIDTH):
        raise ValueError(f"unexpected DepthART engine output: {depth.shape}")
    height, width = output_hw
    xs = np.linspace(0, WIDTH - 1, width, dtype=np.float32)
    ys = np.linspace(0, HEIGHT - 1, height, dtype=np.float32)
    x0 = np.floor(xs).astype(np.intp)
    y0 = np.floor(ys).astype(np.intp)
    x1 = np.minimum(x0 + 1, WIDTH - 1)
    y1 = np.minimum(y0 + 1, HEIGHT - 1)
    ax = (xs - x0)[None]
    ay = (ys - y0)[:, None]
    top = depth[y0[:, None], x0[None, :]] * (1 - ax) + depth[y0[:, None], x1[None, :]] * ax
    bottom = depth[y1[:, None], x0[None, :]] * (1 - ax) + depth[y1[:, None], x1[None, :]] * ax
    return np.asarray(top * (1 - ay) + bottom * ay, dtype=np.float32)


class DepthARTSession:
    """One plugin-registered TensorRT engine behind the existing vision contract."""

    def __init__(self, engine_path: Path = ENGINE, plugin_path: Path = PLUGIN):
        engine_path = Path(engine_path)
        plugin_path = Path(plugin_path)
        if not engine_path.is_file() or not plugin_path.is_file():
            raise FileNotFoundError(f"DepthART engine/plugin missing: {engine_path}, {plugin_path}")
        self._plugin = ctypes.CDLL(str(plugin_path), mode=ctypes.RTLD_GLOBAL)
        self._plugin.depthart_selective_scan_trt_version.restype = ctypes.c_char_p
        version = self._plugin.depthart_selective_scan_trt_version().decode()
        if version != "SelectiveScan-1":
            raise RuntimeError(f"unexpected DepthART Selective Scan plugin: {version}")
        from utils.tensorrt_runtime import TensorRTEngine

        self._engine = TensorRTEngine(engine_path)
        if self._engine.input_shape != (1, 3, HEIGHT, WIDTH) or len(self._engine.output_names) != 1:
            self._engine.close()
            raise ValueError("DepthART engine does not match the fixed one-input graph")
        self._lock = threading.Lock()

    @property
    def input_size(self) -> tuple[int, int]:
        return WIDTH, HEIGHT

    def infer(self, frame: np.ndarray) -> tuple[list[np.ndarray], None]:
        blob = prepare_image(frame, self._engine.input_dtype)
        with self._lock:
            outputs = self._engine.infer(blob)
        depth = restore_align_corners(outputs[0], frame.shape[:2])
        return [depth], None

    def close(self) -> None:
        with self._lock:
            self._engine.close()
