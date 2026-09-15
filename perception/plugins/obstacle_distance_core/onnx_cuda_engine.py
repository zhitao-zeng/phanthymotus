"""CUDA ONNX engine for a fixed-shape indoor metric depth model."""

from __future__ import annotations

import threading

import numpy as np

from .contracts import ErrorCode, ObstacleDistanceError


class CudaOnnxEngine:
    def __init__(self, path: str) -> None:
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise ObstacleDistanceError(ErrorCode.MODEL_ERROR, "ONNX CUDA provider is unavailable")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        cuda_options = {
            "device_id": 0,
            "gpu_mem_limit": 512 * 1024**2,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
            "cudnn_conv_use_max_workspace": "0",
            "do_copy_in_default_stream": "1",
            "use_tf32": "0",
        }
        session = ort.InferenceSession(
            path, sess_options=options,
            providers=[("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"],
        )
        session.disable_fallback()
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise ObstacleDistanceError(ErrorCode.MODEL_ERROR, "ONNX CUDA provider failed to initialize")
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if (len(inputs) != 1 or inputs[0].type != "tensor(float)"
                or inputs[0].shape != [1, 3, 384, 512]
                or len(outputs) != 1 or outputs[0].shape != [1, 1, 384, 512]
                or outputs[0].type != "tensor(float)"):
            raise ObstacleDistanceError(ErrorCode.MODEL_ERROR, "ONNX indoor model shape or dtype is incompatible")
        self.input_name = inputs[0].name
        self.input_shape = tuple(inputs[0].shape)
        self.output_names = [outputs[0].name]
        self._lock = threading.Lock()
        self._session = session

    def infer(self, image: np.ndarray) -> tuple[np.ndarray, ...]:
        if tuple(image.shape) != self.input_shape:
            raise ObstacleDistanceError(ErrorCode.MODEL_ERROR, "ONNX indoor input shape is incompatible")
        with self._lock:
            if self._session is None:
                raise ObstacleDistanceError(ErrorCode.MODEL_ERROR, "ONNX indoor engine is closed")
            return tuple(self._session.run(self.output_names, {
                self.input_name: np.ascontiguousarray(image, dtype=np.float32),
            }))

    def close(self) -> None:
        with self._lock:
            self._session = None
