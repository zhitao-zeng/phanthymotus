"""Real-model backends: Lifelong-MonoDepth depth + YOLO26n-seg segmentation.

`create_backends` 是 `backend_factory` 入口（深度在前、分割在后）。
权重文件约定:
- 深度: `depth_model_dir/NSK_int8.pth`（per-channel INT8 weight-only, 加载时反量化回 FP16）
- 分割: `segmentation_model_dir/model.pt`（ultralytics YOLO26n-seg checkpoint）
"""

from __future__ import annotations

import logging
import os
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

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# NSK 训练顺序: NYU -> ScanNet -> KITTI, 故 head0=NYU(室内), head2=KITTI(车辆)
_DEPTH_HEAD_INDOOR = 0
_DEPTH_HEAD_VEHICLE = 2
_DEPTH_INPUT_WIDTH = 608
_DEPTH_INPUT_HEIGHT = 228

_SEG_INPUT_SIZE = 640
# backend 输出下限(全量候选), 最终置信度过滤由 estimator 的 vehicle.min_confidence 负责
_SEG_CONF_FLOOR = 0.05

_DEPTH_MODEL_NAMES = (
    "NSK_int8.pth",
    "NSK.pth.tar",
    "NKS.pth.tar",
    "NK.pth.tar",
    "model.pth",
)
_SEG_MODEL_NAMES = ("model.pt", "yolo26n-seg.pt")


def _monotonic() -> float:
    try:
        return float(time.monotonic())
    except Exception:
        return 0.0


def _check_deadline(deadline_monotonic: float) -> None:
    if deadline_monotonic > 0 and _monotonic() >= deadline_monotonic:
        raise ObstacleDistanceError(ErrorCode.TIMEOUT, "model inference timed out")


def _decode_image(image_bytes: bytes) -> np.ndarray:
    import cv2

    try:
        data = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        raise ObstacleDistanceError(
            ErrorCode.INVALID_IMAGE, "image bytes could not be decoded"
        ) from None
    if image is None or image.size == 0:
        raise ObstacleDistanceError(
            ErrorCode.INVALID_IMAGE, "image bytes could not be decoded"
        )
    return image


def _pick_model_file(directory: object, preferred: Sequence[str]) -> str | None:
    if not isinstance(directory, str) or not directory:
        return None
    if os.path.isfile(directory):
        return directory
    if not os.path.isdir(directory):
        return None
    for name in preferred:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate
    for name in sorted(os.listdir(directory)):
        if name.endswith((".pth", ".pth.tar", ".pt")):
            return os.path.join(directory, name)
    return None


def _default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _letterbox(
    image: np.ndarray, size: int
) -> tuple[np.ndarray, float, int, int]:
    import cv2

    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    unpad_h = int(round(height * ratio))
    unpad_w = int(round(width * ratio))
    dw = (size - unpad_w) // 2
    dh = (size - unpad_h) // 2
    resized = cv2.resize(
        image, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR
    )
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[dh : dh + unpad_h, dw : dw + unpad_w] = resized
    return canvas, ratio, dw, dh


class LifelongDepthBackend:
    """Lifelong-MonoDepth 深度后端 (NSK 权重, per-channel INT8 -> FP16 反量化)。"""

    def __init__(
        self,
        model_dir: object,
        device: object = None,
    ) -> None:
        self._device = device or _default_device()
        path = _pick_model_file(model_dir, _DEPTH_MODEL_NAMES)
        if path is None:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                f"depth model not found under {model_dir!r}",
            )
        self._model_path = path
        self._model = self._load(path)
        log.info(
            "[obstacle] depth backend loaded %s (device=%s)",
            path,
            self._device,
        )

    def _load(self, path: str):
        import torch

        from .lifelong_monodepth import modules, net, resnet

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        raw = checkpoint.get("state_dict", checkpoint)
        scales = checkpoint.get("scales")
        if scales is not None:
            state: dict = {}
            for key, value in raw.items():
                if key in scales:
                    scale = scales[key]
                    if value.dim() > 1:
                        scale = scale.view(-1, *([1] * (value.dim() - 1)))
                    state[key] = (value.float() * scale).half()
                else:
                    state[key] = value
        else:
            state = {
                key: (value.half() if value.is_floating_point() else value)
                for key, value in raw.items()
            }

        encoder = modules.E_resnet(resnet.resnet34(pretrained=False))
        backbone = net.backbone(
            encoder, num_features=512, block_channel=[64, 128, 256, 512]
        )
        model = net.model_ll(
            backbone, num_tasks=3, block_channel=[64, 128, 256, 512]
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            log.warning(
                "[obstacle] depth state_dict mismatch missing=%s unexpected=%s",
                missing,
                unexpected,
            )
        model.to(self._device).eval()
        return model

    def predict_depth(
        self,
        image_bytes: bytes,
        domain: SceneDomain,
        deadline_monotonic: float,
    ) -> DepthPrediction:
        _check_deadline(deadline_monotonic)
        if not isinstance(domain, SceneDomain):
            try:
                domain = SceneDomain(str(domain).strip().lower())
            except Exception:
                raise ObstacleDistanceError(
                    ErrorCode.MISSING_SCENE, "scene domain is invalid"
                ) from None
        head = (
            _DEPTH_HEAD_INDOOR
            if domain is SceneDomain.INDOOR
            else _DEPTH_HEAD_VEHICLE
        )

        image = _decode_image(image_bytes)
        orig_h, orig_w = image.shape[:2]

        import cv2
        import torch
        import torch.nn.functional as F

        resized = cv2.resize(
            image,
            (_DEPTH_INPUT_WIDTH, _DEPTH_INPUT_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(
            self._device
        )

        _check_deadline(deadline_monotonic)
        with torch.no_grad():
            outputs, _ = self._model(tensor)
        depth = outputs[head][0]
        uncertainty = outputs[head][1]
        depth = F.interpolate(
            depth, size=(orig_h, orig_w), mode="bilinear", align_corners=True
        )
        uncertainty = F.interpolate(
            uncertainty,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=True,
        )
        _check_deadline(deadline_monotonic)

        depth_map = depth.squeeze(0).squeeze(0).float().cpu().numpy()
        uncertainty_map = (
            uncertainty.squeeze(0).squeeze(0).float().cpu().numpy()
        )
        if not np.isfinite(depth_map).any():
            raise ObstacleDistanceError(
                ErrorCode.INVALID_DEPTH,
                "depth model returned no finite depth",
            )
        return DepthPrediction(
            depth_m=depth_map,
            source_height=orig_h,
            source_width=orig_w,
            uncertainty=uncertainty_map,
        )


class YoloSegBackend:
    """YOLO26n-seg 实例分割后端 (mask 逆 letterbox 回原图坐标系)。"""

    def __init__(
        self,
        model_dir: object,
        device: object = None,
    ) -> None:
        self._device = device or _default_device()
        path = _pick_model_file(model_dir, _SEG_MODEL_NAMES)
        if path is None:
            raise ObstacleDistanceError(
                ErrorCode.MODEL_ERROR,
                f"segmentation model not found under {model_dir!r}",
            )
        from ultralytics import YOLO

        self._model = YOLO(path, task="segment")
        self._names = self._model.names
        self._model_path = path
        log.info(
            "[obstacle] segmentation backend loaded %s (device=%s)",
            path,
            self._device,
        )

    def predict_instances(
        self,
        image_bytes: bytes,
        deadline_monotonic: float,
    ) -> Sequence[InstanceMask]:
        _check_deadline(deadline_monotonic)
        image = _decode_image(image_bytes)
        orig_h, orig_w = image.shape[:2]

        import cv2

        letterboxed, ratio, dw, dh = _letterbox(image, _SEG_INPUT_SIZE)
        _check_deadline(deadline_monotonic)
        results = self._model.predict(
            letterboxed,
            imgsz=_SEG_INPUT_SIZE,
            conf=_SEG_CONF_FLOOR,
            verbose=False,
            device=self._device,
        )
        _check_deadline(deadline_monotonic)
        if not results or results[0].masks is None:
            return ()

        result = results[0]
        boxes = result.boxes
        masks = result.masks.data.cpu().numpy()
        unpad_h = int(round(orig_h * ratio))
        unpad_w = int(round(orig_w * ratio))
        unpad_h = min(unpad_h, letterboxed.shape[0] - dh)
        unpad_w = min(unpad_w, letterboxed.shape[1] - dw)

        instances: list[InstanceMask] = []
        for index in range(len(boxes)):
            class_id = int(boxes.cls[index].item())
            confidence = float(boxes.conf[index].item())
            mask_crop = masks[index, dh : dh + unpad_h, dw : dw + unpad_w]
            mask_bool = cv2.resize(
                (mask_crop > 0.5).astype(np.uint8),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            instances.append(
                InstanceMask(
                    class_name=str(self._names[class_id]),
                    confidence=confidence,
                    mask=mask_bool,
                )
            )
        return instances


def create_backends(
    config: Mapping,
) -> tuple[DepthBackend, InstanceSegmentationBackend]:
    """backend_factory: 返回 (深度后端, 分割后端)。"""
    depth_device = config.get("depth_device") or config.get("device")
    segmentation_device = config.get(
        "segmentation_device"
    ) or config.get("device")
    depth_backend = LifelongDepthBackend(
        config.get("depth_model_dir"), depth_device
    )
    segmentation_backend = YoloSegBackend(
        config.get("segmentation_model_dir"), segmentation_device
    )
    return depth_backend, segmentation_backend
