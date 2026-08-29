"""End-to-end face detection, alignment, embedding and gallery matching."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np

from .alignment import align_face
from .backends import build_backend
from .detector import SCRFDDetector
from .gallery import GalleryQualityConfig, IdentityGallery, select_primary_face
from .matcher import IdentityMatcher
from .postprocess import face_payload
from .recognizer import FaceRecognizer
from .schema import empty_face_payload

log = logging.getLogger(__name__)


def _pair(value, *, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a pair")
    return int(value[0]), int(value[1])


def _resolve_model_path(
    cfg: dict,
    *,
    key: str,
    default_name: str,
    backend: str,
    family: str | None,
) -> str:
    explicit = cfg.get(key)
    if explicit:
        path = str(explicit)
        if family:
            path = path.format(family=family)
    else:
        model_dir = Path(str(cfg.get("model_dir", "/models/face")))
        if backend == "tensorrt":
            path = str(model_dir / str(family) / f"{default_name}.engine")
        elif backend == "opencv":
            path = str(model_dir / "cpu" / f"{default_name}.onnx")
        else:
            path = str(model_dir / f"{default_name}.onnx")
    if not Path(path).is_file():
        raise FileNotFoundError(f"face model does not exist: {path}")
    return path


def _has_explicit_model_pair(cfg: dict) -> bool:
    return bool(cfg.get("detector_model") and cfg.get("recognizer_model"))


class FaceIdentityEngine:
    """Thread-safe batch-one face identification pipeline."""

    def __init__(
        self,
        detector,
        recognizer,
        gallery: IdentityGallery,
        matcher: IdentityMatcher,
        *,
        query_flip_tta: bool = False,
        bbox_x_scale: float = 1.0,
        bbox_y_scale: float = 1.0,
        bbox_y_shift: float = 0.0,
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.gallery = gallery
        self.matcher = matcher
        self.query_flip_tta = bool(query_flip_tta)
        self.bbox_x_scale = float(bbox_x_scale)
        self.bbox_y_scale = float(bbox_y_scale)
        self.bbox_y_shift = float(bbox_y_shift)
        self._lock = threading.Lock()
        self._closed = False

    def infer_face_identity(self, image_bytes: bytes) -> dict:
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("could not decode JPEG/PNG face input")
        return self.infer_image(image)

    def infer_image(self, image: np.ndarray) -> dict:
        with self._lock:
            if self._closed:
                raise RuntimeError("face identity engine is closed")
            detections = self.detector.detect(image)
            detection = select_primary_face(detections, image.shape)
            if detection is None:
                return empty_face_payload()
            aligned = align_face(image, detection.landmarks)
            embedding = self.recognizer.embed(
                aligned,
                flip_tta=self.query_flip_tta,
            )
            match = self.matcher.match(embedding)
            confidence = None if match is None else self.matcher.confidence(match.score)
            return face_payload(
                detection,
                match,
                image.shape,
                x_scale=self.bbox_x_scale,
                y_scale=self.bbox_y_scale,
                y_shift=self.bbox_y_shift,
                match_confidence=confidence,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.recognizer.close()
            self.detector.close()


def build_face_engine(cfg: dict) -> FaceIdentityEngine:
    """Build a real local face engine from explicit model files and gallery."""

    cfg = dict(cfg)
    env_overrides = {
        "FACE_BACKEND": "backend",
        "FACE_MODEL_DIR": "model_dir",
        "FACE_DETECTOR_MODEL": "detector_model",
        "FACE_RECOGNIZER_MODEL": "recognizer_model",
        "FACE_RECOGNIZER": "recognizer",
        "FACE_DB_DIR": "face_db_dir",
    }
    for env_name, config_name in env_overrides.items():
        value = os.environ.get(env_name)
        if value:
            cfg[config_name] = value
    backend_name = str(cfg.get("backend", "tensorrt")).strip().lower()
    if backend_name == "onnxruntime":
        backend_name = "onnx"
    if backend_name in {"opencv-dnn", "cpu"}:
        backend_name = "opencv"
    explicit_model_pair = _has_explicit_model_pair(cfg)
    family = None
    if backend_name == "tensorrt":
        from utils.tensorrt_runtime import tensorrt_family

        family = tensorrt_family()
        if not explicit_model_pair:
            from utils.model_downloader import ensure_face_model

            ensure_face_model(str(cfg.get("model_dir", "/models/face")), family=family)
    elif backend_name == "opencv":
        if not explicit_model_pair:
            from utils.model_downloader import ensure_face_cpu_model

            ensure_face_cpu_model(str(cfg.get("model_dir", "/models/face")))
        family = "cpu"
    recognizer_type = str(cfg.get("recognizer", "lvface")).strip().lower()
    recognizer_file = {
        "lvface": "lvface_t_glint360k",
        "lvface-t": "lvface_t_glint360k",
        "mobilefacenet": "mobilefacenet_webface600k",
        "mbf": "mobilefacenet_webface600k",
    }.get(recognizer_type)
    if recognizer_file is None:
        raise ValueError(f"unsupported face recognizer: {recognizer_type}")
    detector_path = _resolve_model_path(
        cfg,
        key="detector_model",
        default_name="scrfd_500m_kps",
        backend=backend_name,
        family=family,
    )
    recognizer_path = _resolve_model_path(
        cfg,
        key="recognizer_model",
        default_name=recognizer_file,
        backend=backend_name,
        family=family,
    )
    gallery_dir = str(cfg.get("face_db_dir") or "/workspace/face_db")
    device_id = int(cfg.get("device_id", 0))
    providers = cfg.get("onnx_providers")
    detector_backend = build_backend(
        backend_name,
        detector_path,
        device_id=device_id,
        providers=providers,
    )
    recognizer_backend = None
    detector = None
    recognizer = None
    try:
        recognizer_backend = build_backend(
            backend_name,
            recognizer_path,
            device_id=device_id,
            providers=providers,
        )
        detector = SCRFDDetector(
            detector_backend,
            input_size=_pair(cfg.get("detector_input_size", [640, 640]), name="detector_input_size"),
            score_threshold=float(cfg.get("detector_score_threshold", 0.2)),
            nms_threshold=float(cfg.get("detector_nms_threshold", 0.4)),
        )
        recognizer = FaceRecognizer(
            recognizer_backend,
            model_type=recognizer_type,
        )
        quality_cfg = dict(cfg.get("gallery_quality") or {})
        quality = GalleryQualityConfig(
            min_detection_score=float(
                quality_cfg.get("min_detection_score", cfg.get("detector_score_threshold", 0.2))
            ),
            min_face_size=float(quality_cfg.get("min_face_size", 24.0)),
            min_eye_distance=float(quality_cfg.get("min_eye_distance", 8.0)),
            min_blur_variance=float(quality_cfg.get("min_blur_variance", 0.0)),
            flip_tta=bool(quality_cfg.get("flip_tta", True)),
            max_subcenters=int(quality_cfg.get("max_subcenters", 3)),
            subcenter_min_images=int(quality_cfg.get("subcenter_min_images", 4)),
        )
        gallery = IdentityGallery.build(
            gallery_dir,
            detector,
            recognizer,
            quality=quality,
        )
        unknown = cfg.get("unknown_threshold")
        matcher = IdentityMatcher(
            gallery,
            centroid_weight=float(cfg.get("centroid_weight", 0.6)),
            unknown_threshold=None if unknown in (None, "") else float(unknown),
        )
        bbox_cfg = dict(cfg.get("bbox_calibration") or {})
        engine = FaceIdentityEngine(
            detector,
            recognizer,
            gallery,
            matcher,
            query_flip_tta=bool(cfg.get("query_flip_tta", False)),
            bbox_x_scale=float(bbox_cfg.get("x_scale", 1.0)),
            bbox_y_scale=float(bbox_cfg.get("y_scale", 1.0)),
            bbox_y_shift=float(bbox_cfg.get("y_shift", 0.0)),
        )
        log.info(
            "[face] engine ready: backend=%s family=%s recognizer=%s gallery=%d",
            backend_name,
            family or "host",
            recognizer_type,
            len(gallery.templates),
        )
        return engine
    except Exception:
        if recognizer is not None:
            recognizer.close()
        elif recognizer_backend is not None:
            recognizer_backend.close()
        if detector is not None:
            detector.close()
        else:
            detector_backend.close()
        raise
