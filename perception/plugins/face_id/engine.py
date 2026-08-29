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

_RUNTIME_PROFILES = {
    "mobile_cpu": {
        "backend": "opencv",
        "detector_backend": "opencv",
        "recognizer_backend": "opencv",
        "recognizer": "mobilefacenet",
    },
    "lvface_cpu": {
        "backend": "opencv",
        "detector_backend": "opencv",
        "recognizer_backend": "onnx",
        "recognizer": "lvface",
        "onnx_providers": ["CPUExecutionProvider"],
        "onnx_intra_op_threads": 1,
        "onnx_inter_op_threads": 1,
    },
}


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


def _normalize_backend_name(value: str) -> str:
    name = str(value).strip().lower()
    if name == "onnxruntime":
        return "onnx"
    if name in {"opencv-dnn", "cpu"}:
        return "opencv"
    return name


def _apply_runtime_profile(cfg: dict) -> dict:
    profile_name = str(cfg.get("runtime_profile") or "").strip().lower()
    if not profile_name:
        return dict(cfg)
    try:
        profile = _RUNTIME_PROFILES[profile_name]
    except KeyError as error:
        raise ValueError(f"unsupported face runtime profile: {profile_name}") from error
    resolved = {**cfg, **profile}
    if profile_name == "lvface_cpu" and not cfg.get("recognizer_model"):
        resolved["recognizer_model"] = str(
            Path(str(cfg.get("model_dir", "/models/face")))
            / "lvface_cpu"
            / "lvface_t_glint360k.onnx"
        )
    return resolved


class FaceIdentityEngine:
    """Thread-safe batch-one face identification pipeline."""

    def __init__(
        self,
        detector,
        recognizer,
        gallery: IdentityGallery,
        matcher: IdentityMatcher,
        *,
        face_selection: str = "primary",
        query_flip_tta: bool = False,
        bbox_x_scale: float = 1.0,
        bbox_y_scale: float = 1.0,
        bbox_y_shift: float = 0.0,
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.gallery = gallery
        self.matcher = matcher
        self.face_selection = str(face_selection).strip().lower()
        if self.face_selection not in {"primary", "gallery_match"}:
            raise ValueError(f"unsupported face selection: {face_selection}")
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
            if not detections:
                return empty_face_payload()
            if self.face_selection == "primary":
                detection = select_primary_face(detections, image.shape)
                match = self._match_detection(image, detection)
            else:
                detection, match = self._select_by_gallery(image, detections)
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

    def _embedding(self, image: np.ndarray, detection) -> np.ndarray:
        aligned = align_face(image, detection.landmarks)
        return self.recognizer.embed(aligned, flip_tta=self.query_flip_tta)

    def _match_detection(self, image: np.ndarray, detection):
        return self.matcher.match(self._embedding(image, detection))

    def _select_by_gallery(self, image: np.ndarray, detections):
        candidates = []
        for detection in detections:
            ranked = self.matcher.rank(self._embedding(image, detection), top_k=2)
            top = ranked[0]
            margin = top.score - ranked[1].score if len(ranked) > 1 else 0.0
            candidates.append((top.score, margin, detection.score, detection, top))
        _score, _margin, _detection_score, detection, match = max(
            candidates,
            key=lambda item: item[:3],
        )
        if (
            self.matcher.unknown_threshold is not None
            and match.score < self.matcher.unknown_threshold
        ):
            match = None
        return detection, match

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
    runtime_profile = os.environ.get("FACE_RUNTIME_PROFILE")
    if runtime_profile:
        cfg["runtime_profile"] = runtime_profile
    cfg = _apply_runtime_profile(cfg)
    common_backend = os.environ.get("FACE_BACKEND")
    if common_backend:
        cfg["backend"] = common_backend
        cfg["detector_backend"] = common_backend
        cfg["recognizer_backend"] = common_backend
    env_overrides = {
        "FACE_DETECTOR_BACKEND": "detector_backend",
        "FACE_RECOGNIZER_BACKEND": "recognizer_backend",
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
    backend_name = _normalize_backend_name(cfg.get("backend", "tensorrt"))
    detector_backend_name = _normalize_backend_name(
        cfg.get("detector_backend", backend_name)
    )
    recognizer_backend_name = _normalize_backend_name(
        cfg.get("recognizer_backend", backend_name)
    )
    recognizer_type = str(cfg.get("recognizer", "lvface")).strip().lower()
    recognizer_file = {
        "lvface": "lvface_t_glint360k",
        "lvface-t": "lvface_t_glint360k",
        "mobilefacenet": "mobilefacenet_webface600k",
        "mbf": "mobilefacenet_webface600k",
    }.get(recognizer_type)
    if recognizer_file is None:
        raise ValueError(f"unsupported face recognizer: {recognizer_type}")
    explicit_model_pair = _has_explicit_model_pair(cfg)
    trt_family = None
    if "tensorrt" in {detector_backend_name, recognizer_backend_name}:
        from utils.tensorrt_runtime import tensorrt_family

        trt_family = tensorrt_family()
        if not explicit_model_pair:
            from utils.model_downloader import ensure_face_model

            ensure_face_model(
                str(cfg.get("model_dir", "/models/face")), family=trt_family
            )
    if (
        "opencv" in {detector_backend_name, recognizer_backend_name}
        and not explicit_model_pair
    ):
        from utils.model_downloader import ensure_face_cpu_model

        ensure_face_cpu_model(str(cfg.get("model_dir", "/models/face")))
    if (
        recognizer_type in {"lvface", "lvface-t"}
        and recognizer_backend_name == "onnx"
        and not Path(str(cfg.get("recognizer_model", ""))).is_file()
    ):
        from utils.model_downloader import ensure_face_lvface_cpu_model

        ensure_face_lvface_cpu_model(str(cfg.get("model_dir", "/models/face")))
    detector_path = _resolve_model_path(
        cfg,
        key="detector_model",
        default_name="scrfd_500m_kps",
        backend=detector_backend_name,
        family=trt_family if detector_backend_name == "tensorrt" else None,
    )
    recognizer_path = _resolve_model_path(
        cfg,
        key="recognizer_model",
        default_name=recognizer_file,
        backend=recognizer_backend_name,
        family=trt_family if recognizer_backend_name == "tensorrt" else None,
    )
    gallery_dir = str(cfg.get("face_db_dir") or "/workspace/face_db")
    device_id = int(cfg.get("device_id", 0))
    providers = cfg.get("onnx_providers")
    onnx_intra_op_threads = cfg.get("onnx_intra_op_threads")
    onnx_inter_op_threads = cfg.get("onnx_inter_op_threads")
    detector_backend = build_backend(
        detector_backend_name,
        detector_path,
        device_id=device_id,
        providers=providers,
        intra_op_threads=(
            None if onnx_intra_op_threads is None else int(onnx_intra_op_threads)
        ),
        inter_op_threads=(
            None if onnx_inter_op_threads is None else int(onnx_inter_op_threads)
        ),
    )
    recognizer_backend = None
    detector = None
    recognizer = None
    try:
        recognizer_backend = build_backend(
            recognizer_backend_name,
            recognizer_path,
            device_id=device_id,
            providers=providers,
            intra_op_threads=(
                None if onnx_intra_op_threads is None else int(onnx_intra_op_threads)
            ),
            inter_op_threads=(
                None if onnx_inter_op_threads is None else int(onnx_inter_op_threads)
            ),
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
            face_selection=str(cfg.get("face_selection", "primary")),
            query_flip_tta=bool(cfg.get("query_flip_tta", False)),
            bbox_x_scale=float(bbox_cfg.get("x_scale", 1.0)),
            bbox_y_scale=float(bbox_cfg.get("y_scale", 1.0)),
            bbox_y_shift=float(bbox_cfg.get("y_shift", 0.0)),
        )
        log.info(
            "[face] engine ready: detector_backend=%s recognizer_backend=%s "
            "family=%s recognizer=%s gallery=%d",
            detector_backend_name,
            recognizer_backend_name,
            trt_family or "host",
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
