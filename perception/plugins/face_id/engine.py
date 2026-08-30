"""End-to-end face detection, alignment, embedding and gallery matching."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np

from .alignment import align_face
from .backends import build_backend
from .detector import SCRFDDetector
from .diagnostics import detection_quality, ranked_matches
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
        diagnostics_enabled: bool = False,
        diagnostics_top_k: int = 5,
        diagnostics_retry_thresholds: tuple[float, ...] = (),
        diagnostics_max_retry_candidates: int = 5,
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
        self.diagnostics_enabled = bool(diagnostics_enabled)
        self.diagnostics_top_k = int(diagnostics_top_k)
        if self.diagnostics_top_k < 2:
            raise ValueError("diagnostics_top_k must be at least two")
        self.diagnostics_retry_thresholds = tuple(
            float(value) for value in diagnostics_retry_thresholds
        )
        self.diagnostics_max_retry_candidates = int(
            diagnostics_max_retry_candidates
        )
        if self.diagnostics_max_retry_candidates < 1:
            raise ValueError("diagnostics_max_retry_candidates must be positive")
        self.bbox_x_scale = float(bbox_x_scale)
        self.bbox_y_scale = float(bbox_y_scale)
        self.bbox_y_shift = float(bbox_y_shift)
        self._lock = threading.Lock()
        self._closed = False
        self._diagnostic_sequence = 0

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
            self._diagnostic_sequence += 1
            sequence = self._diagnostic_sequence
            detections = self.detector.detect(image)
            if not detections:
                if self.diagnostics_enabled:
                    self._log_empty_diagnostics(sequence, image)
                return empty_face_payload()
            if self.face_selection == "primary":
                detection = select_primary_face(detections, image.shape)
                selected_index = next(
                    index
                    for index, candidate in enumerate(detections)
                    if candidate is detection
                )
                ranked, diagnostic = self._analyze_detection(
                    image,
                    detection,
                    selected_index,
                )
                match = self._accepted_match(ranked[0])
                diagnostic_candidates = [diagnostic]
            else:
                (
                    detection,
                    match,
                    selected_index,
                    diagnostic_candidates,
                ) = self._select_by_gallery(image, detections)
            if self.diagnostics_enabled:
                self._log_diagnostics(
                    sequence,
                    image,
                    detections,
                    selected_index,
                    diagnostic_candidates,
                )
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

    def _embedding(self, image: np.ndarray, detection) -> tuple[np.ndarray, np.ndarray]:
        aligned = align_face(image, detection.landmarks)
        embedding = self.recognizer.embed(
            aligned,
            flip_tta=self.query_flip_tta,
        )
        return embedding, aligned

    def _analyze_detection(self, image: np.ndarray, detection, index: int):
        embedding, aligned = self._embedding(image, detection)
        ranked = self.matcher.rank(embedding, top_k=2)
        diagnostic = None
        if self.diagnostics_enabled:
            try:
                diagnostic_ranked = self.matcher.rank(
                    embedding,
                    top_k=self.diagnostics_top_k,
                )
                margin = (
                    diagnostic_ranked[0].score - diagnostic_ranked[1].score
                    if len(diagnostic_ranked) > 1
                    else 0.0
                )
                diagnostic = {
                    "candidate_index": int(index),
                    "detection_score": round(float(detection.score), 6),
                    "margin": round(float(margin), 6),
                    "quality": detection_quality(image, detection, aligned),
                    "top": ranked_matches(diagnostic_ranked),
                }
            except Exception as error:  # diagnostics must not affect inference
                diagnostic = {
                    "candidate_index": int(index),
                    "diagnostic_error": f"{type(error).__name__}: {error}",
                }
        return ranked, diagnostic

    def _accepted_match(self, match):
        if (
            self.matcher.unknown_threshold is not None
            and match.score < self.matcher.unknown_threshold
        ):
            return None
        return match

    def _select_by_gallery(self, image: np.ndarray, detections):
        candidates = []
        diagnostics = []
        for index, detection in enumerate(detections):
            ranked, diagnostic = self._analyze_detection(image, detection, index)
            top = ranked[0]
            margin = top.score - ranked[1].score if len(ranked) > 1 else 0.0
            candidates.append(
                (top.score, margin, detection.score, -index, index, detection, top)
            )
            if diagnostic is not None:
                diagnostics.append(diagnostic)
        _score, _margin, _detection_score, _order, index, detection, match = max(
            candidates,
            key=lambda item: item[:4],
        )
        return detection, self._accepted_match(match), index, diagnostics

    def _diagnostic_base(self, sequence: int, image: np.ndarray) -> dict:
        return {
            "detections": 0,
            "image_height": int(image.shape[0]),
            "image_width": int(image.shape[1]),
            "sequence": int(sequence),
            "version": 1,
        }

    def _log_diagnostics(
        self,
        sequence: int,
        image: np.ndarray,
        detections,
        selected_index: int,
        diagnostic_candidates: list[dict | None],
    ) -> None:
        record = self._diagnostic_base(sequence, image)
        record.update(
            {
                "candidates": [
                    candidate
                    for candidate in diagnostic_candidates
                    if candidate is not None
                ],
                "detections": len(detections),
                "selected_candidate_index": int(selected_index),
            }
        )
        log.info(
            "[face-diagnostic] %s",
            json.dumps(record, ensure_ascii=True, separators=(",", ":")),
        )

    def _log_empty_diagnostics(self, sequence: int, image: np.ndarray) -> None:
        record = self._diagnostic_base(sequence, image)
        probes = []
        try:
            for threshold in self.diagnostics_retry_thresholds:
                detections = self.detector.detect(
                    image,
                    score_threshold=threshold,
                )
                candidates = []
                for index, detection in enumerate(
                    detections[: self.diagnostics_max_retry_candidates]
                ):
                    ranked, diagnostic = self._analyze_detection(
                        image,
                        detection,
                        index,
                    )
                    if diagnostic is None:
                        continue
                    diagnostic["official_top1_score"] = round(
                        float(ranked[0].score),
                        6,
                    )
                    candidates.append(diagnostic)
                probes.append(
                    {
                        "candidates": candidates,
                        "max_detection_score": round(
                            max((item.score for item in detections), default=0.0),
                            6,
                        ),
                        "raw_detections": len(detections),
                        "score_threshold": round(float(threshold), 6),
                    }
                )
        except Exception as error:  # diagnostics must never change the payload
            record["diagnostic_error"] = f"{type(error).__name__}: {error}"
        record["empty_detection_probes"] = probes
        log.info(
            "[face-diagnostic] %s",
            json.dumps(record, ensure_ascii=True, separators=(",", ":")),
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
        diagnostics_cfg = dict(cfg.get("diagnostics") or {})
        retry_thresholds = tuple(
            float(value)
            for value in diagnostics_cfg.get("retry_score_thresholds", [])
        )
        if any(
            value < 0.0 or value >= detector.score_threshold
            for value in retry_thresholds
        ):
            raise ValueError(
                "diagnostic retry thresholds must be non-negative and lower "
                "than detector_score_threshold"
            )
        engine = FaceIdentityEngine(
            detector,
            recognizer,
            gallery,
            matcher,
            face_selection=str(cfg.get("face_selection", "primary")),
            query_flip_tta=bool(cfg.get("query_flip_tta", False)),
            diagnostics_enabled=bool(diagnostics_cfg.get("enabled", False)),
            diagnostics_top_k=int(diagnostics_cfg.get("top_k", 5)),
            diagnostics_retry_thresholds=retry_thresholds,
            diagnostics_max_retry_candidates=int(
                diagnostics_cfg.get("max_retry_candidates", 5)
            ),
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
