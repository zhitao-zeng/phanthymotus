"""End-to-end face detection, alignment, embedding and gallery matching."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np

from .alignment import alignment_rmse, align_face, rescue_face_alignments
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
        fallback_recognizer=None,
        fallback_matcher: IdentityMatcher | None = None,
        fallback_mobile_margin_max: float = 0.0,
        fallback_margin_min: float = 0.0,
        empty_detection_retry_threshold: float | None = None,
        empty_detection_retry_rotations: tuple[int, ...] = (),
        empty_detection_retry_tile_fraction: float | None = None,
        empty_detection_retry_min_face_ratio: float = 0.0,
        alignment_rescue_rmse_min: float | None = None,
        alignment_rescue_min_score_gain: float = 0.0,
        alignment_rescue_min_margin_gain: float = 0.0,
        subcenter_rescue_margin_max: float | None = None,
        subcenter_rescue_min_margin_gain: float = 0.0,
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
        self.fallback_recognizer = fallback_recognizer
        self.fallback_matcher = fallback_matcher
        self.fallback_mobile_margin_max = float(fallback_mobile_margin_max)
        self.fallback_margin_min = float(fallback_margin_min)
        if (self.fallback_recognizer is None) != (self.fallback_matcher is None):
            raise ValueError("fallback recognizer and matcher must be provided together")
        self.empty_detection_retry_threshold = (
            None
            if empty_detection_retry_threshold is None
            else float(empty_detection_retry_threshold)
        )
        self.empty_detection_retry_rotations = tuple(
            int(value) for value in empty_detection_retry_rotations
        )
        self.empty_detection_retry_tile_fraction = (
            None
            if empty_detection_retry_tile_fraction is None
            else float(empty_detection_retry_tile_fraction)
        )
        self.empty_detection_retry_min_face_ratio = float(
            empty_detection_retry_min_face_ratio
        )
        if not 0.0 <= self.empty_detection_retry_min_face_ratio <= 1.0:
            raise ValueError("empty detection retry min_face_ratio must be in [0, 1]")
        self.alignment_rescue_rmse_min = (
            None
            if alignment_rescue_rmse_min is None
            else float(alignment_rescue_rmse_min)
        )
        self.alignment_rescue_min_score_gain = float(
            alignment_rescue_min_score_gain
        )
        self.alignment_rescue_min_margin_gain = float(
            alignment_rescue_min_margin_gain
        )
        if (
            self.alignment_rescue_rmse_min is not None
            and self.alignment_rescue_rmse_min < 0.0
        ):
            raise ValueError("alignment rescue RMSE threshold must be non-negative")
        if self.alignment_rescue_min_score_gain < 0.0:
            raise ValueError("alignment rescue score gain must be non-negative")
        if self.alignment_rescue_min_margin_gain < 0.0:
            raise ValueError("alignment rescue margin gain must be non-negative")
        self.subcenter_rescue_margin_max = (
            None
            if subcenter_rescue_margin_max is None
            else float(subcenter_rescue_margin_max)
        )
        self.subcenter_rescue_min_margin_gain = float(
            subcenter_rescue_min_margin_gain
        )
        if (
            self.subcenter_rescue_margin_max is not None
            and self.subcenter_rescue_margin_max < 0.0
        ):
            raise ValueError("subcenter rescue margin threshold must be non-negative")
        if self.subcenter_rescue_min_margin_gain < 0.0:
            raise ValueError("subcenter rescue margin gain must be non-negative")
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
            if self.empty_detection_retry_threshold is None:
                detections = self.detector.detect(image)
            else:
                detections = self.detector.detect_with_empty_retry(
                    image,
                    retry_score_threshold=self.empty_detection_retry_threshold,
                    rotations=self.empty_detection_retry_rotations,
                    tile_fraction=self.empty_detection_retry_tile_fraction,
                    rescue_min_face_ratio=self.empty_detection_retry_min_face_ratio,
                )
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
                ranked, diagnostic, aligned = self._analyze_detection(
                    image,
                    detection,
                    selected_index,
                )
                diagnostic_candidates = [diagnostic]
            else:
                (
                    detection,
                    selected_index,
                    diagnostic_candidates,
                    ranked,
                    aligned,
                ) = self._select_by_gallery(image, detections)
            match = self._route_identity(ranked, aligned)
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
        alignment_hypothesis = "standard"
        if self.alignment_rescue_rmse_min is not None:
            residual = alignment_rmse(detection.landmarks)
            if residual >= self.alignment_rescue_rmse_min:
                baseline_score = ranked[0].score
                baseline_margin = (
                    ranked[0].score - ranked[1].score
                    if len(ranked) > 1
                    else 0.0
                )
                best = None
                for name, candidate_aligned in rescue_face_alignments(
                    image,
                    detection.landmarks,
                ):
                    candidate_embedding = self.recognizer.embed(
                        candidate_aligned,
                        flip_tta=self.query_flip_tta,
                    )
                    candidate_ranked = self.matcher.rank(
                        candidate_embedding,
                        top_k=2,
                    )
                    candidate_margin = (
                        candidate_ranked[0].score - candidate_ranked[1].score
                        if len(candidate_ranked) > 1
                        else 0.0
                    )
                    candidate_key = (
                        candidate_ranked[0].score,
                        candidate_margin,
                    )
                    if best is None or candidate_key > best[0]:
                        best = (
                            candidate_key,
                            name,
                            candidate_embedding,
                            candidate_ranked,
                            candidate_aligned,
                        )
                if best is not None:
                    (
                        (candidate_score, candidate_margin),
                        name,
                        candidate_embedding,
                        candidate_ranked,
                        candidate_aligned,
                    ) = best
                    if (
                        candidate_score
                        >= baseline_score + self.alignment_rescue_min_score_gain
                        and candidate_margin
                        >= baseline_margin + self.alignment_rescue_min_margin_gain
                    ):
                        embedding = candidate_embedding
                        ranked = candidate_ranked
                        aligned = candidate_aligned
                        alignment_hypothesis = name
        ranked = self._rank_identity(embedding, top_k=2)
        diagnostic = None
        if self.diagnostics_enabled:
            try:
                diagnostic_ranked = self._rank_identity(
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
                    "alignment_hypothesis": alignment_hypothesis,
                    "margin": round(float(margin), 6),
                    "quality": detection_quality(image, detection, aligned),
                    "top": ranked_matches(diagnostic_ranked),
                }
            except Exception as error:  # diagnostics must not affect inference
                diagnostic = {
                    "candidate_index": int(index),
                    "diagnostic_error": f"{type(error).__name__}: {error}",
                }
        return ranked, diagnostic, aligned

    def _rank_identity(self, embedding: np.ndarray, *, top_k: int):
        if self.subcenter_rescue_margin_max is None:
            return self.matcher.rank(embedding, top_k=top_k)
        return self.matcher.rank_with_subcenter_rescue(
            embedding,
            margin_max=self.subcenter_rescue_margin_max,
            min_margin_gain=self.subcenter_rescue_min_margin_gain,
            top_k=top_k,
        )

    def _route_identity(self, mobile_ranked, aligned: np.ndarray):
        match = mobile_ranked[0]
        if self.fallback_recognizer is not None:
            mobile_margin = (
                mobile_ranked[0].score - mobile_ranked[1].score
                if len(mobile_ranked) > 1
                else 0.0
            )
            if mobile_margin <= self.fallback_mobile_margin_max:
                embedding = self.fallback_recognizer.embed(
                    aligned,
                    flip_tta=False,
                )
                fallback_ranked = self.fallback_matcher.rank(
                    embedding,
                    top_k=2,
                )
                fallback_margin = (
                    fallback_ranked[0].score - fallback_ranked[1].score
                    if len(fallback_ranked) > 1
                    else 0.0
                )
                if fallback_margin >= self.fallback_margin_min:
                    match = fallback_ranked[0]
        return self._accepted_match(match)

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
            ranked, diagnostic, aligned = self._analyze_detection(
                image,
                detection,
                index,
            )
            top = ranked[0]
            margin = top.score - ranked[1].score if len(ranked) > 1 else 0.0
            candidates.append(
                (
                    top.score,
                    margin,
                    detection.score,
                    -index,
                    index,
                    detection,
                    ranked,
                    aligned,
                )
            )
            if diagnostic is not None:
                diagnostics.append(diagnostic)
        (
            _score,
            _margin,
            _detection_score,
            _order,
            index,
            detection,
            ranked,
            aligned,
        ) = max(
            candidates,
            key=lambda item: item[:4],
        )
        return detection, index, diagnostics, ranked, aligned

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
                    ranked, diagnostic, _aligned = self._analyze_detection(
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
            if self.fallback_recognizer is not None:
                self.fallback_recognizer.close()
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
        "adaface": "adaface_ir18_webface4m",
        "adaface-ir18": "adaface_ir18_webface4m",
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
    fallback_backend = None
    fallback_recognizer = None
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
        fallback_cfg = dict(cfg.get("recognizer_fallback") or {})
        fallback_matcher = None
        if bool(fallback_cfg.get("enabled", False)):
            fallback_type = str(
                fallback_cfg.get("recognizer", "adaface-ir18")
            ).strip().lower()
            fallback_path = Path(str(fallback_cfg.get("model") or ""))
            if not fallback_path.is_file():
                raise FileNotFoundError(
                    f"fallback face model does not exist: {fallback_path}"
                )
            fallback_backend_name = _normalize_backend_name(
                fallback_cfg.get("backend", "onnx")
            )
            fallback_providers = fallback_cfg.get("onnx_providers", providers)
            fallback_intra_threads = fallback_cfg.get(
                "onnx_intra_op_threads",
                onnx_intra_op_threads,
            )
            fallback_inter_threads = fallback_cfg.get(
                "onnx_inter_op_threads",
                onnx_inter_op_threads,
            )
            fallback_backend = build_backend(
                fallback_backend_name,
                str(fallback_path),
                device_id=device_id,
                providers=fallback_providers,
                intra_op_threads=(
                    None
                    if fallback_intra_threads is None
                    else int(fallback_intra_threads)
                ),
                inter_op_threads=(
                    None
                    if fallback_inter_threads is None
                    else int(fallback_inter_threads)
                ),
            )
            fallback_recognizer = FaceRecognizer(
                fallback_backend,
                model_type=fallback_type,
            )
            fallback_gallery = IdentityGallery.build(
                gallery_dir,
                detector,
                fallback_recognizer,
                quality=quality,
            )
            fallback_matcher = IdentityMatcher(
                fallback_gallery,
                centroid_weight=float(fallback_cfg.get("centroid_weight", 0.6)),
            )
        bbox_cfg = dict(cfg.get("bbox_calibration") or {})
        empty_retry_cfg = dict(cfg.get("empty_detection_retry") or {})
        empty_retry_threshold = None
        if bool(empty_retry_cfg.get("enabled", False)):
            empty_retry_threshold = float(
                empty_retry_cfg.get("score_threshold", 0.10)
            )
            if (
                empty_retry_threshold < 0.0
                or empty_retry_threshold >= detector.score_threshold
            ):
                raise ValueError(
                    "empty detection retry threshold must be non-negative and "
                    "lower than detector_score_threshold"
                )
        empty_retry_rotations = tuple(
            int(value) for value in empty_retry_cfg.get("rotations", [])
        )
        if any(value not in {90, 180, 270} for value in empty_retry_rotations):
            raise ValueError("empty detection retry rotations must use 90/180/270")
        empty_retry_tile_fraction = empty_retry_cfg.get("tile_fraction")
        if empty_retry_tile_fraction is not None:
            empty_retry_tile_fraction = float(empty_retry_tile_fraction)
            if not 0.5 <= empty_retry_tile_fraction < 1.0:
                raise ValueError(
                    "empty detection retry tile_fraction must be in [0.5, 1.0)"
                )
        empty_retry_min_face_ratio = float(
            empty_retry_cfg.get("min_face_ratio", 0.0)
        )
        alignment_rescue_cfg = dict(cfg.get("alignment_rescue") or {})
        alignment_rescue_rmse_min = None
        if bool(alignment_rescue_cfg.get("enabled", False)):
            alignment_rescue_rmse_min = float(
                alignment_rescue_cfg.get("rmse_min", 8.0)
            )
        subcenter_rescue_cfg = dict(cfg.get("subcenter_rescue") or {})
        subcenter_rescue_margin_max = None
        if bool(subcenter_rescue_cfg.get("enabled", False)):
            subcenter_rescue_margin_max = float(
                subcenter_rescue_cfg.get("margin_max", 0.035)
            )
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
            fallback_recognizer=fallback_recognizer,
            fallback_matcher=fallback_matcher,
            fallback_mobile_margin_max=float(
                fallback_cfg.get("mobile_margin_max", 0.0)
            ),
            fallback_margin_min=float(fallback_cfg.get("margin_min", 0.0)),
            empty_detection_retry_threshold=empty_retry_threshold,
            empty_detection_retry_rotations=empty_retry_rotations,
            empty_detection_retry_tile_fraction=empty_retry_tile_fraction,
            empty_detection_retry_min_face_ratio=empty_retry_min_face_ratio,
            alignment_rescue_rmse_min=alignment_rescue_rmse_min,
            alignment_rescue_min_score_gain=float(
                alignment_rescue_cfg.get("min_score_gain", 0.02)
            ),
            alignment_rescue_min_margin_gain=float(
                alignment_rescue_cfg.get("min_margin_gain", 0.0)
            ),
            subcenter_rescue_margin_max=subcenter_rescue_margin_max,
            subcenter_rescue_min_margin_gain=float(
                subcenter_rescue_cfg.get("min_margin_gain", 0.04)
            ),
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
            "family=%s recognizer=%s gallery=%d fallback=%s",
            detector_backend_name,
            recognizer_backend_name,
            trt_family or "host",
            recognizer_type,
            len(gallery.templates),
            str(fallback_cfg.get("recognizer", "disabled"))
            if fallback_matcher is not None
            else "disabled",
        )
        return engine
    except Exception:
        if fallback_recognizer is not None:
            fallback_recognizer.close()
        elif fallback_backend is not None:
            fallback_backend.close()
        if recognizer is not None:
            recognizer.close()
        elif recognizer_backend is not None:
            recognizer_backend.close()
        if detector is not None:
            detector.close()
        else:
            detector_backend.close()
        raise
