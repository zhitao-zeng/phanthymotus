"""Identity-gallery construction, quality filtering and template aggregation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .alignment import align_face
from .schema import FaceDetection, l2_normalize

log = logging.getLogger(__name__)

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class GalleryBuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class GalleryQualityConfig:
    min_detection_score: float = 0.2
    min_face_size: float = 24.0
    min_eye_distance: float = 8.0
    min_blur_variance: float = 0.0
    flip_tta: bool = True
    max_subcenters: int = 3
    subcenter_min_images: int = 4


@dataclass(frozen=True)
class IdentityTemplates:
    person_id: str
    centroid: np.ndarray
    subcenters: np.ndarray
    exemplars: np.ndarray
    source_count: int


class IdentityGallery:
    def __init__(self, templates: list[IdentityTemplates]):
        if not templates:
            raise GalleryBuildError("identity gallery is empty")
        ids = [template.person_id for template in templates]
        if len(set(ids)) != len(ids):
            raise GalleryBuildError("identity gallery contains duplicate person IDs")
        self.templates = tuple(sorted(templates, key=lambda item: item.person_id))

    @classmethod
    def build(
        cls,
        directory: str | Path,
        detector,
        recognizer,
        *,
        quality: GalleryQualityConfig | None = None,
    ) -> "IdentityGallery":
        config = quality or GalleryQualityConfig()
        root = Path(directory)
        if not root.is_dir():
            raise GalleryBuildError(f"face gallery directory does not exist: {root}")
        person_dirs = sorted(path for path in root.iterdir() if path.is_dir())
        if not person_dirs:
            raise GalleryBuildError(f"face gallery has no identity directories: {root}")

        templates: list[IdentityTemplates] = []
        failures: list[str] = []
        for person_dir in person_dirs:
            try:
                template = _build_identity_templates(
                    person_dir,
                    detector,
                    recognizer,
                    config,
                )
            except GalleryBuildError as error:
                failures.append(str(error))
            else:
                templates.append(template)
        if failures:
            joined = "; ".join(failures[:8])
            if len(failures) > 8:
                joined += f"; and {len(failures) - 8} more"
            raise GalleryBuildError(f"gallery identities without usable faces: {joined}")
        log.info(
            "[face] gallery ready: %d identities, %d source images",
            len(templates),
            sum(template.source_count for template in templates),
        )
        return cls(templates)


def select_primary_face(
    detections: list[FaceDetection], image_shape: tuple[int, ...]
) -> FaceDetection | None:
    if not detections:
        return None
    height, width = image_shape[:2]
    image_area = max(1.0, float(height * width))

    def rank(detection: FaceDetection) -> float:
        x1, y1, x2, y2 = detection.bbox
        area_ratio = max(0.0, float((x2 - x1) * (y2 - y1))) / image_area
        center_x = (x1 + x2) * 0.5 / max(1.0, width)
        center_y = (y1 + y2) * 0.5 / max(1.0, height)
        center_penalty = (center_x - 0.5) ** 2 + (center_y - 0.5) ** 2
        return float(detection.score) + 0.25 * np.sqrt(area_ratio) - 0.05 * center_penalty

    return max(detections, key=rank)


def face_quality(
    image: np.ndarray,
    detection: FaceDetection,
    aligned: np.ndarray,
    config: GalleryQualityConfig,
) -> tuple[bool, float]:
    x1, y1, x2, y2 = detection.bbox
    face_size = min(float(x2 - x1), float(y2 - y1))
    eye_distance = float(np.linalg.norm(detection.landmarks[0] - detection.landmarks[1]))
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    accepted = (
        detection.score >= config.min_detection_score
        and face_size >= config.min_face_size
        and eye_distance >= config.min_eye_distance
        and blur_variance >= config.min_blur_variance
    )
    # Detection confidence is the stable base weight. Blur and geometry only
    # decide whether to admit a registration image; they are deliberately not
    # combined into a fragile hand-tuned score.
    return accepted, max(1e-3, float(detection.score))


def weighted_centroid(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    if features.ndim != 2 or features.shape[0] != weights.shape[0]:
        raise ValueError("features and weights have incompatible shapes")
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("template weights must contain a positive value")
    return l2_normalize(np.sum(features * weights[:, None], axis=0))


def spherical_subcenters(
    features: np.ndarray,
    weights: np.ndarray,
    *,
    max_subcenters: int,
    enable_from: int,
    iterations: int = 20,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    count = features.shape[0]
    if count == 0:
        raise ValueError("cannot cluster an empty identity")
    if max_subcenters <= 1:
        return weighted_centroid(features, weights)[None]
    if count < enable_from:
        # For sparse identities, preserve the observed modes instead of
        # inventing clusters from one or two samples.
        return np.vstack([l2_normalize(row) for row in features[:max_subcenters]])
    clusters = min(max_subcenters, max(1, count // 2))
    centers = [features[int(np.argmax(weights))]]
    while len(centers) < clusters:
        similarities = features @ np.vstack(centers).T
        distance = 1.0 - np.max(similarities, axis=1)
        centers.append(features[int(np.argmax(distance))])
    centers_array = np.vstack([l2_normalize(row) for row in centers])
    assignments = np.full(count, -1, dtype=np.int32)
    for _ in range(max(1, iterations)):
        updated_assignments = np.argmax(features @ centers_array.T, axis=1)
        if np.array_equal(updated_assignments, assignments):
            break
        assignments = updated_assignments
        new_centers = []
        for cluster in range(clusters):
            members = np.flatnonzero(assignments == cluster)
            if members.size == 0:
                nearest = np.max(features @ centers_array.T, axis=1)
                new_centers.append(features[int(np.argmin(nearest))])
            else:
                new_centers.append(
                    weighted_centroid(features[members], weights[members])
                )
        centers_array = np.vstack(new_centers)
    return np.asarray(centers_array, dtype=np.float32)


def _build_identity_templates(
    person_dir: Path,
    detector,
    recognizer,
    config: GalleryQualityConfig,
) -> IdentityTemplates:
    image_paths = sorted(
        path
        for path in person_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not image_paths:
        raise GalleryBuildError(f"{person_dir.name}: no registration images")

    accepted: list[tuple[np.ndarray, float]] = []
    rejected: list[tuple[float, np.ndarray, float]] = []
    detected_images = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            log.warning("[face] unreadable gallery image: %s", image_path)
            continue
        detection = select_primary_face(detector.detect(image), image.shape)
        if detection is None:
            continue
        detected_images += 1
        try:
            aligned = align_face(image, detection.landmarks)
        except ValueError:
            log.warning("[face] invalid gallery landmarks: %s", image_path)
            continue
        passed, weight = face_quality(image, detection, aligned, config)
        if passed:
            feature = recognizer.embed(aligned, flip_tta=config.flip_tta)
            accepted.append((feature, weight))
        else:
            rejected.append((weight, aligned, detection.score))

    if not accepted and rejected:
        # Do not silently delete an identity merely because its only source is
        # softer than a quality threshold. Keep its best detected registration
        # image and make the fallback visible in logs.
        weight, aligned, _ = max(rejected, key=lambda item: item[0])
        accepted.append((recognizer.embed(aligned, flip_tta=config.flip_tta), weight))
        log.warning(
            "[face] %s: all registrations failed quality filters; using best detection",
            person_dir.name,
        )
    if not accepted:
        detail = "no face detected" if detected_images == 0 else "alignment failed"
        raise GalleryBuildError(f"{person_dir.name}: {detail}")

    features = np.vstack([feature for feature, _weight in accepted]).astype(np.float32)
    weights = np.asarray([weight for _feature, weight in accepted], dtype=np.float32)
    centroid = weighted_centroid(features, weights)
    subcenters = spherical_subcenters(
        features,
        weights,
        max_subcenters=config.max_subcenters,
        enable_from=config.subcenter_min_images,
    )
    return IdentityTemplates(
        person_id=person_dir.name,
        centroid=centroid,
        subcenters=subcenters,
        exemplars=features,
        source_count=len(image_paths),
    )
