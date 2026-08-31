"""SCRFD face detector preprocessing and output decoding."""

from __future__ import annotations

import cv2
import numpy as np

from .backends import InferenceBackend
from .schema import FaceDetection


def distance_to_bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    distance = np.asarray(distance, dtype=np.float32)
    return np.stack(
        [
            points[:, 0] - distance[:, 0],
            points[:, 1] - distance[:, 1],
            points[:, 0] + distance[:, 2],
            points[:, 1] + distance[:, 3],
        ],
        axis=-1,
    )


def distance_to_landmarks(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    distance = np.asarray(distance, dtype=np.float32)
    if distance.shape[1] != 10:
        raise ValueError(f"SCRFD keypoint output must have 10 columns: {distance.shape}")
    result = np.empty_like(distance, dtype=np.float32)
    for index in range(0, 10, 2):
        result[:, index] = points[:, 0] + distance[:, index]
        result[:, index + 1] = points[:, 1] + distance[:, index + 1]
    return result.reshape(-1, 5, 2)


def nms(boxes: np.ndarray, threshold: float) -> list[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2, scores = boxes.T
    areas = np.maximum(0.0, x2 - x1 + 1.0) * np.maximum(0.0, y2 - y1 + 1.0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        width = np.maximum(0.0, xx2 - xx1 + 1.0)
        height = np.maximum(0.0, yy2 - yy1 + 1.0)
        overlap = width * height
        union = areas[current] + areas[rest] - overlap
        iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
        order = rest[np.where(iou <= threshold)[0]]
    return keep


class SCRFDDetector:
    """Decode SCRFD KPS models with 3 or 5 feature-map levels."""

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        input_size: tuple[int, int] = (640, 640),
        score_threshold: float = 0.2,
        nms_threshold: float = 0.4,
    ):
        self.backend = backend
        self.input_size = tuple(int(value) for value in input_size)
        if len(self.input_size) != 2 or min(self.input_size) <= 0:
            raise ValueError(f"invalid SCRFD input size: {input_size}")
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self._center_cache: dict[tuple[int, int, int, int], np.ndarray] = {}

    def detect(
        self,
        image: np.ndarray,
        *,
        score_threshold: float | None = None,
    ) -> list[FaceDetection]:
        threshold = (
            self.score_threshold
            if score_threshold is None
            else float(score_threshold)
        )
        self._validate_threshold(threshold)
        outputs, input_height, input_width, scale = self._infer(image)
        return self._postprocess(
            outputs,
            input_height,
            input_width,
            scale,
            image.shape,
            score_threshold=threshold,
        )

    def detect_with_empty_retry(
        self,
        image: np.ndarray,
        *,
        retry_score_threshold: float,
        rotations: tuple[int, ...] = (),
        tile_fraction: float | None = None,
        rescue_min_face_ratio: float = 0.0,
    ) -> list[FaceDetection]:
        """Retry an empty result with cheap decode and optional transformed views."""

        retry_threshold = float(retry_score_threshold)
        self._validate_threshold(retry_threshold)
        if retry_threshold >= self.score_threshold:
            raise ValueError(
                "SCRFD empty retry threshold must be lower than the default "
                f"score threshold: {retry_threshold} >= {self.score_threshold}"
            )
        outputs, input_height, input_width, scale = self._infer(image)
        detections = self._postprocess(
            outputs,
            input_height,
            input_width,
            scale,
            image.shape,
            score_threshold=self.score_threshold,
        )
        if detections:
            return detections
        detections = self._postprocess(
            outputs,
            input_height,
            input_width,
            scale,
            image.shape,
            score_threshold=retry_threshold,
        )
        if detections:
            return detections
        rotated_detections = self._detect_rotated_views(
            image,
            rotations=rotations,
            score_threshold=retry_threshold,
            min_face_ratio=float(rescue_min_face_ratio),
        )
        if rotated_detections:
            return self._merge_detections(rotated_detections)
        if tile_fraction is None:
            return []
        return self._merge_detections(
            self._detect_tiled_views(
                image,
                tile_fraction=float(tile_fraction),
                score_threshold=retry_threshold,
                min_face_ratio=float(rescue_min_face_ratio),
            )
        )

    def _detect_rotated_views(
        self,
        image: np.ndarray,
        *,
        rotations: tuple[int, ...],
        score_threshold: float,
        min_face_ratio: float,
    ) -> list[FaceDetection]:
        detections: list[FaceDetection] = []
        for degrees in rotations:
            rotated, inverse = self._rotate_view(image, int(degrees))
            for detection in self.detect(
                rotated,
                score_threshold=score_threshold,
            ):
                mapped = self._map_detection(
                    detection,
                    inverse,
                    image.shape,
                )
                if mapped is not None and self._rescue_geometry_valid(
                    mapped,
                    image.shape,
                    min_face_ratio=min_face_ratio,
                ):
                    detections.append(mapped)
        return detections

    def _detect_tiled_views(
        self,
        image: np.ndarray,
        *,
        tile_fraction: float,
        score_threshold: float,
        min_face_ratio: float,
    ) -> list[FaceDetection]:
        if not 0.5 <= tile_fraction < 1.0:
            raise ValueError("SCRFD tile fraction must be in [0.5, 1.0)")
        height, width = image.shape[:2]
        crop_width = max(1, int(round(width * tile_fraction)))
        crop_height = max(1, int(round(height * tile_fraction)))
        max_x = max(0, width - crop_width)
        max_y = max(0, height - crop_height)
        origins = [
            (max_x // 2, max_y // 2),
            (0, 0),
            (max_x, 0),
            (0, max_y),
            (max_x, max_y),
        ]
        detections: list[FaceDetection] = []
        for offset_x, offset_y in dict.fromkeys(origins):
            crop = image[
                offset_y : offset_y + crop_height,
                offset_x : offset_x + crop_width,
            ]
            affine = np.array(
                [[1.0, 0.0, offset_x], [0.0, 1.0, offset_y]],
                dtype=np.float32,
            )
            for detection in self.detect(
                crop,
                score_threshold=score_threshold,
            ):
                mapped = self._map_detection(detection, affine, image.shape)
                if mapped is not None and self._rescue_geometry_valid(
                    mapped,
                    image.shape,
                    min_face_ratio=min_face_ratio,
                ):
                    detections.append(mapped)
        return detections

    @staticmethod
    def _rotate_view(
        image: np.ndarray,
        degrees: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        if degrees == 90:
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            forward = np.array(
                [[0.0, -1.0, height - 1.0], [1.0, 0.0, 0.0]],
                dtype=np.float32,
            )
        elif degrees == 180:
            rotated = cv2.rotate(image, cv2.ROTATE_180)
            forward = np.array(
                [[-1.0, 0.0, width - 1.0], [0.0, -1.0, height - 1.0]],
                dtype=np.float32,
            )
        elif degrees == 270:
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            forward = np.array(
                [[0.0, 1.0, 0.0], [-1.0, 0.0, width - 1.0]],
                dtype=np.float32,
            )
        else:
            raise ValueError("SCRFD rescue rotations must be 90, 180, or 270")
        return rotated, cv2.invertAffineTransform(forward).astype(np.float32)

    @staticmethod
    def _map_detection(
        detection: FaceDetection,
        affine: np.ndarray,
        image_shape: tuple[int, ...],
    ) -> FaceDetection | None:
        matrix = np.asarray(affine, dtype=np.float32)
        if matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("face detection mapping requires a finite 2x3 affine")
        x1, y1, x2, y2 = detection.bbox
        corners = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )
        homogeneous_corners = np.column_stack(
            [corners, np.ones(4, dtype=np.float32)]
        )
        homogeneous_landmarks = np.column_stack(
            [detection.landmarks, np.ones(5, dtype=np.float32)]
        )
        mapped_corners = homogeneous_corners @ matrix.T
        mapped_landmarks = homogeneous_landmarks @ matrix.T
        height, width = image_shape[:2]
        mapped_bbox = np.array(
            [
                np.min(mapped_corners[:, 0]),
                np.min(mapped_corners[:, 1]),
                np.max(mapped_corners[:, 0]),
                np.max(mapped_corners[:, 1]),
            ],
            dtype=np.float32,
        )
        mapped_bbox[[0, 2]] = np.clip(mapped_bbox[[0, 2]], 0, width - 1)
        mapped_bbox[[1, 3]] = np.clip(mapped_bbox[[1, 3]], 0, height - 1)
        mapped_landmarks[:, 0] = np.clip(mapped_landmarks[:, 0], 0, width - 1)
        mapped_landmarks[:, 1] = np.clip(mapped_landmarks[:, 1], 0, height - 1)
        if mapped_bbox[2] <= mapped_bbox[0] or mapped_bbox[3] <= mapped_bbox[1]:
            return None
        return FaceDetection(
            mapped_bbox,
            detection.score,
            mapped_landmarks,
        )

    def _merge_detections(
        self,
        detections: list[FaceDetection],
    ) -> list[FaceDetection]:
        if not detections:
            return []
        boxes = np.vstack(
            [
                np.append(detection.bbox, detection.score)
                for detection in detections
            ]
        ).astype(np.float32)
        return [detections[index] for index in nms(boxes, self.nms_threshold)]

    @staticmethod
    def _rescue_geometry_valid(
        detection: FaceDetection,
        image_shape: tuple[int, ...],
        *,
        min_face_ratio: float,
    ) -> bool:
        if not 0.0 <= min_face_ratio <= 1.0:
            raise ValueError("SCRFD rescue min_face_ratio must be in [0, 1]")
        height, width = image_shape[:2]
        x1, y1, x2, y2 = detection.bbox
        face_ratio = min(float(x2 - x1), float(y2 - y1)) / max(
            1.0,
            min(height, width),
        )
        if face_ratio < min_face_ratio:
            return False
        eye_distance = float(
            np.linalg.norm(detection.landmarks[0] - detection.landmarks[1])
        )
        return eye_distance >= 0.12 * min(float(x2 - x1), float(y2 - y1))

    @staticmethod
    def _validate_threshold(threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"invalid SCRFD score threshold: {threshold}")

    def _infer(
        self,
        image: np.ndarray,
    ) -> tuple[list[np.ndarray], int, int, float]:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("SCRFD expects a decoded BGR HWC image")
        model_image, scale = self._resize_and_pad(image)
        blob = cv2.dnn.blobFromImage(
            model_image,
            scalefactor=1.0 / 128.0,
            size=self.input_size,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        outputs = self.backend.infer(blob)
        return outputs, int(blob.shape[2]), int(blob.shape[3]), scale

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        input_height: int,
        input_width: int,
        scale: float,
        image_shape: tuple[int, ...],
        *,
        score_threshold: float,
    ) -> list[FaceDetection]:
        boxes, landmarks = self._decode(
            outputs,
            input_height,
            input_width,
            score_threshold=score_threshold,
        )
        if boxes.size == 0:
            return []
        boxes[:, :4] /= scale
        landmarks /= scale
        height, width = image_shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
        landmarks[:, :, 0] = np.clip(landmarks[:, :, 0], 0, width - 1)
        landmarks[:, :, 1] = np.clip(landmarks[:, :, 1], 0, height - 1)
        order = boxes[:, 4].argsort()[::-1]
        boxes = boxes[order]
        landmarks = landmarks[order]
        keep = nms(boxes, self.nms_threshold)
        detections = []
        for index in keep:
            box = boxes[index]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            detections.append(
                FaceDetection(box[:4], float(box[4]), landmarks[index])
            )
        return detections

    def _resize_and_pad(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        target_width, target_height = self.input_size
        source_height, source_width = image.shape[:2]
        scale = min(target_width / source_width, target_height / source_height)
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height))
        padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        # InsightFace's SCRFD reference uses top-left padding. Keeping that
        # convention means decoded coordinates only need division by scale.
        padded[:resized_height, :resized_width] = resized
        return padded, float(scale)

    def _decode(
        self,
        outputs: list[np.ndarray],
        input_height: int,
        input_width: int,
        *,
        score_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        count = len(outputs)
        if count == 9:
            strides, anchors = (8, 16, 32), 2
        elif count == 15:
            strides, anchors = (8, 16, 32, 64, 128), 1
        else:
            raise ValueError(
                "SCRFD-*_KPS must expose 9 or 15 outputs; "
                f"got {count} ({getattr(self.backend, 'output_names', [])})"
            )
        output_map: dict[tuple[int, int], np.ndarray] = {}
        for output in outputs:
            array = np.asarray(output)
            if array.ndim == 3 and array.shape[0] == 1:
                array = array[0]
            if array.ndim == 1:
                channels = 1
            else:
                channels = int(array.shape[-1])
            if channels not in {1, 4, 10}:
                raise ValueError(f"unexpected SCRFD output shape: {array.shape}")
            rows = int(array.size // channels)
            key = (rows, channels)
            if key in output_map:
                raise ValueError(f"duplicate SCRFD output shape: {array.shape}")
            output_map[key] = array.reshape(rows, channels)
        all_boxes: list[np.ndarray] = []
        all_landmarks: list[np.ndarray] = []
        for stride in strides:
            rows = (input_height // stride) * (input_width // stride) * anchors
            try:
                scores = output_map[(rows, 1)].reshape(-1)
                bbox_distance = output_map[(rows, 4)] * stride
                kps_distance = output_map[(rows, 10)] * stride
            except KeyError as error:
                available = sorted(output_map)
                raise ValueError(
                    f"SCRFD stride {stride} outputs are missing; available={available}"
                ) from error
            centers = self._anchor_centers(
                input_height // stride,
                input_width // stride,
                stride,
                anchors,
            )
            if len(centers) != rows:
                raise ValueError(
                    f"SCRFD stride {stride} anchor mismatch: rows={rows}, anchors={len(centers)}"
                )
            selected = np.flatnonzero(scores >= score_threshold)
            if selected.size == 0:
                continue
            decoded_boxes = distance_to_bbox(centers, bbox_distance)
            decoded_landmarks = distance_to_landmarks(centers, kps_distance)
            all_boxes.append(
                np.column_stack([decoded_boxes[selected], scores[selected]])
            )
            all_landmarks.append(decoded_landmarks[selected])
        if not all_boxes:
            return (
                np.empty((0, 5), dtype=np.float32),
                np.empty((0, 5, 2), dtype=np.float32),
            )
        return (
            np.vstack(all_boxes).astype(np.float32, copy=False),
            np.vstack(all_landmarks).astype(np.float32, copy=False),
        )

    def _anchor_centers(
        self,
        height: int,
        width: int,
        stride: int,
        anchors: int,
    ) -> np.ndarray:
        key = (height, width, stride, anchors)
        cached = self._center_cache.get(key)
        if cached is not None:
            return cached
        centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(
            np.float32
        )
        centers = (centers * stride).reshape(-1, 2)
        if anchors > 1:
            centers = np.repeat(centers, anchors, axis=0)
        self._center_cache[key] = centers
        return centers

    def close(self) -> None:
        self.backend.close()
