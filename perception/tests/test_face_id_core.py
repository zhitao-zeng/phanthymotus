"""Host-side tests for the model-independent face-identification core."""

from __future__ import annotations

import cv2
import numpy as np

from plugins.face_id.alignment import ARCFACE_112_TEMPLATE, align_face
from plugins.face_id.detector import SCRFDDetector, distance_to_bbox, nms
from plugins.face_id.engine import FaceIdentityEngine, _has_explicit_model_pair
from plugins.face_id.gallery import (
    IdentityGallery,
    IdentityTemplates,
    spherical_subcenters,
    weighted_centroid,
)
from plugins.face_id.matcher import IdentityMatcher
from plugins.face_id.postprocess import calibrate_bbox, normalized_xywh
from plugins.face_id.recognizer import FaceRecognizer
from plugins.face_id.schema import FaceDetection


class _StaticBackend:
    def __init__(self, outputs):
        self.outputs = outputs
        self.output_names = [f"output_{index}" for index in range(len(outputs))]
        self.inputs = []
        self.closed = False

    def infer(self, array):
        self.inputs.append(np.array(array, copy=True))
        return [np.array(output, copy=True) for output in self.outputs]

    def close(self):
        self.closed = True


def test_explicit_model_pair_requires_both_paths():
    assert _has_explicit_model_pair(
        {
            "detector_model": "/models/det.onnx",
            "recognizer_model": "/models/rec.onnx",
        }
    )
    assert not _has_explicit_model_pair({"detector_model": "/models/det.onnx"})
    assert not _has_explicit_model_pair({"recognizer_model": "/models/rec.onnx"})


def _scrfd_outputs(input_size=64):
    outputs = []
    score_outputs, box_outputs, landmark_outputs = [], [], []
    for stride in (8, 16, 32):
        count = (input_size // stride) * (input_size // stride) * 2
        scores = np.zeros((count, 1), dtype=np.float32)
        boxes = np.zeros((count, 4), dtype=np.float32)
        landmarks = np.zeros((count, 10), dtype=np.float32)
        score_outputs.append(scores)
        box_outputs.append(boxes)
        landmark_outputs.append(landmarks)
    score_outputs[0][0] = 0.9
    box_outputs[0][0] = [0.0, 0.0, 2.0, 2.0]
    landmark_outputs[0][0] = [0.5, 0.5, 1.5, 0.5, 1.0, 1.0, 0.6, 1.5, 1.4, 1.5]
    outputs.extend(score_outputs)
    outputs.extend(box_outputs)
    outputs.extend(landmark_outputs)
    return outputs


def test_scrfd_decodes_kps_outputs_and_clips_box():
    backend = _StaticBackend(_scrfd_outputs())
    detector = SCRFDDetector(
        backend,
        input_size=(64, 64),
        score_threshold=0.2,
        nms_threshold=0.4,
    )
    detections = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    assert len(detections) == 1
    detection = detections[0]
    np.testing.assert_allclose(detection.bbox, [0, 0, 16, 16])
    np.testing.assert_allclose(detection.landmarks[0], [4, 4])
    assert detection.score == np.float32(0.9)
    assert backend.inputs[0].shape == (1, 3, 64, 64)


def test_scrfd_accepts_tensorrt_interleaved_output_order():
    grouped = _scrfd_outputs()
    interleaved = [
        grouped[0], grouped[3], grouped[6],
        grouped[1], grouped[4], grouped[7],
        grouped[2], grouped[5], grouped[8],
    ]
    detector = SCRFDDetector(
        _StaticBackend(interleaved),
        input_size=(64, 64),
        score_threshold=0.2,
    )
    detections = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))
    assert len(detections) == 1
    np.testing.assert_allclose(detections[0].bbox, [0, 0, 16, 16])


def test_distance_decode_and_nms():
    boxes = distance_to_bbox(
        np.array([[10, 10], [11, 11]], dtype=np.float32),
        np.array([[5, 5, 5, 5], [5, 5, 5, 5]], dtype=np.float32),
    )
    np.testing.assert_allclose(boxes[0], [5, 5, 15, 15])
    scored = np.column_stack([boxes, [0.9, 0.8]])
    assert nms(scored, 0.4) == [0]


def test_alignment_is_identity_for_template_landmarks():
    image = np.arange(112 * 112 * 3, dtype=np.uint8).reshape(112, 112, 3)
    aligned = align_face(image, ARCFACE_112_TEMPLATE)
    assert aligned.shape == image.shape
    assert np.mean(np.abs(aligned.astype(np.int16) - image.astype(np.int16))) < 1.0


def test_recognizer_normalizes_embedding_and_rgb_input():
    backend = _StaticBackend([np.array([[3.0, 4.0]], dtype=np.float32)])
    recognizer = FaceRecognizer(backend, model_type="lvface")
    face = np.zeros((112, 112, 3), dtype=np.uint8)
    face[:, :, 2] = 255  # BGR red becomes RGB channel 0.
    embedding = recognizer.embed(face)
    np.testing.assert_allclose(embedding, [0.6, 0.8])
    tensor = backend.inputs[0]
    assert tensor.shape == (1, 3, 112, 112)
    assert tensor[0, 0, 0, 0] == 1.0
    assert tensor[0, 2, 0, 0] == -1.0


def test_weighted_templates_and_matcher_use_subcenter_signal():
    features = np.array(
        [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.05, 0.95]],
        dtype=np.float32,
    )
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    weights = np.ones(4, dtype=np.float32)
    centroid = weighted_centroid(features, weights)
    subcenters = spherical_subcenters(
        features,
        weights,
        max_subcenters=2,
        enable_from=4,
    )
    assert centroid.shape == (2,)
    assert subcenters.shape == (2, 2)
    gallery = IdentityGallery(
        [
            IdentityTemplates("alice", centroid, subcenters, features, 4),
            IdentityTemplates(
                "bob",
                np.array([-1.0, 0.0], dtype=np.float32),
                np.array([[-1.0, 0.0]], dtype=np.float32),
                np.array([[-1.0, 0.0]], dtype=np.float32),
                1,
            ),
        ]
    )
    match = IdentityMatcher(gallery, centroid_weight=0.5).match([0.0, 1.0])
    assert match is not None and match.person_id == "alice"
    assert match.subcenter_score > match.centroid_score


def test_bbox_calibration_and_normalized_xywh():
    calibrated = calibrate_bbox(
        np.array([20, 20, 60, 80]),
        (100, 200, 3),
        x_scale=1.5,
        y_scale=1.2,
        y_shift=0.1,
    )
    np.testing.assert_allclose(calibrated, [10, 20, 70, 92])
    assert normalized_xywh(calibrated, (100, 200, 3)) == [0.05, 0.2, 0.3, 0.72]


class _Detector:
    def __init__(self):
        self.closed = False

    def detect(self, image):
        return [
            FaceDetection(
                np.array([0, 0, 112, 112], dtype=np.float32),
                0.95,
                ARCFACE_112_TEMPLATE.copy(),
            )
        ]

    def close(self):
        self.closed = True


class _Recognizer:
    def __init__(self, embedding):
        self.embedding = np.asarray(embedding, dtype=np.float32)
        self.closed = False

    def embed(self, aligned, *, flip_tta=False):
        assert aligned.shape == (112, 112, 3)
        return self.embedding

    def close(self):
        self.closed = True


def test_end_to_end_engine_returns_current_single_face_schema():
    template = IdentityTemplates(
        "n000001",
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
        1,
    )
    gallery = IdentityGallery([template])
    detector = _Detector()
    recognizer = _Recognizer([1.0, 0.0])
    engine = FaceIdentityEngine(
        detector,
        recognizer,
        gallery,
        IdentityMatcher(gallery),
    )
    ok, encoded = cv2.imencode(".jpg", np.zeros((112, 112, 3), dtype=np.uint8))
    assert ok
    payload = engine.infer_face_identity(encoded.tobytes())
    assert payload == {
        "detect_confidence": 0.95,
        "bbox_relative": [0.0, 0.0, 1.0, 1.0],
        "identity": {"person_id": "n000001", "confidence": 1.0},
    }
    engine.close()
    assert detector.closed and recognizer.closed
