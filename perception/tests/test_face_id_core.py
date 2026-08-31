"""Host-side tests for the model-independent face-identification core."""

from __future__ import annotations

import cv2
import json
import numpy as np

from plugins.face_id.alignment import (
    ARCFACE_112_TEMPLATE,
    alignment_rmse,
    align_face,
    rescue_face_alignments,
)
from plugins.face_id.detector import SCRFDDetector, distance_to_bbox, nms
from plugins.face_id.engine import (
    FaceIdentityEngine,
    _apply_runtime_profile,
    _has_explicit_model_pair,
)
from plugins.face_id.gallery import (
    IdentityGallery,
    IdentityTemplates,
    spherical_subcenters,
    weighted_centroid,
)
from plugins.face_id.matcher import IdentityMatcher
from plugins.face_id.loo import evaluate_gallery_leave_one_out
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


class _SequenceBackend(_StaticBackend):
    def __init__(self, output_batches):
        self.output_batches = list(output_batches)
        super().__init__(self.output_batches[0])

    def infer(self, array):
        self.inputs.append(np.array(array, copy=True))
        outputs = self.output_batches.pop(0)
        return [np.array(output, copy=True) for output in outputs]


def test_explicit_model_pair_requires_both_paths():
    assert _has_explicit_model_pair(
        {
            "detector_model": "/models/det.onnx",
            "recognizer_model": "/models/rec.onnx",
        }
    )
    assert not _has_explicit_model_pair({"detector_model": "/models/det.onnx"})
    assert not _has_explicit_model_pair({"recognizer_model": "/models/rec.onnx"})


def test_runtime_profiles_share_detector_and_select_recognizer_backend():
    mobile = _apply_runtime_profile({"runtime_profile": "mobile_cpu"})
    assert mobile["detector_backend"] == "opencv"
    assert mobile["recognizer_backend"] == "opencv"
    assert mobile["recognizer"] == "mobilefacenet"
    lvface = _apply_runtime_profile({"runtime_profile": "lvface_cpu"})
    assert lvface["detector_backend"] == "opencv"
    assert lvface["recognizer_backend"] == "onnx"
    assert lvface["recognizer"] == "lvface"
    assert lvface["onnx_intra_op_threads"] == 1
    adaface = _apply_runtime_profile({"runtime_profile": "adaface_cpu"})
    assert adaface["detector_backend"] == "opencv"
    assert adaface["recognizer_backend"] == "onnx"
    assert adaface["recognizer"] == "adaface-ir18"
    assert adaface["onnx_intra_op_threads"] == 1


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


def test_scrfd_threshold_override_is_per_call():
    outputs = _scrfd_outputs()
    outputs[0][0] = 0.1
    detector = SCRFDDetector(
        _StaticBackend(outputs),
        input_size=(64, 64),
        score_threshold=0.2,
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    assert detector.detect(image) == []
    assert len(detector.detect(image, score_threshold=0.05)) == 1
    assert detector.detect(image) == []


def test_scrfd_empty_retry_reuses_one_inference_for_weak_detection():
    outputs = _scrfd_outputs()
    outputs[0][0] = 0.11
    backend = _StaticBackend(outputs)
    detector = SCRFDDetector(
        backend,
        input_size=(64, 64),
        score_threshold=0.15,
    )
    detections = detector.detect_with_empty_retry(
        np.zeros((64, 64, 3), dtype=np.uint8),
        retry_score_threshold=0.10,
    )
    assert len(detections) == 1
    assert detections[0].score == np.float32(0.11)
    assert len(backend.inputs) == 1


def test_scrfd_empty_retry_keeps_default_detection_and_one_inference():
    backend = _StaticBackend(_scrfd_outputs())
    detector = SCRFDDetector(
        backend,
        input_size=(64, 64),
        score_threshold=0.15,
    )
    detections = detector.detect_with_empty_retry(
        np.zeros((64, 64, 3), dtype=np.uint8),
        retry_score_threshold=0.10,
    )
    assert len(detections) == 1
    assert detections[0].score == np.float32(0.9)
    assert len(backend.inputs) == 1


def test_scrfd_empty_retry_stays_empty_below_retry_threshold():
    outputs = _scrfd_outputs()
    outputs[0][0] = 0.09
    backend = _StaticBackend(outputs)
    detector = SCRFDDetector(
        backend,
        input_size=(64, 64),
        score_threshold=0.15,
    )
    detections = detector.detect_with_empty_retry(
        np.zeros((64, 64, 3), dtype=np.uint8),
        retry_score_threshold=0.10,
    )
    assert detections == []
    assert len(backend.inputs) == 1


def test_scrfd_rotation_rescue_maps_detection_back_to_source_coordinates():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    rotated, inverse = SCRFDDetector._rotate_view(image, 90)
    assert rotated.shape == (60, 40, 3)
    detection = FaceDetection(
        np.array([10, 20, 30, 40], dtype=np.float32),
        0.8,
        np.array(
            [[12, 22], [28, 22], [20, 30], [14, 38], [26, 38]],
            dtype=np.float32,
        ),
    )
    mapped = SCRFDDetector._map_detection(detection, inverse, image.shape)
    assert mapped is not None
    np.testing.assert_allclose(mapped.bbox, [20, 9, 40, 29])
    np.testing.assert_allclose(mapped.landmarks[0], [22, 27])


def test_scrfd_rotation_rescue_runs_only_after_original_is_empty():
    empty = _scrfd_outputs()
    empty[0][0] = 0.0
    backend = _SequenceBackend([empty, _scrfd_outputs()])
    detector = SCRFDDetector(
        backend,
        input_size=(64, 64),
        score_threshold=0.15,
    )
    detections = detector.detect_with_empty_retry(
        np.zeros((40, 60, 3), dtype=np.uint8),
        retry_score_threshold=0.10,
        rotations=(90,),
    )
    assert len(detections) == 1
    assert len(backend.inputs) == 2


def test_scrfd_tile_rescue_maps_crop_detection_and_runs_all_tiles():
    empty = _scrfd_outputs()
    empty[0][0] = 0.0
    backend = _SequenceBackend(
        [empty, _scrfd_outputs(), empty, empty, empty, empty]
    )
    detector = SCRFDDetector(
        backend,
        input_size=(64, 64),
        score_threshold=0.15,
    )
    detections = detector.detect_with_empty_retry(
        np.zeros((80, 100, 3), dtype=np.uint8),
        retry_score_threshold=0.10,
        tile_fraction=0.75,
    )
    assert len(detections) == 1
    assert detections[0].bbox[0] >= 12
    assert detections[0].bbox[1] >= 10
    assert len(backend.inputs) == 6


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


def test_alignment_rescue_generates_hypotheses_for_corrupted_landmark():
    landmarks = ARCFACE_112_TEMPLATE.copy()
    landmarks += np.array(
        [[2, -1], [-2, 1], [0, 0], [1, 2], [-1, -2]],
        dtype=np.float32,
    )
    landmarks[2] += [28.0, 20.0]
    assert alignment_rmse(landmarks) > 7.0
    hypotheses = rescue_face_alignments(
        np.zeros((112, 112, 3), dtype=np.uint8),
        landmarks,
    )
    assert "eyes_only" in [name for name, _aligned in hypotheses]
    assert len(hypotheses) >= 3
    assert all(aligned.shape == (112, 112, 3) for _name, aligned in hypotheses)


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


def test_adaface_alias_uses_shared_rgb_normalization():
    backend = _StaticBackend([np.array([[0.0, 2.0]], dtype=np.float32)])
    recognizer = FaceRecognizer(backend, model_type="adaface-ir18")
    face = np.zeros((112, 112, 3), dtype=np.uint8)
    face[:, :, 2] = 255
    np.testing.assert_allclose(recognizer.embed(face), [0.0, 1.0])
    assert backend.inputs[0][0, 0, 0, 0] == 1.0


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


def test_matcher_rank_and_leave_one_out_exclude_query_template():
    def template(person_id, rows):
        features = np.asarray(rows, dtype=np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        weights = np.ones(len(features), dtype=np.float32)
        return IdentityTemplates(
            person_id,
            weighted_centroid(features, weights),
            features.copy(),
            features,
            len(features),
            exemplar_weights=weights,
            query_exemplars=features.copy(),
        )

    gallery = IdentityGallery(
        [
            template("alice", [[1.0, 0.0], [0.98, 0.02]]),
            template("bob", [[0.0, 1.0], [0.02, 0.98]]),
            template("singleton", [[-1.0, 0.0]]),
        ]
    )
    ranked = IdentityMatcher(gallery).rank([0.1, 0.9], top_k=2)
    assert [item.person_id for item in ranked] == ["bob", "alice"]
    result = evaluate_gallery_leave_one_out(gallery, centroid_weight=0.6)
    assert result["queries"] == 4
    assert result["eligible_identities"] == 2
    assert result["skipped_singletons"] == 1
    assert result["top1_accuracy"] == 1.0
    assert result["top5_accuracy"] == 1.0


def test_subcenter_rescue_flips_only_a_low_margin_top2_tie():
    alice_subcenter = np.array([0.7, np.sqrt(1.0 - 0.7**2)], dtype=np.float32)
    bob_centroid = np.array([0.75, np.sqrt(1.0 - 0.75**2)], dtype=np.float32)
    alice = IdentityTemplates(
        "alice",
        np.array([1.0, 0.0], dtype=np.float32),
        alice_subcenter[None],
        alice_subcenter[None],
        1,
    )
    bob = IdentityTemplates(
        "bob",
        bob_centroid,
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
        1,
    )
    matcher = IdentityMatcher(IdentityGallery([alice, bob]))
    baseline = matcher.rank([1.0, 0.0], top_k=2)
    assert [item.person_id for item in baseline] == ["alice", "bob"]
    reranked = matcher.rank_with_subcenter_rescue(
        [1.0, 0.0],
        margin_max=0.05,
        min_margin_gain=0.04,
        top_k=2,
    )
    assert [item.person_id for item in reranked] == ["bob", "alice"]
    gated = matcher.rank_with_subcenter_rescue(
        [1.0, 0.0],
        margin_max=0.01,
        min_margin_gain=0.04,
        top_k=2,
    )
    assert [item.person_id for item in gated] == ["alice", "bob"]


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


class _EmptyRetryDetector(_Detector):
    def __init__(self):
        super().__init__()
        self.retry_thresholds = []

    def detect(self, image):
        raise AssertionError("configured empty retry must use its dedicated path")

    def detect_with_empty_retry(
        self,
        image,
        *,
        retry_score_threshold,
        rotations=(),
        tile_fraction=None,
        rescue_min_face_ratio=0.0,
    ):
        self.retry_thresholds.append(
            (
                retry_score_threshold,
                tuple(rotations),
                tile_fraction,
                rescue_min_face_ratio,
            )
        )
        return super().detect(image)


class _Recognizer:
    def __init__(self, embedding):
        self.embedding = np.asarray(embedding, dtype=np.float32)
        self.closed = False
        self.calls = 0

    def embed(self, aligned, *, flip_tta=False):
        assert aligned.shape == (112, 112, 3)
        self.calls += 1
        return self.embedding

    def close(self):
        self.closed = True


class _SequenceRecognizer(_Recognizer):
    def __init__(self, embeddings):
        self.embeddings = [np.asarray(item, dtype=np.float32) for item in embeddings]
        self.closed = False
        self.calls = 0

    def embed(self, aligned, *, flip_tta=False):
        assert aligned.shape == (112, 112, 3)
        self.calls += 1
        return self.embeddings.pop(0)


class _MultiDetector(_Detector):
    def detect(self, image):
        return [
            FaceDetection(
                np.array([0, 0, 56, 112], dtype=np.float32),
                0.99,
                ARCFACE_112_TEMPLATE.copy(),
            ),
            FaceDetection(
                np.array([56, 0, 112, 112], dtype=np.float32),
                0.90,
                ARCFACE_112_TEMPLATE.copy(),
            ),
        ]


class _CorruptLandmarkDetector(_Detector):
    def detect(self, image):
        landmarks = ARCFACE_112_TEMPLATE.copy()
        landmarks += np.array(
            [[2, -1], [-2, 1], [0, 0], [1, 2], [-1, -2]],
            dtype=np.float32,
        )
        landmarks[2] += [28.0, 20.0]
        return [
            FaceDetection(
                np.array([0, 0, 112, 112], dtype=np.float32),
                0.95,
                landmarks,
            )
        ]


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


def test_end_to_end_engine_routes_empty_detection_retry_through_matching():
    gallery = _diagnostic_gallery()
    detector = _EmptyRetryDetector()
    engine = FaceIdentityEngine(
        detector,
        _Recognizer([1.0, 0.0]),
        gallery,
        IdentityMatcher(gallery),
        empty_detection_retry_threshold=0.10,
        empty_detection_retry_rotations=(90, 180),
        empty_detection_retry_tile_fraction=0.75,
        empty_detection_retry_min_face_ratio=0.12,
    )
    payload = engine.infer_image(np.zeros((112, 112, 3), dtype=np.uint8))
    assert detector.retry_thresholds == [(0.10, (90, 180), 0.75, 0.12)]
    assert payload["identity"]["person_id"] == "alice"


def test_high_residual_alignment_rescue_can_replace_baseline_identity():
    gallery = _diagnostic_gallery()
    probe = _CorruptLandmarkDetector().detect(
        np.zeros((112, 112, 3), dtype=np.uint8)
    )[0]
    alternate_count = len(
        rescue_face_alignments(
            np.zeros((112, 112, 3), dtype=np.uint8),
            probe.landmarks,
        )
    )
    embeddings = [[0.65, 0.75]]
    embeddings.extend([[0.65, 0.75]] * alternate_count)
    embeddings[3] = [1.0, 0.0]
    recognizer = _SequenceRecognizer(embeddings)
    engine = FaceIdentityEngine(
        _CorruptLandmarkDetector(),
        recognizer,
        gallery,
        IdentityMatcher(gallery),
        alignment_rescue_rmse_min=7.0,
        alignment_rescue_min_score_gain=0.02,
    )
    payload = engine.infer_image(np.zeros((112, 112, 3), dtype=np.uint8))
    assert payload["identity"]["person_id"] == "alice"
    assert recognizer.calls == alternate_count + 1


def test_low_residual_alignment_uses_only_standard_embedding():
    gallery = _diagnostic_gallery()
    recognizer = _Recognizer([1.0, 0.0])
    engine = FaceIdentityEngine(
        _Detector(),
        recognizer,
        gallery,
        IdentityMatcher(gallery),
        alignment_rescue_rmse_min=7.0,
    )
    payload = engine.infer_image(np.zeros((112, 112, 3), dtype=np.uint8))
    assert payload["identity"]["person_id"] == "alice"
    assert recognizer.calls == 1


def _diagnostic_gallery():
    return IdentityGallery(
        [
            IdentityTemplates(
                "alice",
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([[1.0, 0.0]], dtype=np.float32),
                np.array([[1.0, 0.0]], dtype=np.float32),
                1,
            ),
            IdentityTemplates(
                "bob",
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([[0.0, 1.0]], dtype=np.float32),
                np.array([[0.0, 1.0]], dtype=np.float32),
                1,
            ),
        ]
    )


def _diagnostic_records(caplog):
    prefix = "[face-diagnostic] "
    return [
        json.loads(record.message[len(prefix) :])
        for record in caplog.records
        if record.message.startswith(prefix)
    ]


def test_diagnostics_log_topk_without_changing_payload(caplog):
    gallery = _diagnostic_gallery()
    engine = FaceIdentityEngine(
        _Detector(),
        _Recognizer([1.0, 0.0]),
        gallery,
        IdentityMatcher(gallery),
        face_selection="gallery_match",
        diagnostics_enabled=True,
        diagnostics_top_k=5,
    )
    with caplog.at_level("INFO", logger="plugins.face_id.engine"):
        payload = engine.infer_image(np.zeros((112, 112, 3), dtype=np.uint8))
    assert payload == {
        "detect_confidence": 0.95,
        "bbox_relative": [0.0, 0.0, 1.0, 1.0],
        "identity": {"person_id": "alice", "confidence": 1.0},
    }
    records = _diagnostic_records(caplog)
    assert len(records) == 1
    assert records[0]["sequence"] == 1
    assert records[0]["selected_candidate_index"] == 0
    assert [item["person_id"] for item in records[0]["candidates"][0]["top"]] == [
        "alice",
        "bob",
    ]
    assert records[0]["candidates"][0]["quality"]["alignment_rmse"] < 0.01


def test_low_margin_fallback_changes_identity_on_the_same_face():
    gallery = _diagnostic_gallery()
    mobile = _Recognizer([0.70, 0.71])
    fallback = _Recognizer([1.0, 0.0])
    engine = FaceIdentityEngine(
        _Detector(),
        mobile,
        gallery,
        IdentityMatcher(gallery),
        fallback_recognizer=fallback,
        fallback_matcher=IdentityMatcher(gallery),
        fallback_mobile_margin_max=0.05,
        fallback_margin_min=0.05,
    )
    payload = engine.infer_image(np.zeros((112, 112, 3), dtype=np.uint8))
    assert payload["bbox_relative"] == [0.0, 0.0, 1.0, 1.0]
    assert payload["identity"]["person_id"] == "alice"
    assert mobile.calls == 1
    assert fallback.calls == 1
    engine.close()
    assert fallback.closed


def test_high_margin_mobile_result_skips_fallback():
    gallery = _diagnostic_gallery()
    fallback = _Recognizer([0.0, 1.0])
    engine = FaceIdentityEngine(
        _Detector(),
        _Recognizer([1.0, 0.0]),
        gallery,
        IdentityMatcher(gallery),
        fallback_recognizer=fallback,
        fallback_matcher=IdentityMatcher(gallery),
        fallback_mobile_margin_max=0.05,
        fallback_margin_min=0.05,
    )
    payload = engine.infer_image(np.zeros((112, 112, 3), dtype=np.uint8))
    assert payload["identity"]["person_id"] == "alice"
    assert fallback.calls == 0


class _NoFaceThenWeakDetector(_Detector):
    def __init__(self):
        super().__init__()
        self.thresholds = []

    def detect(self, image, *, score_threshold=None):
        self.thresholds.append(score_threshold)
        if score_threshold == 0.05:
            return [
                FaceDetection(
                    np.array([0, 0, 112, 112], dtype=np.float32),
                    0.07,
                    ARCFACE_112_TEMPLATE.copy(),
                )
            ]
        return []


def test_empty_diagnostics_probe_weak_faces_but_keep_empty_payload(caplog):
    gallery = _diagnostic_gallery()
    detector = _NoFaceThenWeakDetector()
    engine = FaceIdentityEngine(
        detector,
        _Recognizer([1.0, 0.0]),
        gallery,
        IdentityMatcher(gallery),
        diagnostics_enabled=True,
        diagnostics_retry_thresholds=(0.1, 0.05),
    )
    with caplog.at_level("INFO", logger="plugins.face_id.engine"):
        payload = engine.infer_image(np.zeros((112, 112, 3), dtype=np.uint8))
    assert payload == {
        "detect_confidence": 0.0,
        "bbox_relative": None,
        "identity": None,
    }
    assert detector.thresholds == [None, 0.1, 0.05]
    record = _diagnostic_records(caplog)[0]
    assert record["detections"] == 0
    assert [probe["raw_detections"] for probe in record["empty_detection_probes"]] == [
        0,
        1,
    ]
    assert record["empty_detection_probes"][1]["candidates"][0]["top"][0][
        "person_id"
    ] == "alice"


def test_gallery_selection_recognizes_all_faces_before_choosing():
    gallery = IdentityGallery(
        [
            IdentityTemplates(
                "alice",
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([[1.0, 0.0]], dtype=np.float32),
                np.array([[1.0, 0.0]], dtype=np.float32),
                1,
            ),
            IdentityTemplates(
                "bob",
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([[0.0, 1.0]], dtype=np.float32),
                np.array([[0.0, 1.0]], dtype=np.float32),
                1,
            ),
        ]
    )
    engine = FaceIdentityEngine(
        _MultiDetector(),
        _SequenceRecognizer([[0.6, 0.8], [1.0, 0.0]]),
        gallery,
        IdentityMatcher(gallery),
        face_selection="gallery_match",
    )
    payload = engine.infer_image(np.zeros((112, 112, 3), dtype=np.uint8))
    assert payload["bbox_relative"] == [0.5, 0.0, 0.5, 1.0]
    assert payload["identity"]["person_id"] == "alice"


def test_leave_one_out_can_select_best_query_face():
    features = np.array([[1.0, 0.0], [0.98, 0.02]], dtype=np.float32)
    weights = np.ones(2, dtype=np.float32)
    gallery = IdentityGallery(
        [
            IdentityTemplates(
                "alice",
                weighted_centroid(features, weights),
                features.copy(),
                features,
                2,
                exemplar_weights=weights,
                query_exemplars=features.copy(),
            ),
            IdentityTemplates(
                "bob",
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([[0.0, 1.0]], dtype=np.float32),
                np.array([[0.0, 1.0]], dtype=np.float32),
                1,
            ),
        ]
    )
    result = evaluate_gallery_leave_one_out(
        gallery,
        candidate_queries={
            ("alice", 0): np.array([[0.5, 0.5], [1.0, 0.0]], dtype=np.float32)
        },
    )
    first = next(
        item
        for item in result["details"]
        if item["person_id"] == "alice" and item["exemplar_index"] == 0
    )
    assert first["candidate_count"] == 2
    assert first["selected_candidate_index"] == 1
    assert first["true_rank"] == 1
