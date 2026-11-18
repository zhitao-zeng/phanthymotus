"""Received-segment boundaries, real node ingress, and batched frame order."""
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from plugins import pose
from plugins.pose_segment import SegmentBuffer, SegmentActionBackend, rgb_frame_indices
from plugins.vision_runtime import VisionEngineSession
from test_pose_plugin import _plugin, _publisher, _standing_row
from test_pose_idle_finalize import _FinalClassifier
from vision_stubs import _FakeCompressedImage, _wait_until, frame_bytes


def test_buffer_separates_a_pause_even_when_worker_has_not_polled():
    buffer = SegmentBuffer(quiet_s=.2)
    for i in range(4):
        buffer.append(bytes([i]), i*.01)
    for i in range(4, 8):
        buffer.append(bytes([i]), 1.+(i-4)*.01)
    first, stats = buffer.pop_ready(1.04)
    assert [x[1] for x in first] == [bytes([i]) for i in range(4)]
    assert stats['received_frames'] == 4
    assert buffer.pop_ready(1.1) is None
    second, _ = buffer.pop_ready(1.3)
    assert [x[1] for x in second] == [bytes([i]) for i in range(4, 8)]
    assert buffer.pop_ready(2.) is None


def test_buffer_reports_truncation_and_reset_discards_queued_segments():
    buffer = SegmentBuffer(quiet_s=.2, max_bytes=6)
    for i in range(5):
        buffer.append(b'ab', i*.01)
    frames, stats = buffer.pop_ready(1.)
    assert len(frames) == 3 and stats['received_frames'] == 5
    assert stats['truncated_frames'] == 2
    buffer.append(b'old', 2.)
    buffer.append(b'new', 3.)
    buffer.reset()
    assert buffer.pop_ready(4.) is None


class _SegmentModel:
    last_error = None

    def __init__(self, wait_for_stop=False):
        self.clips = []
        self.entered = threading.Event()
        self.wait_for_stop = wait_for_stop

    def recognize(self, received, stop_event):
        self.clips.append(received)
        self.entered.set()
        if self.wait_for_stop:
            stop_event.wait(2.)
        row = np.array(_standing_row(), np.float32)
        return {'image': np.zeros((480, 640, 3), np.uint8),
                'boxes': row[None, :4], 'scores': row[None, 4],
                'keypoints': row[6:].reshape(1, 17, 3), 'primary_box': row[:4],
                'activity': {'name': 'brush teeth', 'name_zh': '刷牙',
                             'score': .81, 'action_id': 3},
                'evidence': {'frames_used': len(received)}}


def _start_segment_node(backend):
    plugin, _ = _plugin({'fps': 1, 'activity_mode': 'segment',
                          'activity_idle_flush_s': .12, 'label_hold': 1})
    plugin._segment_backend_for = lambda cfg: backend
    plugin._classifier_for = lambda cfg: _FinalClassifier()
    plugin.dispatch('pose', {'action': 'start', 'input_topic': '/cam/segment'})
    return plugin._nodes['/cam/segment']


def test_real_node_buffers_all_received_images_before_its_fps_gate():
    backend = _SegmentModel()
    node = _start_segment_node(backend)
    try:
        for _ in range(6):
            node.subscriptions[0].callback(_FakeCompressedImage(frame_bytes(640, 480)))
        _wait_until(lambda: node._detect_count == 1)
        lean = _publisher(node, node._output_topic)
        assert all('activity' not in p for msg in lean.messages
                   for p in json.loads(msg)['persons'])
        _wait_until(lambda: node._segment_count == 1)
        _wait_until(lambda: len(lean.messages) == 2)
        assert len(backend.clips) == 1 and len(backend.clips[0]) == 1
        assert backend.clips[0][0][2] is not None
        assert json.loads(lean.messages[-1])['persons'][0]['activity_score'] == .81
        assert node._detect_count == 1
        assert len(_publisher(node, node._skeleton_topic).messages) == 1
        assert node._last_prediction['received_frames'] == 6
        assert node._last_prediction['processed_observations'] == 1
    finally:
        node.stop()


def test_stop_cancels_publication_and_discards_input_from_a_stopped_subscription():
    backend = _SegmentModel(wait_for_stop=True)
    node = _start_segment_node(backend)
    try:
        for _ in range(4):
            node._image_cb(_FakeCompressedImage(frame_bytes(640, 480)))
        assert backend.entered.wait(2.)
        node.stop()
        node._image_cb(_FakeCompressedImage(frame_bytes(640, 480)))
        assert node._segment_buffer.pop_ready(1e30) is None
        assert node._segment_count == 0
        assert len(_publisher(node, node._output_topic).messages) == 1
    finally:
        node.stop()


@pytest.mark.parametrize('fps,pause,expected', [
    # Fractional FPS retains the existing integer configuration conversion.
    (30., 0., .15), (15., 0., .15), (7.15, 0., 2/7), (1., 0., 2.),
    (30., .4, .4), (1., .4, .4), (1., .12, .12),
])
def test_segment_auto_pause_respects_frame_period_and_explicit_config(fps, pause, expected):
    plugin, _ = _plugin({'fps': fps, 'activity_mode': 'segment',
                          'activity_idle_flush_s': pause})
    plugin._segment_backend_for = lambda cfg: _SegmentModel()
    plugin._classifier_for = lambda cfg: _FinalClassifier()
    plugin.dispatch('pose', {'action': 'start', 'input_topic': '/cam/auto_pause'})
    node = plugin._nodes['/cam/auto_pause']
    try:
        assert node._activity_idle_flush_s == pytest.approx(expected)
        assert node._segment_buffer.quiet_s == pytest.approx(pause or max(.4, expected))
    finally:
        node.stop()


def test_auto_segment_retains_frames_across_short_pause_and_publishes_once(monkeypatch):
    now = 10.
    monkeypatch.setattr(pose, 'time', SimpleNamespace(
        monotonic=lambda: now, perf_counter=time.perf_counter, time=time.time))
    backend = _SegmentModel()
    node = pose._PoseNode('/cam/segment', None, .35, 30., 'quiet',
                          classifier=_FinalClassifier(), segment_backend=backend)
    messages = _publisher(node, node._output_topic).messages
    row = np.array(_standing_row(), np.float32)
    buffer = node._segment_buffer

    def record_frames(start):
        for i in range(4):
            timestamp = start+i*.03
            buffer.append(bytes([i]), timestamp)
            buffer.record_pose(timestamp, row[None, :4], row[None, 4],
                               row[6:].reshape(1, 17, 3), (480, 640, 3))

    record_frames(10.)
    now = 10.26
    node._flush_received_segment()
    assert [len(clip) for clip in backend.clips] == [4]
    assert messages == []

    record_frames(10.34)
    now = 10.60
    node._flush_received_segment()
    assert [len(clip) for clip in backend.clips] == [4, 8]
    assert messages == []

    now = 10.84
    node._flush_received_segment()
    node._flush_received_segment()
    assert [len(clip) for clip in backend.clips] == [4, 8]
    assert len(messages) == 1
    evidence = node._last_prediction
    assert evidence['received_frames'] == evidence['processed_observations'] == 8


def test_resume_during_preview_reclassifies_full_segment_in_real_worker(monkeypatch):
    now = 10.
    monkeypatch.setattr(pose, 'time', SimpleNamespace(
        monotonic=lambda: now, perf_counter=time.perf_counter, time=time.time))
    release_first = threading.Event()

    class BlockedPreview(_SegmentModel):
        def recognize(self, received, stop_event):
            result = super().recognize(received, stop_event)
            if len(self.clips) == 1:
                if not release_first.wait(2.):
                    raise RuntimeError('Test did not release the first forward')
                result['activity'] = {**result['activity'], 'name': 'drink water'}
            return result

    backend = BlockedPreview()
    plugin, _ = _plugin({'activity_mode': 'segment', 'label_hold': 1})
    plugin._segment_backend_for = lambda cfg: backend
    plugin._classifier_for = lambda cfg: _FinalClassifier()
    topic = '/cam/segment'
    plugin.dispatch('pose', {'action': 'config', 'instance_id': topic, 'fps': 30})
    plugin.dispatch('pose', {'action': 'start', 'input_topic': topic})
    node = plugin._nodes[topic]
    callback = node.subscriptions[0].callback
    lean = _publisher(node, node._output_topic)
    frame = _FakeCompressedImage(frame_bytes(640, 480))

    def feed(when, expected):
        nonlocal now
        now = when
        callback(frame)
        assert _wait_until(lambda: node._detect_count == expected)

    def activities():
        return [person['activity'] for raw in lean.messages
                for person in json.loads(raw)['persons'] if 'activity' in person]

    try:
        for index in range(4):
            feed(10. + index*.05, index+1)
        now = 10.32
        assert backend.entered.wait(2.)
        assert [len(x) for x in backend.clips] == [4]
        assert activities() == []

        # Deliver a frame while the first inference is still in flight.
        now = 10.40
        callback(frame)
        release_first.set()
        assert _wait_until(lambda: node._detect_count == 5)
        for index in range(1, 4):
            feed(10.40 + index*.05, index+5)
        now = 10.73
        assert _wait_until(lambda: len(backend.clips) == 2)
        assert [len(x) for x in backend.clips] == [4, 8]
        assert activities() == []

        now = 11.0
        assert _wait_until(lambda: activities() == ['brush teeth'])
        assert len(backend.clips) == 2  # Publishing reused the complete preview.
        assert node._detect_count == 8
        assert node._last_prediction['processed_observations'] == 8
    finally:
        release_first.set()
        node.stop()


def test_dynamic_pose_batches_preserve_input_order_and_individual_letterboxes():
    class Engine:
        input_dtype = np.float32
        profiles = [((1, 3, 32, 32), (3, 3, 32, 32), (3, 3, 32, 32))]

        def __init__(self):
            self.calls = []
            self.count = 0

        def infer(self, blob):
            self.calls.append(blob.shape)
            n = len(blob)
            values = np.arange(self.count, self.count+n).reshape(n, 1, 1)
            self.count += n
            return [values]

    session = VisionEngineSession.__new__(VisionEngineSession)
    session._engine = Engine()
    session._in_w = session._in_h = 32
    images = [np.zeros((20+i, 30+i, 3), np.uint8) for i in range(5)]
    results = session.infer_batch(images)
    assert session._engine.calls == [(3, 3, 32, 32), (2, 3, 32, 32)]
    assert [int(output[0].item()) for output, _ in results] == list(range(5))
    assert [(meta.orig_h, meta.orig_w) for _, meta in results] == [(20+i, 30+i) for i in range(5)]


def test_pose_finishing_after_a_gap_stays_with_its_received_segment():
    buffer = SegmentBuffer(quiet_s=.2)
    buffer.append(b'old', 0.)
    buffer.append(b'new', 1.)
    boxes = np.array([[1., 2., 3., 4.]])
    buffer.record_pose(0., boxes, np.array([.9]), np.zeros((1, 17, 3)), (480, 640, 3))
    boxes[0, 0] = 999.
    frames, _ = buffer.pop_ready(1.01)
    assert len(frames) == 1 and frames[0][1] == b'old'
    assert frames[0][2][0][0, 0] == 1.
    buffer.reset()
    assert not buffer._by_time


@pytest.mark.parametrize('rgb_frames', [8, 24])
@pytest.mark.parametrize('handface_state', ['disabled', 'skipped', 'applied', 'fallback'])
@pytest.mark.parametrize('defer_handface', [False, True])
@pytest.mark.parametrize('handface_frames', [4, 8])
def test_final_classification_reuses_poses_and_never_calls_the_pose_engine(
        rgb_frames, handface_state, defer_handface, handface_frames, monkeypatch):
    import cv2

    class PoseMustNotRun:
        def infer(self, image):
            raise AssertionError('pose inference after the input pause')

        def infer_batch(self, images):
            raise AssertionError('pose batch after the input pause')

    class Head:
        def __init__(self, shape):
            self.input_shape = shape

        def infer(self, blob):
            assert blob.shape == self.input_shape
            p = np.zeros((1, 126), np.float32)
            p[0, 2], p[0, 37] = .9, .1
            return [p]

    class HandFace:
        FRAME_COUNT = 8
        wholebody_frame_count = handface_frames

        def should_run(self, probabilities, fps):
            assert probabilities.argmax() == 2
            assert fps == pytest.approx(10)
            return handface_state != 'skipped'

        def predict(self, poses, images, reconstruction):
            assert len(poses) == 5 and len(images) == handface_frames
            if handface_frames == 4:
                assert reconstruction.shape == (8,)
                assert reconstruction.max() < 4
            else:
                assert reconstruction is None
            if handface_state == 'fallback':
                raise RuntimeError('HandFace engine stopped')
            return np.ones(126, dtype=np.float32) / 126, .5

        def fuse(self, body, rgb, handface, quality, fps):
            assert fps == pytest.approx(10)
            p = np.zeros(126, dtype=np.float32)
            p[8] = 1.
            return p

    row = np.array(_standing_row(), np.float32)
    decoded = (row[None, :4], row[None, 4], row[6:].reshape(1, 17, 3), (480, 640))
    received = [(i*.1, frame_bytes(640, 480), decoded) for i in range(5)]
    backend = SegmentActionBackend(PoseMustNotRun(), Head((1, 2, 100, 17, 6)),
                                   Head((1, rgb_frames, 3, 192, 192)),
                                   handface=None if handface_state == 'disabled' else HandFace())
    original_decode = cv2.imdecode
    calls = []

    def count_decode(data, flags):
        calls.append(1)
        return original_decode(data, flags)

    monkeypatch.setattr(cv2, 'imdecode', count_decode)
    result = backend.recognize(received, defer_handface=defer_handface)
    # The last primary frame is already one of the RGB classifier's inputs.
    expected_decodes = set(rgb_frame_indices(5, rgb_frames//2))
    if defer_handface and handface_state in ('applied', 'fallback'):
        assert result['activity']['action_id'] == 3
        assert result['evidence']['handface_state'] == 'pending'
        assert len(calls) == len(expected_decodes)
        result = backend.complete_handface(result)
    if handface_state in ('applied', 'fallback') and handface_frames == 8:
        expected_decodes.update(np.linspace(0, 4, 8).round().astype(int))
    assert len(calls) == len(expected_decodes)
    assert '_handface_pending' not in result
    assert np.array_equal(result['image'], original_decode(
        np.frombuffer(received[-1][1], np.uint8), cv2.IMREAD_COLOR))
    assert result['activity']['action_id'] == (9 if handface_state == 'applied' else 3)
    assert result['activity']['score'] == (1. if handface_state == 'applied' else .9)
    if handface_state != 'disabled':
        assert result['evidence']['handface_state'] == handface_state
    else:
        assert 'handface_state' not in result['evidence']
    assert result['evidence']['poses_computed_during_reception'] is True
    assert result['evidence']['RGB_frames_per_view'] == rgb_frames//2
    assert backend.recognize(received[:3]) is None


def test_handface_uses_sparse_observations_with_a_relevant_candidate():
    from plugins.pose_handface import HandFaceActionBackend

    clusters = np.zeros(126, dtype=np.int64)
    clusters[8] = 1
    backend = HandFaceActionBackend(None, None, clusters, np.ones(17))
    probabilities = np.zeros(126, dtype=np.float32)
    probabilities[2], probabilities[8] = .66, .34
    assert backend.should_run(probabilities, 8)
    assert backend.should_run(probabilities, 10)
    assert not backend.should_run(probabilities, 15)
    probabilities[3], probabilities[8] = .34, 0.
    assert not backend.should_run(probabilities, 8)


@pytest.mark.parametrize('with_handface', [False, True])
def test_model_bundle_spec_survives_creation_of_an_instance_backend(tmp_path, monkeypatch, with_handface):
    from utils import tensorrt_runtime

    loaded = []

    class Engine:
        def __init__(self, path, **kwargs):
            loaded.append(path.name)
            self.input_shape = {
                'body-new.engine': (1, 2, 100, 17, 6),
                'teacher-new.engine': (1, 24, 3, 192, 192),
                'wholebody.engine': (8, 3, 256, 192),
                'handface.engine': (1, 8, 894),
            }[path.name]

        def close(self):
            pass

    monkeypatch.setattr(tensorrt_runtime, 'TensorRTEngine', Engine)
    bundle = {'body_engine': 'body-new.engine', 'rgb_engine': 'teacher-new.engine', 'rgb_ntu_weight': .5}
    expected_models = ['body-new.engine', 'teacher-new.engine']
    if with_handface:
        bundle['handface'] = {
            'wholebody_engine': 'wholebody.engine', 'action_engine': 'handface.engine', 'gate': 'gate.json',
        }
        (tmp_path/'gate.json').write_text(json.dumps({'cluster_ids': [3] * 126, 'theta_3way_8f': [1.] * 17}))
        expected_models.extend(['wholebody.engine', 'handface.engine'])
    (tmp_path/'models.json').write_text(json.dumps(bundle))
    backend = SegmentActionBackend.from_directory(object(), tmp_path)
    assert loaded == expected_models
    plugin, _ = _plugin({'activity_mode': 'segment'})
    plugin._segment_model = backend
    monkeypatch.setenv('ACTION_SEGMENT_MODEL_DIR', str(tmp_path))
    instance = plugin._segment_backend_for(plugin._merged_config('new-instance'))
    assert instance.rgb_n_per_view == 12
    assert instance.ntu_rgb_weight == .5
    assert (instance.handface is not None) == with_handface
    assert instance.handface is backend.handface


def test_raw_buffer_unlocks_full_fps_rgb_and_handface_while_gating_on_posed_fps():
    """Opt #2 + Opt #3: un-posed buffered JPEGs feed RGB and HandFace cleanly."""
    import cv2
    from plugins.pose_segment import interpolate_keypoints_at_times, time_uniform_pick_indices

    def make_colored_jpeg(marker: int):
        img = np.full((120, 160, 3), marker, dtype=np.uint8)
        ok, enc = cv2.imencode('.jpg', img)
        assert ok
        return bytes(enc)

    # 21 raw frames at 20 Hz (markers 10..210), but YOLO only posed 6 (~5 Hz)
    # leaving a mid-clip stall between t=0.2 and t=0.8.
    raw_received = []
    posed_indices = {0, 2, 4, 16, 18, 20}
    row = np.array(_standing_row(), np.float32)
    for i in range(21):
        t = i * 0.05
        jpg = make_colored_jpeg(10 + i * 10)
        pose_tuple = None
        if i in posed_indices:
            kps = row[6:].reshape(1, 17, 3).copy()
            kps[0, 9, :2] = [80.0 + 40.0 * t, 60.0]  # wrist moves linearly with t
            kps[0, 9, 2] = 0.9
            pose_tuple = (row[None, :4].copy(), row[None, 4].copy(), kps, (120, 160))
        raw_received.append([t, jpg, pose_tuple])
    received = [r for r in raw_received if r[2] is not None]

    # Verify timestamp pick & linear keypoint interpolation directly.
    picked = time_uniform_pick_indices([r[0] for r in raw_received], 4)
    assert picked.tolist() == [0, 7, 13, 20]
    interp = interpolate_keypoints_at_times(
        np.stack([r[2][2][0] for r in received]),
        np.array([r[0] for r in received]),
        np.array([0.0, 0.5, 1.0]),
    )
    assert interp[1, 9, 0] == pytest.approx(100.0, abs=1e-4)

    class Head:
        def __init__(self, shape):
            self.input_shape = shape
            self.seen_blobs = []

        def infer(self, blob):
            self.seen_blobs.append(blob.copy())
            p = np.zeros((1, 126), np.float32)
            p[0, 2], p[0, 37] = .9, .1
            return [p]

    seen_hf_markers = []

    class HandFace:
        FRAME_COUNT = 8
        wholebody_frame_count = 8

        def should_run(self, probabilities, fps):
            # Posed FPS = (6 - 1) / 1.0 = 5 Hz (<= 10 Hz), NOT 20 Hz raw!
            assert fps == pytest.approx(5.0)
            return True

        def predict(self, poses, images, reconstruction):
            assert len(images) == 8 and reconstruction is None
            # Each image is solid marker color (before JPEG loss ~+-2); assert distinct un-posed mid frames included!
            markers = [int(round(float(img[:, :, 0].mean()))) for img in images]
            seen_hf_markers.extend(markers)
            return np.ones(126, dtype=np.float32) / 126, .8

        def fuse(self, body, rgb, handface, quality, fps):
            assert fps == pytest.approx(5.0)
            p = np.zeros(126, dtype=np.float32)
            p[8] = 1.
            return p

    rgb_head = Head((1, 8, 3, 192, 192))
    backend = SegmentActionBackend(object(), Head((1, 2, 100, 17, 6)), rgb_head, handface=HandFace())
    result = backend.recognize(received, raw_received=raw_received)
    assert result['evidence']['rgb_source'] == 'raw-buffer'
    assert result['evidence']['raw_frames_available'] == 21
    assert result['evidence']['handface_state'] == 'applied'
    assert result['evidence']['handface_input_fps'] == pytest.approx(5.0)
    # 8 HandFace frames span 0..20 uniformly -> middle frames (e.g., idx 6~11 => marker ~70..120) must NOT all come from posed edges!
    assert any(55 <= m <= 165 for m in seen_hf_markers)


def test_worker_next_timeout_wakes_at_exact_preview_and_confirm_deadlines(monkeypatch):
    """Opt #4b + #4a: exact wakeup deadlines avoid 0.1s polling tail lag."""
    from pathlib import Path

    now = 10.0
    monkeypatch.setattr(pose, 'time', SimpleNamespace(
        monotonic=lambda: now, perf_counter=time.perf_counter, time=time.time))
    node = pose._PoseNode('/cam/exact', None, .35, 30., 'quiet',
                          classifier=_FinalClassifier(), segment_backend=_SegmentModel())
    node._activity_idle_flush_s = 0.15
    assert node._next_worker_timeout() == pytest.approx(0.1)

    node._last_image_received_at = 10.0
    node._segment_buffer.append(b'x', 10.0)
    # 0.12s elapsed -> only 0.03s left before 0.15s preview deadline
    now = 10.12
    assert node._next_worker_timeout() == pytest.approx(0.03)
    # Preview computed for last_t=10.0 -> wait exact remaining to quiet_s=0.40s
    node._segment_preview = ((10.0, 1), None, 10.0)
    now = 10.34
    assert node._next_worker_timeout() == pytest.approx(0.06)

    dockerfile = Path(__file__).resolve().parents[1] / 'Dockerfile.jetson'
    assert 'FASTDDS_BUILTIN_TRANSPORTS=UDPv4' in dockerfile.read_text()


def test_pose_node_flush_passes_raw_received_and_defers_handface_past_preview(monkeypatch):
    """End-to-end _PoseNode check: preview defers HandFace, confirm runs it cleanly."""
    now = 10.0
    monkeypatch.setattr(pose, 'time', SimpleNamespace(
        monotonic=lambda: now, perf_counter=time.perf_counter, time=time.time))

    events = []

    class NodeBackend:
        supports_raw_received = True
        handface = object()
        last_error = None

        def recognize(self, received, stop_event, *, defer_handface=False, raw_received=None):
            events.append(('recognize', len(received), len(raw_received), defer_handface))
            row = np.array(_standing_row(), np.float32)
            result = {
                'image': np.zeros((480, 640, 3), np.uint8),
                'boxes': row[None, :4], 'scores': row[None, 4],
                'keypoints': row[6:].reshape(1, 17, 3), 'primary_box': row[:4],
                'activity': {'name': 'drink water', 'name_zh': '喝水', 'score': .81, 'action_id': 1},
                'evidence': {'frames_used': len(received), 'handface_state': 'pending'},
                '_handface_pending': ('token',),
            }
            return result if defer_handface else self.complete_handface(result, stop_event)

        def complete_handface(self, result, stop_event=None):
            events.append(('complete', result.pop('_handface_pending', None)))
            result['activity'] = {'name': 'brush teeth', 'name_zh': '刷牙', 'score': .92, 'action_id': 3}
            result['evidence']['handface_state'] = 'applied'
            return result

    node = pose._PoseNode('/cam/node_hf', None, .35, 30., 'quiet',
                          classifier=_FinalClassifier(), segment_backend=NodeBackend())
    messages = _publisher(node, node._output_topic).messages
    row = np.array(_standing_row(), np.float32)
    for i in range(6):
        ts = 10.0 + i * 0.03
        node._segment_buffer.append(bytes([i]), ts)
        if i % 2 == 0:
            node._segment_buffer.record_pose(
                ts, row[None, :4], row[None, 4], row[6:].reshape(1, 17, 3), (480, 640, 3))
    # Preview at 0.25s (< 0.40s quiet_s): defers HandFace, no publish yet.
    now = 10.40
    node._flush_received_segment()
    assert events == [('recognize', 3, 6, True)]
    assert messages == []
    # Confirm at 0.60s (> 0.40s quiet_s): completes HandFace without re-running recognize.
    now = 10.75
    node._flush_received_segment()
    assert events == [('recognize', 3, 6, True), ('complete', ('token',))]
    assert len(messages) == 1
    assert json.loads(messages[0])['persons'][0]['activity'] == 'brush teeth'


def test_real_handface_predict_fuse_and_segment_warmup():
    """Exercise real HandFaceActionBackend.predict/fuse & warmup against mock TRT."""
    from plugins.pose_handface import HandFaceActionBackend

    class WholeBodyMock:
        optimization_shape = (8, 3, 256, 192)
        input_dtype = np.float32
        calls = 0

        def infer(self, blob):
            assert blob.shape == (8, 3, 256, 192)
            self.calls += 1
            x_scores = np.zeros((8, 133, 384), dtype=np.float32)
            y_scores = np.zeros((8, 133, 512), dtype=np.float32)
            x_scores[:, :, 200] = 0.9
            y_scores[:, :, 250] = 0.85
            return [x_scores, y_scores]

    class ActionMock:
        input_shape = (1, 8, 894)
        input_dtype = np.float32
        calls = 0

        def infer(self, blob):
            assert blob.shape == (1, 8, 894)
            self.calls += 1
            logits = np.zeros((1, 126), dtype=np.float32)
            logits[0, 8] = 8.0
            return [logits]

    class SimpleHead:
        def __init__(self, shape):
            self.input_shape = shape
            self.calls = 0

        def infer(self, blob):
            self.calls += 1
            return [np.zeros((1, 126), dtype=np.float32)]

    clusters = [0] * 126
    clusters[8] = 1
    params = [1.0, 1.0, 1.0, 0.5, -1.0, -0.5, -1.5, -0.5, 1.2, -0.8, -1.2, 1.0, 0.8, 1.2, 0.2, 0.5, 0.3]
    wb, act = WholeBodyMock(), ActionMock()
    hf = HandFaceActionBackend(wb, act, clusters, params)
    row = np.array(_standing_row(), np.float32)
    poses = [row[6:].reshape(17, 3) for _ in range(4)]
    frames = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(8)]
    hf_prob, qual = hf.predict(poses, frames)
    assert hf_prob.shape == (126,) and np.isfinite(hf_prob).all() and hf_prob.argmax() == 8
    assert qual == pytest.approx(0.85, abs=0.02)

    body = np.full(126, 1.0 / 126.0, dtype=np.float32)
    rgb = np.full(126, 1.0 / 126.0, dtype=np.float32)
    fused = hf.fuse(body, rgb, hf_prob, qual, observation_fps=8.0)
    assert fused.shape == (126,) and np.isfinite(fused).all()
    assert fused.sum() == pytest.approx(1.0, abs=1e-5)
    assert fused.argmax() == 8

    body_head = SimpleHead((1, 2, 100, 17, 6))
    rgb_head = SimpleHead((1, 8, 3, 192, 192))
    seg = SegmentActionBackend(object(), body_head, rgb_head, handface=hf)
    seg.warmup()
    assert body_head.calls == 1 and rgb_head.calls == 1 and wb.calls == 2 and act.calls == 2


def test_deployment_environment_keeps_core_instance_overrides(tmp_path, monkeypatch):
    engine = tmp_path/'pose.engine'
    engine.write_bytes(b'path-resolution-only')
    monkeypatch.setenv('POSE_ACTIVITY_MODE', 'segment')
    monkeypatch.setenv('POSE_ENGINE_PATH', str(engine))
    plugin, _ = _plugin()
    assert plugin._merged_config('fresh')['activity_mode'] == 'segment'
    plugin._instance_configs['core'] = {'fps': 30, 'activity_mode': 'rolling'}
    assert plugin._merged_config('core')['activity_mode'] == 'rolling'
    assert plugin._resolve_engine() == str(engine)


def test_lean_segment_frames_preserve_tracking_and_both_head_inputs(monkeypatch):
    from plugins import pose_segment
    from plugins.pose_action import PoseFrame

    class Head:
        def __init__(self, shape):
            self.input_shape = shape
            self.inputs = []

        def infer(self, blob):
            self.inputs.append(blob.copy())
            p = np.zeros((1, 126), np.float32)
            p[0, 2], p[0, 37] = .9, .1
            return [p]

    def run():
        body = Head((1, 2, 100, 17, 6))
        rgb = Head((1, 8, 3, 192, 192))
        backend = SegmentActionBackend(object(), body, rgb)
        result = backend.recognize(received)
        return result, body.inputs, rgb.inputs

    row = np.array(_standing_row(), np.float32)
    received = []
    # Multiple people, reordered detections, invisible joints, and a gap long
    # enough to expire tracks exercise both IoU and centroid association.
    for i in range(12):
        rows = np.stack([row.copy(), row.copy()])
        rows[1, [0, 2]] += 180
        rows[1, 6:].reshape(17, 3)[:, 0] += 180
        if i == 4:
            rows = rows[::-1].copy()
        if i in (5, 6):
            rows[0, 6:].reshape(17, 3)[:5, 2] = 0
        pose = (rows[:, :4], rows[:, 4], rows[:, 6:].reshape(2, 17, 3), (480, 640))
        received.append((i*.1+(3. if i >= 8 else 0.),
                         frame_bytes(640, 480), pose))
    lean, lean_body, lean_rgb = run()
    monkeypatch.setattr(pose_segment, '_SkeletonFrame', PoseFrame)
    reference, reference_body, reference_rgb = run()
    assert lean['evidence'] == reference['evidence']
    assert lean['activity'] == reference['activity']
    np.testing.assert_array_equal(lean['probabilities'], reference['probabilities'])
    np.testing.assert_array_equal(lean_body[0], reference_body[0])
    np.testing.assert_array_equal(lean_rgb[0], reference_rgb[0])


def test_build_rgb_input_selects_clip_constant_crop_e_modes():
    from plugins.pose_segment import _clip_crop_mode, build_rgb_input

    def make_person_kps(lw_xy, rw_xy):
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[0] = [320., 90., .9]
        kp[5] = [270., 150., .9]
        kp[6] = [370., 150., .9]
        kp[7] = [260., 220., .9]
        kp[8] = [380., 220., .9]
        kp[9] = [lw_xy[0], lw_xy[1], .9]
        kp[10] = [rw_xy[0], rw_xy[1], .9]
        kp[11] = [285., 290., .9]
        kp[12] = [355., 290., .9]
        kp[15] = [290., 430., .9]
        kp[16] = [350., 430., .9]
        return kp

    close_kps = np.stack([make_person_kps((305., 240.), (335., 242.)) for _ in range(8)])
    wide_kps = np.stack([make_person_kps((220., 160.), (420., 240.)) for _ in range(8)])
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(8)]
    for img in frames:
        img[130:170, 260:380] = (200, 40, 40)  # shoulder/chest marker

    assert _clip_crop_mode(close_kps, 640, 480, side=442.) == 'B'
    assert _clip_crop_mode(wide_kps, 640, 480, side=442.) == 'C'

    blob_b = build_rgb_input(frames, close_kps, frame_keypoints=close_kps)
    blob_c = build_rgb_input(frames, wide_kps, frame_keypoints=wide_kps)
    assert blob_b.shape == (1, 16, 3, 192, 192)
    assert blob_c.shape == (1, 16, 3, 192, 192)
    assert np.isfinite(blob_b).all() and np.isfinite(blob_c).all()
    assert not np.allclose(blob_b[:, 8:], blob_c[:, 8:])


def test_segment_bundle_declaration_stays_strictly_three_engines():
    from utils import model_downloader as md

    files = md.SEGMENT_ACTION_MODEL_BUNDLES['jp61']['files']
    engines = [name for name in files if name.endswith('.engine')]
    assert len(engines) == 3
    assert 'models.json' in files
    assert not any('handface' in name or 'dwpose' in name for name in files)
