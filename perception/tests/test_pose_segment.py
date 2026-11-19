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
def test_final_classification_reuses_poses_and_never_calls_the_pose_engine(rgb_frames, monkeypatch):
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

    row = np.array(_standing_row(), np.float32)
    decoded = (row[None, :4], row[None, 4], row[6:].reshape(1, 17, 3), (480, 640))
    received = [(i*.1, frame_bytes(640, 480), decoded) for i in range(5)]
    backend = SegmentActionBackend(PoseMustNotRun(), Head((1, 2, 100, 17, 6)),
                                   Head((1, rgb_frames, 3, 192, 192)))
    original_decode = cv2.imdecode
    calls = []

    def count_decode(data, flags):
        calls.append(1)
        return original_decode(data, flags)

    monkeypatch.setattr(cv2, 'imdecode', count_decode)
    result = backend.recognize(received)
    # The last primary frame is already one of the RGB classifier's inputs.
    assert len(calls) == len(set(rgb_frame_indices(5, rgb_frames//2)))
    assert np.array_equal(result['image'], original_decode(
        np.frombuffer(received[-1][1], np.uint8), cv2.IMREAD_COLOR))
    assert result['activity']['action_id'] == 3
    assert result['activity']['score'] == .9
    assert result['evidence']['poses_computed_during_reception'] is True
    assert result['evidence']['RGB_frames_per_view'] == rgb_frames//2
    assert backend.recognize(received[:3]) is None


def test_model_bundle_spec_survives_creation_of_an_instance_backend(tmp_path, monkeypatch):
    from utils import tensorrt_runtime

    loaded = []

    class Engine:
        def __init__(self, path, **kwargs):
            loaded.append(path.name)
            self.input_shape = ((1, 2, 100, 17, 6) if path.name == 'body-new.engine'
                                else (1, 24, 3, 192, 192))

        def close(self):
            pass

    monkeypatch.setattr(tensorrt_runtime, 'TensorRTEngine', Engine)
    (tmp_path/'models.json').write_text(json.dumps({'body_engine': 'body-new.engine',
        'rgb_engine': 'teacher-new.engine', 'rgb_ntu_weight': .5}))
    backend = SegmentActionBackend.from_directory(object(), tmp_path)
    assert loaded == ['body-new.engine', 'teacher-new.engine']
    plugin, _ = _plugin({'activity_mode': 'segment'})
    plugin._segment_model = backend
    monkeypatch.setenv('ACTION_SEGMENT_MODEL_DIR', str(tmp_path))
    instance = plugin._segment_backend_for(plugin._merged_config('new-instance'))
    assert instance.rgb_n_per_view == 12
    assert instance.ntu_rgb_weight == .5


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


def test_crop_e_router_and_warmup_propagate_from_manifest(tmp_path, monkeypatch):
    from utils import tensorrt_runtime
    from plugins.pose_segment import build_rgb_input, fuse_calibrated_router
    from plugins.pose_stgcn import resample_clip

    warm_calls = []

    class Engine:
        input_dtype = np.float32

        def __init__(self, path, **kwargs):
            self.name = path.name
            self.input_shape = ((1, 2, 100, 17, 6) if path.name == 'body.engine'
                                else (1, 32, 3, 192, 192))

        def infer(self, blob):
            warm_calls.append((self.name, blob.shape, blob.dtype))
            out = np.zeros((1, 126), np.float32)
            out[0, 55 if self.name == 'rgb.engine' else 0] = 6.0
            return [out]

        def close(self):
            pass

    monkeypatch.setattr(tensorrt_runtime, 'TensorRTEngine', Engine)
    theta = [1.3584, 1.876, -1.3663, -1.197, 0.472, -0.7761, -1.506, 1.1615, 0.5428, 0.2666]
    (tmp_path/'models.json').write_text(json.dumps({
        'body_engine': 'body.engine',
        'rgb_engine': 'rgb.engine',
        'crop_mode': 'E',
        'router': {'mode': 'poe_gate', 'theta': theta},
    }))
    backend = SegmentActionBackend.from_directory(object(), tmp_path)
    backend.warmup()
    assert warm_calls == [
        ('body.engine', (1, 2, 100, 17, 6), np.dtype('float32')),
        ('rgb.engine', (1, 32, 3, 192, 192), np.dtype('float32')),
    ]

    plugin, _ = _plugin({'activity_mode': 'segment'})
    plugin._segment_model = backend
    monkeypatch.setenv('ACTION_SEGMENT_MODEL_DIR', str(tmp_path))
    instance = plugin._segment_backend_for(plugin._merged_config('inst-v4'))
    assert instance.crop_mode == 'E'
    assert instance.rgb_n_per_view == 16
    assert instance.router == {'mode': 'poe_gate', 'theta': theta}

    row = np.array(_standing_row(), np.float32)
    decoded = (row[None, :4], row[None, 4], row[6:].reshape(1, 17, 3), (480, 640))
    received = [(i*.08, frame_bytes(640, 480), decoded) for i in range(16)]
    res = instance.recognize(received)
    assert res is not None and res['probabilities'].shape == (126,)
    assert np.isclose(res['probabilities'].sum(), 1.0, atol=1e-5)

    # Verify Crop E supports both close-wrist tabletop B and raised-chest C modes.
    imgs = [np.zeros((480, 640, 3), np.uint8) for _ in range(16)]
    kps = np.zeros((16, 17, 3), np.float32)
    kps[:, :, 2] = .9
    kps[:, 9, :2] = (300., 300.)
    kps[:, 10, :2] = (310., 300.)
    kps[:, 5, :2] = (260., 180.)
    kps[:, 6, :2] = (360., 180.)
    blob_b = build_rgb_input(imgs, kps, crop_mode='E')
    kps[:, 10, :2] = (480., 300.)
    blob_c = build_rgb_input(imgs, kps, crop_mode='E')
    assert blob_b.shape == blob_c.shape == (1, 32, 3, 192, 192)

    # Verify physical-timestamp resampling handles non-uniform frame gaps cleanly.
    clip = np.zeros((3, 17, 6), np.float32)
    clip[1] = 1.0
    clip[2] = 3.0
    ts = np.array([0.0, 0.25, 1.0], dtype=np.float64)
    sampled = resample_clip(clip, 5, timestamps=ts)
    assert sampled.shape == (5, 17, 6)
    assert np.isclose(sampled[1, 0, 0], 1.0, atol=1e-5)
    assert np.isclose(sampled[2, 0, 0], 1.0 + (0.5 - 0.25) / 0.75 * 2.0, atol=1e-5)

    # Verify router low-FPS gate shifts weight toward RGB when FPS drops from 30 -> 12.
    pb = np.full(126, .1 / 125, np.float32)
    pr = np.full(126, .1 / 125, np.float32)
    pb[0] = .9
    pr[55] = .9
    f_hi = fuse_calibrated_router(pb, pr, {'mode': 'poe_gate', 'theta': theta}, observation_fps=30.0)
    f_lo = fuse_calibrated_router(pb, pr, {'mode': 'poe_gate', 'theta': theta}, observation_fps=12.0)
    assert f_lo[55] > f_hi[55]


def test_preview_finishing_after_quiet_boundary_publishes_immediately_and_evicts_unposed_jpegs(monkeypatch):
    now = 20.0
    monkeypatch.setattr(pose, 'time', SimpleNamespace(
        monotonic=lambda: now, perf_counter=time.perf_counter, time=time.time))

    class SlowPreview(_SegmentModel):
        def recognize(self, received, stop_event):
            nonlocal now
            result = super().recognize(received, stop_event)
            now = 20.60  # Cross quiet_s=0.40s while segment inference is in flight.
            return result

    backend = SlowPreview()
    node = pose._PoseNode('/cam/segment', None, .35, 10., 'slow_preview',
                          classifier=_FinalClassifier(), segment_backend=backend)
    messages = _publisher(node, node._output_topic).messages
    row = np.array(_standing_row(), np.float32)
    buffer = node._segment_buffer

    # Simulate throttled + queue-evicted frames releasing JPEG bytes while keeping counts.
    for i in range(4):
        now = 20.0 + i * .03
        node._image_cb(_FakeCompressedImage(b'x' * 4096))
    # Drain the single queued posed item so _frame_queue is empty, and attach 4 poses.
    while not node._frame_queue.empty():
        node._frame_queue.get_nowait()
    # Throttled/evicted frames should have released their raw JPEG bytes before posing.
    assert buffer._bytes <= 4096
    for i in range(4):
        ts = 20.0 + i * .03
        buffer._frames[i][1] = b'j' * 128
        buffer.record_pose(ts, row[None, :4], row[None, 4],
                           row[6:].reshape(1, 17, 3), (480, 640, 3))

    now = 20.32  # Past preview_s (20.09 + 0.20 = 20.29) before quiet_s (20.49) -> starts preview.
    node._flush_received_segment()
    # Because SlowPreview crossed quiet_s=0.40s with no new frames, it confirms in the same call!
    assert len(messages) == 1
    assert node._segment_preview is None
    assert node._last_prediction['received_frames'] == 4
    assert node._last_prediction['processed_observations'] == 4


def test_single_view_rgb_head_and_streaming_crop_parity(tmp_path, monkeypatch):
    from utils import tensorrt_runtime
    from plugins.pose_segment import (
        build_rgb_input, build_rgb_views_from_crops,
        crop_rgb_pair_for_frame, prepare_rgb_crop_geometry,
    )

    rng = np.random.default_rng(123)
    imgs = [rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8) for _ in range(16)]
    kps = rng.uniform(20., 200., size=(16, 17, 3)).astype(np.float32)
    kps[:, :, 2] = .9

    for mode in ('A', 'E'):
        ref = build_rgb_input(imgs, kps, crop_mode=mode)
        geom = prepare_rgb_crop_geometry(kps, imgs[0].shape[:2], crop_mode=mode)
        pairs = [crop_rgb_pair_for_frame(imgs[i], i, geom) for i in range(16)]
        vp, vh = build_rgb_views_from_crops([p[0] for p in pairs], [p[1] for p in pairs])
        np.testing.assert_array_equal(np.concatenate([vp, vh], axis=1), ref)

    seen_shapes = []

    class SVEngine:
        input_dtype = np.float32

        def __init__(self, path, **kwargs):
            self.name = path.name
            self.input_shape = ((1, 2, 100, 17, 6) if path.name == 'body.engine'
                                else (1, 16, 3, 192, 192))

        def infer(self, blob):
            seen_shapes.append((self.name, blob.shape))
            if self.name == 'body.engine':
                out = np.zeros((1, 126), np.float32)
                out[0, 0] = 5.0
                return [out]
            return [np.ones((1, 16, 8), np.float32) * 0.25]

        def close(self):
            pass

    monkeypatch.setattr(tensorrt_runtime, 'TensorRTEngine', SVEngine)
    np.savez(
        tmp_path / 'head.npz',
        att_f_w=np.zeros((4, 16), np.float32),
        att_d_w=np.zeros((4, 16), np.float32),
        att_e_w=np.zeros((4, 1), np.float32),
        att_e_b=np.zeros((4,), np.float32),
        att_out_w=np.zeros((1, 4), np.float32),
        att_out_b=np.zeros((1,), np.float32),
        gamma_ev=np.zeros((1, 16), np.float32),
        gamma_dv=np.zeros((1, 16), np.float32),
        fc_w=np.zeros((126, 16), np.float32),
        fc_b=np.linspace(0., 6., 126, dtype=np.float32),
    )
    (tmp_path / 'models.json').write_text(json.dumps({
        'body_engine': 'body.engine',
        'rgb_engine': 'sv.engine',
        'rgb_head': 'head.npz',
        'crop_mode': 'E',
    }))
    backend = SegmentActionBackend.from_directory(object(), tmp_path)
    plugin, _ = _plugin({'activity_mode': 'segment'})
    plugin._segment_model = backend
    monkeypatch.setenv('ACTION_SEGMENT_MODEL_DIR', str(tmp_path))
    instance = plugin._segment_backend_for(plugin._merged_config('sv-inst'))
    assert instance.rgb_single_view is True
    assert instance.rgb_n_per_view == 16

    row = np.array(_standing_row(), np.float32)
    decoded = (row[None, :4], row[None, 4], row[6:].reshape(1, 17, 3), (480, 640))
    received = [(i * .08, frame_bytes(640, 480), decoded) for i in range(16)]
    res = instance.recognize(received)
    assert res is not None and np.isclose(res['probabilities'].sum(), 1.0, atol=1e-5)
    assert seen_shapes.count(('sv.engine', (1, 16, 3, 192, 192))) == 2

    decoded_hd = (row[None, :4] * 2., row[None, 4],
                  (row[6:].reshape(1, 17, 3) * np.array([2., 2., 1.], np.float32)),
                  (960, 1280))
    received_hd = [(i * .08, frame_bytes(1280, 960), decoded_hd) for i in range(16)]
    res_lean = instance.recognize(received_hd, need_image=False)
    assert res_lean is not None
    assert res_lean['image'].shape == (960, 1280, 3)
    assert not isinstance(res_lean['image'], np.ndarray)


