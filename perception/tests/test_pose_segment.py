"""Received-segment boundaries, real node ingress, and batched frame order."""
import json
import threading

import numpy as np

from plugins.pose_segment import SegmentBuffer
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
        assert len(backend.clips) == 1 and len(backend.clips[0]) == 6
        assert json.loads(lean.messages[-1])['persons'][0]['activity_score'] == .81
        assert node._detect_count == 1
        assert len(_publisher(node, node._skeleton_topic).messages) == 1
        assert node._last_prediction['received_frames'] == 6
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
