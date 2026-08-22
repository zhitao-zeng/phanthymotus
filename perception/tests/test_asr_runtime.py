import sys
import types
import unittest
from collections import deque
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugins import asr_runtime


class _FakeArray:
    def __init__(self, pcm):
        self.shape = (len(pcm) // 2,)

    def astype(self, _dtype):
        return self


class _FakeNumpy:
    float32 = object()

    @staticmethod
    def frombuffer(pcm, dtype):
        del dtype
        return _FakeArray(pcm)


class _SequenceDetector:
    def __init__(self, results):
        self.results = deque(results)
        self.calls = 0

    def detect(self, _samples):
        self.calls += 1
        result = self.results.popleft()
        if isinstance(result, Exception):
            raise result
        return result


class FireRedVadSessionTest(unittest.TestCase):
    def _session(self, detector, *, silence_ms=400, pre_roll_ms=0):
        fake_module = types.ModuleType("plugins.firered_vad")
        fake_module.FireRedVadOnnx = mock.Mock(return_value=detector)
        modules = mock.patch.dict(
            sys.modules, {"plugins.firered_vad": fake_module}
        )
        numpy = mock.patch.object(asr_runtime, "_np", _FakeNumpy())
        modules.start()
        numpy.start()
        self.addCleanup(modules.stop)
        self.addCleanup(numpy.stop)
        return asr_runtime._FireRedVadSession(
            threshold=0.4,
            silence_ms=silence_ms,
            pre_roll_ms=pre_roll_ms,
            model_dir="/unused",
        )

    def test_preserves_one_continuous_span_across_detected_segments(self):
        detector = _SequenceDetector([[(0.4, 1.0), (1.2, 1.8)]])
        session = self._session(detector, pre_roll_ms=500)
        session._pcm.extend(b"\x01\x00" * int(asr_runtime.SAMPLE_RATE * 3.0))

        session._run_detect()

        utterance, _, _ = session._completed.popleft()
        # 0.0s start (0.4 - 0.5 pre-roll), 1.7s end
        # (1.8 - 0.4 possible silence + 0.3 tail).
        self.assertEqual(
            len(utterance), int(asr_runtime.SAMPLE_RATE * 1.7) * 2
        )

    def test_does_not_trim_segment_that_reaches_buffer_end(self):
        detector = _SequenceDetector([[(0.5, 2.0)]])
        session = self._session(detector)
        session._pcm.extend(b"\x01\x00" * int(asr_runtime.SAMPLE_RATE * 2.0))

        session._run_detect()

        utterance, _, _ = session._completed.popleft()
        self.assertEqual(
            len(utterance), int(asr_runtime.SAMPLE_RATE * 1.5) * 2
        )

    def test_reports_absolute_timestamps_for_detected_span(self):
        detector = _SequenceDetector([[(0.5, 1.0)]])
        session = self._session(detector)
        session._pcm.extend(b"\x01\x00" * int(asr_runtime.SAMPLE_RATE * 2.0))
        session._buffer_start_ts = 100.0

        session._run_detect()

        _, start_ts, end_ts = session._completed.popleft()
        self.assertAlmostEqual(start_ts, 100.5)
        self.assertAlmostEqual(end_ts, 100.9)

    def test_empty_detection_keeps_session_open_for_late_audio(self):
        # FireRed reports the completed segment after the configured 400ms
        # possible-silence run: speech is [1.0, 1.6], segment is [1.0, 2.0].
        detector = _SequenceDetector([[], [(1.0, 2.0)]])
        session = self._session(detector)
        silence = b"\x00\x00" * asr_runtime.SAMPLE_RATE
        speech = b"\x01\x00" * int(asr_runtime.SAMPLE_RATE * 0.6)

        self.assertIsNone(session.process_chunk(silence, 1.0))
        self.assertFalse(session._detected)

        self.assertIsNone(session.process_chunk(speech, 2.0))
        result = session.process_chunk(silence, 3.0)

        self.assertIsNotNone(result)
        self.assertTrue(session._detected)
        self.assertEqual(detector.calls, 2)
        utterance, _, _ = result
        self.assertIn(speech, utterance)

    def test_idle_retries_only_after_buffer_grows(self):
        detector = _SequenceDetector([[], [(0.0, 0.8)]])
        session = self._session(detector)
        speech = b"\x01\x00" * int(asr_runtime.SAMPLE_RATE * 0.8)

        self.assertIsNone(session.process_chunk(speech, 10.0))
        self.assertIsNone(session.notify_idle(12.0))
        self.assertEqual(detector.calls, 1)

        self.assertIsNone(session.notify_idle(13.0))
        self.assertEqual(detector.calls, 1)

        session.process_chunk(speech, 13.1)
        result = session.notify_idle(15.0)
        self.assertIsNotNone(result)
        self.assertEqual(detector.calls, 2)

    def test_detect_failure_is_retryable(self):
        detector = _SequenceDetector(
            [RuntimeError("temporary failure"), [(0.0, 0.8)]]
        )
        session = self._session(detector)
        speech = b"\x01\x00" * int(asr_runtime.SAMPLE_RATE * 0.8)

        session.process_chunk(speech, 10.0)
        with self.assertLogs("asr_runtime.firered", level="WARNING"):
            self.assertIsNone(session.notify_idle(12.0))
        result = session.notify_idle(13.0)

        self.assertIsNotNone(result)
        self.assertEqual(detector.calls, 2)

    def test_diagnostics_explain_empty_detection(self):
        detector = _SequenceDetector([[]])
        session = self._session(detector)
        speech = b"\x64\x00" * int(asr_runtime.SAMPLE_RATE * 0.4)
        silence = b"\x00\x00" * asr_runtime.SAMPLE_RATE

        session.process_chunk(speech, 1.0)
        with self.assertLogs("asr_runtime.firered", level="INFO") as logs:
            self.assertIsNone(session.process_chunk(silence, 2.0))

        diagnostics = session.diagnostics()
        self.assertEqual(diagnostics["chunks_seen"], 2)
        self.assertEqual(diagnostics["detect_calls"], 1)
        self.assertEqual(diagnostics["empty_detects"], 1)
        self.assertEqual(diagnostics["segments_detected"], 0)
        self.assertEqual(diagnostics["detect_triggers"], {"zero_tail": 1})
        self.assertGreater(diagnostics["active_samples"], 0)
        self.assertGreater(diagnostics["rms"], 0)
        self.assertTrue(any("segments=0" in message for message in logs.output))


class IngressSessionDiagnosticsTest(unittest.TestCase):
    def test_separates_callbacks_enqueues_and_queue_drops(self):
        diagnostics = asr_runtime.IngressSessionDiagnostics()
        diagnostics.start(3, now=10.0)

        self.assertTrue(diagnostics.record_callback(b"\x00" * 4, 100.0))
        diagnostics.record_enqueued(b"\x00" * 4)
        self.assertFalse(diagnostics.record_callback(b"\x01\x00" * 3, 100.1))
        diagnostics.record_drop()

        self.assertEqual(
            diagnostics.snapshot(now=10.25),
            {
                "session_id": 3,
                "callback_chunks": 2,
                "callback_bytes": 10,
                "callback_nonzero_chunks": 1,
                "queued_chunks": 1,
                "queued_bytes": 4,
                "pcm_queue_drops": 1,
                "first_header_ts": 100.0,
                "last_header_ts": 100.1,
                "elapsed_ms": 250,
            },
        )

    def test_start_resets_previous_session(self):
        diagnostics = asr_runtime.IngressSessionDiagnostics()
        diagnostics.start(1, now=1.0)
        diagnostics.record_callback(b"\x01\x00", 20.0)
        diagnostics.record_enqueued(b"\x01\x00")

        diagnostics.start(2, now=2.0)

        snapshot = diagnostics.snapshot(now=2.0)
        self.assertEqual(snapshot["session_id"], 2)
        self.assertEqual(snapshot["callback_chunks"], 0)
        self.assertEqual(snapshot["queued_chunks"], 0)
        self.assertIsNone(snapshot["first_header_ts"])


if __name__ == "__main__":
    unittest.main()
