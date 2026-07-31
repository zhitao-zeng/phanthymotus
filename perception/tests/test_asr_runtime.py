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
    def _session(self, detector):
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
            silence_ms=400,
            pre_roll_ms=0,
            model_dir="/unused",
        )

    def test_empty_detection_keeps_session_open_for_late_audio(self):
        detector = _SequenceDetector([[], [(1.0, 1.6)]])
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


if __name__ == "__main__":
    unittest.main()
