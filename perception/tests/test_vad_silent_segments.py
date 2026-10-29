"""New VAD cores need signal; look-back context cannot create an utterance."""
import queue
import struct
import sys
import threading
from types import SimpleNamespace

import pytest

from plugins import asr


def run_scripted_worker(monkeypatch, samples, segments, pre_roll_ms=250,
                        recover_quiet=False):
    from utils import model_downloader

    class ScriptedVad:
        def __init__(self, *args, **kwargs):
            self.received = []
            self.events = list(segments)
            self.completed = []

        def accept_waveform(self, chunk):
            self.received.extend(chunk)
            while self.events and len(self.received) >= self.events[0][1] + 11200:
                begin, end = self.events.pop(0)
                self.completed.append(SimpleNamespace(
                    start=begin, samples=self.received[begin:end]))

        def is_speech_detected(self):
            return False

        def empty(self):
            return not self.completed

        @property
        def front(self):
            return self.completed[0]

        def pop(self):
            self.completed.pop(0)

    monkeypatch.setitem(sys.modules, 'sherpa_onnx', SimpleNamespace(
        VoiceActivityDetector=ScriptedVad,
        VadModelConfig=lambda **kw: SimpleNamespace(**kw),
        SileroVadModelConfig=lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setattr(model_downloader, 'ensure_model', lambda *args: None)
    raw = struct.pack('<%dh' % len(samples), *samples)
    stop = threading.Event()

    class Input:
        position = 0

        def get(self, timeout=1):
            if self.position >= len(raw):
                stop.set()
                raise queue.Empty
            i = self.position
            self.position += 1024
            return raw[i:i + 1024].ljust(1024, b'\0'), i / 32000, True

    class Output:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    out = Output()
    asr._vad_worker(Input(), out, stop, 'sherpa_onnx', .5, 700,
                    pre_roll_ms=pre_roll_ms, recover_quiet=recover_quiet)
    return [p for p in out.items if isinstance(p[0], bytes)], raw


@pytest.mark.parametrize('recover_quiet', [False, True])
@pytest.mark.parametrize('pre_roll_ms', [250, 500])
def test_silent_core_cannot_reuse_the_previous_utterance_tail(
        monkeypatch, recover_quiet, pre_roll_ms):
    samples = [4096] * 12000 + [0] * 24000
    packets, raw = run_scripted_worker(
        monkeypatch, samples, [(0, 12000), (12000, 16224)],
        pre_roll_ms, recover_quiet)
    assert len(packets) == 1
    assert packets[0][0] == raw[:24000]


@pytest.mark.parametrize('level', [1, -1, 32, 4096])
@pytest.mark.parametrize('pre_roll_ms', [0, 250, 500])
def test_nonzero_core_preserves_faint_signal_and_original_context(
        monkeypatch, level, pre_roll_ms):
    samples = [0] * 8000 + [level] * 9600 + [0] * 20000
    packets, raw = run_scripted_worker(
        monkeypatch, samples, [(8000, 17600)], pre_roll_ms, True)
    begin = max(0, 8000 - pre_roll_ms * 16)
    assert len(packets) == 1
    assert packets[0][0] == raw[begin * 2:35200]
    assert packets[0][3] == (8000 - begin) * 2


@pytest.mark.parametrize('level', [1, -1, 32, 4096])
def test_short_speech_keeps_legacy_accumulation_with_a_silent_followup(
        monkeypatch, level):
    samples = [0] * 4000 + [level] * 1600 + [0] * 24800
    packets, raw = run_scripted_worker(
        monkeypatch, samples, [(4000, 5600), (5600, 10400)])
    assert len(packets) == 1
    assert packets[0][0] == raw[:20800]
    assert packets[0][3] == 8000


def test_buffered_silence_before_short_speech_keeps_the_legacy_packet(monkeypatch):
    samples = [0] * 5600 + [32] * 3200 + [0] * 22000
    packets, raw = run_scripted_worker(
        monkeypatch, samples, [(4000, 5600), (5600, 7200), (7200, 8800)])
    assert len(packets) == 1
    assert packets[0][0] == raw[:17600]


@pytest.mark.parametrize('recover_quiet', [False, True])
def test_rejected_empty_utterance_does_not_pollute_the_next_real_one(
        monkeypatch, recover_quiet):
    samples = [0] * 24000 + [32] * 9600 + [0] * 24000
    packets, raw = run_scripted_worker(
        monkeypatch, samples, [(4000, 8224), (24000, 33600)],
        recover_quiet=recover_quiet)
    assert len(packets) == 1
    assert packets[0][0] == raw[40000:67200]
    assert packets[0][3] == 8000
