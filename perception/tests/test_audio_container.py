"""Container recovery must preserve PCM and handle padded fragmented M4A."""

import shutil
import struct
import subprocess
import threading
from types import SimpleNamespace
from unittest import mock

import pytest

from vision_stubs import PERCEPTION_ROOT  # noqa: F401
from plugins.audio_container import Mpeg4AudioInput


def box(kind, content=b''):
    return struct.pack('>I4s', len(content) + 8, kind) + content


def test_pcm_passes_through_byte_for_byte():
    reader = Mpeg4AudioInput()
    for data in [b'\x01\x00' * 512, b'\x00' * 1024, b'odd']:
        assert reader.feed(data) == (data, False)


def test_fragmented_boxes_decode_once_and_ignore_transport_padding():
    encoded = box(b'ftyp', b'M4A ' + b'\0' * 16) + box(b'mdat', b'abc' * 1000) + box(b'moov', b'metadata')
    reader = Mpeg4AudioInput()
    result = mock.Mock(stdout=b'\x01\x00' * 640)
    chunks = [encoded[i:i + 1024].ljust(1024, b'\0') for i in range(0, len(encoded), 1024)]
    with mock.patch('plugins.audio_container.subprocess.run', return_value=result) as run:
        for chunk in chunks[:-1]:
            assert reader.feed(chunk) == (b'', False)
        assert reader.feed(chunks[-1]) == (result.stdout, True)
        assert reader.feed(b'\0' * 1024) == (b'\0' * 1024, False)
        assert run.call_count == 1


def test_incomplete_input_is_not_forwarded_as_pcm():
    reader = Mpeg4AudioInput()
    assert reader.feed(box(b'ftyp', b'M4A ' + b'\0' * 16)) == (b'', False)
    assert reader.feed(struct.pack('>I4s', 1000, b'mdat') + b'\0' * 16) == (b'', False)


def test_decode_failure_does_not_forward_remaining_container_bytes():
    reader = Mpeg4AudioInput()
    data = box(b'ftyp', b'M4A ') + box(b'moov') + box(b'mdat')
    with mock.patch('plugins.audio_container.subprocess.run', side_effect=RuntimeError('decoder unavailable')):
        with pytest.raises(RuntimeError, match='decoder unavailable'):
            reader.feed(data)
    assert reader.feed(b'compressed remainder') == (b'', False)


def test_recovered_minute_uses_silero_windows_backpressure_and_eof_silence():
    from plugins.asr import _ASRNode

    node = _ASRNode.__new__(_ASRNode)
    node._stop_event = threading.Event()
    node._first_chunk_event = threading.Event()
    node._vad_proc = None
    node._vad_silence_ms = 700
    node._pcm_queue = mock.Mock()
    decoded = b'\x01\x00' * (16000 * 61)
    def checked(fmt, data):
        node._audio_container_decoded = True
        return decoded
    node._check_audio_contract = checked
    stamp = SimpleNamespace(sec=1800000000, nanosec=0)
    node._audio_cb(SimpleNamespace(format='audio/pcm-16k', data=b'container',
                                  header=SimpleNamespace(stamp=stamp)))
    frames = [call.args[0] for call in node._pcm_queue.put.call_args_list]
    assert all(len(frame[0]) <= 1024 for frame in frames)
    assert b''.join(frame[0] for frame in frames) == decoded + b'\0' * 25600
    assert all(frame[2] is False for frame in frames)
    assert all(call.kwargs['timeout'] == 1 for call in node._pcm_queue.put.call_args_list)
    node._pcm_queue.put_nowait.assert_not_called()


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg is required for the image audio path')
def test_actual_aac_file_with_trailing_sample_table(tmp_path):
    audio = tmp_path / 'sample.m4a'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'lavfi', '-i',
                    'sine=frequency=440:duration=0.2', '-c:a', 'aac', str(audio)], check=True)
    encoded = audio.read_bytes()
    expected = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(audio),
                               '-f', 's16le', '-ac', '1', '-ar', '16000', 'pipe:1'],
                              check=True, capture_output=True).stdout
    reader = Mpeg4AudioInput()
    output = []
    for offset in range(0, len(encoded), 1024):
        pcm, converted = reader.feed(encoded[offset:offset + 1024].ljust(1024, b'\0'))
        if converted:
            output.append(pcm)
        else:
            assert pcm == b''
    assert output == [expected]
