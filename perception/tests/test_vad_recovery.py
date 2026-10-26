"""Quiet endpoint handling and side-effect ordering contracts."""
import queue
import struct
import threading
from types import SimpleNamespace
from unittest import mock

import pytest

from plugins import asr
from plugins.vad_recovery import QuietEndpointer


def tracker(level=.001):
    endpoint=QuietEndpointer(16000,11200)
    endpoint.observe([level]*4096)
    endpoint.begin()
    return endpoint


def test_quiet_endpoint_waits_for_two_silence_windows():
    endpoint=tracker()
    endpoint.observe([0.0]*(42*512))
    assert not endpoint.ready()
    endpoint.observe([0.0]*(2*512))
    assert endpoint.ready()
    endpoint.end()
    assert not endpoint.ready()


def test_no_endpoint_is_invented_from_silence_or_steady_noise():
    for level,tail in [(0.0,0.0),(1/32768,0.0),(.001,.001)]:
        endpoint=tracker(level)
        endpoint.observe([tail]*(100*512))
        assert not endpoint.ready()


def test_normal_level_utterances_keep_model_endpointing():
    endpoint=tracker(.1)
    endpoint.observe([0.0]*(100*512))
    assert not endpoint.quiet
    assert not endpoint.ready()


def test_faint_speech_tail_is_not_silence_just_because_the_start_was_louder():
    endpoint=tracker(.003)
    endpoint.observe([.00015]*(60*512))
    assert not endpoint.ready()
    endpoint.observe([0.0]*(44*512))
    assert endpoint.ready()


def test_initial_large_packet_remembers_normal_speech_before_its_silent_tail():
    endpoint=QuietEndpointer(16000,11200)
    endpoint.observe([.1]*4096+[0.0]*8192)
    endpoint.begin()
    assert not endpoint.quiet


def test_container_input_provides_enough_silence_for_quiet_endpoint_recovery():
    node=object.__new__(asr._ASRNode)
    node._stop_event=threading.Event();node._first_chunk_event=threading.Event()
    node._vad_proc=None;node._vad_silence_ms=700
    node._adapter=mock.Mock(spec=['transcribe_recovered'])
    node._pcm_queue=mock.Mock()
    body=b'\x20\0'*2000
    def decode(*args):node._audio_container_decoded=True;return body
    node._check_audio_contract=decode
    message=SimpleNamespace(format='audio/pcm-16k',data=b'container',
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1700000000,nanosec=0)))
    node._audio_cb(message)
    emitted=b''.join(call.args[0][0] for call in node._pcm_queue.put.call_args_list)
    assert emitted==body+b'\0'*(1500*32)
    first=node._pcm_queue.put.call_args_list[0].args[0]
    assert first[1]==pytest.approx(1700000000-(len(body)+800*32-1024)/32000)
    assert all(not call.args[0][2] for call in node._pcm_queue.put.call_args_list)


def test_energy_frames_do_not_depend_on_packet_boundaries():
    audio=[.001]*4096+[0.0]*(44*512)
    whole=tracker();whole.observe(audio)
    split=tracker()
    for i in range(0,len(audio),317):split.observe(audio[i:i+317])
    assert whole.ready() and split.ready()


@pytest.mark.parametrize('kind,text,accepted', [
    ('quiet_flush','',False),('quiet_flush','.',False),
    ('quiet_flush','confirmed words',True),('quiet_primary','original words',True),
    (None,'normal words',True)])
def test_quiet_confirmation_precedes_speaker_hook_and_publish(kind,text,accepted):
    node=object.__new__(asr._ASRNode)
    node._kws_cfg={'trigger_mode':'vad'};node._worker_ready=threading.Event();node._stop_event=threading.Event()
    node._direction_lock=threading.Lock();node._direction_windows=[];node._direction_topic=''
    node._language='zh-CN';node._emit_emotion=False;node._emit_audio_tags=False
    events=[]
    def decode(*args):events.append('confirm');return text
    def ordinary(*args):events.append('ordinary');return text
    node._adapter=mock.Mock();node._adapter.transcribe_recovered.side_effect=decode
    node._adapter.transcribe.side_effect=ordinary
    node._start_identify=mock.Mock(side_effect=lambda *args:events.append('speaker'))
    node._collect_identity=mock.Mock(return_value={})
    node._fire_hearing_hook=mock.Mock(side_effect=lambda:events.append('hearing'));node._pub=mock.Mock()
    class Input:
        used=False
        def get(self,timeout):
            if not self.used:
                self.used=True
                packet=(b'\x01\0'*16000,0.,1.,0,True)
                return packet+(kind,) if kind else packet
            node._stop_event.set();raise queue.Empty
    node._utterance_queue=Input()
    node._worker_inner()
    if kind=='quiet_flush':
        node._adapter.transcribe_recovered.assert_called_once()
        node._adapter.transcribe.assert_not_called()
        expected=['confirm']+(['hearing','speaker'] if accepted else [])
    elif kind=='quiet_primary':
        node._adapter.transcribe_recovered.assert_not_called()
        expected=['ordinary','hearing','speaker']
    else:
        expected=['speaker','ordinary']
    assert events==expected
    assert node._pub.publish.call_count==int(accepted)


def test_reset_after_forced_endpoint_preserves_original_pcm_and_absolute_offsets(monkeypatch):
    import sys
    from utils import model_downloader
    class StickyVad:
        def __init__(self,*args,**kwargs):self.reset()
        def reset(self):self.samples=[];self.start=None;self.completed=[]
        def accept_waveform(self,samples):
            if self.start is None and any(samples):self.start=len(self.samples)
            self.samples.extend(samples)
        def is_speech_detected(self):return self.start is not None
        def flush(self):
            self.completed.append(SimpleNamespace(start=self.start,samples=self.samples[self.start:]))
            self.start=None
        def empty(self):return not self.completed
        @property
        def front(self):return self.completed[0]
        def pop(self):self.completed.pop(0)
    monkeypatch.setitem(sys.modules, 'sherpa_onnx', SimpleNamespace(
        VoiceActivityDetector=StickyVad,
        VadModelConfig=lambda **kwargs:SimpleNamespace(**kwargs),
        SileroVadModelConfig=lambda **kwargs:SimpleNamespace(**kwargs)))
    monkeypatch.setattr(model_downloader,'ensure_model',lambda *args:None)
    voice=struct.pack('<512h',*([32]*512));quiet=b'\0\0'*512
    chunks=([voice]*20+[quiet]*50)*2
    raw=b''.join(chunks);stop=threading.Event()
    class Input:
        position=0
        def get(self,timeout):
            if self.position==len(chunks):stop.set();raise queue.Empty
            i=self.position;self.position+=1;return chunks[i],i*.032,True
    class Output:
        packets=[]
        def put(self,item):self.packets.append(item)
    out=Output()
    asr._vad_worker(Input(),out,stop,'sherpa_onnx',.5,700,pre_roll_ms=250,recover_quiet=True)
    assert len(out.packets)==2
    first,second=out.packets
    assert first[0]==raw[:64*1024]
    assert second[0]==raw[(70*512-4000)*2:(70+64)*1024]
    assert first[1:3]==pytest.approx((0.,64*.032))
    assert second[1:3]==pytest.approx(((70*512-4000)/16000,(70+64)*.032))
    assert first[5]==second[5]=='quiet_flush'
