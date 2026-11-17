"""Exercise the shape-specific activation-memory contract without CUDA."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from utils import tensorrt_runtime as runtime


class FakeCuda:
    stream_handle = 1

    def __init__(self):
        self.next_pointer = 1

    def malloc(self, size):
        self.next_pointer += 1
        return self.next_pointer

    def free(self, pointer):
        pass

    def copy_host_to_device(self, pointer, array):
        pass

    def copy_device_to_host(self, array, pointer):
        array.fill(0)

    def synchronize(self):
        pass

    def close(self):
        pass


class FakeContext:
    def __init__(self, size):
        self.size = size
        self.memory_calls = []
        self.shape = None
        if size is None:
            self.update_device_memory_size_for_shapes = None

    def set_input_shape(self, name, shape):
        self.shape = shape
        return True

    def update_device_memory_size_for_shapes(self):
        assert self.shape is not None
        return self.size * self.shape[0]

    def get_tensor_shape(self, name):
        return self.shape

    def set_tensor_address(self, name, pointer):
        pass

    def set_device_memory(self, pointer, size):
        self.memory_calls.append((pointer, size))

    def execute_async_v3(self, stream):
        return True


@pytest.fixture
def make_engine(monkeypatch):
    monkeypatch.setattr(runtime, "_SHARED_SCRATCH_BUFFERS", {})
    engines = []

    def create(size, shared=True):
        engine = runtime.TensorRTEngine.__new__(runtime.TensorRTEngine)
        engine.path = "fake.engine"
        engine.input_dtype = np.dtype("float32")
        engine.input_name = "input"
        engine.output_names = ["output"]
        engine.output_dtypes = {"output": np.dtype("float32")}
        engine.auxiliary_shapes = {}
        engine.profiles = [((1, 1), (4, 1), (4, 1))]
        engine._lock = threading.Lock()
        engine._uses_shared_scratch = shared
        engine._scratch_required = 0
        engine._device_id = 0
        engine._device_buffers = {}
        engine._device_capacity = {}
        engine._active_profile = 0
        engine._active_shape = None
        engine._cuda = FakeCuda()
        engine._context = FakeContext(size)
        engine._engine = SimpleNamespace(device_memory_size_v2=2048)
        engines.append(engine)
        return engine

    yield create
    for engine in engines:
        engine.close()


@pytest.mark.parametrize("reported_size,expected", [(512, 512), (0, 2048), (None, 2048)])
def test_scratch_uses_current_shape_or_engine_fallback(make_engine, reported_size, expected):
    engine = make_engine(reported_size)
    engine.infer(np.zeros((1, 1), dtype=np.float32))
    assert engine._context.memory_calls[-1][1] == expected
    assert runtime._SHARED_SCRATCH_BUFFERS[0][1] == expected


def test_scratch_grows_when_the_input_batch_grows(make_engine):
    engine = make_engine(512)
    engine.infer(np.zeros((1, 1), dtype=np.float32))
    assert runtime._SHARED_SCRATCH_BUFFERS[0][1] == 512
    engine.infer(np.zeros((4, 1), dtype=np.float32))
    assert engine._context.memory_calls[-1][1] == 2048
    assert runtime._SHARED_SCRATCH_BUFFERS[0][1] == 2048


def test_scratch_leaves_driver_managed_contexts_unchanged(make_engine):
    engine = make_engine(512, shared=False)
    engine.infer(np.zeros((1, 1), dtype=np.float32))
    assert engine._context.memory_calls == []
    assert runtime._SHARED_SCRATCH_BUFFERS == {}
