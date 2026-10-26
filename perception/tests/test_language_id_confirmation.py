"""Exercise the native worker's packet, confirmation and model ownership contracts."""

import json
from pathlib import Path
import shutil
import struct
import subprocess

import pytest


@pytest.fixture(scope="module")
def worker(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    root = tmp_path_factory.mktemp("language-id")
    (root / "whisper.h").write_text(r'''
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
struct whisper_context_params { bool use_gpu = true; };
struct whisper_context { std::string model; int packet = 0; };
static int live_contexts = 0;
inline whisper_context_params whisper_context_default_params() { return {}; }
inline whisper_context *whisper_init_from_file_with_params(const char *path, whisper_context_params p) {
    if (p.use_gpu || live_contexts) std::abort();
    if (std::string(path) == "broken") return nullptr;
    ++live_contexts;
    std::cerr << "load:" << path << "\n";
    return new whisper_context{path};
}
inline void whisper_free(whisper_context *c) {
    if (c) { --live_contexts; delete c; }
}
inline int whisper_lang_id(const char *name) {
    return !std::strcmp(name,"en") ? 0 : !std::strcmp(name,"fr") ? 1 : !std::strcmp(name,"ja") ? 2 : -1;
}
inline int whisper_lang_max_id() { return 2; }
inline const char *whisper_lang_str(int id) { return id == 0 ? "en" : id == 1 ? "fr" : "ja"; }
inline int whisper_pcm_to_mel(whisper_context *c, const float *samples, int, int) {
    c->packet = std::lround(samples[0] * 32768); return 0;
}
inline int whisper_lang_auto_detect(whisper_context *c, int, int, float *p) {
    p[0] = p[1] = p[2] = .01f;
    if (c->model == "primary") p[0] = c->packet == 2 ? .95f : .30f;
    else if (c->packet == 3) p[1] = .60f;
    else p[2] = .95f;
    return 0;
}
''')
    binary = root / "worker"
    source = Path(__file__).resolve().parents[1] / "native/asr_language_id.cc"
    subprocess.run([compiler, "-std=c++17", "-I", str(root), str(source), "-o", str(binary)], check=True)
    return binary


def request(worker, packets, verifier=None):
    command = [str(worker), "primary", "2", "en,fr,ja"]
    if verifier is not None:
        command.append(verifier)
    payload = b"".join(struct.pack("<Ih", 1, value) for value in packets)
    result = subprocess.run(command, input=payload, capture_output=True, timeout=5, check=True)
    messages = [json.loads(line) for line in result.stdout.splitlines()]
    assert messages.pop(0) == {"ready": True}
    return messages, result.stderr.decode()


def test_legacy_worker_keeps_primary_result(worker):
    messages, log = request(worker, [1, 2])
    assert [x["language"] for x in messages] == ["en", "en"]
    assert log.count("load:") == 1


def test_high_confidence_primary_does_not_load_verifier(worker):
    messages, log = request(worker, [2], "verifier")
    assert messages[0]["language"] == "en"
    assert "load:verifier" not in log


def test_confirmation_replays_same_audio_and_reloads_primary_for_next_packet(worker):
    # The backend aborts if two contexts coexist. A later packet must not use
    # the prior packet's verification decision or its model as the primary.
    messages, log = request(worker, [1, 2, 3], "verifier")
    assert [x["language"] for x in messages] == ["ja", "en", "en"]
    assert log.count("load:primary") == 2
    assert log.count("load:verifier") == 2


def test_unavailable_optional_verifier_retains_primary_result(worker):
    messages, _ = request(worker, [1, 2], "broken")
    assert [x["language"] for x in messages] == ["en", "en"]
