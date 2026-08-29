"""TensorRT face-engine builder compatibility helpers."""

from __future__ import annotations

import types

from tools import build_face_engines


def test_trtexec_version_banner_jp511(monkeypatch):
    monkeypatch.setattr(
        build_face_engines.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(
            stdout="&&&& RUNNING TensorRT.trtexec [TensorRT v8502] # trtexec --help",
            returncode=0,
        ),
    )
    assert build_face_engines._trtexec_major("trtexec") == 8


def test_trtexec_version_banner_jp61(monkeypatch):
    monkeypatch.setattr(
        build_face_engines.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(
            stdout="&&&& RUNNING TensorRT.trtexec [TensorRT v100300] # trtexec --help",
            returncode=0,
        ),
    )
    assert build_face_engines._trtexec_major("trtexec") == 10
