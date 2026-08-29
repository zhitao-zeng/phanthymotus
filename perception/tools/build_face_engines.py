#!/usr/bin/env python3
"""Build fixed-shape FP16 TensorRT engines for SCRFD and a face recognizer."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


def _trtexec_major(executable: str) -> int:
    result = subprocess.run(
        [executable, "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    match = re.search(r"TensorRT\s+v(\d{4,6})\b", result.stdout)
    if match:
        encoded = match.group(1)
        # NVIDIA banners encode 8.5.2 as 8502 and 10.3.0 as 100300.
        return int(encoded[0]) if len(encoded) == 4 else int(encoded[:2])
    match = re.search(r"TensorRT[^0-9]*([0-9]+)(?:\.[0-9]+)+", result.stdout)
    if not match:
        match = re.search(r"\b([0-9]+)(?:\.[0-9]+){1,3}\b", result.stdout)
    if not match:
        raise RuntimeError(f"could not parse TensorRT version from: {result.stdout.strip()}")
    return int(match.group(1))


def _build(
    executable: str,
    onnx_path: Path,
    engine_path: Path,
    *,
    fp16: bool,
    workspace_mib: int,
    trt_major: int,
    input_name: str,
    input_shape: tuple[int, ...],
) -> None:
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    shape = "x".join(str(value) for value in input_shape)
    profile = f"{input_name}:{shape}"
    command = [
        executable,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={profile}",
        f"--optShapes={profile}",
        f"--maxShapes={profile}",
        "--skipInference" if trt_major >= 10 else "--buildOnly",
    ]
    if fp16:
        command.append("--fp16")
    if trt_major >= 10:
        command.append(f"--memPoolSize=workspace:{workspace_mib}")
    else:
        command.append(f"--workspace={workspace_mib}")
    subprocess.run(command, check=True)
    if not engine_path.is_file() or engine_path.stat().st_size == 0:
        raise RuntimeError(f"trtexec did not produce a usable engine: {engine_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-onnx", type=Path)
    parser.add_argument("--recognizer-onnx", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--workspace-mib", type=int, default=2048)
    parser.add_argument("--fp32", action="store_true", help="disable FP16")
    parser.add_argument("--detector-name", default="scrfd_500m_kps.engine")
    parser.add_argument("--recognizer-name", default="lvface_t_glint360k.engine")
    parser.add_argument("--detector-input-name", default="input.1")
    parser.add_argument("--recognizer-input-name", default="data")
    args = parser.parse_args()
    if args.detector_onnx is None and args.recognizer_onnx is None:
        parser.error("at least one of --detector-onnx/--recognizer-onnx is required")

    major = _trtexec_major(args.trtexec)
    if args.detector_onnx is not None:
        _build(
            args.trtexec,
            args.detector_onnx.resolve(),
            args.output_dir.resolve() / args.detector_name,
            fp16=not args.fp32,
            workspace_mib=args.workspace_mib,
            trt_major=major,
            input_name=args.detector_input_name,
            input_shape=(1, 3, 640, 640),
        )
    if args.recognizer_onnx is not None:
        _build(
            args.trtexec,
            args.recognizer_onnx.resolve(),
            args.output_dir.resolve() / args.recognizer_name,
            fp16=not args.fp32,
            workspace_mib=args.workspace_mib,
            trt_major=major,
            input_name=args.recognizer_input_name,
            input_shape=(1, 3, 112, 112),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
