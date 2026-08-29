#!/usr/bin/env python3
"""Freeze one ONNX input shape while preserving the downloaded original."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--shape", required=True, help="comma-separated dimensions")
    args = parser.parse_args()
    source = args.input.resolve()
    destination = args.output.resolve()
    if source == destination:
        raise ValueError("input and output must differ; preserve the original model")
    shape = [int(value) for value in args.shape.split(",")]
    if not shape or any(value <= 0 for value in shape):
        raise ValueError("all fixed input dimensions must be positive")
    model = onnx.load(str(source), load_external_data=True)
    if len(model.graph.input) != 1:
        raise ValueError(f"expected one graph input, got {len(model.graph.input)}")
    dimensions = model.graph.input[0].type.tensor_type.shape.dim
    if len(dimensions) != len(shape):
        raise ValueError(
            f"shape rank mismatch: graph={len(dimensions)}, requested={len(shape)}"
        )
    for dimension, value in zip(dimensions, shape):
        dimension.ClearField("dim_param")
        dimension.dim_value = value
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))
    print(f"input={model.graph.input[0].name}")
    print(f"shape={shape}")
    print(f"output={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
