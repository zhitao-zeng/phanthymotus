#!/usr/bin/env python3
"""Count ONNX initializer elements for the face-leaderboard parameter cap."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import onnx


def _tensor_elements(tensor) -> int:
    return int(math.prod(int(dimension) for dimension in tensor.dims))


def _walk_graph(graph):
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _walk_graph(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    yield from _walk_graph(child)


def inspect_model(path: Path) -> dict:
    model = onnx.load(str(path), load_external_data=False)
    initializers = []
    constant_tensors = []
    for graph in _walk_graph(model.graph):
        initializers.extend(graph.initializer)
        for node in graph.node:
            if node.op_type != "Constant":
                continue
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    constant_tensors.append(attribute.t)
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    constant_tensors.extend(attribute.tensors)
    initializer_elements = sum(_tensor_elements(item) for item in initializers)
    constant_elements = sum(_tensor_elements(item) for item in constant_tensors)
    return {
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "initializer_tensors": len(initializers),
        "initializer_parameters": initializer_elements,
        # Report separately: Constant nodes often encode shapes and anchors,
        # not trainable parameters. The leaderboard must decide whether they
        # belong in its cap instead of this tool silently mixing definitions.
        "constant_tensors": len(constant_tensors),
        "constant_elements": constant_elements,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", type=Path)
    args = parser.parse_args()
    models = [inspect_model(path.resolve()) for path in args.models]
    payload = {
        "models": models,
        "total_initializer_parameters": sum(
            item["initializer_parameters"] for item in models
        ),
        "total_constant_elements": sum(item["constant_elements"] for item in models),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
