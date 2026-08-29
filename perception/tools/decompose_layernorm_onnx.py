#!/usr/bin/env python3
"""Replace ONNX LayerNormalization with TensorRT-8-compatible primitive ops."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def _attribute(node, name: str, default):
    for attribute in node.attribute:
        if attribute.name == name:
            return helper.get_attribute_value(attribute)
    return default


def decompose_layer_normalization(model: onnx.ModelProto) -> int:
    """Rewrite single-output LayerNormalization nodes in-place."""

    replacements = []
    replaced = 0
    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "LayerNormalization":
            replacements.append(node)
            continue
        if len(node.input) < 2 or len(node.output) != 1:
            raise ValueError(
                f"unsupported LayerNormalization signature at {node.name or node_index}"
            )
        axis = int(_attribute(node, "axis", -1))
        epsilon = float(_attribute(node, "epsilon", 1e-5))
        if axis != -1:
            raise ValueError(
                f"only final-axis LayerNormalization is supported; got axis={axis}"
            )
        prefix = f"__face_ln_{node_index}"
        mean = f"{prefix}_mean"
        centered = f"{prefix}_centered"
        squared = f"{prefix}_squared"
        variance = f"{prefix}_variance"
        stabilized = f"{prefix}_stabilized"
        stddev = f"{prefix}_stddev"
        normalized = f"{prefix}_normalized"
        scaled = f"{prefix}_scaled"
        exponent = f"{prefix}_exponent"
        epsilon_name = f"{prefix}_epsilon"
        replacements.extend(
            [
                helper.make_node(
                    "ReduceMean",
                    [node.input[0]],
                    [mean],
                    name=f"{prefix}/mean",
                    axes=[-1],
                    keepdims=1,
                ),
                helper.make_node(
                    "Sub",
                    [node.input[0], mean],
                    [centered],
                    name=f"{prefix}/center",
                ),
                helper.make_node(
                    "Constant",
                    [],
                    [exponent],
                    name=f"{prefix}/exponent",
                    value=helper.make_tensor(
                        f"{prefix}_exponent_value",
                        TensorProto.FLOAT,
                        [],
                        [2.0],
                    ),
                ),
                helper.make_node(
                    "Pow",
                    [centered, exponent],
                    [squared],
                    name=f"{prefix}/square",
                ),
                helper.make_node(
                    "ReduceMean",
                    [squared],
                    [variance],
                    name=f"{prefix}/variance",
                    axes=[-1],
                    keepdims=1,
                ),
                helper.make_node(
                    "Constant",
                    [],
                    [epsilon_name],
                    name=f"{prefix}/epsilon",
                    value=helper.make_tensor(
                        f"{prefix}_epsilon_value",
                        TensorProto.FLOAT,
                        [],
                        [epsilon],
                    ),
                ),
                helper.make_node(
                    "Add",
                    [variance, epsilon_name],
                    [stabilized],
                    name=f"{prefix}/stabilize",
                ),
                helper.make_node(
                    "Sqrt",
                    [stabilized],
                    [stddev],
                    name=f"{prefix}/sqrt",
                ),
                helper.make_node(
                    "Div",
                    [centered, stddev],
                    [normalized],
                    name=f"{prefix}/normalize",
                ),
                helper.make_node(
                    "Mul",
                    [normalized, node.input[1]],
                    [scaled],
                    name=f"{prefix}/scale",
                ),
            ]
        )
        if len(node.input) >= 3 and node.input[2]:
            replacements.append(
                helper.make_node(
                    "Add",
                    [scaled, node.input[2]],
                    list(node.output),
                    name=f"{prefix}/bias",
                )
            )
        else:
            replacements.append(
                helper.make_node(
                    "Identity",
                    [scaled],
                    list(node.output),
                    name=f"{prefix}/output",
                )
            )
        replaced += 1
    del model.graph.node[:]
    model.graph.node.extend(replacements)
    return replaced


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.input.resolve()
    destination = args.output.resolve()
    if source == destination:
        raise ValueError("input and output must be different; preserve the original model")
    model = onnx.load(str(source), load_external_data=True)
    count = decompose_layer_normalization(model)
    if count == 0:
        raise ValueError("model contains no LayerNormalization nodes")
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))
    print(f"decomposed_layer_normalization={count}")
    print(f"output={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
