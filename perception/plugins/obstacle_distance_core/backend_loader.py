from __future__ import annotations

import importlib
from typing import Mapping

from .contracts import DepthBackend, InstanceSegmentationBackend


_FACTORY_PATH_ERROR = "backend_factory must use module.path:function_name"


def load_backend_factory(path: str):
    if (
        not isinstance(path, str)
        or path.count(":") != 1
        or path != path.strip()
    ):
        raise ValueError(_FACTORY_PATH_ERROR)

    module_name, function_name = path.split(":", 1)
    if (
        not module_name
        or not function_name
        or any(not part.isidentifier() for part in module_name.split("."))
        or not function_name.isidentifier()
    ):
        raise ValueError(_FACTORY_PATH_ERROR)

    try:
        module = importlib.import_module(module_name)
    except Exception:
        raise RuntimeError("backend factory module could not be imported") from None
    try:
        factory = getattr(module, function_name)
    except AttributeError:
        raise RuntimeError("backend factory attribute was not found") from None
    if not callable(factory):
        raise TypeError("backend factory is not callable")
    return factory


def create_model_backends(
    config: Mapping[str, object],
) -> tuple[DepthBackend | None, InstanceSegmentationBackend | None]:
    if not isinstance(config, Mapping):
        raise ValueError("backend configuration must be a mapping")

    mode = config.get("mode", "model")
    if mode == "diagnostic_constant":
        return None, None
    if mode != "model":
        raise ValueError("backend mode must be model or diagnostic_constant")

    path = config.get("backend_factory")
    if not isinstance(path, str) or not path:
        raise ValueError("backend_factory is required in model mode")
    factory = load_backend_factory(path)
    try:
        result = factory(config)
    except Exception:
        raise RuntimeError("backend factory failed") from None

    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise TypeError("backend factory must return exactly two backends")
    depth_backend, segmentation_backend = result
    if not isinstance(depth_backend, DepthBackend):
        raise TypeError("backend factory returned an invalid depth backend")
    if not isinstance(segmentation_backend, InstanceSegmentationBackend):
        raise TypeError(
            "backend factory returned an invalid segmentation backend"
        )
    return depth_backend, segmentation_backend
