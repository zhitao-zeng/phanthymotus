from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .contracts import ErrorCode, ObstacleDistanceError, SceneDomain


def _missing_scene(message: str) -> ObstacleDistanceError:
    return ObstacleDistanceError(ErrorCode.MISSING_SCENE, message)


def _coerce_scene(value: SceneDomain | str) -> SceneDomain:
    if isinstance(value, SceneDomain):
        return value
    if isinstance(value, str):
        try:
            return SceneDomain(value.strip().lower())
        except ValueError:
            pass
    raise _missing_scene("scene selection is invalid")


def resolve_scene(
    *,
    scene_hint: SceneDomain | str | None = None,
    source_name: str | None = None,
    fixed_scene: SceneDomain | str | None = None,
    suffix_map: Mapping[str, SceneDomain | str] | None = None,
) -> SceneDomain:
    if scene_hint is not None:
        return _coerce_scene(scene_hint)

    if source_name is not None and suffix_map is not None:
        source_suffix = Path(source_name).suffix.lower()
        for suffix, configured_scene in suffix_map.items():
            if suffix.lower() == source_suffix:
                return _coerce_scene(configured_scene)

    if fixed_scene is not None:
        return _coerce_scene(fixed_scene)

    raise _missing_scene("scene selection is required")
