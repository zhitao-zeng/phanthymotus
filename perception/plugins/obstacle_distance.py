#!/usr/bin/env python3
"""ROS/MCP lifecycle integration for nearest-obstacle distance estimation."""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
from copy import deepcopy
from dataclasses import asdict
from numbers import Real
from typing import Mapping

from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from .obstacle_distance_core import backend_loader
from .obstacle_distance_core.estimator import (
    DistanceResult,
    ObstacleDistanceEstimator,
)


log = logging.getLogger(__name__)

_CAMERA_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)
_RESULT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

TOOLS = [
    {
        "name": "obstacle_distance",
        "type": "processor",
        "multiInstance": True,
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "info", "config"],
                },
                "input_topic": {"type": "string"},
                "scene_hint": {
                    "type": "string",
                    "enum": ["indoor", "vehicle"],
                },
            },
            "required": ["action"],
        },
        "topic_in": [
            {"format": "image/jpeg", "desc": "front camera image"}
        ],
        "topic_out": [
            {
                "format": "data/json",
                "desc": "nearest obstacle distance",
            }
        ],
    }
]


def _obstacle_distance_output_topic(input_topic: str) -> str:
    return f"{input_topic}/obstacle_distance"


def _finite_number(
    value: object,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    if positive and converted <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and converted < 0:
        raise ValueError(f"{name} must be nonnegative")
    return converted


def _scene_hint(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("scene_hint must be indoor or vehicle")
    normalized = value.strip().lower()
    if normalized not in {"indoor", "vehicle"}:
        raise ValueError("scene_hint must be indoor or vehicle")
    return normalized


def _safe_timestamp(message: object) -> float:
    try:
        stamp = message.header.stamp
        value = float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0
        if math.isfinite(value):
            return value
    except (AttributeError, TypeError, ValueError, OverflowError):
        pass
    try:
        value = float(time.time())
        if math.isfinite(value):
            return value
    except (TypeError, ValueError, OverflowError):
        pass
    return 0.0


def _safe_monotonic() -> float:
    try:
        value = float(time.monotonic())
        if math.isfinite(value):
            return value
    except (TypeError, ValueError, OverflowError):
        pass
    return 0.0


class _ObstacleDistanceNode(Node):
    """One synchronous estimator worker for one ROS image topic."""

    def __init__(
        self,
        input_topic: str,
        estimator: ObstacleDistanceEstimator,
        scene_hint: str,
        inference_lock: threading.Lock,
        *,
        node_suffix: str = "",
        min_interval: float = 0.0,
        fallback_distance_m: float = 3.0,
        decision_threshold_m: float = 1.0,
    ) -> None:
        node_name = (
            f"obstacle_distance_{node_suffix}"
            if node_suffix
            else "obstacle_distance"
        )
        super().__init__(node_name)
        self._input_topic = input_topic
        self._output_topic = _obstacle_distance_output_topic(input_topic)
        self._estimator = estimator
        self._scene_hint = _scene_hint(scene_hint)
        self._inference_lock = inference_lock
        self._min_interval = _finite_number(
            min_interval,
            name="min_interval",
            nonnegative=True,
        )
        self._fallback_distance_m = _finite_number(
            fallback_distance_m,
            name="fallback_distance_m",
            nonnegative=True,
        )
        self._decision_threshold_m = _finite_number(
            decision_threshold_m,
            name="decision_threshold_m",
            positive=True,
        )

        self.state = "idle"
        self._sub = None
        self._pub = self.create_publisher(
            String, self._output_topic, _RESULT_QOS
        )
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._queue_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._publish_gate = threading.Lock()
        self._stop_event = threading.Event()
        self._stop_event.set()
        self._generation = 0
        self._worker_thread: threading.Thread | None = None
        self._worker_threads: list[threading.Thread] = []

    def _status_dict(self) -> dict:
        return {
            "state": self.state,
            "scene": self._scene_hint,
            "topic_in": [
                {
                    "topic": self._input_topic,
                    "format": "image/jpeg",
                    "desc": "front camera image",
                }
            ],
            "topic_out": [
                {
                    "topic": self._output_topic,
                    "format": "data/json",
                    "desc": "nearest obstacle distance",
                }
            ],
        }

    @property
    def worker_alive(self) -> bool:
        with self._lifecycle_lock:
            return any(thread.is_alive() for thread in self._worker_threads)

    def start(self) -> dict:
        with self._lifecycle_lock:
            if self.state == "running":
                return self._status_dict()
            self._worker_threads = [
                thread for thread in self._worker_threads if thread.is_alive()
            ]
            if self._worker_threads:
                raise RuntimeError("obstacle distance worker is still stopping")

            self._generation += 1
            generation = self._generation
            stop_event = threading.Event()
            frame_queue: queue.Queue = queue.Queue(maxsize=1)
            self._stop_event = stop_event
            self._frame_queue = frame_queue
            if self._sub is None:
                self._sub = self.create_subscription(
                    CompressedImage,
                    self._input_topic,
                    self._image_cb,
                    _CAMERA_QOS,
                )
            self.state = "running"
            worker = None
            try:
                worker = threading.Thread(
                    target=self._worker,
                    args=(generation, stop_event, frame_queue),
                    daemon=True,
                    name=f"{self.name}_worker",
                )
                self._worker_thread = worker
                self._worker_threads.append(worker)
                worker.start()
            except Exception:
                self.state = "idle"
                self._generation += 1
                stop_event.set()
                self._clear_queue(frame_queue)
                if worker is not None:
                    self._worker_threads = [
                        item
                        for item in self._worker_threads
                        if item is not worker
                    ]
                self._worker_thread = None
                raise RuntimeError(
                    "obstacle distance worker could not start"
                ) from None
            return self._status_dict()

    @staticmethod
    def _clear_queue(frame_queue: queue.Queue) -> None:
        while True:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                return

    def stop(self) -> dict:
        with self._lifecycle_lock:
            self.state = "idle"
            self._generation += 1
            self._stop_event.set()
            self._clear_queue(self._frame_queue)
            workers = list(self._worker_threads)

        deadline = time.monotonic() + 3.0
        for worker in workers:
            if worker.is_alive():
                worker.join(timeout=max(0.0, deadline - time.monotonic()))

        with self._lifecycle_lock:
            self._worker_threads = [
                thread for thread in self._worker_threads if thread.is_alive()
            ]
            return self._status_dict()

    def configure(
        self,
        estimator: ObstacleDistanceEstimator,
        scene_hint: str,
        *,
        min_interval: float,
        fallback_distance_m: float,
        decision_threshold_m: float,
    ) -> None:
        with self._lifecycle_lock:
            if self.state == "running":
                raise RuntimeError(
                    "obstacle distance node must be stopped before config"
                )
            self._estimator = estimator
            self._scene_hint = _scene_hint(scene_hint)
            self._min_interval = _finite_number(
                min_interval,
                name="min_interval",
                nonnegative=True,
            )
            self._fallback_distance_m = _finite_number(
                fallback_distance_m,
                name="fallback_distance_m",
                nonnegative=True,
            )
            self._decision_threshold_m = _finite_number(
                decision_threshold_m,
                name="decision_threshold_m",
                positive=True,
            )

    def _image_cb(self, message: CompressedImage) -> None:
        with self._lifecycle_lock:
            generation = self._generation
            stop_event = self._stop_event
            frame_queue = self._frame_queue
            if self.state != "running" or stop_event.is_set():
                return
        try:
            image_bytes = bytes(message.data)
        except Exception:
            image_bytes = b""
        item = (image_bytes, _safe_timestamp(message))
        with self._lifecycle_lock:
            if (
                self.state != "running"
                or self._generation != generation
                or self._stop_event is not stop_event
                or self._frame_queue is not frame_queue
                or stop_event.is_set()
            ):
                return
            with self._queue_lock:
                try:
                    frame_queue.put_nowait(item)
                    return
                except queue.Full:
                    pass
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    frame_queue.put_nowait(item)
                except queue.Full:
                    pass

    def _is_generation_active(
        self,
        generation: int,
        stop_event: threading.Event,
    ) -> bool:
        with self._lifecycle_lock:
            return (
                self.state == "running"
                and self._generation == generation
                and self._stop_event is stop_event
                and not stop_event.is_set()
            )

    def _acquire_inference(
        self,
        generation: int,
        stop_event: threading.Event,
    ) -> bool:
        if not self._is_generation_active(generation, stop_event):
            return False
        while not stop_event.is_set():
            if self._inference_lock.acquire(timeout=0.05):
                if self._is_generation_active(generation, stop_event):
                    return True
                self._inference_lock.release()
                return False
        return False

    def _fallback_result(self, timestamp: float) -> DistanceResult:
        safe_timestamp = timestamp if math.isfinite(timestamp) else 0.0
        return DistanceResult(
            distance_m=self._fallback_distance_m,
            near_obstacle=(
                self._fallback_distance_m < self._decision_threshold_m
            ),
            decision_threshold_m=self._decision_threshold_m,
            scene=self._scene_hint,
            status="error",
            error_code="model_error",
            fallback=True,
            approximate_geometry=False,
            latency_ms=0.0,
            timestamp=safe_timestamp,
        )

    def _serialized_result(
        self,
        result: DistanceResult,
        timestamp: float,
    ) -> str:
        try:
            return json.dumps(asdict(result), allow_nan=False)
        except (TypeError, ValueError, OverflowError):
            return json.dumps(
                asdict(self._fallback_result(timestamp)),
                allow_nan=False,
            )

    def _publish_if_active(
        self,
        message: String,
        generation: int,
        stop_event: threading.Event,
    ) -> bool:
        with self._publish_gate:
            with self._lifecycle_lock:
                if (
                    self.state != "running"
                    or self._generation != generation
                    or self._stop_event is not stop_event
                ):
                    return False
            # An external publish already in progress cannot be cancelled.
            # This final token check prevents starting one after stop().
            if stop_event.is_set():
                return False
            self._pub.publish(message)
            return True

    def _worker(
        self,
        generation: int,
        stop_event: threading.Event,
        frame_queue: queue.Queue,
    ) -> None:
        while not stop_event.is_set():
            try:
                image_bytes, timestamp = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            started = _safe_monotonic()
            if not self._acquire_inference(generation, stop_event):
                continue
            try:
                try:
                    result = self._estimator.estimate(
                        image_bytes,
                        scene_hint=self._scene_hint,
                        timestamp=timestamp,
                    )
                except Exception:
                    log.error("obstacle distance estimator failed")
                    result = self._fallback_result(timestamp)
            finally:
                self._inference_lock.release()

            if not self._is_generation_active(generation, stop_event):
                continue
            try:
                message = String()
                message.data = self._serialized_result(result, timestamp)
                self._publish_if_active(message, generation, stop_event)
            except Exception:
                log.error("obstacle distance result publication failed")
                continue

            if self._min_interval > 0:
                elapsed = max(0.0, _safe_monotonic() - started)
                remaining = self._min_interval - elapsed
                if remaining > 0:
                    stop_event.wait(remaining)


class ObstacleDistancePlugin:
    PREFIX = "obstacle_distance"

    _CONTROL_FIELDS = {"action", "instance_id", "input_topic"}
    _INSTANCE_CONFIG_FIELDS = {
        "scene_hint",
        "decision_threshold_m",
        "min_interval_ms",
    }
    _BACKEND_CONFIG_FIELDS = {
        "mode",
        "backend_factory",
        "depth_backend",
        "segmentation_backend",
        "depth_model_dir",
        "segmentation_model_dir",
    }

    def __init__(self, plugin_cfg: Mapping[str, object], executor) -> None:
        if not isinstance(plugin_cfg, Mapping):
            raise ValueError("obstacle distance config must be a mapping")
        self._plugin_cfg = deepcopy(dict(plugin_cfg))
        self._executor = executor
        self._lifecycle_lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._nodes: dict[str, _ObstacleDistanceNode] = {}
        self._instance_configs: dict[str, dict] = {}
        self._retired_nodes: list[_ObstacleDistanceNode] = []
        self._executor_nodes: dict[int, _ObstacleDistanceNode] = {}
        self._destroyed_node_ids: set[int] = set()
        self._depth_backend, self._segmentation_backend = (
            backend_loader.create_model_backends(self._plugin_cfg)
        )
        self._base_estimator = ObstacleDistanceEstimator(
            self._depth_backend,
            self._segmentation_backend,
            deepcopy(self._plugin_cfg),
        )

    def get_tools(self) -> list:
        return TOOLS

    @staticmethod
    def _node_suffix(node_key: str) -> str:
        suffix = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in node_key
        )
        return suffix.strip("_") or "instance"

    def _merged_config(self, instance_id: str) -> dict:
        config = deepcopy(self._plugin_cfg)
        if instance_id:
            config.update(deepcopy(self._instance_configs.get(instance_id, {})))
        return config

    def _build_estimator(self, instance_id: str) -> ObstacleDistanceEstimator:
        return ObstacleDistanceEstimator(
            self._depth_backend,
            self._segmentation_backend,
            self._merged_config(instance_id),
        )

    def _resolve_start_scene(self, args: Mapping, instance_id: str) -> str:
        explicit = args.get("scene_hint")
        if explicit not in (None, ""):
            return _scene_hint(explicit)
        configured = self._instance_configs.get(instance_id, {}).get(
            "scene_hint"
        )
        if configured not in (None, ""):
            return _scene_hint(configured)
        if self._plugin_cfg.get("scene_mode", "metadata") == "metadata":
            raise ValueError("scene_hint is required for metadata scene mode")
        fixed = self._plugin_cfg.get("fixed_scene")
        if fixed in (None, ""):
            raise ValueError("scene_hint is required for start action")
        return _scene_hint(fixed)

    def _node_settings(self, instance_id: str) -> tuple[float, float, float]:
        config = self._merged_config(instance_id)
        min_interval_ms = _finite_number(
            config.get("min_interval_ms", 0),
            name="min_interval_ms",
            nonnegative=True,
        )
        fallback_distance_m = _finite_number(
            config.get("fallback_distance_m", 3.0),
            name="fallback_distance_m",
            nonnegative=True,
        )
        decision_threshold_m = _finite_number(
            config.get("decision_threshold_m", 1.0),
            name="decision_threshold_m",
            positive=True,
        )
        return (
            min_interval_ms / 1000.0,
            fallback_distance_m,
            decision_threshold_m,
        )

    @staticmethod
    def _unique_nodes(nodes) -> list[_ObstacleDistanceNode]:
        unique = []
        seen = set()
        for node in nodes:
            identity = id(node)
            if identity not in seen:
                seen.add(identity)
                unique.append(node)
        return unique

    def _remove_executor_node(self, node: _ObstacleDistanceNode) -> None:
        executor_nodes = getattr(self, "_executor_nodes", {})
        identity = id(node)
        if identity not in executor_nodes:
            return
        try:
            removed = self._executor.remove_node(node)
        except Exception:
            raise RuntimeError(
                "obstacle distance executor removal failed"
            ) from None
        if removed is False:
            raise RuntimeError("obstacle distance executor removal failed")
        executor_nodes.pop(identity, None)

    def _remember_retired(self, node: _ObstacleDistanceNode) -> None:
        if node not in self._retired_nodes:
            self._retired_nodes.append(node)

    def _destroy_unregistered_node(
        self, node: _ObstacleDistanceNode
    ) -> None:
        identity = id(node)
        try:
            destroyed = node.destroy_node()
            if destroyed is False:
                raise RuntimeError("node destruction returned false")
        except Exception:
            log.error("obstacle distance node destruction failed")
            self._remember_retired(node)
            return
        self._destroyed_node_ids.add(identity)

    def _retire_node(self, node_key: str) -> None:
        node = self._nodes.pop(node_key)
        try:
            node.stop()
        finally:
            try:
                self._remove_executor_node(node)
            finally:
                self._remember_retired(node)

    def _configure_node(
        self,
        node: _ObstacleDistanceNode,
        instance_id: str,
        scene: str,
    ) -> None:
        estimator = self._build_estimator(instance_id)
        min_interval, fallback_distance, decision_threshold = (
            self._node_settings(instance_id)
        )
        node.configure(
            estimator,
            scene,
            min_interval=min_interval,
            fallback_distance_m=fallback_distance,
            decision_threshold_m=decision_threshold,
        )

    def _start(self, args: Mapping) -> dict:
        input_topic = args.get("input_topic")
        if not isinstance(input_topic, str) or not input_topic.strip():
            raise ValueError("input_topic is required for start action")
        input_topic = input_topic.strip()
        instance_id = args.get("instance_id", "")
        if not isinstance(instance_id, str):
            raise ValueError("instance_id must be a string")
        scene = self._resolve_start_scene(args, instance_id)
        node_key = instance_id or input_topic
        existing = self._nodes.get(node_key)
        if existing is not None and existing._input_topic != input_topic:
            self._retire_node(node_key)
            existing = None

        if existing is None:
            estimator = self._build_estimator(instance_id)
            min_interval, fallback_distance, decision_threshold = (
                self._node_settings(instance_id)
            )
            node = _ObstacleDistanceNode(
                input_topic,
                estimator,
                scene,
                self._inference_lock,
                node_suffix=self._node_suffix(node_key),
                min_interval=min_interval,
                fallback_distance_m=fallback_distance,
                decision_threshold_m=decision_threshold,
            )
            node._instance_id = instance_id
            try:
                added = self._executor.add_node(node)
                if added is False:
                    raise RuntimeError("executor add returned false")
            except Exception:
                self._destroy_unregistered_node(node)
                raise RuntimeError(
                    "obstacle distance executor add failed"
                ) from None
            self._executor_nodes[id(node)] = node
            self._nodes[node_key] = node
            existing = node
            created = True
        else:
            created = False
        if not created and existing._scene_hint != scene:
            existing.stop()
            self._configure_node(existing, instance_id, scene)
        try:
            return existing.start()
        except Exception:
            if created:
                self._nodes.pop(node_key, None)
                try:
                    existing.stop()
                except Exception:
                    log.error(
                        "obstacle distance failed node stop failed"
                    )
                try:
                    self._remove_executor_node(existing)
                except Exception:
                    log.error(
                        "obstacle distance executor removal failed"
                    )
                self._remember_retired(existing)
            raise RuntimeError(
                "obstacle distance node could not start"
            ) from None

    def _stop(self, args: Mapping) -> dict:
        instance_id = args.get("instance_id", "")
        if instance_id:
            node = self._nodes.get(instance_id)
            if node is None:
                return {"state": "idle"}
            return node.stop()
        for node in self._unique_nodes(self._nodes.values()):
            try:
                node.stop()
            except Exception:
                log.error("obstacle distance node stop failed")
        return {"state": "idle"}

    def _info(self, args: Mapping) -> dict:
        instance_id = args.get("instance_id", "")
        if instance_id:
            node = self._nodes.get(instance_id)
            if node is not None:
                return node._status_dict()
            return {"state": "idle", "scene": None, "topic_in": [], "topic_out": []}
        nodes = self._unique_nodes(self._nodes.values())
        if not nodes:
            return {"state": "idle", "scene": None, "topic_in": [], "topic_out": []}
        return {
            "state": (
                "running"
                if any(node.state == "running" for node in nodes)
                else "idle"
            ),
            "scene": (
                nodes[0]._scene_hint
                if len({node._scene_hint for node in nodes}) == 1
                else "mixed"
            ),
            "topic_in": [
                item for node in nodes for item in node._status_dict()["topic_in"]
            ],
            "topic_out": [
                item for node in nodes for item in node._status_dict()["topic_out"]
            ],
        }

    def _validated_config(self, args: Mapping) -> dict:
        provided = {
            key: value
            for key, value in args.items()
            if key not in self._CONTROL_FIELDS and value not in (None, "")
        }
        if any(key in self._BACKEND_CONFIG_FIELDS for key in provided):
            raise ValueError("backend configuration cannot be changed live")
        unknown = set(provided) - self._INSTANCE_CONFIG_FIELDS
        if unknown:
            raise ValueError("unsupported obstacle distance config field")
        if "scene_hint" in provided:
            provided["scene_hint"] = _scene_hint(provided["scene_hint"])
        if "decision_threshold_m" in provided:
            provided["decision_threshold_m"] = _finite_number(
                provided["decision_threshold_m"],
                name="decision_threshold_m",
                positive=True,
            )
        if "min_interval_ms" in provided:
            provided["min_interval_ms"] = _finite_number(
                provided["min_interval_ms"],
                name="min_interval_ms",
                nonnegative=True,
            )
        return provided

    def _config(self, args: Mapping) -> dict:
        instance_id = args.get("instance_id", "")
        updates = self._validated_config(args)
        if instance_id:
            previous = self._instance_configs.get(instance_id, {})
            self._instance_configs[instance_id] = {**previous, **updates}
            node = self._nodes.get(instance_id)
            if node is not None:
                node.stop()
                scene = self._resolve_start_scene({}, instance_id)
                self._configure_node(node, instance_id, scene)
            return {"status": "configured", "instance_id": instance_id}

        if "scene_hint" in updates:
            self._plugin_cfg["fixed_scene"] = updates.pop("scene_hint")
        self._plugin_cfg.update(updates)
        for node_key, node in list(self._nodes.items()):
            node.stop()
            instance = getattr(node, "_instance_id", "")
            try:
                scene = self._resolve_start_scene({}, instance)
            except ValueError:
                scene = node._scene_hint
            self._configure_node(node, instance, scene)
        return {"status": "configured"}

    def dispatch(self, name: str, args: dict) -> dict | None:
        if not isinstance(args, Mapping):
            raise ValueError("obstacle distance arguments must be a mapping")
        if not isinstance(args.get("instance_id", ""), str):
            raise ValueError("instance_id must be a string")
        with self._lifecycle_lock:
            action = args.get("action") if name == self.PREFIX else name
            if action == "start":
                return self._start(args)
            if action == "stop":
                return self._stop(args)
            if action == "info":
                return self._info(args)
            if action == "config":
                return self._config(args)
            return None

    def prepare_shutdown(self) -> None:
        with self._lifecycle_lock:
            nodes = self._unique_nodes(
                [*self._nodes.values(), *self._retired_nodes]
            )
            for node in nodes:
                try:
                    node.stop()
                except Exception:
                    log.error("obstacle distance shutdown stop failed")

    def destroy_nodes(self) -> None:
        with self._lifecycle_lock:
            nodes = self._unique_nodes(
                [*self._nodes.values(), *self._retired_nodes]
            )
            destroyed = getattr(self, "_destroyed_node_ids", set())
            self._destroyed_node_ids = destroyed
            pending = []
            for node in nodes:
                identity = id(node)
                if identity in destroyed:
                    continue
                try:
                    node.stop()
                except Exception:
                    log.error("obstacle distance final stop failed")
                    pending.append(node)
                    continue
                try:
                    self._remove_executor_node(node)
                except Exception:
                    log.error("obstacle distance executor removal failed")
                    pending.append(node)
                    continue
                try:
                    result = node.destroy_node()
                    if result is False:
                        raise RuntimeError(
                            "node destruction returned false"
                        )
                except Exception:
                    log.error("obstacle distance node destruction failed")
                    pending.append(node)
                    continue
                destroyed.add(identity)
            self._nodes.clear()
            self._retired_nodes = self._unique_nodes(pending)
