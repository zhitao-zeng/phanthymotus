#!/usr/bin/env python3
"""
plugins/obstacle.py — ObstacleDistancePlugin: obstacle distance estimation.

Subscribes to image/jpeg topics, estimates obstacle distance from camera,
publishes distance results to ROS2 topic.
Supports multi-instance (one instance per input topic).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from copy import deepcopy

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from .obstacle_distance_core.contracts import SceneDomain

log = logging.getLogger(__name__)


_LOW_LAT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    durability=DurabilityPolicy.VOLATILE,
)

_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

TOOLS = [
    {
        "name": "obstacle",
        "type": "processor",
        "multiInstance": True,
        "description": "Obstacle Distance Estimation — estimate distance to obstacles from camera feed",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "info", "config"],
                    "description": "Action to perform"
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 image topic to subscribe (e.g. /hostname/camera/rgb, required for action=start)"
                },
            },
            "required": ["action"]
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "provider": {"type": "string", "enum": ["local"], "description": "Distance estimation provider", "scope": "shared"},
            },
            "required": ["provider"]
        },
        "topic_in":  [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [{"format": "data/json",  "desc": "obstacle distance estimation result"}],
    }
]


# ── Distance Estimation Pipeline ──────────────────────────────────────────────


class LocalDistanceAdapter:
    """本地深度 + 分割管线（基于 obstacle_distance_core 估算器）。

    加载深度与分割后端，由 ObstacleDistanceEstimator
    计算最近障碍物距离。
    模型模式初始化失败时直接抛错，禁止用随机值伪装成有效预测。
    """

    def __init__(self, cfg: dict):
        from .obstacle_distance_core.estimator import ObstacleDistanceEstimator
        from .obstacle_distance_core.zipdepth_tensorrt_backends import (
            create_backends,
        )

        self._cfg = deepcopy(cfg or {})
        scene_mode = self._cfg.get("scene_mode", "fixed")
        environment_scene = os.environ.get("OBSTACLE_FIXED_SCENE")
        self._resolution_scene_map: dict[tuple[int, int], str] = {}
        if environment_scene or scene_mode == "fixed":
            configured_scene = (
                environment_scene
                or self._cfg.get("fixed_scene")
                or self._cfg.get("scene_hint")
            )
            try:
                self._scene_hint = SceneDomain(
                    str(configured_scene).strip().lower()
                ).value
            except Exception:
                raise ValueError(
                    "fixed_scene must be indoor or vehicle"
                ) from None
            self._scene_mode = "fixed"
        elif scene_mode == "resolution":
            configured_map = self._cfg.get("resolution_scene_map")
            if not isinstance(configured_map, dict) or not configured_map:
                raise ValueError(
                    "resolution_scene_map must be a nonempty mapping"
                )
            for resolution, configured_scene in configured_map.items():
                try:
                    width_text, height_text = str(resolution).lower().split("x")
                    width = int(width_text)
                    height = int(height_text)
                    scene = SceneDomain(
                        str(configured_scene).strip().lower()
                    ).value
                except Exception:
                    raise ValueError(
                        "resolution_scene_map entries must use WIDTHxHEIGHT "
                        "and indoor or vehicle"
                    ) from None
                if width <= 0 or height <= 0:
                    raise ValueError(
                        "resolution_scene_map dimensions must be positive"
                    )
                key = (width, height)
                if key in self._resolution_scene_map:
                    raise ValueError(
                        "resolution_scene_map contains duplicate dimensions"
                    )
                self._resolution_scene_map[key] = scene
            self._scene_hint = None
            self._scene_mode = "resolution"
            # Unknown resolutions must produce missing_scene rather than silently
            # falling back to a stale fixed_scene value.
            self._cfg["fixed_scene"] = None
        else:
            raise ValueError(
                "obstacle ROS input has no scene metadata; scene_mode must be "
                "fixed or resolution"
            )
        depth_backend, segmentation_backend = create_backends(self._cfg)
        self._estimator = ObstacleDistanceEstimator(
            depth_backend,
            segmentation_backend,
            self._cfg,
        )

    def estimate(self, image_bytes: bytes) -> dict:
        """估算最近障碍物距离。"""
        scene_hint = self._scene_hint
        estimator_image_bytes = image_bytes
        if self._scene_mode == "resolution":
            try:
                import cv2
                import numpy as np

                image = cv2.imdecode(
                    np.frombuffer(image_bytes, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
            except Exception:
                image = None
            if image is None or image.size == 0:
                # Preserve the estimator's structured invalid_image fallback.
                estimator_image_bytes = b""
            else:
                height, width = image.shape[:2]
                scene_hint = self._resolution_scene_map.get((width, height))
        result = self._estimator.estimate(
            estimator_image_bytes,
            scene_hint=scene_hint,
        )
        return {
            "pred_distance": result.distance_m,
            "distance_m": result.distance_m,
            "near_obstacle": result.near_obstacle,
            "scene": result.scene,
            "status": result.status,
            "error_code": result.error_code,
            "fallback": result.fallback,
            "approximate_geometry": result.approximate_geometry,
            "latency_ms": result.latency_ms,
        }


def _build_distance_adapter(cfg: dict) -> LocalDistanceAdapter:
    if cfg.get("provider", "local") != "local":
        raise ValueError("obstacle provider must be local")
    return LocalDistanceAdapter(cfg)


# ── ROS2 Node (one per instance/topic) ────────────────────────────────────────

class _ObstacleNode(Node):
    """Per-topic obstacle distance estimation node."""

    def __init__(self, input_topic: str, node_suffix: str):
        super().__init__(f"obstacle_{node_suffix}")
        self._input_topic = input_topic
        self._output_topic = f"{input_topic}/obstacle"
        self._pub = self.create_publisher(String, self._output_topic, _PUB_QOS)
        self._sub: object | None = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=10)
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()
        self._detect_count = 0
        self.state = "idle"

    def start(self, adapter: LocalDistanceAdapter) -> dict:
        with self._lifecycle_lock:
            if self.state == "running":
                return {"state": "running", "input": self._input_topic, "output": self._output_topic}
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("previous obstacle worker is still stopping")
            if self._sub is None:
                self._sub = self.create_subscription(
                    CompressedImage, self._input_topic, self._image_cb, _LOW_LAT_QOS
                )
            while True:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break
            # Use a distinct event for every worker generation.  If an old
            # inference ever outlives stop()'s join timeout, a later start
            # must not clear that worker's stop signal and revive it.
            stop_event = threading.Event()
            self._stop_event = stop_event
            self._worker = threading.Thread(
                target=self._inference_worker,
                args=(stop_event, adapter),
                daemon=True,
                name=f"obstacle_worker_{self._input_topic}",
            )
            self._worker.start()
            self.state = "running"
        log.info(f"[obstacle] started: {self._input_topic} -> {self._output_topic}")
        return {"state": "running", "input": self._input_topic, "output": self._output_topic}

    def stop(self) -> dict:
        # Keep the ROS node and subscription registered with the spinning
        # executor. Destroying either entity from the MCP HTTP thread races
        # rclpy's wait set, so stop only pauses the inference worker.
        with self._lifecycle_lock:
            self.state = "idle"
            self._stop_event.set()
            worker = self._worker
            if worker and worker.is_alive():
                # Wake queue.get() instead of waiting for its polling timeout.
                try:
                    self._frame_queue.put_nowait(b"")
                except queue.Full:
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._frame_queue.put_nowait(b"")
                    except queue.Full:
                        pass
                worker.join(timeout=3.0)
            if worker is None or not worker.is_alive():
                self._worker = None
            else:
                log.warning(
                    "[obstacle] worker still stopping after timeout: %s",
                    self._input_topic,
                )
        log.info(f"[obstacle] stopped: {self._input_topic}")
        return {"state": "idle", "input": self._input_topic}

    def _image_cb(self, msg: CompressedImage):
        if self.state != "running":
            return
        try:
            image_bytes = bytes(msg.data)
        except Exception:
            log.warning(
                "[obstacle] received image frame with invalid data on %s",
                self._input_topic,
            )
            return
        log.debug(
            f"[obstacle] received image frame: size={len(image_bytes)} bytes, format={msg.format}, topic={self._input_topic}")
        # Drop old frame if queue full (no backpressure)
        try:
            self._frame_queue.put_nowait(image_bytes)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(image_bytes)
            except queue.Full:
                pass

    def _inference_worker(
        self,
        stop_event: threading.Event,
        adapter: LocalDistanceAdapter,
    ):
        while not stop_event.is_set():
            try:
                jpeg_bytes = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if stop_event.is_set():
                break
            try:
                result = adapter.estimate(jpeg_bytes)
                if result.get("fallback"):
                    log.debug(
                        "[obstacle] fallback scene=%s error_code=%s "
                        "latency_ms=%.1f distance_m=%s",
                        result.get("scene"),
                        result.get("error_code"),
                        float(result.get("latency_ms", 0.0)),
                        result.get("pred_distance"),
                    )
                else:
                    log.debug(
                        "[obstacle] result scene=%s latency_ms=%.1f "
                        "distance_m=%s",
                        result.get("scene"),
                        float(result.get("latency_ms", 0.0)),
                        result.get("pred_distance"),
                    )
                if not stop_event.is_set():
                    self._publish_result(result)
            except Exception as e:
                log.error(f"[obstacle] inference error: {e}", exc_info=True)

    def _publish_result(self, result: dict):
        self._detect_count += 1
        msg = String()
        msg.data = json.dumps({
            "pred_distance": result.get("pred_distance", 10.0),
        }, ensure_ascii=False)
        self._pub.publish(msg)


# ── Plugin class ──────────────────────────────────────────────────────────────

class ObstacleDistancePlugin:
    PREFIX = "obstacle"

    def __init__(self, plugin_cfg: dict, executor):
        self._executor = executor
        self._plugin_cfg = deepcopy(plugin_cfg or {})
        self._provider = self._plugin_cfg.get("provider", "local")
        self._nodes: dict[str, _ObstacleNode] = {}
        self._instance_configs: dict[str, dict] = {}
        self._adapter_cache: dict[
            str, tuple[dict, LocalDistanceAdapter]
        ] = {}
        self._adapter_lock = threading.Lock()

        log.info("[obstacle] plugin registered: provider=%s", self._provider)

    def _adapter_for_start(
        self,
        node_key: str,
        effective_cfg: dict,
    ) -> LocalDistanceAdapter:
        """Initialize model runtime only for the first valid start action."""
        cache_key = (
            node_key if self._instance_configs.get(node_key) else "__shared__"
        )
        with self._adapter_lock:
            cached = self._adapter_cache.get(cache_key)
            if cached is not None and cached[0] == effective_cfg:
                return cached[1]
            adapter = _build_distance_adapter(effective_cfg)
            self._adapter_cache[cache_key] = (deepcopy(effective_cfg), adapter)
            return adapter

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action", name)
        instance_id = args.get("instance_id", "")

        if action == "info":
            instances = {}
            for key, node in self._nodes.items():
                instances[key] = {
                    "input": node._input_topic,
                    "output": node._output_topic,
                    "detect_count": node._detect_count,
                    "state": node.state,
                }
            input_topic = args.get("input_topic", "")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                input_topic = node._input_topic
            elif not input_topic and self._nodes:
                first_node = next(iter(self._nodes.values()))
                input_topic = first_node._input_topic
            topics_in = [{"topic": input_topic, "format": "image/jpeg"}] if input_topic else []
            topics_out = [{"topic": f"{input_topic}/obstacle", "format": "data/json"}] if input_topic else []
            state = (
                "running"
                if any(node.state == "running" for node in self._nodes.values())
                else "idle"
            )
            return {
                "name": "ObstacleDistance", "manufacture": "Embodied", "model": "obstacle",
                "state": state,
                "instances": instances,
                "topic_in": topics_in,
                "topic_out": topics_out,
                "desc": "Obstacle distance estimation from camera feed",
            }

        elif action == "start":
            input_topic = args.get("input_topic")
            if not input_topic:
                topics_list = args.get("input_topics") or []
                if topics_list:
                    input_topic = topics_list[0]
            if not input_topic:
                raise ValueError("input_topic is required")
            node_key = instance_id or input_topic
            merged_cfg = deepcopy(self._plugin_cfg)
            merged_cfg.update(self._instance_configs.get(node_key, {}))
            adapter = self._adapter_for_start(node_key, merged_cfg)
            if node_key not in self._nodes:
                suffix = node_key.replace("/", "_").replace("-", "_").lstrip("_")
                node = _ObstacleNode(input_topic, suffix)
                self._executor.add_node(node)
                self._nodes[node_key] = node
            return self._nodes[node_key].start(adapter)

        elif action == "stop":
            if instance_id and instance_id in self._nodes:
                node = self._nodes[instance_id]
                return node.stop()
            elif not instance_id and self._nodes:
                results = []
                for key, node in self._nodes.items():
                    node.stop()
                    results.append(key)
                return {"state": "idle", "stopped_instances": results}
            return {"state": "idle"}

        elif action == "config":
            cfg = {k: v for k, v in args.items() if k not in ('action', 'instance_id') and v is not None and v != ''}
            if instance_id:
                previous_cfg = self._instance_configs.get(instance_id, {})
                self._instance_configs[instance_id] = cfg
                if instance_id in self._nodes:
                    node = self._nodes[instance_id]
                    if node.state == "running":
                        node.stop()
                if cfg != previous_cfg:
                    with self._adapter_lock:
                        self._adapter_cache.pop(instance_id, None)
                return {"status": "configured", "instance_id": instance_id, "config": cfg}
            else:
                merged_cfg = deepcopy(self._plugin_cfg)
                merged_cfg.update(cfg)
                if merged_cfg != self._plugin_cfg:
                    self._plugin_cfg = merged_cfg
                    self._provider = merged_cfg.get("provider", "local")
                    for node in self._nodes.values():
                        if node.state == "running":
                            node.stop()
                    with self._adapter_lock:
                        self._adapter_cache.clear()
                    log.info("[obstacle] configuration updated")
                else:
                    log.debug("[obstacle] configuration unchanged")
                return {"status": "configured", "config": cfg}

        return None
