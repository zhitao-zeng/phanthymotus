"""MCP/ROS2 wrapper for the on-device face-identification engine."""

from __future__ import annotations

import json
import logging
import threading
import time

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from utils.latest_frame import LatestFrame
from utils.qos import CAMERA_QOS
from utils.ros_lifecycle import dispose_node

from .engine import build_face_engine
from .schema import empty_face_payload

log = logging.getLogger(__name__)

_RESULT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)


TOOLS = [
    {
        "name": "face",
        "type": "processor",
        "multiInstance": True,
        "description": "Face Recognition — detect and identify a face using the mounted identity gallery",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop", "info", "config"],
                    "description": "Action to perform",
                },
                "input_topic": {
                    "type": "string",
                    "description": "ROS2 compressed-image topic (required for start)",
                },
            },
            "required": ["action"],
        },
        "configSchema": {
            "type": "object",
            "properties": {
                "min_interval_ms": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 0,
                    "description": "Minimum interval between processed frames; 0 disables throttling",
                    "scope": "instance",
                },
            },
        },
        "topic_in": [{"format": "image/jpeg", "desc": "camera image input"}],
        "topic_out": [
            {
                "format": "data/json",
                "desc": "normalized face box and gallery identity",
            }
        ],
    }
]


def _output_topic(input_topic: str) -> str:
    return f"{input_topic}/face"


class _FaceNode(Node):
    def __init__(
        self,
        input_topic: str,
        engine,
        *,
        node_suffix: str,
        min_interval: float = 0.0,
    ):
        super().__init__(f"face_{node_suffix}" if node_suffix else "face")
        self._input_topic = input_topic
        self._output_topic = _output_topic(input_topic)
        self._engine = engine
        self._min_interval = max(0.0, float(min_interval))
        self._publisher = self.create_publisher(String, self._output_topic, _RESULT_QOS)
        self._subscription = None
        self._frames = LatestFrame()
        self._frames.close()
        self._stop_event = threading.Event()
        self._stop_event.set()
        self._worker: threading.Thread | None = None
        self._workers: list[threading.Thread] = []
        self._generation = 0
        self._node_lock = threading.RLock()
        self._retired = False
        self.state = "idle"
        self.detect_count = 0

    def start(self) -> dict:
        with self._node_lock:
            if self._retired:
                return self._status()
            if self.state == "running":
                return self._status()
            self._generation += 1
            generation = self._generation
            stop_event = threading.Event()
            frames = LatestFrame()
            self._stop_event = stop_event
            self._frames = frames
            if self._subscription is None:
                self._subscription = self.create_subscription(
                    CompressedImage,
                    self._input_topic,
                    self._image_callback,
                    CAMERA_QOS,
                )
            self.state = "running"
            self._workers = [worker for worker in self._workers if worker.is_alive()]
            self._worker = threading.Thread(
                target=self._inference_worker,
                args=(generation, stop_event, frames),
                name=f"face-worker-{generation}",
                daemon=True,
            )
            self._workers.append(self._worker)
            self._worker.start()
            log.info("[face] started: %s -> %s", self._input_topic, self._output_topic)
            return self._status()

    def stop(self) -> dict:
        with self._node_lock:
            self.state = "idle"
            self._stop_event.set()
            self._frames.close()
            deadline = time.monotonic() + 3.0
            for worker in self._workers:
                if worker.is_alive():
                    worker.join(timeout=max(0.0, deadline - time.monotonic()))
            self._workers = [worker for worker in self._workers if worker.is_alive()]
            if self._workers:
                log.warning(
                    "[face] %d worker(s) still stopping: %s",
                    len(self._workers),
                    self._input_topic,
                )
            return {"state": "idle", "input": self._input_topic}

    def retire(self) -> dict:
        with self._node_lock:
            self._retired = True
            return self.stop()

    @property
    def worker_alive(self) -> bool:
        return any(worker.is_alive() for worker in self._workers)

    def _image_callback(self, message: CompressedImage) -> None:
        stop_event = self._stop_event
        frames = self._frames
        if self.state != "running" or stop_event.is_set():
            return
        frames.push(bytes(message.data))

    def _generation_active(
        self,
        generation: int,
        stop_event: threading.Event,
    ) -> bool:
        return (
            self.state == "running"
            and self._generation == generation
            and self._stop_event is stop_event
            and not stop_event.is_set()
        )

    def _inference_worker(
        self,
        generation: int,
        stop_event: threading.Event,
        frames: LatestFrame,
    ) -> None:
        while not stop_event.is_set():
            image_bytes = frames.pop(timeout=1.0)
            if image_bytes is None:
                continue
            started = time.monotonic()
            try:
                payload = self._engine.infer_face_identity(image_bytes)
            except Exception:  # noqa: BLE001 - publish a valid no-result payload
                log.exception("[face] inference failed")
                payload = empty_face_payload()
            if self._generation_active(generation, stop_event):
                message = String()
                message.data = json.dumps(payload, ensure_ascii=False)
                self._publisher.publish(message)
                self.detect_count += 1
            if self._min_interval > 0:
                remaining = self._min_interval - (time.monotonic() - started)
                if remaining > 0:
                    stop_event.wait(remaining)

    def _status(self) -> dict:
        return {
            "state": self.state,
            "input": self._input_topic,
            "output": self._output_topic,
        }


class FaceRecognitionPlugin:
    PREFIX = "face"

    def __init__(self, plugin_cfg: dict, executor):
        self._plugin_cfg = dict(plugin_cfg)
        self._executor = executor
        self._state_lock = threading.Lock()
        self._engine_lock = threading.Lock()
        self._engine = None
        self._engine_state = "idle"
        self._load_error: str | None = None
        self._config_generation = 0
        self._nodes: dict[str, _FaceNode] = {}
        self._instance_configs: dict[str, dict] = {}
        log.info(
            "[face] plugin initialized: backend=%s recognizer=%s",
            self._plugin_cfg.get("backend", "tensorrt"),
            self._plugin_cfg.get("recognizer", "lvface"),
        )

    def get_tools(self) -> list:
        return TOOLS

    def dispatch(self, name: str, args: dict) -> dict | None:
        action = args.get("action") if name == "face" else name
        instance_id = str(args.get("instance_id", ""))
        if action == "info":
            return self._info(instance_id, str(args.get("input_topic", "")))
        if action == "start":
            return self._start(instance_id, args)
        if action == "stop":
            return self._stop(instance_id)
        if action == "config":
            return self._config(instance_id, args)
        return None

    def _ensure_engine(self):
        with self._engine_lock:
            while True:
                with self._state_lock:
                    if self._engine is not None:
                        return self._engine
                    generation = self._config_generation
                    cfg = dict(self._plugin_cfg)
                    self._engine_state = "loading"
                    self._load_error = None
                try:
                    candidate = build_face_engine(cfg)
                except Exception as error:
                    with self._state_lock:
                        if generation == self._config_generation:
                            self._engine_state = "error"
                            self._load_error = str(error)
                    raise
                with self._state_lock:
                    if generation == self._config_generation:
                        self._engine = candidate
                        self._engine_state = "ready"
                        return candidate
                candidate.close()

    def _start(self, instance_id: str, args: dict) -> dict:
        input_topic = str(args.get("input_topic") or "")
        if not input_topic:
            topics = args.get("input_topics") or []
            input_topic = str(topics[0]) if topics else ""
        if not input_topic:
            raise ValueError("input_topic is required for start action")
        node_key = instance_id or input_topic

        retired = None
        with self._state_lock:
            existing = self._nodes.get(node_key)
            if existing is not None and existing._input_topic != input_topic:
                retired = self._nodes.pop(node_key)
                existing = None
        if retired is not None:
            self._dispose(node_key, retired)
        if existing is not None:
            return existing.start()

        # The face Judge publishes immediately after start returns. Build the
        # engines and gallery synchronously on the first start, then keep them
        # cached across the per-case start/stop lifecycle.
        engine = self._ensure_engine()
        cfg = {**self._plugin_cfg, **self._instance_configs.get(node_key, {})}
        suffix = node_key.replace("/", "_").replace("-", "_").lstrip("_")
        node = _FaceNode(
            input_topic,
            engine,
            node_suffix=suffix,
            min_interval=float(cfg.get("min_interval_ms", 0)) / 1000.0,
        )
        registered = False
        with self._state_lock:
            current = self._nodes.get(node_key)
            if current is None and engine is self._engine:
                self._executor.add_node(node)
                self._nodes[node_key] = node
                registered = True
        if not registered:
            node.destroy_node()
            if current is not None:
                return current.start()
            return {"state": "idle", "input": input_topic, "output": _output_topic(input_topic)}
        try:
            return node.start()
        except Exception:
            with self._state_lock:
                if self._nodes.get(node_key) is node:
                    del self._nodes[node_key]
            self._dispose(node_key, node)
            raise

    def _stop(self, instance_id: str) -> dict:
        with self._state_lock:
            if instance_id:
                node = self._nodes.pop(instance_id, None)
                disposed = [] if node is None else [(instance_id, node)]
            else:
                disposed = list(self._nodes.items())
                self._nodes = {}
        for node_key, node in disposed:
            self._dispose(node_key, node)
        return {"state": "idle"}

    def _dispose(self, node_key: str, node: _FaceNode) -> None:
        try:
            node.retire()
        finally:
            dispose_node(self._executor, node, label=f"face/{node_key}")

    def _config(self, instance_id: str, args: dict) -> dict:
        cfg = {
            key: value
            for key, value in args.items()
            if key not in {"action", "instance_id"} and value not in (None, "")
        }
        if instance_id:
            unsupported = set(cfg) - {"min_interval_ms"}
            if unsupported:
                raise ValueError(
                    "face model and gallery settings are shared: "
                    + ", ".join(sorted(unsupported))
                )
            with self._state_lock:
                previous = self._instance_configs.get(instance_id, {})
                self._instance_configs[instance_id] = {**previous, **cfg}
                node = self._nodes.get(instance_id)
                if node is not None and "min_interval_ms" in cfg:
                    node._min_interval = max(0.0, float(cfg["min_interval_ms"]) / 1000.0)
            return {"status": "configured", "instance_id": instance_id}

        if not cfg:
            with self._state_lock:
                return {
                    "status": "configured",
                    "engine_loaded": self._engine is not None,
                    "reused": self._engine is not None,
                }
        lightweight = set(cfg) <= {"min_interval_ms"}
        if lightweight:
            with self._state_lock:
                self._plugin_cfg = {**self._plugin_cfg, **cfg}
                for node in self._nodes.values():
                    node._min_interval = max(
                        0.0, float(self._plugin_cfg.get("min_interval_ms", 0)) / 1000.0
                    )
                loaded = self._engine is not None
            return {"status": "configured", "engine_loaded": loaded, "reused": loaded}

        with self._state_lock:
            self._plugin_cfg = {**self._plugin_cfg, **cfg}
            self._config_generation += 1
            nodes = list(self._nodes.items())
            self._nodes = {}
            stale_engine = self._engine
            self._engine = None
            self._engine_state = "idle"
            self._load_error = None
        for node_key, node in nodes:
            self._dispose(node_key, node)
        if stale_engine is not None:
            stale_engine.close()
        return {"status": "configured", "engine_loaded": False, "reused": False}

    def _info(self, instance_id: str, input_topic: str) -> dict:
        with self._state_lock:
            if instance_id:
                node = self._nodes.get(instance_id)
                topic = node._input_topic if node is not None else input_topic
                state = node.state if node is not None else self._engine_state
                if state == "ready":
                    state = "idle"
                result = self._info_base(state)
                result["topic_in"] = (
                    [{"topic": topic, "format": "image/jpeg", "desc": ""}]
                    if topic
                    else []
                )
                result["topic_out"] = (
                    [{"topic": _output_topic(topic), "format": "data/json", "desc": ""}]
                    if topic
                    else []
                )
                return result
            instances = {
                key: {"state": node.state, "detect_count": node.detect_count}
                for key, node in self._nodes.items()
            }
            if any(item["state"] == "running" for item in instances.values()):
                state = "running"
            elif self._engine_state in {"loading", "error"}:
                state = self._engine_state
            else:
                state = "idle"
            result = self._info_base(state)
            if instances:
                result["instances"] = instances
            return result

    def _info_base(self, state: str) -> dict:
        description = "On-device SCRFD and gallery face identification"
        result = {
            "name": "FaceRecognition",
            "manufacture": "Embodied",
            "model": str(self._plugin_cfg.get("recognizer", "lvface")),
            "state": state,
            "desc": description,
            "topic_in": [],
            "topic_out": [],
        }
        if state == "loading":
            result["desc"] = "Loading face engines and identity gallery..."
        if state == "error" and self._load_error:
            result["desc"] = f"Face engine load failed: {self._load_error}"
            result["error"] = self._load_error
        return result
