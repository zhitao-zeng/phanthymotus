"""Scalar obstacle-distance interface backed by the visual depth plugin.

The depth tool keeps its regional summaries. This interface additionally
publishes the forward region's nearest robust distance on /obstacle.
"""
from __future__ import annotations

import copy
import json
import logging
import queue
import time
from typing import Optional

import numpy as np
from std_msgs.msg import String

from plugins.visual_depth import (
    DEPTH_HEIGHT, DEPTH_WIDTH, TOOLS as DEPTH_TOOLS,
    VideoDepthPerceptionPlugin, _DepthNode, _PUB_QOS,
)

log = logging.getLogger(__name__)
TOOLS = copy.deepcopy(DEPTH_TOOLS)
TOOLS[0]["name"] = "obstacle"
TOOLS[0]["description"] = "Estimate forward obstacle distance in metres"
TOOLS[0]["inputSchema"]["properties"]["action"]["enum"] = ["start", "stop", "info", "config"]
TOOLS[0]["inputSchema"]["x-action-params"] = {
    name: entry for name, entry in TOOLS[0]["inputSchema"]["x-action-params"].items()
    if name in ("start", "stop", "info", "config")
}


def forward_distance(depth: np.ndarray, height: int, width: int) -> float:
    import cv2

    x = np.linspace(0, depth.shape[1] - 1, width, dtype=np.float32)
    y = np.linspace(0, depth.shape[0] - 1, height, dtype=np.float32)
    mx, my = np.meshgrid(x, y)
    restored = cv2.remap(depth.astype(np.float32), mx, my,
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    roi = restored[:round(height * 300 / 480),
                   round(width * 213 / 640):round(width * 426 / 640)]
    valid = roi[np.isfinite(roi)]
    if valid.size < 64:
        raise ValueError("Insufficient valid depth in the forward region")
    return float(np.clip(np.percentile(valid, 1), .3, 10.))


class _ObstacleNode(_DepthNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._obstacle_pub = self.create_publisher(
            String, self._input_topic + "/obstacle", _PUB_QOS)

    def _inference_worker(self):
        import cv2
        from plugins.vision_runtime import decode_depth

        while not self._stop_event.is_set():
            try:
                payload = self._frame_queue.get(timeout=1.)
            except queue.Empty:
                continue
            began = time.perf_counter()
            try:
                frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Invalid camera image")
                outputs, meta = self._model.infer(frame)
                raw = decode_depth(outputs, meta)
                self._last_raw_depth = raw
                distance = forward_distance(raw, *frame.shape[:2])
                display = cv2.resize(raw, (DEPTH_WIDTH, DEPTH_HEIGHT), interpolation=cv2.INTER_NEAREST)
                self._publish(display)
                result = {"status": "ok", "pred_distance": distance,
                          "fallback": False, "latency_ms": (time.perf_counter() - began) * 1000}
            except Exception as error:
                log.exception("[obstacle] depth inference failed")
                result = {"status": "error", "pred_distance": None,
                          "fallback": False, "detail": str(error)}
            message = String()
            message.data = json.dumps(result)
            self._obstacle_pub.publish(message)


class ObstacleDepthPlugin(VideoDepthPerceptionPlugin):
    PREFIX = "obstacle"
    ALIASES = ()

    def get_tools(self):
        return TOOLS

    def _ensure_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    from plugins.vision_runtime import VisionEngineSession
                    self._model = VisionEngineSession(
                        "/opt/vision-depth/indoor-metric.engine", resize_mode="stretch")

    def dispatch(self, name, args):
        if args.get("action", name) not in ("start", "stop", "info", "config"):
            return {"state": "error", "message": "Unsupported obstacle action"}
        return super().dispatch(name, args)

    def _start_node(self, node_key: str, input_topic: Optional[str]):
        with self._nodes_lock:
            if node_key in self._nodes:
                return
            node = _ObstacleNode(
                input_topic or None, self._model, fps=self._fps,
                cal_a=1., cal_b=0., max_depth_m=self._max_depth_m,
                node_suffix=node_key.replace("/", "_").replace("-", "_").lstrip("_"),
            )
            self._executor.add_node(node)
            self._nodes[node_key] = node
        node.start()
