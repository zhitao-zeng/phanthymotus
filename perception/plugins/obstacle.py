"""Scalar obstacle distances using the shared visual-depth runtime and node worker."""
from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
from numbers import Real

import numpy as np
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from plugins.visual_depth import VideoDepthPerceptionPlugin, _DepthNode
from utils.ros_lifecycle import dispose_node

log = logging.getLogger(__name__)
_RESULT_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=10,
                         durability=DurabilityPolicy.VOLATILE)
_DEFAULTS = {'fps': 2, 'decision_threshold_m': 1.0,
             'soft_timeout_s': 2.5, 'fallback_distance_m': 3.0}

TOOLS = [{
    'name': 'obstacle', 'type': 'processor', 'multiInstance': True,
    'description': 'Estimate forward obstacle distance in metres',
    'inputSchema': {'type': 'object', 'required': ['action'], 'properties': {
        'action': {'type': 'string', 'enum': ['start', 'stop', 'info', 'config']},
        'input_topic': {'type': 'string', 'description': 'Camera topic, required for start'},
    }},
    'configSchema': {'type': 'object', 'properties': {
        'fps': {'type': 'integer', 'minimum': 1, 'default': 2, 'scope': 'instance'},
        'decision_threshold_m': {'type': 'number', 'exclusiveMinimum': 0,
                                 'default': 1.0, 'scope': 'instance'},
        'soft_timeout_s': {'type': 'number', 'exclusiveMinimum': 0,
                           'default': 2.5, 'scope': 'instance'},
        'fallback_distance_m': {'type': 'number', 'minimum': 0,
                                'default': 3.0, 'scope': 'instance'},
    }},
    'topic_in': [{'format': 'image/jpeg', 'desc': 'camera image input'}],
    'topic_out': [{'format': 'data/json', 'desc': 'obstacle distance and validity'}],
}]


def _validate_config(values):
    unknown = set(values) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f'Unsupported obstacle settings: {sorted(unknown)}')
    result = {}
    for key, value in values.items():
        if key == 'fps':
            if type(value) is not int or value <= 0:
                raise ValueError('fps must be a positive integer')
        elif isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f'{key} must be a finite number')
        elif value < 0 or (key != 'fallback_distance_m' and value == 0):
            raise ValueError(f'{key} is out of range')
        result[key] = value
    return result


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
        raise ValueError('Insufficient valid depth in the forward region')
    return float(np.clip(np.percentile(valid, 1), .3, 10.))


class _ObstacleNode(_DepthNode):
    def __init__(self, input_topic, model, config, node_suffix):
        super().__init__(input_topic, model, config['fps'], 1., 0., 10., node_suffix)
        self._decision_threshold_m = config['decision_threshold_m']
        self._soft_timeout_s = config['soft_timeout_s']
        self._fallback_distance_m = config['fallback_distance_m']
        self._retired = False

    def _create_publishers(self):
        self._obstacle_pub = self.create_publisher(String, self._input_topic + '/obstacle', _RESULT_QOS)

    def _state(self, state):
        return {'state': state, 'input': self._input_topic, 'output': self._input_topic + '/obstacle'}

    def start(self):
        with self._lifecycle_lock:
            if self._retired:
                return self._state('idle')
            return super().start()

    def request_stop(self):
        with self._lifecycle_lock:
            self._retired = True
            super().request_stop()

    def _result(self, distance, began, error_code=None):
        return {'pred_distance': distance, 'distance_m': distance,
                'near_obstacle': distance < self._decision_threshold_m, 'scene': 'indoor',
                'status': 'fallback' if error_code else 'ok', 'error_code': error_code,
                'fallback': error_code is not None, 'approximate_geometry': False,
                'latency_ms': (time.perf_counter() - began) * 1000}

    def _inference_worker(self):
        import cv2
        from plugins.vision_runtime import decode_depth

        while not self._stop_event.is_set():
            try:
                payload = self._frame_queue.get(timeout=1.)
            except queue.Empty:
                continue
            began = time.perf_counter()
            error_code = 'invalid_image'
            try:
                frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError('Invalid camera image')
                error_code = 'model_error'
                outputs, meta = self._model.infer(frame)
                raw = decode_depth(outputs, meta)
                error_code = 'no_valid_depth'
                distance = forward_distance(raw, *frame.shape[:2])
                error_code = 'timeout'
                if time.perf_counter() - began > self._soft_timeout_s:
                    raise TimeoutError('Depth inference timed out')
                result = self._result(distance, began)
            except Exception:
                log.warning('[obstacle] inference failed: %s', error_code)
                result = self._result(self._fallback_distance_m, began, error_code)
            # stop may have retired the ROS handles while inference was in flight.
            if self._stop_event.is_set():
                continue
            self._frame_count += 1
            message = String()
            message.data = json.dumps(result)
            self._obstacle_pub.publish(message)


class ObstacleDepthPlugin(VideoDepthPerceptionPlugin):
    PREFIX = 'obstacle'
    ALIASES = ()

    def __init__(self, plugin_cfg, namespace, executor):
        super().__init__(plugin_cfg, namespace, executor)
        configured = {k: v for k, v in plugin_cfg.items() if k in _DEFAULTS}
        self._defaults = dict(_DEFAULTS, **_validate_config(configured))
        self._wanted_starts = {}

    def get_tools(self):
        return TOOLS

    def _ensure_model(self):
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return
            from plugins.vision_runtime import VisionEngineSession, decode_depth
            model = VisionEngineSession('/opt/vision-depth/indoor-metric.engine', resize_mode='stretch')
            try:
                width, height = model.input_size
                outputs, meta = model.infer(np.zeros((height, width, 3), np.uint8))
                depth = decode_depth(outputs, meta)
                if depth.shape != (height, width) or not np.isfinite(depth).all():
                    raise ValueError('Invalid depth output during warmup')
            except Exception:
                model.close()
                raise
            self._model = model

    def _dispose(self, key, node):
        node.request_stop()
        try:
            node.stop()
        finally:
            dispose_node(self._executor, node, label=f'obstacle/{key}')

    def _start_node(self, node_key=None, input_topic=None):
        # One completed model load serves all requests that have not been cancelled.
        created = []
        with self._nodes_lock:
            for key, topic in list(self._wanted_starts.items()):
                if key in self._nodes:
                    self._wanted_starts.pop(key)
                    continue
                config = dict(self._defaults, **self._instance_configs.get(key, {}))
                suffix = key.replace('/', '_').replace('-', '_').lstrip('_')
                node = _ObstacleNode(topic, self._model, config, suffix)
                try:
                    self._executor.add_node(node)
                except Exception:
                    node.destroy_node()
                    raise
                self._nodes[key] = node
                self._wanted_starts.pop(key)
                created.append((key, node))
        for key, node in created:
            try:
                node.start()
            except Exception:
                with self._nodes_lock:
                    if self._nodes.get(key) is node:
                        self._nodes.pop(key)
                self._dispose(key, node)
                raise

    def _load_pending(self):
        try:
            self._ensure_model()
            self._start_node()
        except Exception as error:
            with self._nodes_lock:
                self._model_load_error = str(error) if self._wanted_starts else None
            log.exception('[obstacle] startup failed')
        finally:
            with self._nodes_lock:
                self._model_loading = False

    def _stop(self, instance_id):
        with self._nodes_lock:
            keys = [instance_id] if instance_id else list(self._nodes)
            if instance_id:
                self._wanted_starts.pop(instance_id, None)
            else:
                self._wanted_starts.clear()
            removed = [(key, self._nodes.pop(key)) for key in keys if key in self._nodes]
            if not self._wanted_starts:
                self._model_load_error = None
        for key, node in removed:
            self._dispose(key, node)
        return {'state': 'idle', 'stopped_instances': [key for key, _ in removed]}

    def _info(self, args):
        key = args.get('instance_id', '')
        with self._nodes_lock:
            nodes = {k: n for k, n in self._nodes.items() if not key or k == key}
            wanted = {k: t for k, t in self._wanted_starts.items() if not key or k == key}
            state = ('loading' if self._model_loading and wanted else
                     'error' if self._model_load_error and wanted else
                     'running' if nodes else 'idle')
            topic = args.get('input_topic') or next((n._input_topic for n in nodes.values()), '')
            topic = topic or next(iter(wanted.values()), '')
            if key in nodes:
                topic = nodes[key]._input_topic
            elif key in wanted:
                topic = wanted[key]
            result = {'name': 'ObstacleDistance', 'manufacture': 'Embodied', 'model': 'yolo26s-depth',
                      'state': state, 'instances': {k: {'input': n._input_topic, 'fps': n._fps,
                       'frame_count': n._frame_count} for k, n in nodes.items()},
                      'topic_in': [{'topic': topic, 'format': 'image/jpeg'}] if topic else [],
                      'topic_out': [{'topic': topic + '/obstacle', 'format': 'data/json'}] if topic else []}
            if state == 'error':
                result['error'] = self._model_load_error
            return result

    def dispatch(self, name, args):
        action = args.get('action', name)
        key = args.get('instance_id', '')
        if action == 'info':
            return self._info(args)
        if action == 'stop':
            return self._stop(key)
        if action == 'config':
            cfg = _validate_config({k: v for k, v in args.items()
                                    if k not in ('action', 'instance_id') and v is not None and v != ''})
            with self._nodes_lock:
                previous = self._instance_configs.get(key, {}) if key else self._defaults
                merged = dict(previous, **cfg)
                if merged == previous:
                    return {'status': 'configured', 'config': merged}
                if key:
                    self._instance_configs[key] = merged
                    keys = [key]
                else:
                    self._defaults = merged
                    keys = list(self._nodes)
                removed = [(k, self._nodes.pop(k)) for k in keys if k in self._nodes]
            for k, node in removed:
                self._dispose(k, node)
            return {'status': 'configured', 'instance_id': key, 'config': merged}
        if action == 'start':
            topic = args.get('input_topic') or next(iter(args.get('input_topics') or []), None)
            if not isinstance(topic, str) or not topic.strip():
                raise ValueError('input_topic is required')
            key = key or topic
            with self._nodes_lock:
                existing = self._nodes.get(key)
                rebound = existing if existing is not None and existing._input_topic != topic else None
                if rebound is not None:
                    self._nodes.pop(key)
                    existing = None
                if existing is None:
                    self._wanted_starts[key] = topic
                    self._model_load_error = None
                cold = self._model is None
                if cold and not self._model_loading:
                    self._model_loading = True
                    threading.Thread(target=self._load_pending, daemon=True,
                                     name='obstacle_model_load').start()
            if rebound is not None:
                self._dispose(key, rebound)
            if cold:
                return {'state': 'loading', 'input': topic, 'output': topic + '/obstacle'}
            if existing is not None:
                return existing.start()
            self._start_node()
            with self._nodes_lock:
                node = self._nodes.get(key)
            return node._state('running' if node._running else 'idle') if node else {'state': 'idle'}
        return {'state': 'error', 'message': 'Unsupported obstacle action'}
