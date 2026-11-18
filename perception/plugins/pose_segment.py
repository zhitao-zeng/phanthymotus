"""Recognize a received video segment once, after the input becomes quiet.

The JPEG buffer is independent of the live pose frame-rate gate. All selected
observations came from this subscription; no file boundaries or future input
are needed. The frozen action models use the same person and hand crops.
"""
from collections import deque
from pathlib import Path
import json
import logging
import threading

import numpy as np

from plugins.action_vocab import OFFICIAL_126_CLASSES
from plugins.pose_action import PoseTracker, body_height
from plugins.pose_stgcn import SkeletonActionBackend, looks_like_probabilities, softmax


log = logging.getLogger(__name__)


class _SkeletonFrame:
    """Tracker/head fields without posture geometry that neither consumes."""

    __slots__ = ('t', 'box', 'keypoints', 'image_size', 'height')

    def __init__(self, t, box, keypoints, min_conf, image_size=None):
        shape = getattr(keypoints, 'shape', None)
        if shape is None or len(shape) != 2 or shape[0] != 17 or shape[1] < 3:
            raise ValueError(f'keypoints must be (17, 3+); got {shape}')
        self.t = float(t)
        self.box = [float(v) for v in box]
        self.keypoints = keypoints
        self.image_size = (tuple(int(v) for v in image_size)
                           if image_size and image_size[0] and image_size[1]
                           else None)
        self.height = body_height(box)


class SegmentBuffer:
    def __init__(self, quiet_s=.4, max_s=30., max_bytes=32*1024**2):
        self.quiet_s = float(quiet_s)
        self.max_s = float(max_s)
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self._frames = deque()
        self._ready = deque(maxlen=2)
        self._by_time = {}
        self._bytes = 0
        self._received = 0
        self.dropped_segments = 0

    def _snapshot(self):
        return (list(self._frames), {'received_frames': self._received,
                'retained_frames': len(self._frames),
                'truncated_frames': self._received-len(self._frames)})

    def _take(self):
        result = self._snapshot()
        self._frames.clear()
        self._bytes = self._received = 0
        return result

    def append(self, payload, now):
        with self._lock:
            if self._frames and now-self._frames[-1][0] >= self.quiet_s:
                if len(self._ready) == self._ready.maxlen:
                    self.dropped_segments += 1
                    self._forget(self._ready.popleft()[0])
                self._ready.append(self._take())
            data = bytes(payload)
            record = [float(now), data, None]
            self._frames.append(record)
            self._by_time[float(now)] = record
            self._bytes += len(data)
            self._received += 1
            while self._frames and (self._bytes > self.max_bytes or
                                    now-self._frames[0][0] > self.max_s):
                removed = self._frames.popleft()
                self._bytes -= len(removed[1])
                self._by_time.pop(removed[0], None)

    def record_pose(self, now, boxes, scores, keypoints, image_shape):
        """Attach the real worker result to its received JPEG, even after a gap."""
        with self._lock:
            record = self._by_time.get(float(now))
            if record is not None:
                record[2] = (np.array(boxes, copy=True), np.array(scores, copy=True),
                             np.array(keypoints, copy=True), tuple(image_shape[:2]))

    def _forget(self, frames):
        for record in frames:
            self._by_time.pop(record[0], None)

    def next_wait_s(self, now, preview_s=None):
        """Return seconds until the active segment reaches preview or quiet confirm."""
        with self._lock:
            if self._ready:
                return 0.0
            if not self._frames:
                return None
            target_s = self.quiet_s
            if preview_s is not None and 0.0 < float(preview_s) < self.quiet_s:
                target_s = float(preview_s)
            return max(0.0, self._frames[-1][0] + target_s - float(now))

    def snapshot_if_quiet(self, now, quiet_s):
        """Read a provisional segment without closing it on a brief pause."""
        with self._lock:
            if self._frames and now-self._frames[-1][0] >= quiet_s:
                return self._snapshot()
        return None

    def pop_ready(self, now):
        with self._lock:
            if self._ready:
                result = self._ready.popleft()
                self._forget(result[0])
                return result
            if self._frames and now-self._frames[-1][0] >= self.quiet_s:
                result = self._take()
                self._forget(result[0])
                return result
        return None

    def reset(self):
        with self._lock:
            self._frames.clear()
            self._ready.clear()
            self._by_time.clear()
            self._bytes = self._received = 0
            self.dropped_segments = 0


def crop_square(rgb, cx, cy, side, out_hw=192):
    import cv2
    h, w = rgb.shape[:2]
    half = max(side*.5, 24.)
    if half*2 >= max(h, w):
        return cv2.resize(rgb, (out_hw, out_hw), interpolation=cv2.INTER_LINEAR)
    x0, y0 = int(round(cx-half)), int(round(cy-half))
    x1, y1 = x0+int(round(2*half)), y0+int(round(2*half))
    if x0 < 0:
        x1, x0 = min(w, x1-x0), 0
    if y0 < 0:
        y1, y0 = min(h, y1-y0), 0
    if x1 > w:
        x0, x1 = max(0, x0-(x1-w)), w
    if y1 > h:
        y0, y1 = max(0, y0-(y1-h)), h
    patch = rgb[y0:y1, x0:x1]
    if patch.size == 0:
        patch = rgb
    return cv2.resize(patch, (out_hw, out_hw), interpolation=cv2.INTER_LINEAR)


def rgb_frame_indices(length, n_per_view=4):
    # Preserve the saved models' 16-observation cache -> observations per view.
    sixteen = np.linspace(0, length-1, 16).round().astype(int)
    return sixteen[np.linspace(0, 15, n_per_view).round().astype(int)]


def time_uniform_pick_indices(timestamps, count):
    """Pick `count` frame indices uniformly over physical timestamps."""
    length = len(timestamps)
    if length <= 0:
        raise ValueError('cannot sample from an empty timeline')
    if length == 1:
        return np.zeros(count, dtype=int)
    times = np.asarray(timestamps, dtype=np.float64)
    if float(times[-1] - times[0]) <= 1e-6 or not bool(np.all(np.diff(times) >= -1e-9)):
        indices = np.linspace(0, length - 1, count).round().astype(int)
        return indices
    grid = np.linspace(float(times[0]), float(times[-1]), count, dtype=np.float64)
    hi = np.clip(np.searchsorted(times, grid, side='left'), 0, length - 1)
    lo = np.maximum(hi - 1, 0)
    pick_lo = np.abs(times[lo] - grid) < np.abs(times[hi] - grid)
    out = np.where(pick_lo, lo, hi).astype(int)
    if count >= 2:
        out[0] = 0
        out[-1] = length - 1
    return out


def interpolate_keypoints_at_times(posed_keypoints, posed_times, target_times):
    """Linearly interpolate (T, 17, 3) primary-track poses onto target timestamps."""
    kps = np.asarray(posed_keypoints, dtype=np.float32)
    t_target = np.asarray(target_times, dtype=np.float64)
    if len(kps) == 0:
        return np.zeros((len(t_target), 17, 3), dtype=np.float32)
    t_posed = np.asarray(posed_times, dtype=np.float64)
    if len(kps) == 1 or float(t_posed[-1] - t_posed[0]) <= 1e-6 or not bool(np.all(np.diff(t_posed) >= -1e-9)):
        return np.repeat(kps[:1], len(t_target), axis=0).astype(np.float32)
    hi = np.clip(np.searchsorted(t_posed, t_target, side='right'), 1, len(kps) - 1)
    lo = hi - 1
    dt = np.maximum(t_posed[hi] - t_posed[lo], 1e-6)
    alpha = np.clip((t_target - t_posed[lo]) / dt, 0.0, 1.0).astype(np.float32)[:, None, None]
    lo_kp, hi_kp = kps[lo], kps[hi]
    lo_ok, hi_ok = lo_kp[..., 2:3] > .20, hi_kp[..., 2:3] > .20
    xy_interp = lo_kp[..., :2] * (1.0 - alpha) + hi_kp[..., :2] * alpha
    xy = np.where(lo_ok & hi_ok, xy_interp, np.where(lo_ok, lo_kp[..., :2], hi_kp[..., :2]))
    sc = lo_kp[..., 2:3] * (1.0 - alpha) + hi_kp[..., 2:3] * alpha
    return np.concatenate([xy, sc], axis=-1).astype(np.float32)


def _decode_rgb_jpeg(data, error_message='segment contains an undecodable JPEG'):
    import cv2

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(error_message)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_rgb_input(rgb_frames, keypoints, frame_keypoints=None):
    """RGB source images plus whole-segment (or aligned per-frame) COCO-17 poses."""
    p0 = np.asarray(keypoints, np.float32)
    h, w = rgb_frames[0].shape[:2]
    cx, cy, side = w*.5, h*.5, float(min(h, w))
    valid = p0[..., 2] > .20
    if valid.any():
        xs, ys = p0[..., 0][valid], p0[..., 1][valid]
        cx = .5*(float(np.percentile(xs, 5))+float(np.percentile(xs, 95)))
        cy = .5*(float(np.percentile(ys, 3))+float(np.percentile(ys, 97)))
        side = max(float(np.percentile(xs, 97)-np.percentile(xs, 3)),
                   float(np.percentile(ys, 98)-np.percentile(ys, 2)))*1.30
    wrist_scores = []
    for wi in [9, 10]:
        ok = p0[:, wi, 2] > .25
        wrist_scores.append(float(-p0[ok, wi, 1].mean()) if ok.any() else -9999.)
    prefer_left = wrist_scores[0] >= wrist_scores[1]
    per_frame_kps = (
        [np.asarray(kp, np.float32) for kp in frame_keypoints]
        if frame_keypoints is not None
        else [p0[fi] for fi in rgb_frame_indices(len(p0), len(rgb_frames))]
    )
    people, hands = [], []
    for rgb, kp in zip(rgb_frames, per_frame_kps):
        h, w = rgb.shape[:2]
        people.append(crop_square(rgb, cx, cy, max(side, 120.)).transpose(2, 0, 1))
        hx, hy, hs = cx, cy-.12*side, max(.42*side, 96.)
        points = []
        for wi, ei in [(9, 7), (10, 8)]:
            if kp[wi, 2] > .20:
                x, y = float(kp[wi, 0]), float(kp[wi, 1])
                if kp[ei, 2] > .20:
                    x += .30*(x-float(kp[ei, 0]))
                    y += .30*(y-float(kp[ei, 1]))
                points.append((np.clip(x, 0, w-1), np.clip(y, 0, h-1)))
        if len(points) == 2:
            (x1, y1), (x2, y2) = points
            distance = float(np.hypot(x1-x2, y1-y2))
            if distance <= .50*side:
                hx, hy = .5*(x1+x2), .5*(y1+y2)
                hs = max(distance+.34*side, .38*side, 96.)
            else:
                hx, hy = points[0 if prefer_left else 1]
                hs = max(.38*side, 96.)
        elif points:
            hx, hy = points[0]
            hs = max(.38*side, 96.)
        hands.append(crop_square(rgb, hx, hy, hs).transpose(2, 0, 1))
    frames = np.concatenate([people, hands]).astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53], np.float32)[None, :, None, None]
    std = np.array([58.395, 57.12, 57.375], np.float32)[None, :, None, None]
    return ((frames-mean)/std)[None]


def fuse_probabilities(body, rgb, ntu_rgb_weight=.08):
    # These model heads share this fixed ontology recipe. Non-NTU classes are
    # the kitchen actions, none, and the traffic/control gesture family.
    weights = np.full(126, ntu_rgb_weight, np.float32)
    weights[np.array([38]+list(range(103, 117)))-1] = .30
    weights[12:28] = .65
    result = (1-weights)*np.asarray(body)+weights*np.asarray(rgb)
    return result/result.sum()


class SegmentActionBackend:
    supports_raw_received = True

    def __init__(self, pose_model, body_engine, rgb_engine, min_score=.02,
                 confidence=.35, observations=48, ntu_rgb_weight=.08, handface=None):
        self.pose_model = pose_model
        self.body = SkeletonActionBackend(engine=body_engine, modality='jb',
                                         prenorm_mode='fix', min_motion=0.)
        self.rgb_engine = rgb_engine
        shape = getattr(rgb_engine, 'input_shape', None) or getattr(
            rgb_engine, 'optimization_shape', None)
        if (shape is None or len(shape) != 5 or tuple(shape[2:]) != (3, 192, 192)
                or int(shape[1]) <= 0 or int(shape[1]) % 2):
            raise ValueError('RGB action engine must take (N, 2*K, 3, 192, 192)')
        self.rgb_n_per_view = int(shape[1])//2
        self.ntu_rgb_weight = float(ntu_rgb_weight)
        self.min_score = float(min_score)
        self.confidence = float(confidence)
        self.observations = int(observations)
        self.handface = handface
        self.last_error = None

    def warmup(self):
        """Run one dummy pass per segment TensorRT engine so case 1 sees warm latency."""
        engines = [self.body._engine, self.rgb_engine]
        if self.handface is not None:
            engines.extend([self.handface.wholebody_engine, self.handface.handface_engine])
        for engine in engines:
            infer = getattr(engine, 'infer', None)
            shape = getattr(engine, 'optimization_shape', None) or getattr(engine, 'input_shape', None)
            if not callable(infer) or not shape:
                continue
            try:
                dims = tuple(int(d) for d in shape)
                if any(d <= 0 for d in dims):
                    continue
                dtype = getattr(engine, 'input_dtype', None) or np.float32
                infer(np.zeros(dims, dtype=dtype))
            except Exception as error:
                log.warning('Segment engine warmup skipped: %s', error)

    @classmethod
    def from_directory(cls, pose_model, directory, **kwargs):
        from utils.tensorrt_runtime import TensorRTEngine
        root = Path(directory)
        manifest = root/'models.json'
        files = json.loads(manifest.read_text()) if manifest.exists() else {}
        body = TensorRTEngine(root/files.get('body_engine', 'stgcnpp-action126-v2p.engine'), shared_scratch=True)
        try:
            rgb = TensorRTEngine(root/files.get('rgb_engine', 'tsm-mnv2-dual8f-action126.engine'), shared_scratch=True)
        except Exception:
            body.close()
            raise
        handface = None
        try:
            kwargs.setdefault('ntu_rgb_weight', files.get('rgb_ntu_weight', .08))
            if 'handface' in files and 'handface' not in kwargs:
                from plugins.pose_handface import HandFaceActionBackend
                try:
                    handface = HandFaceActionBackend.from_directory(root, files['handface'])
                except Exception as error:
                    log.warning('HandFace bundle unavailable: %s', error)
                kwargs['handface'] = handface
            return cls(pose_model, body, rgb, **kwargs)
        except Exception:
            body.close()
            rgb.close()
            if handface is not None:
                handface.close()
            raise

    def _refine_handface(self, body, rgb, source_frames, source_times,
                         primary_records, cached_rgb_images, observation_fps, stop_event):
        if source_times is not None:
            target = time_uniform_pick_indices(source_times, self.handface.FRAME_COUNT)
        else:
            target = np.linspace(0, len(source_frames) - 1,
                                 self.handface.FRAME_COUNT).round().astype(int)
        count = self.handface.wholebody_frame_count
        indices, reconstruction = target, None
        if count < len(target):
            if source_times is not None:
                indices = time_uniform_pick_indices(source_times, count)
            else:
                indices = rgb_frame_indices(len(source_frames), count)
            reconstruction = np.abs(indices[:, None] - target[None, :]).argmin(axis=0)
        for index in indices:
            if stop_event is not None and stop_event.is_set():
                return None
            if index not in cached_rgb_images:
                cached_rgb_images[index] = _decode_rgb_jpeg(
                    source_frames[index][1],
                    'HandFace observation contains an undecodable JPEG',
                )
        handface, quality = self.handface.predict(
            primary_records, [cached_rgb_images[i] for i in indices], reconstruction)
        if stop_event is not None and stop_event.is_set():
            return None
        return self.handface.fuse(body, rgb, handface, quality, observation_fps)

    def _set_decision(self, result, probabilities, source):
        action_id = int(probabilities.argmax())+1
        score = float(probabilities[action_id-1])
        activity = None
        if score >= self.min_score:
            zh, name = OFFICIAL_126_CLASSES[action_id]
            activity = {'action_id': action_id, 'name': name, 'name_zh': zh,
                        'score': round(score, 3), 'source': source}
        result.update(activity=activity, probabilities=probabilities)
        result['evidence'].update(best_action_id=action_id, best_score=round(score, 3))

    def complete_handface(self, result, stop_event=None):
        pending = result.pop('_handface_pending', None)
        if pending is None:
            return result
        try:
            refined = self._refine_handface(*pending, stop_event)
        except Exception as error:
            log.warning('HandFace inference unavailable: %s', error)
            result['evidence']['handface_state'] = 'fallback'
            return result
        if refined is None:
            return None
        self._set_decision(result, refined, 'segment-tri')
        result['evidence']['handface_state'] = 'applied'
        return result

    def recognize(self, received, stop_event=None, *, defer_handface=False, raw_received=None):
        import cv2
        from plugins.vision_runtime import decode_poses
        if len(received) < 4:
            return None
        indices = np.linspace(0, len(received)-1,
                              min(self.observations, len(received))).round().astype(int)
        samples = [received[i] for i in indices]
        cached_poses = len(samples[0]) == 3

        raw_timeline = list(raw_received) if raw_received is not None else None
        raw_times = (
            np.array([float(r[0]) for r in raw_timeline], dtype=np.float64)
            if raw_timeline is not None else None
        )
        use_raw_buffer = (
            cached_poses
            and raw_timeline is not None
            and len(raw_timeline) > len(received)
            and len(raw_timeline) >= 4
            and float(raw_times[-1] - raw_times[0]) > 1e-6
            and bool(np.all(np.diff(raw_times) >= -1e-9))
        )

        if use_raw_buffer:
            rgb_raw_indices = list(time_uniform_pick_indices(raw_times, self.rgb_n_per_view))
            posed_image_set = set()
        else:
            rgb_raw_indices = list(rgb_frame_indices(len(samples), self.rgb_n_per_view))
            posed_image_set = set(rgb_raw_indices)

        rgb_images, raw_rgb_images, frame_records = {}, {}, []
        tracker = PoseTracker(timeout_s=2., history_s=30.,
                              frame_factory=_SkeletonFrame)
        track_weights = {}
        base_time = samples[0][0]

        def record_detections(fi, shape, boxes, scores, kps):
            h, w = shape
            order = np.argsort(np.asarray(scores, np.float32))[::-1][:4]
            boxes, scores, kps = boxes[order], scores[order], kps[order]
            timestamp = samples[fi][0]-base_time
            tracks = tracker.update(boxes, kps, timestamp, image_size=(w, h))
            records = []
            for track, box, score, kp in zip(tracks, boxes, scores, kps):
                area = max(1., float(box[2]-box[0])*float(box[3]-box[1]))/max(1., float(w*h))
                track_weights[track.id] = track_weights.get(track.id, 0.)+float(score)*(.5+area)
                records.append((track.id, box, float(score), kp))
            frame_records.append((timestamp, (w, h), records))

        if cached_poses:
            for fi, (_, data, pose) in enumerate(samples):
                if stop_event is not None and stop_event.is_set():
                    return None
                if pose is None:
                    raise ValueError('segment observation has no processed pose')
                boxes, scores, kps, shape = pose
                record_detections(fi, shape, boxes, scores, kps)
                if fi in posed_image_set:
                    rgb_images[fi] = _decode_rgb_jpeg(data)
            if use_raw_buffer:
                for ri in rgb_raw_indices:
                    if stop_event is not None and stop_event.is_set():
                        return None
                    if ri not in raw_rgb_images:
                        raw_rgb_images[ri] = _decode_rgb_jpeg(raw_timeline[ri][1])
        else:
            for start in range(0, len(samples), 8):
                if stop_event is not None and stop_event.is_set():
                    return None
                images = [cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                          for _, data in samples[start:start+8]]
                if any(image is None for image in images):
                    raise ValueError('segment contains an undecodable JPEG')
                infer_many = getattr(self.pose_model, 'infer_batch', None)
                outputs = (infer_many(images) if infer_many else
                           [self.pose_model.infer(image) for image in images])
                for offset, (image, (raw, meta)) in enumerate(zip(images, outputs)):
                    fi = start+offset
                    h, w = image.shape[:2]
                    if fi in posed_image_set:
                        rgb_images[fi] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    boxes, scores, kps = decode_poses(raw, meta, self.confidence)
                    record_detections(fi, (h, w), boxes, scores, kps)
                del images, outputs
        if not track_weights:
            return None
        primary = max(track_weights, key=track_weights.get)
        poses, primary_records = [], []
        snapshot_index = None
        for fi, (timestamp, size, records) in enumerate(frame_records):
            chosen = next((r for r in records if r[0] == primary), None)
            if chosen is not None:
                snapshot_index = fi
            if chosen is None and records:
                chosen = records[0]
            box = chosen[1] if chosen else np.zeros(4, np.float32)
            kp = chosen[3] if chosen else np.zeros((17, 3), np.float32)
            poses.append(_SkeletonFrame(timestamp, box, kp, .3, image_size=size))
            primary_records.append(kp)
        if snapshot_index is None:
            return None
        blob = self.body._build_input(poses, self.body._window_frames())
        body = self.body._run(blob)
        if not looks_like_probabilities(body):
            body = softmax(body)

        if use_raw_buffer:
            posed_times = np.array([float(s[0]) for s in samples], dtype=np.float64)
            rgb_kps = interpolate_keypoints_at_times(
                np.asarray(primary_records, dtype=np.float32),
                posed_times,
                raw_times[rgb_raw_indices],
            )
            rgb_blob = build_rgb_input(
                [raw_rgb_images[i] for i in rgb_raw_indices],
                primary_records,
                frame_keypoints=rgb_kps,
            )
        else:
            rgb_blob = build_rgb_input([rgb_images[i] for i in rgb_raw_indices], primary_records)

        raw_rgb = np.asarray(self.rgb_engine.infer(rgb_blob)[0], np.float32).reshape(-1)
        rgb = raw_rgb if looks_like_probabilities(raw_rgb) else softmax(raw_rgb)
        if body.shape != (126,) or rgb.shape != (126,):
            raise ValueError('segment action heads must both emit 126 classes')
        probabilities = fuse_probabilities(body, rgb, self.ntu_rgb_weight)

        handface_evidence = {}
        pending = None
        if self.handface is not None:
            handface_state = 'skipped'
            elapsed = received[-1][0] - received[0][0]
            observation_fps = (len(received)-1) / max(elapsed, 1e-6)
            if self.handface.should_run(probabilities, observation_fps):
                hf_source = raw_timeline if use_raw_buffer else samples
                hf_times = raw_times if use_raw_buffer else None
                hf_cache = raw_rgb_images if use_raw_buffer else rgb_images
                pending = (body, rgb, hf_source, hf_times, primary_records, hf_cache, observation_fps)
                handface_state = 'pending'
            handface_evidence = {
                'handface_state': handface_state,
                'handface_input_fps': round(observation_fps, 3),
            }

        _, size, records = frame_records[snapshot_index]
        if snapshot_index in rgb_images:
            image = cv2.cvtColor(rgb_images[snapshot_index], cv2.COLOR_RGB2BGR)
        elif use_raw_buffer:
            snap_ts, snap_data = samples[snapshot_index][0], samples[snapshot_index][1]
            cached_rgb = next(
                (raw_rgb_images[ri] for ri in raw_rgb_images
                 if raw_timeline[ri][0] == snap_ts and raw_timeline[ri][1] == snap_data),
                None,
            )
            if cached_rgb is not None:
                image = cv2.cvtColor(cached_rgb, cv2.COLOR_RGB2BGR)
            else:
                image = cv2.imdecode(np.frombuffer(snap_data, np.uint8), cv2.IMREAD_COLOR)
        else:
            image = cv2.imdecode(np.frombuffer(samples[snapshot_index][1], np.uint8), cv2.IMREAD_COLOR)

        primary_box = next(r[1] for r in records if r[0] == primary)
        self.last_error = None
        evidence = {
            'frames_used': len(samples),
            'primary_track': primary,
            'poses_computed_during_reception': cached_poses,
            'RGB_frames_per_view': self.rgb_n_per_view,
            'RGB_NTU_weight': self.ntu_rgb_weight,
            'source': 'received-segment',
            **handface_evidence,
        }
        if use_raw_buffer:
            evidence['raw_frames_available'] = len(raw_timeline)
            evidence['rgb_source'] = 'raw-buffer'
        result = {
            'evidence': evidence,
            'image': image,
            'boxes': np.array([r[1] for r in records]),
            'scores': np.array([r[2] for r in records]),
            'keypoints': np.array([r[3] for r in records]),
            'primary_box': primary_box,
        }
        self._set_decision(result, probabilities, 'segment-dual')
        if pending is not None:
            result['_handface_pending'] = pending
            if not defer_handface:
                return self.complete_handface(result, stop_event)
        return result
