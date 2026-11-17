"""Recognize a received video segment once, after the input becomes quiet.

The JPEG buffer is independent of the live pose frame-rate gate. All selected
observations came from this subscription; no file boundaries or future input
are needed. The frozen action models use the same person and hand crops.
"""
from collections import deque
from pathlib import Path
import threading

import numpy as np

from plugins.action_vocab import OFFICIAL_126_CLASSES
from plugins.pose_action import PoseFrame, PoseTracker
from plugins.pose_stgcn import SkeletonActionBackend, looks_like_probabilities, softmax


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

    def _take(self):
        result = (list(self._frames), {'received_frames': self._received,
                  'retained_frames': len(self._frames),
                  'truncated_frames': self._received-len(self._frames)})
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


def rgb_frame_indices(length):
    # Preserve the saved model's 16-observation cache -> four per view policy.
    sixteen = np.linspace(0, length-1, 16).round().astype(int)
    return sixteen[np.linspace(0, 15, 4).round().astype(int)]


def build_rgb_input(rgb_frames, keypoints):
    """Four RGB source images plus the whole segment's primary COCO-17 poses."""
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
    people, hands = [], []
    for rgb, fi in zip(rgb_frames, rgb_frame_indices(len(p0))):
        h, w = rgb.shape[:2]
        people.append(crop_square(rgb, cx, cy, max(side, 120.)).transpose(2, 0, 1))
        hx, hy, hs = cx, cy-.12*side, max(.42*side, 96.)
        kp = p0[fi]
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


def fuse_probabilities(body, rgb):
    # These model heads share this fixed ontology recipe. Non-NTU classes are
    # the kitchen actions, none, and the traffic/control gesture family.
    weights = np.full(126, .08, np.float32)
    weights[np.array([38]+list(range(103, 117)))-1] = .30
    weights[12:28] = .65
    result = (1-weights)*np.asarray(body)+weights*np.asarray(rgb)
    return result/result.sum()


class SegmentActionBackend:
    def __init__(self, pose_model, body_engine, rgb_engine, min_score=.02,
                 confidence=.35, observations=48):
        self.pose_model = pose_model
        self.body = SkeletonActionBackend(engine=body_engine, modality='jb',
                                         prenorm_mode='fix', min_motion=0.)
        self.rgb_engine = rgb_engine
        self.min_score = float(min_score)
        self.confidence = float(confidence)
        self.observations = int(observations)
        self.last_error = None

    @classmethod
    def from_directory(cls, pose_model, directory, **kwargs):
        from utils.tensorrt_runtime import TensorRTEngine
        root = Path(directory)
        body = TensorRTEngine(root/'stgcnpp-action126-v2p.engine', shared_scratch=True)
        try:
            rgb = TensorRTEngine(root/'tsm-mnv2-dual8f-action126.engine', shared_scratch=True)
        except Exception:
            body.close()
            raise
        return cls(pose_model, body, rgb, **kwargs)

    def recognize(self, received, stop_event=None):
        import cv2
        from plugins.vision_runtime import decode_poses
        if len(received) < 4:
            return None
        indices = np.linspace(0, len(received)-1,
                              min(self.observations, len(received))).round().astype(int)
        samples = [received[i] for i in indices]
        image_indices = list(rgb_frame_indices(len(samples)))
        rgb_images, frame_records = {}, []
        tracker = PoseTracker(timeout_s=2., history_s=30.)
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

        cached_poses = len(samples[0]) == 3
        if cached_poses:
            for fi, (_, data, pose) in enumerate(samples):
                if stop_event is not None and stop_event.is_set():
                    return None
                if pose is None:
                    raise ValueError('segment observation has no processed pose')
                boxes, scores, kps, shape = pose
                record_detections(fi, shape, boxes, scores, kps)
                if fi in image_indices:
                    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError('segment contains an undecodable JPEG')
                    rgb_images[fi] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for start in ([] if cached_poses else range(0, len(samples), 8)):
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
                if fi in image_indices:
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
            poses.append(PoseFrame(timestamp, box, kp, .3, image_size=size))
            primary_records.append(kp)
        if snapshot_index is None:
            return None
        blob = self.body._build_input(poses, self.body._window_frames())
        body = self.body._run(blob)
        if not looks_like_probabilities(body):
            body = softmax(body)
        rgb_blob = build_rgb_input([rgb_images[i] for i in image_indices], primary_records)
        raw_rgb = np.asarray(self.rgb_engine.infer(rgb_blob)[0], np.float32).reshape(-1)
        rgb = raw_rgb if looks_like_probabilities(raw_rgb) else softmax(raw_rgb)
        if body.shape != (126,) or rgb.shape != (126,):
            raise ValueError('segment action heads must both emit 126 classes')
        probabilities = fuse_probabilities(body, rgb)
        action_id = int(probabilities.argmax())+1
        score = float(probabilities[action_id-1])
        activity = None
        if score >= self.min_score:
            zh, name = OFFICIAL_126_CLASSES[action_id]
            activity = {'action_id': action_id, 'name': name, 'name_zh': zh,
                        'score': round(score, 3), 'source': 'segment-dual'}
        _, size, records = frame_records[snapshot_index]
        image = cv2.imdecode(np.frombuffer(samples[snapshot_index][1], np.uint8), cv2.IMREAD_COLOR)
        primary_box = next(r[1] for r in records if r[0] == primary)
        self.last_error = None
        return {'activity': activity, 'probabilities': probabilities,
                'evidence': {'frames_used': len(samples), 'primary_track': primary,
                             'poses_computed_during_reception': cached_poses,
                             'source': 'received-segment', 'best_action_id': action_id,
                             'best_score': round(score, 3)},
                'image': image, 'boxes': np.array([r[1] for r in records]),
                'scores': np.array([r[2] for r in records]),
                'keypoints': np.array([r[3] for r in records]),
                'primary_box': primary_box}
