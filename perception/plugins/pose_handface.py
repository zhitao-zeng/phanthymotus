"""Optional whole-body branch for the received-clip action classifier."""
import json
import math

import numpy as np

from plugins.pose_stgcn import looks_like_probabilities, softmax


HAND_BONES = [
    (1, 0), (2, 1), (3, 2), (4, 3),
    (5, 0), (6, 5), (7, 6), (8, 7),
    (9, 0), (10, 9), (11, 10), (12, 11),
    (13, 0), (14, 13), (15, 14), (16, 15),
    (17, 0), (18, 17), (19, 18), (20, 19),
]


FINGERTIPS = [4, 8, 12, 16, 20]


TIP_PAIRS = [(FINGERTIPS[i], FINGERTIPS[j]) for i in range(5) for j in range(i + 1, 5)]


def build_handface_geom_features(kp133: np.ndarray) -> np.ndarray:
    """Pure geometric + kinematic feature extractor: (T=8, 133, 3) -> (T=8, F_dim) float32."""
    kp = kp133.astype(np.float32)
    T = kp.shape[0]
    feats = []

    # Shoulder scale for craniofacial & bimanual relational normalization
    l_sh, r_sh = kp[:, 5, :2], kp[:, 6, :2]
    sh_span = np.maximum(np.linalg.norm(l_sh - r_sh, axis=-1, keepdims=True), 0.12)  # (T, 1)
    neck = 0.5 * (l_sh + r_sh)  # (T, 2)

    # Upper-body normalized joints (0..10: nose, eyes, ears, shoulders, elbows, wrists)
    ub_xy = (kp[:, :11, :2] - neck[:, None, :]) / sh_span[:, :, None]  # (T, 11, 2)
    ub_sc = kp[:, :11, 2:3]
    feats.append(ub_xy.reshape(T, -1))
    feats.append(ub_sc.reshape(T, -1))

    # Craniofacial anchors: nose_tip(53), mouth_center(mean 71..82), L_eye(1), R_eye(2), L_ear(3), R_ear(4), forehead(mean 42,47)
    nose_tip = kp[:, 53, :2]
    mouth_ctr = kp[:, 71:83, :2].mean(axis=1)
    mouth_open = np.linalg.norm(kp[:, 23 + 66, :2] - kp[:, 23 + 62, :2], axis=-1, keepdims=True) / sh_span
    forehead = 0.5 * (kp[:, 23 + 19, :2] + kp[:, 23 + 24, :2])
    head_anchors = [mouth_ctr, nose_tip, kp[:, 1, :2], kp[:, 2, :2], kp[:, 3, :2], kp[:, 4, :2], forehead, l_sh, r_sh]
    feats.append(mouth_open)

    # Process Left Hand (91..111) and Right Hand (112..132)
    hand_slices = [kp[:, 91:112], kp[:, 112:133]]
    for h_kp in hand_slices:
        h_xy = h_kp[:, :, :2]  # (T, 21, 2)
        h_sc = h_kp[:, :, 2]   # (T, 21)
        wrist = h_xy[:, 0:1, :]
        p_span1 = np.linalg.norm(h_xy[:, 9] - h_xy[:, 0], axis=-1, keepdims=True)
        p_span2 = np.linalg.norm(h_xy[:, 5] - h_xy[:, 17], axis=-1, keepdims=True)
        p_scale = np.maximum(np.maximum(p_span1, p_span2), 0.035)  # (T, 1)

        # 1) Wrist-centered palm-normalized 21 finger joints + scores (63d)
        norm_xy = np.clip((h_xy - wrist) / p_scale[:, :, None], -4.0, 4.0)
        feats.append(norm_xy.reshape(T, -1))
        feats.append(h_sc)

        # 2) 20 anatomical finger bone vectors (40d)
        bones = np.stack([norm_xy[:, c] - norm_xy[:, p] for c, p in HAND_BONES], axis=1)
        feats.append(bones.reshape(T, -1))

        # 3) 10 pairwise fingertip pinch/spread distances + 5 tip-to-wrist curl ratios (15d)
        pinch = np.stack([np.linalg.norm(norm_xy[:, a] - norm_xy[:, b], axis=-1) for a, b in TIP_PAIRS], axis=-1)
        curl = np.stack([np.linalg.norm(norm_xy[:, tip], axis=-1) for tip in FINGERTIPS], axis=-1)
        feats.append(np.clip(pinch, 0.0, 5.0))
        feats.append(np.clip(curl, 0.0, 5.0))

        # 4) Hand-to-craniofacial/shoulder relational vectors & distances for (wrist, thumb_tip=4, index_tip=8)
        for j_idx in (0, 4, 8):
            pt = h_xy[:, j_idx, :]  # (T, 2)
            for anc in head_anchors:
                dv = np.clip((pt - anc) / sh_span, -4.0, 4.0)
                dist = np.clip(np.linalg.norm(dv, axis=-1, keepdims=True), 0.0, 6.0)
                feats.append(dv)
                feats.append(dist)

    # Bimanual Left-to-Right finger/palm interaction vectors & distances (normalized by sh_span)
    lh_xy, rh_xy = hand_slices[0][:, :, :2], hand_slices[1][:, :, :2]
    for j_idx in (0, 4, 8, 12, 9):
        dv = np.clip((lh_xy[:, j_idx] - rh_xy[:, j_idx]) / sh_span, -4.0, 4.0)
        dist = np.clip(np.linalg.norm(dv, axis=-1, keepdims=True), 0.0, 6.0)
        feats.append(dv)
        feats.append(dist)

    static_f = np.concatenate(feats, axis=-1)  # (T, D)
    vel_f = np.zeros_like(static_f)
    vel_f[1:] = static_f[1:] - static_f[:-1]
    return np.concatenate([static_f, vel_f], axis=-1).astype(np.float32)


def norm_entropy(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-7, 1.0)
    return -(pc * np.log(pc)).sum(axis=-1, keepdims=True) / math.log(126.0)


def temp_sharpen(p: np.ndarray, inv_temp: float) -> np.ndarray:
    lp = np.log(np.clip(p, 1e-7, 1.0)) * inv_temp
    lp -= lp.max(axis=-1, keepdims=True)
    ep = np.exp(lp)
    return ep / ep.sum(axis=-1, keepdims=True)


_IMAGE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)[None, :, None, None]
_IMAGE_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)[None, :, None, None]


def upper_body_affine(poses, width, height):
    """Map the robust upper-body extent to the DWPose input canvas."""
    center_x, center_y = .5 * width, .45 * height
    box_height = max(.75 * height, 140.)
    upper = poses[:, :13]
    visible = upper[..., 2] > .20
    if visible.sum() >= 4:
        x, y = upper[..., 0][visible], upper[..., 1][visible]
        x5, x95 = np.percentile(x, [5, 95])
        y2, y98 = np.percentile(y, [2, 98])
        x4, x96 = np.percentile(x, [4, 96])
        center_x, center_y = (x5+x95)/2, (y2+y98)/2
        box_height = max(y98-y2, (x96-x4)*4/3, 120.) * 1.38
    scale_x, scale_y = 192 / (.75 * box_height), 256 / box_height
    return np.array([[scale_x, 0., 96-center_x*scale_x],
                     [0., scale_y, 128-center_y*scale_y]], dtype=np.float32)


def decode_simcc(x_scores, y_scores):
    x = (x_scores.argmax(axis=-1).astype(np.float32) / 2 - 96) / 96
    y = (y_scores.argmax(axis=-1).astype(np.float32) / 2 - 128) / 128
    confidence = np.clip(np.minimum(x_scores.max(axis=-1), y_scores.max(axis=-1)), 0, 2)
    # The HandFace head was trained from an FP16 keypoint cache.
    return np.stack([x, y, confidence], axis=-1).astype(np.float16).astype(np.float32)


class HandFaceActionBackend:
    FRAME_COUNT = 8
    LOW_OBSERVATION_FPS = 10.

    def __init__(self, wholebody_engine, handface_engine, cluster_ids, parameters):
        self.wholebody_engine = wholebody_engine
        self.handface_engine = handface_engine
        shape = getattr(wholebody_engine, 'optimization_shape', None) or getattr(
            wholebody_engine, 'input_shape', None)
        batch = shape[0] if shape else None
        self.wholebody_frame_count = (int(batch) if isinstance(batch, (int, np.integer))
                                      and batch > 0 else self.FRAME_COUNT)
        if self.wholebody_frame_count not in (4, self.FRAME_COUNT):
            raise ValueError('Whole-body model needs four or eight observations per clip')
        groups = np.asarray(cluster_ids, dtype=np.int64)
        self.parameters = np.asarray(parameters, dtype=np.float64)
        if groups.shape != (126,) or not np.isin(groups, [0, 1, 2, 3]).all():
            raise ValueError('HandFace gate needs a cluster identifier for each action')
        if self.parameters.shape != (17,) or not np.isfinite(self.parameters).all():
            raise ValueError('HandFace gate needs 17 finite parameters')
        self.cluster_ids = groups

    @classmethod
    def from_directory(cls, root, options):
        from utils.tensorrt_runtime import TensorRTEngine

        gate = json.loads((root / options['gate']).read_text())
        wholebody = TensorRTEngine(root / options['wholebody_engine'], shared_scratch=True)
        handface = None
        try:
            handface = TensorRTEngine(root / options['action_engine'], shared_scratch=True)
            return cls(wholebody, handface, gate['cluster_ids'], gate['theta_3way_8f'])
        except Exception:
            wholebody.close()
            if handface is not None:
                handface.close()
            raise

    def should_run(self, dual_probabilities, observation_fps):
        if observation_fps > self.LOW_OBSERVATION_FPS:
            return False
        top_two = np.argpartition(dual_probabilities, -2)[-2:]
        return bool((self.cluster_ids[top_two] == 1).any())

    def close(self):
        self.wholebody_engine.close()
        self.handface_engine.close()

    def predict(self, observed_poses, frames, reconstruction=None):
        import cv2

        count = self.wholebody_frame_count
        if len(frames) != count:
            raise ValueError('Whole-body input frame count is incompatible with the engine')
        image_height, image_width = frames[0].shape[:2]
        matrix = upper_body_affine(np.asarray(observed_poses), image_width, image_height)
        crops = np.stack([
            cv2.warpAffine(rgb, matrix, (192, 256), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE).transpose(2, 0, 1)
            for rgb in frames
        ]).astype(np.float32)
        outputs = list(self.wholebody_engine.infer((crops-_IMAGE_MEAN)/_IMAGE_STD))
        if len(outputs) != 2:
            raise ValueError('Whole-body model must emit both SimCC axes')
        x_scores, y_scores = sorted(outputs, key=lambda v: v.shape[-1])
        if x_scores.shape != (count, 133, 384) or y_scores.shape != (count, 133, 512):
            raise ValueError('Unexpected whole-body SimCC shapes')
        keypoints = decode_simcc(x_scores, y_scores)
        if reconstruction is not None:
            keypoints = keypoints[reconstruction]
        if len(keypoints) != self.FRAME_COUNT:
            raise ValueError('HandFace classifier needs eight feature observations')
        hand_quality = float(keypoints[:, 91:133, 2].mean())
        features = build_handface_geom_features(keypoints)[None]
        raw = np.asarray(self.handface_engine.infer(features)[0], dtype=np.float32).reshape(-1)
        if raw.shape != (126,) or not np.isfinite(raw).all():
            raise ValueError('HandFace model must emit 126 finite scores')
        return (raw if looks_like_probabilities(raw) else softmax(raw)), hand_quality

    def fuse(self, body, rgb, handface, hand_quality, observation_fps):
        params = self.parameters
        temperatures = np.clip(params[:3], .5, 2.2)
        streams = np.stack([temp_sharpen(p, temperature) for p, temperature in
                            zip((body, rgb, handface), temperatures)])
        sharpness = 1. - norm_entropy(streams)
        low_fps = float(observation_fps <= self.LOW_OBSERVATION_FPS)
        body_logit = np.full(126, params[11] * sharpness[0, 0], dtype=np.float32)
        rgb_logit = params[3:7][self.cluster_ids] + params[12] * sharpness[1, 0] + params[14] * low_fps
        hand_logit = (params[7:11][self.cluster_ids] + params[13] * sharpness[2, 0]
                      + params[15] * low_fps + params[16] * (hand_quality - .75))
        weights = np.stack([body_logit, rgb_logit, hand_logit])
        weights -= weights.max(axis=0, keepdims=True)
        weights = np.exp(weights)
        result = (weights * streams).sum(axis=0) / weights.sum(axis=0)
        return result / result.sum()
