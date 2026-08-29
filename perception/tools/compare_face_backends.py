#!/usr/bin/env python3
"""Compare FP32 ONNX and TensorRT face outputs on one real image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from plugins.face_id.alignment import align_face  # noqa: E402
from plugins.face_id.backends import OnnxRuntimeBackend, TensorRTBackend  # noqa: E402
from plugins.face_id.detector import SCRFDDetector  # noqa: E402
from plugins.face_id.gallery import select_primary_face  # noqa: E402
from plugins.face_id.recognizer import FaceRecognizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-onnx")
    parser.add_argument("--detector-engine", required=True)
    parser.add_argument("--recognizer-onnx")
    parser.add_argument("--recognizer-engine", required=True)
    parser.add_argument("--recognizer-type", default="lvface")
    parser.add_argument("--image", required=True)
    parser.add_argument("--export-trt", type=Path)
    parser.add_argument("--embedding-cosine-min", type=float, default=0.999)
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode image: {args.image}")
    detector_trt = SCRFDDetector(
        TensorRTBackend(args.detector_engine),
        input_size=(640, 640),
        score_threshold=0.2,
    )
    recognizer_trt = FaceRecognizer(
        TensorRTBackend(args.recognizer_engine), model_type=args.recognizer_type
    )
    detector_onnx = None
    recognizer_onnx = None
    try:
        trt_face = select_primary_face(detector_trt.detect(image), image.shape)
        if trt_face is None:
            raise RuntimeError("TensorRT did not detect a face")
        trt_aligned = align_face(image, trt_face.landmarks)
        trt_embedding = recognizer_trt.embed(trt_aligned)
        if args.export_trt is not None:
            destination = args.export_trt.resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                destination,
                bbox=trt_face.bbox,
                landmarks=trt_face.landmarks,
                detection_score=np.asarray([trt_face.score], dtype=np.float32),
                embedding=trt_embedding,
            )
            print(json.dumps({"trt_output": str(destination)}, ensure_ascii=False))
            return 0
        if not args.detector_onnx or not args.recognizer_onnx:
            parser.error(
                "--detector-onnx and --recognizer-onnx are required unless --export-trt is used"
            )
        detector_onnx = SCRFDDetector(
            OnnxRuntimeBackend(args.detector_onnx),
            input_size=(640, 640),
            score_threshold=0.2,
        )
        recognizer_onnx = FaceRecognizer(
            OnnxRuntimeBackend(args.recognizer_onnx), model_type=args.recognizer_type
        )
        onnx_face = select_primary_face(detector_onnx.detect(image), image.shape)
        if onnx_face is None:
            raise RuntimeError("ONNX Runtime did not detect a face")
        onnx_aligned = align_face(image, onnx_face.landmarks)
        onnx_embedding = recognizer_onnx.embed(onnx_aligned)
        cosine = float(np.dot(onnx_embedding, trt_embedding))
        result = {
            "bbox_max_abs_diff_px": float(
                np.max(np.abs(onnx_face.bbox - trt_face.bbox))
            ),
            "landmark_max_abs_diff_px": float(
                np.max(np.abs(onnx_face.landmarks - trt_face.landmarks))
            ),
            "detection_score_abs_diff": abs(onnx_face.score - trt_face.score),
            "embedding_cosine": cosine,
            "embedding_max_abs_diff": float(
                np.max(np.abs(onnx_embedding - trt_embedding))
            ),
        }
        print(json.dumps(result, ensure_ascii=False))
        return 0 if cosine >= args.embedding_cosine_min else 2
    finally:
        recognizer_trt.close()
        if recognizer_onnx is not None:
            recognizer_onnx.close()
        detector_trt.close()
        if detector_onnx is not None:
            detector_onnx.close()


if __name__ == "__main__":
    raise SystemExit(main())
