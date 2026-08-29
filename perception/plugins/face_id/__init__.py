"""On-device face detection and closed-set identity recognition."""

from .engine import FaceIdentityEngine, build_face_engine
from .schema import FaceDetection, IdentityMatch, empty_face_payload

__all__ = [
    "FaceDetection",
    "FaceIdentityEngine",
    "IdentityMatch",
    "build_face_engine",
    "empty_face_payload",
]
