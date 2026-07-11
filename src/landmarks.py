"""
Hand landmark extraction and feature engineering for SignSpeak.

Pipeline:
    frame (BGR image) --MediaPipe Hands--> 21 (x, y, z) landmarks
        --normalize (translation + scale invariant)--> 63-d vector
        --+ 5 finger-flexion angles--> 68-d feature vector fed to the classifier

Keeping this in one module means both the offline training pipeline
(train.py / generate_synthetic_data.py) and the live FastAPI backend
(backend/app.py) use the exact same feature representation, which is
what actually matters for a model that generalizes.
"""
from __future__ import annotations

import threading

import numpy as np

# mediapipe is only needed when running against real images/webcam frames.
# The synthetic-data + training pipeline can run without a camera, so we
# import it lazily to keep that path lightweight and import-error-free.
_mp_hands = None

# MediaPipe's Hands graph is stateful and NOT thread-safe: it expects
# input timestamps to strictly increase call-over-call. FastAPI runs sync
# endpoints in a threadpool, so two /predict requests can call
# hands.process() concurrently and race -- MediaPipe then raises
# "Packet timestamp mismatch" and the request 500s. Serializing access
# with a lock fixes it; a single MediaPipe Hands call is a few ms, so
# this isn't a real throughput bottleneck for a webcam-polling demo.
_hands_lock = threading.Lock()


def _get_mp_hands():
    global _mp_hands
    if _mp_hands is None:
        import mediapipe as mp

        _mp_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
    return _mp_hands


LANDMARK_COUNT = 21
WRIST = 0
MIDDLE_MCP = 9

# Each finger as a chain of landmark indices used to measure flexion angle
# (MCP -> PIP -> TIP). Thumb uses CMC->MCP->TIP since it only has 4 joints.
FINGER_CHAINS = {
    "thumb": (1, 2, 4),
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}

FEATURE_DIM = LANDMARK_COUNT * 3 + len(FINGER_CHAINS)  # 63 + 5 = 68


def extract_raw_landmarks(frame_bgr: np.ndarray) -> np.ndarray | None:
    """Run MediaPipe Hands on a BGR frame and return a (21, 3) array of
    (x, y, z) landmark coordinates in normalized image space, or None if
    no hand was detected."""
    import cv2

    hands = _get_mp_hands()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    with _hands_lock:
        result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None

    hand = result.multi_hand_landmarks[0]
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
    return pts


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b formed by points a-b-c, in radians."""
    v1 = a - b
    v2 = c - b
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    cos_angle = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def normalize_landmarks(points: np.ndarray) -> np.ndarray:
    """Make landmarks translation- and scale-invariant.

    Translation invariance: subtract the wrist position so the wrist is
    the origin.
    Scale invariance: divide by the wrist->middle_mcp distance, which is
    roughly constant across a given hand pose regardless of how close the
    hand is to the camera.
    """
    origin = points[WRIST]
    shifted = points - origin
    scale = np.linalg.norm(shifted[MIDDLE_MCP]) + 1e-6
    return shifted / scale


def finger_angles(points: np.ndarray) -> np.ndarray:
    """Return one flexion angle per finger (5 values), which encodes how
    curled/extended each finger is -- a strong, pose-invariant signal for
    static sign classification."""
    angles = []
    for _, (i, j, k) in FINGER_CHAINS.items():
        angles.append(_angle(points[i], points[j], points[k]))
    return np.array(angles, dtype=np.float32)


def landmarks_to_feature_vector(points: np.ndarray) -> np.ndarray:
    """Full pipeline: raw (21, 3) landmarks -> flat 68-d feature vector."""
    norm = normalize_landmarks(points)
    angles = finger_angles(points)
    return np.concatenate([norm.flatten(), angles]).astype(np.float32)


def frame_to_feature_vector(frame_bgr: np.ndarray) -> np.ndarray | None:
    """Convenience wrapper: image -> feature vector, or None if no hand."""
    points = extract_raw_landmarks(frame_bgr)
    if points is None:
        return None
    return landmarks_to_feature_vector(points)
