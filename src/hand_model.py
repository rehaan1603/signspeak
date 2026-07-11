"""
A tiny forward-kinematics hand model used ONLY to generate synthetic
training data (see generate_synthetic_data.py).

Why this exists: MediaPipe + a live webcam is the real data source, but
you can't collect webcam samples inside a headless sandbox / CI runner.
This module builds anatomically-plausible 21-point hand landmark arrays
(matching MediaPipe's landmark indexing) from a small "pose spec"
(how curled each finger is), so the rest of the pipeline -- feature
engineering, training, evaluation -- can be built and unit-tested
end-to-end without a camera, then simply pointed at real collected data
later via data_collection.py.
"""
from __future__ import annotations

import numpy as np

WRIST = np.array([0.5, 0.85, 0.0], dtype=np.float32)

_FINGER_GEOMETRY = {
    "thumb": (55.0, 0.10, (0.09, 0.07, 0.06)),
    "index": (15.0, 0.30, (0.11, 0.07, 0.06)),
    "middle": (2.0, 0.32, (0.12, 0.08, 0.06)),
    "ring": (-12.0, 0.30, (0.11, 0.07, 0.06)),
    "pinky": (-25.0, 0.27, (0.09, 0.06, 0.05)),
}

_LANDMARK_SLOTS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def _dir_from_angle(deg: float) -> np.ndarray:
    rad = np.radians(deg)
    return np.array([np.sin(rad), -np.cos(rad), 0.0], dtype=np.float32)


def _rotate_toward_wrist(direction: np.ndarray, bend_rad: float) -> np.ndarray:
    c, s = np.cos(bend_rad), np.sin(bend_rad)
    x, y = direction[0], direction[1]
    new_x = x * c - y * s
    new_y = x * s + y * c
    return np.array([new_x, new_y, direction[2]], dtype=np.float32)


def _rotate_xy_around(points: np.ndarray, center: np.ndarray, deg: float) -> np.ndarray:
    if deg == 0:
        return points
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    shifted = points - center
    x, y = shifted[:, 0].copy(), shifted[:, 1].copy()
    shifted[:, 0] = x * c - y * s
    shifted[:, 1] = x * s + y * c
    return shifted + center


def build_hand(pose: dict, noise: float = 0.0,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Build a (21, 3) landmark array from a pose spec.

    pose: dict with per-finger curl in [0, 1] under keys thumb/index/
    middle/ring/pinky (0 = extended, 1 = fully curled). Optional keys:
    "rotation" (degrees, whole-hand rotation around the wrist) and
    "spread" (dict of finger -> extra base-angle offset in degrees).
    """
    rng = rng or np.random.default_rng()
    rotation = float(pose.get("rotation", 0.0))
    spread = pose.get("spread", {})

    points = np.zeros((21, 3), dtype=np.float32)
    points[0] = WRIST

    for finger, (base_deg, mcp_dist, seg_lengths) in _FINGER_GEOMETRY.items():
        curl = float(pose.get(finger, 0.0))
        base_deg = base_deg + float(spread.get(finger, 0.0))
        max_bend = np.radians(80.0)
        bend = curl * max_bend

        direction = _dir_from_angle(base_deg)
        mcp = WRIST + direction * mcp_dist
        pip = mcp + direction * seg_lengths[0]

        direction2 = _rotate_toward_wrist(direction, bend)
        dip = pip + direction2 * seg_lengths[1]

        direction3 = _rotate_toward_wrist(direction2, bend * 0.7)
        tip = dip + direction3 * seg_lengths[2]

        slot = _LANDMARK_SLOTS[finger]
        for idx, pt in zip(slot, (mcp, pip, dip, tip)):
            points[idx] = pt

    if rotation:
        points = _rotate_xy_around(points, WRIST, rotation)

    if noise > 0:
        points += rng.normal(0.0, noise, size=points.shape).astype(np.float32)

    return points
