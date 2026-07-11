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

# (base_dir_degrees_from_up, mcp_distance, seg_lengths[3]) per finger,
# roughly matching typical proportions of a hand facing the camera.
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
    # "up" in image space is -y; 0 degrees points straight up.
    return np.array([np.sin(rad), -np.cos(rad), 0.0], dtype=np.float32)


def _rotate_toward_wrist(direction: np.ndarray, bend_rad: float) -> np.ndarray:
    """Rotate a 2D direction vector by bend_rad, curling it back down
    (toward +y / the wrist) to simulate a finger folding into the palm."""
    c, s = np.cos(bend_rad), np.sin(bend_rad)
    x, y = direction[0], direction[1]
    new_x = x * c - y * s
    new_y = x * s + y * c
    return np.array([new_x, new_y, direction[2]], dtype=np.float32)


def build_hand(pose: dict[str, float], noise: float = 0.0,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Build a (21, 3) landmark array from a pose spec.

    pose: dict mapping finger name -> curl in [0, 1] (0 = fully
    extended/straight, 1 = fully curled into the palm). Missing fingers
    default to 0 (extended).
    noise: stddev of Gaussian jitter added to every coordinate, to
    simulate camera/tracking noise for augmentation.
    """
    rng = rng or np.random.default_rng()
    points = np.zeros((21, 3), dtype=np.float32)
    points[0] = WRIST

    for finger, (base_deg, mcp_dist, seg_lengths) in _FINGER_GEOMETRY.items():
        curl = float(pose.get(finger, 0.0))
        max_bend = np.radians(80.0)  # per-joint bend at full curl
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

    if noise > 0:
        points += rng.normal(0.0, noise, size=points.shape).astype(np.float32)

    return points
