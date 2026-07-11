"""
Trajectory-based detection for the two ASL letters that are motions, not
static handshapes: J (pinky traces a hook) and Z (index traces a zigzag).

The static classifier (src/train.py + models/sign_model.pkl) only ever
sees one frame at a time, so it structurally cannot represent a letter
that's defined by *movement*. This module is the fix: the backend keeps
a short rolling trail of a fingertip's recent (x, y) positions (in
wrist-relative, scale-normalized coordinates -- see
landmarks.normalize_landmarks -- so it isn't sensitive to how close the
hand is to the camera) and these pure functions classify the shape of
that trail.

Both detectors are deliberately simple and hand-tunable rather than
learned, because there's no realistic way to synthesize training data
for hand *motion* the way hand_model.py does for static poses -- motion
timing and amplitude vary too much to fake convincingly. They're unit
tested against synthetic trajectories in tests/test_motion_gestures.py.
"""
from __future__ import annotations

import math

Point = tuple[float, float]


def _velocities(trail: list[Point]) -> list[Point]:
    return [(trail[i + 1][0] - trail[i][0], trail[i + 1][1] - trail[i][1])
            for i in range(len(trail) - 1)]


def _path_length(trail: list[Point]) -> float:
    return sum(math.hypot(dx, dy) for dx, dy in _velocities(trail))


def _horizontal_direction_reversals(trail: list[Point], deadzone: float = 0.004) -> int:
    """Count how many times the horizontal direction of travel flips
    sign, ignoring tiny jitter below `deadzone`. A zigzag (Z) has
    several; a smooth hook (J) has at most one."""
    signs = []
    for dx, _ in _velocities(trail):
        if dx > deadzone:
            signs.append(1)
        elif dx < -deadzone:
            signs.append(-1)
    return sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])


def detect_z_motion(trail: list[Point], min_points: int = 6,
                     min_reversals: int = 2, min_path_length: float = 0.35) -> bool:
    """Z is drawn as three strokes (right, diagonal down-left, right
    again), so the horizontal direction reverses at least twice over a
    reasonably long path. Path length is in the same normalized units
    as landmarks.normalize_landmarks (roughly hand-widths)."""
    if len(trail) < min_points:
        return False
    if _path_length(trail) < min_path_length:
        return False
    return _horizontal_direction_reversals(trail) >= min_reversals


def detect_j_motion(trail: list[Point], min_points: int = 6,
                     min_turn_deg: float = 40.0, min_path_length: float = 0.25,
                     max_reversals: int = 1) -> bool:
    """J is a single smooth hook: the direction of travel turns by a
    good chunk of a right angle over the trail, without the multiple
    back-and-forth reversals that would make it a Z instead."""
    if len(trail) < min_points:
        return False
    if _path_length(trail) < min_path_length:
        return False
    if _horizontal_direction_reversals(trail) > max_reversals:
        return False

    window = max(2, len(trail) // 3)
    start_vec = (trail[window][0] - trail[0][0], trail[window][1] - trail[0][1])
    end_vec = (trail[-1][0] - trail[-1 - window][0], trail[-1][1] - trail[-1 - window][1])
    if math.hypot(*start_vec) < 1e-6 or math.hypot(*end_vec) < 1e-6:
        return False

    start_angle = math.degrees(math.atan2(start_vec[1], start_vec[0]))
    end_angle = math.degrees(math.atan2(end_vec[1], end_vec[0]))
    turn = abs(((end_angle - start_angle + 180) % 360) - 180)
    return turn >= min_turn_deg


class FingerTrail:
    """Fixed-size rolling buffer of recent (x, y) positions for one
    fingertip, with staleness expiry so a paused/restarted gesture
    doesn't get glued onto an old one."""

    def __init__(self, maxlen: int = 20, stale_after_s: float = 0.8):
        self.maxlen = maxlen
        self.stale_after_s = stale_after_s
        self._points: list[Point] = []
        self._last_t: float | None = None

    def push(self, point: Point, t: float) -> None:
        if self._last_t is not None and (t - self._last_t) > self.stale_after_s:
            self._points.clear()
        self._points.append(point)
        if len(self._points) > self.maxlen:
            self._points.pop(0)
        self._last_t = t

    def clear(self) -> None:
        self._points.clear()
        self._last_t = None

    @property
    def points(self) -> list[Point]:
        return list(self._points)
