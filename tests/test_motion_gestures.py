"""
Unit tests for the J/Z motion detectors (src/motion_gestures.py).

These can't use real webcam recordings (no camera in CI/sandbox), so
instead they build synthetic fingertip trajectories with the shape each
letter is actually defined by -- a zigzag for Z, a smooth hook for J --
plus negative cases (a straight line, pure jitter, a too-short trail)
that should NOT trigger either one. This is exactly the kind of pure
geometric logic that's honestly testable without hardware, unlike the
webcam-dependent parts of the pipeline.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from motion_gestures import FingerTrail, detect_j_motion, detect_z_motion  # noqa: E402


def _line(n=10, dx=0.05, dy=0.0, start=(0.0, 0.0)):
    return [(start[0] + dx * i, start[1] + dy * i) for i in range(n)]


def _zigzag(strokes=3, points_per_stroke=4, step=0.03):
    """Right, down-left, right, ... -- the actual shape of a drawn Z."""
    trail = [(0.0, 0.0)]
    x, y = 0.0, 0.0
    direction = 1
    for s in range(strokes):
        for _ in range(points_per_stroke):
            x += step * direction
            y += step * 0.4
            trail.append((x, y))
        direction *= -1
    return trail


def _hook(n=12, radius=0.15):
    """A smooth ~100 degree curve, like the hook a pinky traces for J."""
    trail = []
    for i in range(n):
        theta = math.radians(100 * i / (n - 1))
        x = radius * math.sin(theta)
        y = radius * (1 - math.cos(theta))
        trail.append((x, y))
    return trail


def _jitter(n=10, amplitude=0.0015, seed=1):
    import random
    rng = random.Random(seed)
    return [(rng.uniform(-amplitude, amplitude), rng.uniform(-amplitude, amplitude))
            for _ in range(n)]


def test_zigzag_triggers_z_not_j():
    trail = _zigzag()
    assert detect_z_motion(trail) is True
    assert detect_j_motion(trail) is False


def test_hook_triggers_j_not_z():
    trail = _hook()
    assert detect_j_motion(trail) is True
    assert detect_z_motion(trail) is False


def test_straight_line_triggers_neither():
    trail = _line()
    assert detect_j_motion(trail) is False
    assert detect_z_motion(trail) is False


def test_jitter_triggers_neither():
    trail = _jitter()
    assert detect_j_motion(trail) is False
    assert detect_z_motion(trail) is False


def test_short_trail_triggers_neither():
    trail = _zigzag()[:3]
    assert detect_j_motion(trail) is False
    assert detect_z_motion(trail) is False


def test_finger_trail_rolls_and_expires():
    trail = FingerTrail(maxlen=5, stale_after_s=0.5)
    for i in range(10):
        trail.push((float(i), 0.0), t=float(i) * 0.01)
    assert len(trail.points) == 5
    assert trail.points[-1] == (9.0, 0.0)

    # A big time gap should be treated as a new gesture, not a
    # continuation of the old trail.
    trail.push((100.0, 0.0), t=100.0)
    assert trail.points == [(100.0, 0.0)]


if __name__ == "__main__":
    test_zigzag_triggers_z_not_j()
    test_hook_triggers_j_not_z()
    test_straight_line_triggers_neither()
    test_jitter_triggers_neither()
    test_short_trail_triggers_neither()
    test_finger_trail_rolls_and_expires()
    print("All motion gesture tests passed.")
