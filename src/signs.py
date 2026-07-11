"""Static ASL letter definitions used by the demo/synthetic pipeline.

Each entry is a "curl spec": 0.0 = finger fully extended/straight,
1.0 = finger fully curled into the fist. These 8 letters were picked
because they're cleanly distinguishable by *which fingers are
extended*, which is exactly the signal the geometric feature vector
(src/landmarks.py) captures well -- a good fit for a lightweight
classical ML classifier (as opposed to full sentence-level ASL, which
involves motion and non-manual markers and needs a very different,
much larger system).

To extend this to more letters/gestures, add entries here (for the
synthetic bootstrap set) and/or collect real samples with
src/data_collection.py -- both write to the same CSV schema.
"""

SIGN_POSES: dict[str, dict[str, float]] = {
    "A": {"thumb": 0.15, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
    "B": {"thumb": 1.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0},
    "C": {"thumb": 0.45, "index": 0.45, "middle": 0.45, "ring": 0.45, "pinky": 0.45},
    "D": {"thumb": 0.6, "index": 0.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
    "I": {"thumb": 1.0, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 0.0},
    "L": {"thumb": 0.0, "index": 0.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
    "V": {"thumb": 1.0, "index": 0.0, "middle": 0.0, "ring": 1.0, "pinky": 1.0},
    "Y": {"thumb": 0.0, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 0.0},
}

LABELS = sorted(SIGN_POSES.keys())
