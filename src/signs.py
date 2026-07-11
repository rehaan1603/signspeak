"""ASL letter definitions used by the demo/synthetic pipeline.

Two kinds of letters here:

STATIC_SIGN_POSES -- 24 letters (all except J and Z) that are a single
held handshape, classified frame-by-frame from a curl spec: 0.0 = finger
fully extended/straight, 1.0 = finger fully curled into the fist.
Two optional keys refine geometry beyond curl alone (see hand_model.py):
"rotation" (whole-hand orientation, e.g. G points sideways, P points
down) and "spread" (separates adjacent extended fingers, e.g. V's spread
index/middle vs U's together index/middle).

A handful of letter pairs are only approximately distinguishable with
this simplified curl+rotation+spread model, because real ASL also uses
detail this model doesn't capture -- finger crossing (R), and exactly
which fingers the thumb tucks between (M vs N vs T). Those are flagged
below; see README for what that means for accuracy.

MOTION_LETTERS -- J and Z are not static poses, they're traced motions
(J: a hook drawn by the pinky tip; Z: a zigzag drawn by the index tip).
A single-frame classifier can't represent motion, so these are detected
separately by src/motion_gestures.py from a short trajectory of recent
fingertip positions, layered on top of the static classifier's output.
"""

STATIC_SIGN_POSES = {
    "A": {"thumb": 0.30, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
    "B": {"thumb": 1.00, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0},
    "C": {"thumb": 0.35, "index": 0.35, "middle": 0.35, "ring": 0.35, "pinky": 0.35},
    "D": {"thumb": 0.60, "index": 0.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
    "E": {"thumb": 0.85, "index": 0.85, "middle": 0.85, "ring": 0.85, "pinky": 0.85},
    "F": {"thumb": 0.50, "index": 0.6, "middle": 0.0, "ring": 0.0, "pinky": 0.0},
    "G": {"thumb": 0.20, "index": 0.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0, "rotation": 90},
    "H": {"thumb": 0.60, "index": 0.0, "middle": 0.0, "ring": 1.0, "pinky": 1.0, "rotation": 90},
    "I": {"thumb": 1.00, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 0.0},
    "K": {"thumb": 0.30, "index": 0.0, "middle": 0.0, "ring": 1.0, "pinky": 1.0, "spread": {"index": 12, "middle": -12}},
    "L": {"thumb": 0.00, "index": 0.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
    "M": {"thumb": 0.90, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
    "N": {"thumb": 0.85, "index": 1.0, "middle": 1.0, "ring": 0.85, "pinky": 1.0},
    "O": {"thumb": 0.70, "index": 0.70, "middle": 0.70, "ring": 0.70, "pinky": 0.70},
    "P": {"thumb": 0.30, "index": 0.0, "middle": 0.0, "ring": 1.0, "pinky": 1.0, "rotation": 160, "spread": {"index": 12, "middle": -12}},
    "Q": {"thumb": 0.20, "index": 0.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0, "rotation": 200},
    "R": {"thumb": 0.60, "index": 0.0, "middle": 0.0, "ring": 1.0, "pinky": 1.0},
    "S": {"thumb": 0.70, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
    "T": {"thumb": 0.90, "index": 1.0, "middle": 0.70, "ring": 1.0, "pinky": 1.0},
    "U": {"thumb": 1.00, "index": 0.0, "middle": 0.0, "ring": 1.0, "pinky": 1.0},
    "V": {"thumb": 1.00, "index": 0.0, "middle": 0.0, "ring": 1.0, "pinky": 1.0, "spread": {"index": 12, "middle": -12}},
    "W": {"thumb": 1.00, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 1.0, "spread": {"index": 14, "middle": 0, "ring": -14}},
    "X": {"thumb": 1.00, "index": 0.5, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
    "Y": {"thumb": 0.00, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 0.0},
}

APPROXIMATE_LETTER_GROUPS = [
    {"M", "N", "T", "E"},
    {"R", "U", "H"},
]

STATIC_LABELS = sorted(STATIC_SIGN_POSES.keys())

MOTION_LETTERS = {
    "J": {"base_pose": "I", "tip_landmark": 20},
    "Z": {"base_pose": "D", "tip_landmark": 8},
}

ALL_LABELS = sorted(list(STATIC_SIGN_POSES.keys()) + list(MOTION_LETTERS.keys()))

SIGN_POSES = STATIC_SIGN_POSES
LABELS = STATIC_LABELS
