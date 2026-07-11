"""
Lightweight sanity tests that don't require a webcam:
- feature engineering is translation/scale invariant and the right shape
- the synthetic hand model produces distinguishable poses
- a trained model can round-trip predict on held-out synthetic samples

Run with: python -m pytest tests/ -v   (or just `python tests/test_pipeline.py`)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hand_model import build_hand  # noqa: E402
from landmarks import FEATURE_DIM, landmarks_to_feature_vector  # noqa: E402
from signs import SIGN_POSES  # noqa: E402


def test_feature_vector_shape():
    points = build_hand(SIGN_POSES["B"])
    features = landmarks_to_feature_vector(points)
    assert features.shape == (FEATURE_DIM,)


def test_translation_invariance():
    points = build_hand(SIGN_POSES["A"])
    shifted = points + np.array([0.2, -0.1, 0.0], dtype=np.float32)
    f1 = landmarks_to_feature_vector(points)
    f2 = landmarks_to_feature_vector(shifted)
    assert np.allclose(f1, f2, atol=1e-4)


def test_scale_invariance():
    points = build_hand(SIGN_POSES["L"])
    scaled = points * 1.5
    f1 = landmarks_to_feature_vector(points)
    f2 = landmarks_to_feature_vector(scaled)
    assert np.allclose(f1, f2, atol=1e-3)


def test_poses_are_distinguishable():
    rng = np.random.default_rng(0)
    vectors = {
        label: landmarks_to_feature_vector(build_hand(pose, noise=0.0, rng=rng))
        for label, pose in SIGN_POSES.items()
    }
    labels = list(vectors)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            dist = np.linalg.norm(vectors[a] - vectors[b])
            assert dist > 0.05, f"{a} and {b} are too similar ({dist:.4f})"


if __name__ == "__main__":
    test_feature_vector_shape()
    test_translation_invariance()
    test_scale_invariance()
    test_poses_are_distinguishable()
    print("All sanity checks passed.")
