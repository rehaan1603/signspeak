"""
Generate a synthetic hand-landmark dataset so the training pipeline can
be built, tested, and demoed without a webcam (e.g. in CI or a sandbox).

For real-world accuracy, replace/augment this with actual samples
collected via data_collection.py -- the CSV schema is identical, so
train.py doesn't care where the rows came from.

Usage:
    python src/generate_synthetic_data.py --samples-per-class 300
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from hand_model import build_hand
from landmarks import FEATURE_DIM, landmarks_to_feature_vector
from signs import STATIC_SIGN_POSES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def generate(samples_per_class: int, noise: float, seed: int) -> list[tuple[list[float], str]]:
    rng = np.random.default_rng(seed)
    rows = []
    for label, pose in STATIC_SIGN_POSES.items():
        for _ in range(samples_per_class):
            points = build_hand(pose, noise=noise, rng=rng)
            features = landmarks_to_feature_vector(points)
            rows.append((features.tolist(), label))
    rng.shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-class", type=int, default=300)
    parser.add_argument("--noise", type=float, default=0.012,
                         help="Gaussian jitter stddev applied to landmark coords")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "synthetic_landmarks.csv"))
    args = parser.parse_args()

    rows = generate(args.samples_per_class, args.noise, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [f"f{i}" for i in range(FEATURE_DIM)] + ["label"]
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for features, label in rows:
            writer.writerow([*features, label])

    print(f"Wrote {len(rows)} samples ({len(STATIC_SIGN_POSES)} classes) to {out_path}")


if __name__ == "__main__":
    main()
