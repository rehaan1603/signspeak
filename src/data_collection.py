"""
Collect REAL labeled hand-landmark samples from your webcam.

Run this locally (needs an actual camera -- it will not work in a
headless sandbox/CI environment). Recommended workflow:

    python src/data_collection.py --label A --samples 200

Controls:
    SPACE  - toggle capturing on/off (move your hand slightly between
             captures so samples aren't all identical)
    q      - quit

Appends rows to data/landmarks.csv (same schema generate_synthetic_data.py
uses), so you can mix real + synthetic data, or drop the synthetic file
entirely once you have enough real samples per letter.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

from landmarks import FEATURE_DIM, extract_raw_landmarks, landmarks_to_feature_vector

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILE = DATA_DIR / "landmarks.csv"


def ensure_header():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        with DATA_FILE.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"f{i}" for i in range(FEATURE_DIM)] + ["label"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Letter this session records, e.g. A")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.08,
                         help="Seconds between captures while capturing is on")
    args = parser.parse_args()

    ensure_header()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    capturing = False
    collected = 0
    last_capture = 0.0

    print(f"Collecting samples for label '{args.label}'. Press SPACE to start/stop, q to quit.")

    with DATA_FILE.open("a", newline="") as f:
        writer = csv.writer(f)
        while collected < args.samples:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            points = extract_raw_landmarks(frame)
            status = "NO HAND"
            if points is not None:
                status = f"HAND OK ({collected}/{args.samples})"
                now = time.time()
                if capturing and now - last_capture >= args.interval:
                    features = landmarks_to_feature_vector(points)
                    writer.writerow([*features.tolist(), args.label])
                    collected += 1
                    last_capture = now

            color = (0, 200, 0) if capturing else (0, 0, 200)
            cv2.putText(frame, f"Label: {args.label}  {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, "SPACE=start/stop  q=quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("SignSpeak - Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                capturing = not capturing
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Collected {collected} samples for '{args.label}' -> {DATA_FILE}")


if __name__ == "__main__":
    main()
