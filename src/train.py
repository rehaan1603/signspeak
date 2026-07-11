"""
Train the sign classifier on collected landmark feature vectors.

Usage:
    python src/train.py --data data/synthetic_landmarks.csv
    python src/train.py --data data/landmarks.csv --model svm

Loads one or more CSVs (any file matching the schema written by
generate_synthetic_data.py / data_collection.py: f0..fN,label), trains
a classifier, prints accuracy + a full classification report on a held
out test split, and saves the fitted model + label classes to
models/sign_model.pkl.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def load_dataset(patterns: list[str]) -> pd.DataFrame:
    frames = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(f"No data files matched: {patterns}")
    return pd.concat(frames, ignore_index=True)


def build_model(kind: str):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        )
    if kind == "svm":
        return SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    raise ValueError(f"Unknown model kind: {kind}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+",
                         default=[str(ROOT / "data" / "*.csv")],
                         help="CSV path(s) or glob pattern(s)")
    parser.add_argument("--model", choices=["rf", "svm"], default="rf")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--out", type=str, default=str(MODELS_DIR / "sign_model.pkl"))
    args = parser.parse_args()

    df = load_dataset(args.data)
    print(f"Loaded {len(df)} samples across {df['label'].nunique()} classes: "
          f"{sorted(df['label'].unique())}")

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    clf = build_model(args.model)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix (rows=true, cols=pred):")
    labels_sorted = sorted(df["label"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    print(pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": clf, "classes": labels_sorted, "model_kind": args.model,
         "feature_dim": X.shape[1]},
        args.out,
    )
    print(f"\nSaved model -> {args.out}")


if __name__ == "__main__":
    main()
