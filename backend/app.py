"""
SignSpeak backend: FastAPI service that takes a webcam frame, runs
MediaPipe hand-landmark extraction + the trained classifier, and
returns the predicted ASL letter.

Run:
    uvicorn backend.app:app --reload --port 8000

Then open http://localhost:8000/ -- this also serves frontend/ directly
so there's a single process to run for the whole demo.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # let us import src/landmarks.py etc.

from landmarks import frame_to_feature_vector  # noqa: E402

MODEL_PATH = ROOT / "models" / "sign_model.pkl"
FRONTEND_DIR = ROOT / "frontend"

app = FastAPI(title="SignSpeak API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_bundle = None  # loaded lazily so the app can still start without a model


def get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "No trained model found. Run "
                    "`python src/generate_synthetic_data.py` then "
                    "`python src/train.py` first."
                ),
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class PredictRequest(BaseModel):
    image: str  # data URL, e.g. "data:image/jpeg;base64,..."


class PredictResponse(BaseModel):
    hand_detected: bool
    letter: str | None = None
    confidence: float | None = None
    top_k: list[dict] | None = None


def decode_data_url(data_url: str) -> np.ndarray:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return frame


@app.get("/health")
def health():
    model_ready = MODEL_PATH.exists()
    return {"status": "ok", "model_ready": model_ready}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    bundle = get_bundle()
    frame = decode_data_url(req.image)

    features = frame_to_feature_vector(frame)
    if features is None:
        return PredictResponse(hand_detected=False)

    model = bundle["model"]
    classes = bundle["classes"]
    features = features.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        order = np.argsort(probs)[::-1]
        top_k = [{"letter": classes[i], "confidence": float(probs[i])} for i in order[:3]]
        best = top_k[0]
        return PredictResponse(
            hand_detected=True,
            letter=best["letter"],
            confidence=best["confidence"],
            top_k=top_k,
        )

    pred = model.predict(features)[0]
    return PredictResponse(hand_detected=True, letter=str(pred), confidence=None)


# Serve the simple HTML/JS frontend at "/" so `uvicorn backend.app:app`
# is the only thing you need to run for the full demo.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
