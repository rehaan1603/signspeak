# 🤟 SignSpeak — Real-Time ASL Alphabet Recognizer

Recognizes static American Sign Language letters from a live webcam feed. A Python backend uses **MediaPipe** to extract 21 hand landmarks per frame, converts them into a translation/scale-invariant geometric feature vector, and classifies the pose with a **scikit-learn** model — served over a **FastAPI** REST endpoint and consumed by a browser frontend that streams webcam frames and builds up words letter by letter.

```
Webcam frame → MediaPipe hand landmarks → geometric feature vector (68-d)
            → ML classifier (Random Forest / SVM) → predicted letter
            → FastAPI /predict endpoint → browser UI (live overlay + word buffer)
```

## Why this approach

Classifying raw pixels for hand signs needs a CNN, a large labeled image dataset, and a GPU to train reasonably. Extracting hand landmarks first (MediaPipe does this in a few ms on CPU) collapses the problem to a small, well-structured feature vector — 21 landmarks × (x, y, z) normalized against the wrist and hand scale, plus 5 per-finger flexion angles. A lightweight classical classifier hits high accuracy on this representation with orders of magnitude less data and no GPU, and inference is fast enough to run live in a browser loop.

## Project structure

```
signspeak/
├── src/
│   ├── landmarks.py            # MediaPipe extraction + feature engineering (shared by training & serving)
│   ├── hand_model.py           # Forward-kinematics hand model (synthetic data only)
│   ├── signs.py                # Letter → hand-pose definitions
│   ├── generate_synthetic_data.py  # Bootstraps a training set without a camera
│   ├── data_collection.py      # Records REAL labeled samples from your webcam
│   └── train.py                # Trains + evaluates the classifier, saves the model
├── backend/
│   └── app.py                  # FastAPI service: /predict, /health, serves the frontend
├── frontend/
│   └── index.html              # Webcam capture UI, live predictions, word buffer
├── tests/
│   └── test_pipeline.py        # Feature-engineering sanity tests (invariance, separability)
├── data/                       # Landmark CSVs (synthetic and/or real)
├── models/                     # Trained model artifacts (.pkl)
└── requirements.txt
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # or your preferred env manager
pip install -r requirements.txt

# 1. Get a training set. Fastest path — synthetic bootstrap (no camera needed):
python src/generate_synthetic_data.py --samples-per-class 300

# 2. Train the classifier:
python src/train.py --data data/synthetic_landmarks.csv

# 3. Run the app (serves the API + the web UI on one port):
uvicorn backend.app:app --reload --port 8000
```

Open `http://localhost:8000`, click **Start camera**, and show a hand sign. Hold a pose steady for about a second and the letter gets appended to the word buffer below the video.

## Using real data instead of (or alongside) synthetic data

The synthetic generator exists so the whole pipeline can be built, tested, and demoed without a camera — it's a stand-in, not the real signal. For a model that actually works well on your hand, collect real samples:

```bash
python src/data_collection.py --label A --samples 200
# repeat for each letter you want to support
python src/train.py --data data/landmarks.csv
```

Both scripts write to the same CSV schema (`f0..f67,label`), so you can freely mix synthetic and real rows, or drop the synthetic file once you have enough real coverage. `train.py --data data/*.csv` (the default) picks up everything in `data/`.

## Currently supported letters

The demo ships with 8 statically-distinguishable ASL letters chosen because they differ cleanly by *which fingers are extended*: **A, B, C, D, I, L, V, Y**. Extending to the full alphabet mostly means collecting real samples for the remaining letters (a few, like J and Z, involve motion and would need a sequence/gesture model rather than single-frame classification — a natural "next step" extension).

## Model performance (synthetic demo set)

Random Forest, 300 samples/class, 80/20 train/test split: **100% test accuracy**, 0 confusion between classes. This reflects how cleanly separated the 8 demo poses are in feature space — real-world accuracy with webcam data will depend on lighting, camera angle, and how consistently you hold each sign, and is normally in the 90s% with a few hundred real samples per letter.

## Tech stack

Python · MediaPipe (hand landmark detection) · scikit-learn (Random Forest / SVM) · FastAPI · vanilla JS (webcam capture, `getUserMedia` + `canvas`) · pandas / numpy for the data pipeline.

## Possible extensions

- Full 26-letter alphabet with real collected data, plus J/Z via a short temporal window (buffer of landmark sequences + a simple RNN or DTW matcher) instead of single-frame classification.
- Two-hand support for signs that use both hands.
- Confidence-based auto-correction / a dictionary layer to turn letter sequences into likely words.
- Swap the browser polling loop for a WebSocket stream for lower latency.
- Package the backend + a pinned frontend build into a Docker image for one-command deploy.

## Resume bullet (if useful)

> Built a real-time American Sign Language alphabet recognizer: MediaPipe hand-landmark extraction feeding a geometry-based feature pipeline into a scikit-learn classifier, served via a FastAPI backend with a browser webcam client — includes a synthetic data generator and automated tests so the ML pipeline is fully testable without hardware.
