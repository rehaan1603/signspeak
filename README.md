# 🤟 SignSpeak — Real-Time ASL Alphabet Recognizer

**🔴 Live demo (recommended):** [signspeak-seven.vercel.app](https://signspeak-seven.vercel.app) — runs 100% client-side (hand tracking + classifier both in-browser via WASM, no server, no video ever leaves your machine). Always on, no cold start.

**Full backend demo:** [signspeak-9rfq.onrender.com](https://signspeak-9rfq.onrender.com) — the actual FastAPI + MediaPipe architecture described below, hosted on Render's free tier. Note: Render's free instance caps out at 512MB RAM, which is tight for MediaPipe's model-loading footprint under real traffic — if `/predict` calls return a 502, that's why. The Vercel demo above uses the same model and feature engineering (ported to JS and numerically verified to match), so it's the more reliable way to actually try it live.

Recognizes the full American Sign Language alphabet from a live webcam feed. A Python backend uses **MediaPipe** to extract 21 hand landmarks per frame, converts them into a translation/scale-invariant geometric feature vector, and classifies the pose with a **scikit-learn** model — served over a **FastAPI** REST endpoint and consumed by a browser frontend that streams webcam frames and builds up words letter by letter. The two motion letters (J, Z) are handled separately by a lightweight fingertip-trajectory detector layered on top of the static classifier.

```
Webcam frame → MediaPipe hand landmarks
            → geometric feature vector (68-d) → ML classifier (24 static letters)
            → fingertip trajectory buffer      → motion detector (J, Z)
            → FastAPI /predict endpoint → browser UI (live overlay + word buffer)
```

## Why this approach

Classifying raw pixels for hand signs needs a CNN, a large labeled image dataset, and a GPU to train reasonably. Extracting hand landmarks first (MediaPipe does this in a few ms on CPU) collapses the problem to a small, well-structured feature vector — 21 landmarks × (x, y, z) normalized against the wrist and hand scale, plus 5 per-finger flexion angles. A lightweight classical classifier hits high accuracy on this representation with orders of magnitude less data and no GPU, and inference is fast enough to run live in a browser loop.

J and Z can't be represented this way at all, because they're defined by *motion* (a traced hook and a traced zigzag), not a held handshape — a single-frame classifier is structurally the wrong tool for them. Rather than skip them or fake a static approximation, the backend keeps a short rolling trail of the relevant fingertip's recent position and a separate geometric detector (`src/motion_gestures.py`) recognizes the shape of that trail.

## Project structure

```
signspeak/
├── src/
│   ├── landmarks.py            # MediaPipe extraction + feature engineering (shared by training & serving)
│   ├── hand_model.py           # Forward-kinematics hand model (synthetic data only)
│   ├── signs.py                # Letter → hand-pose definitions (24 static + 2 motion)
│   ├── motion_gestures.py      # Trajectory-based J/Z detection
│   ├── generate_synthetic_data.py  # Bootstraps a training set without a camera
│   ├── data_collection.py      # Records REAL labeled samples from your webcam
│   └── train.py                # Trains + evaluates the classifier, saves the model
├── backend/
│   └── app.py                  # FastAPI service: /predict, /health, serves the frontend
├── frontend/
│   └── index.html              # Webcam capture UI, live predictions, word buffer
├── tests/
│   ├── test_pipeline.py        # Feature-engineering sanity tests (invariance, separability)
│   └── test_motion_gestures.py # J/Z detector tests against synthetic trajectories
├── data/                       # Landmark CSVs (synthetic and/or real)
├── models/                     # Trained model artifacts (.pkl)
├── web-demo/                   # Zero-backend client-side port (deployed to Vercel)
│   ├── index.html              # Same UI, but hand tracking + classifier both run in-browser
│   ├── pipeline.js             # JS port of landmarks.py + motion_gestures.py (verified numerically to match)
│   └── model.js                # Trained RandomForest exported to a plain JS function via m2cgen
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

Open `http://localhost:8000`, click **Start camera**, and show a hand sign. Hold a static pose steady for about a second and the letter gets appended to the word buffer below the video. For J and Z, hold the base handshape (I for J, D for Z) and trace the motion — a hook for J, a zigzag for Z.

## Using real data instead of (or alongside) synthetic data

The synthetic generator exists so the whole pipeline can be built, tested, and demoed without a camera — it's a stand-in, not the real signal. The shipped model is now trained on a mix of synthetic data and **real photographed ASL hand poses** (`data/real_landmarks.csv` — 1,197 samples across 24 static letters, landmarks extracted with MediaPipe from real hand photos, not synthetic geometry), which is what closed most of the gap between "looks right in a demo" and "recognizes an actual hand."

To add more real samples yourself (from your own webcam):

```bash
python src/data_collection.py --label A --samples 200
# repeat for each letter you want to support
python src/train.py --data data/landmarks.csv
```

Both scripts write to the same CSV schema (`f0..f67,label`), so you can freely mix synthetic and real rows. `train.py --data data/*.csv` (the default) picks up everything in `data/`.

## Supported letters

All 26 letters — the full alphabet, in both live demos:

`A B C D E F G H I J K L M N O P Q R S T U V W X Y Z`

Split across two different mechanisms:

**24 static letters** — `A B C D E F G H I K L M N O P Q R S T U V W X Y` (every letter except J and Z) — classified frame-by-frame by the ML model from hand geometry. Most are cleanly separated; three letter groups are only *approximately* distinguishable with this simplified curl+rotation+spread geometric model, because real ASL also encodes detail this model doesn't capture — finger crossing (R), and exactly which fingers the thumb tucks between (M vs N vs T vs E). Real webcam training data resolves this better than the synthetic model can, since a real hand has all that extra detail for MediaPipe to actually see.

**2 motion letters** — `J` and `Z` — detected from fingertip trajectory rather than the ML classifier — see "Why this approach" above. Hold the base handshape (I for J, D for Z) and trace the motion: a hook for J, a zigzag for Z.

## Model performance (real + synthetic data)

Random Forest trained on 8,397 samples (real photographed hand poses + synthetic) across the 24 static letters, 80/20 train/test split: **94.7% test accuracy**. This is a more honest number than the earlier synthetic-only figure, since real photos are now in both the train and test splits — it reflects actual hand geometry, not just the simplified curl model. Per-letter, most letters are solidly separated; the weakest are the geometrically ambiguous cluster documented above — M (0.69 F1), N (0.77), S (0.82), E (0.75) — where thumb-tuck position and finger-crossing detail are hardest to distinguish even from real landmarks.

## Tech stack

Python · MediaPipe (hand landmark detection) · scikit-learn (Random Forest / SVM) · FastAPI · vanilla JS (webcam capture, `getUserMedia` + `canvas`) · pandas / numpy for the data pipeline.

## Possible extensions

- Real collected data for the M/N/T/E and R/U/H clusters specifically, to resolve the geometric-model ambiguity documented above.
- Two-hand support for signs that use both hands.
- Confidence-based auto-correction / a dictionary layer to turn letter sequences into likely words.
- Swap the browser polling loop for a WebSocket stream for lower latency.
- Package the backend + a pinned frontend build into a Docker image for one-command deploy.
