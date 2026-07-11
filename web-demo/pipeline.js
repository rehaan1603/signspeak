/**
 * JS port of src/landmarks.py (feature engineering) and
 * src/motion_gestures.py (J/Z trajectory detection), so this demo can run
 * fully client-side (no Python backend) using @mediapipe/tasks-vision for
 * hand landmark extraction in the browser.
 *
 * These functions are deliberately written to mirror the Python originals
 * line-for-line where possible -- correctness here means "same output as
 * the training pipeline", not "idiomatic JS". Verified numerically against
 * the Python implementation on synthetic samples before shipping.
 */

// ---- Feature engineering (mirrors src/landmarks.py) ----

const WRIST = 0;
const MIDDLE_MCP = 9;

// Same finger chains, same order, as landmarks.py's FINGER_CHAINS.
const FINGER_CHAINS = [
  ["thumb", 1, 2, 4],
  ["index", 5, 6, 8],
  ["middle", 9, 10, 12],
  ["ring", 13, 14, 16],
  ["pinky", 17, 18, 20],
];

function normalizeLandmarks(points) {
  // points: array of 21 {x,y,z}. Translation invariance: subtract wrist.
  // Scale invariance: divide by wrist->middle_mcp distance.
  const origin = points[WRIST];
  const shifted = points.map((p) => ({
    x: p.x - origin.x,
    y: p.y - origin.y,
    z: p.z - origin.z,
  }));
  const mcp = shifted[MIDDLE_MCP];
  const scale = Math.sqrt(mcp.x * mcp.x + mcp.y * mcp.y + mcp.z * mcp.z) + 1e-6;
  return shifted.map((p) => ({ x: p.x / scale, y: p.y / scale, z: p.z / scale }));
}

function angleAt(a, b, c) {
  // Angle at vertex b formed by points a-b-c, in radians.
  const v1 = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  const v2 = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const n1 = Math.hypot(v1.x, v1.y, v1.z);
  const n2 = Math.hypot(v2.x, v2.y, v2.z);
  const cos = Math.max(-1, Math.min(1, dot / (n1 * n2 + 1e-9)));
  return Math.acos(cos);
}

function fingerAngles(points) {
  // Uses RAW (unnormalized) points, same as landmarks.py -- angles are
  // already translation/scale invariant, so this matches the Python
  // output exactly despite not pre-normalizing.
  return FINGER_CHAINS.map(([, i, j, k]) => angleAt(points[i], points[j], points[k]));
}

function landmarksToFeatureVector(points) {
  // Full pipeline: raw 21x3 landmarks -> flat 68-d feature vector,
  // matching landmarks.landmarks_to_feature_vector exactly (same
  // concatenation order: normalized xyz flattened, then 5 finger angles).
  const norm = normalizeLandmarks(points);
  const angles = fingerAngles(points);
  const flat = [];
  for (const p of norm) flat.push(p.x, p.y, p.z);
  return flat.concat(angles);
}

// ---- Motion detection (mirrors src/motion_gestures.py) ----

function velocities(trail) {
  const out = [];
  for (let i = 0; i < trail.length - 1; i++) {
    out.push([trail[i + 1][0] - trail[i][0], trail[i + 1][1] - trail[i][1]]);
  }
  return out;
}

function pathLength(trail) {
  return velocities(trail).reduce((s, [dx, dy]) => s + Math.hypot(dx, dy), 0);
}

function horizontalDirectionReversals(trail, deadzone = 0.004) {
  const signs = [];
  for (const [dx] of velocities(trail)) {
    if (dx > deadzone) signs.push(1);
    else if (dx < -deadzone) signs.push(-1);
  }
  let count = 0;
  for (let i = 1; i < signs.length; i++) if (signs[i] !== signs[i - 1]) count++;
  return count;
}

function detectZMotion(trail, minPoints = 6, minReversals = 2, minPathLength = 0.35) {
  if (trail.length < minPoints) return false;
  if (pathLength(trail) < minPathLength) return false;
  return horizontalDirectionReversals(trail) >= minReversals;
}

function detectJMotion(
  trail,
  minPoints = 6,
  minTurnDeg = 40.0,
  minPathLength = 0.25,
  maxReversals = 1
) {
  if (trail.length < minPoints) return false;
  if (pathLength(trail) < minPathLength) return false;
  if (horizontalDirectionReversals(trail) > maxReversals) return false;

  const window = Math.max(2, Math.floor(trail.length / 3));
  const startVec = [trail[window][0] - trail[0][0], trail[window][1] - trail[0][1]];
  const last = trail.length - 1;
  const endVec = [trail[last][0] - trail[last - window][0], trail[last][1] - trail[last - window][1]];
  if (Math.hypot(...startVec) < 1e-6 || Math.hypot(...endVec) < 1e-6) return false;

  const startAngle = (Math.atan2(startVec[1], startVec[0]) * 180) / Math.PI;
  const endAngle = (Math.atan2(endVec[1], endVec[0]) * 180) / Math.PI;
  // Python's `%` always returns a non-negative result for a positive
  // modulus; JS's follows the sign of the dividend, so normalize first.
  const turn = Math.abs((((endAngle - startAngle + 180) % 360) + 360) % 360 - 180);
  return turn >= minTurnDeg;
}

class FingerTrail {
  constructor(maxlen = 20, staleAfterS = 0.8) {
    this.maxlen = maxlen;
    this.staleAfterS = staleAfterS;
    this._points = [];
    this._lastT = null;
  }
  push(point, t) {
    if (this._lastT !== null && t - this._lastT > this.staleAfterS) this._points = [];
    this._points.push(point);
    if (this._points.length > this.maxlen) this._points.shift();
    this._lastT = t;
  }
  clear() {
    this._points = [];
    this._lastT = null;
  }
  get points() {
    return [...this._points];
  }
}
