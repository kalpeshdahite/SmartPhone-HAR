import sys
import threading
from collections import deque, Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from live.live_preprocess import (phone_to_dataset, resample_to_grid,
                                  load_final_model, predict_window)

app = Flask(__name__)
model, mean, std, CLASS_NAMES, ck = load_final_model(ROOT / "checkpoints" / "final_model.pt")
print(f"Loaded model (init={ck['init']}), offline test acc {ck['test_accuracy']:.4f}")

CH = ["ax", "ay", "az", "gx", "gy", "gz"]
PRED_EVERY_S = 0.5
lock = threading.Lock()
T_BUF, V_BUF = deque(maxlen=2000), deque(maxlen=2000)
recent = deque(maxlen=3)
last_pred_t = [-1e18]
EMPTY = dict(activity=None, raw_activity=None, raw_pred=None, confidence=None, probs=None)
state = dict(EMPTY, latest=None, buffer_s=0.0)


def update_and_predict():
    if len(T_BUF) < 2:
        return
    t, v = np.array(T_BUF), np.array(V_BUF)
    state["buffer_s"] = round(float(t[-1] - t[0]), 2)
    state["latest"] = dict(zip(CH, np.round(v[-1], 3).tolist()))
    if t[-1] - last_pred_t[0] < PRED_EVERY_S:
        return
    win = resample_to_grid(t, v)
    if win is None:
        return
    probs = predict_window(model, win, mean, std)
    idx = int(probs.argmax())
    recent.append(idx)
    counts = Counter(recent)
    sm = idx if counts[idx] == max(counts.values()) else counts.most_common(1)[0][0]
    last_pred_t[0] = float(t[-1])
    state.update(activity=CLASS_NAMES[sm], raw_activity=CLASS_NAMES[idx], raw_pred=idx,
                 confidence=round(float(probs[sm]), 4),
                 probs={n: round(float(p), 4) for n, p in zip(CLASS_NAMES, probs)})


@app.post("/ingest")
def ingest():
    data = request.get_json(force=True)
    with lock:
        if data.get("reset"):
            T_BUF.clear(); V_BUF.clear(); recent.clear()
            last_pred_t[0] = -1e18
            state.update(EMPTY)
        dataset_space = data.get("space") == "dataset"
        for s in data.get("samples", []):
            t = s["t"] / 1000.0                      # phone sends milliseconds
            if T_BUF and t <= T_BUF[-1]:
                continue                              # drop out-of-order samples
            acc, gyr = np.array(s["acc"], float), np.array(s["gyro"], float)
            v = (np.concatenate([acc, gyr]) if dataset_space
                 else phone_to_dataset(acc[None], gyr[None])[0])
            T_BUF.append(t); V_BUF.append(v)
        update_and_predict()
        return jsonify(state)


@app.get("/state")
def get_state():
    with lock:
        return jsonify(state)


@app.get("/")
def dashboard():
    return send_from_directory(ROOT / "live", "dashboard.html")

@app.get("/phone")
def phone():
    return send_from_directory(ROOT / "live", "phone_sensors.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)