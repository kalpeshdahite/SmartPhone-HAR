import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from preprocessing.har_data import load_split, CLASS_NAMES

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:5000")
ap.add_argument("--demo", action="store_true", help="stream one window per class in real time")
ap.add_argument("--classes", type=int, nargs="+", default=[0, 2, 3, 4])
ap.add_argument("--limit", type=int, default=0, help="0 = all test windows")
args = ap.parse_args()


def post(payload):
    req = urllib.request.Request(args.url + "/ingest", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def samples(w, start=0, stop=150):
    return [{"t": k * 20, "acc": w[:3, k].tolist(), "gyro": w[3:, k].tolist()}
            for k in range(start, stop)]            # 20 ms apart = 50 Hz


X, y = load_split("test")        # RAW values; the server normalizes
if args.demo:
    for c in args.classes:
        w = X[np.where(y == c)[0][0]]
        post({"reset": True, "space": "dataset", "samples": []})
        for i in range(0, 150, 10):                   # 10 samples every 0.2 s
            st = post({"space": "dataset", "samples": samples(w, i, i + 10)})
            time.sleep(0.2)
        print(f"true={CLASS_NAMES[c]:10s} -> predicted {st['activity']} "
              f"({st['confidence']})")
else:
    idx = np.arange(len(y))
    if args.limit:
        idx = np.random.default_rng(0).permutation(len(y))[:args.limit]
    correct, missing = 0, 0
    for n, i in enumerate(idx, 1):
        st = post({"reset": True, "space": "dataset", "samples": samples(X[i])})
        if st["raw_pred"] is None:
            missing += 1
        else:
            correct += int(st["raw_pred"] == y[i])
        if n % 100 == 0:
            print(f"{n}/{len(idx)} windows sent")
    acc = correct / len(idx)
    print(f"\nLIVE-PIPELINE accuracy on {len(idx)} test windows: {acc:.4f} "
          f"(windows with no prediction: {missing})")
    if not args.limit:
        ck = torch.load("checkpoints/final_model.pt", map_location="cpu")
        print(f"OFFLINE accuracy from checkpoint:            {ck['test_accuracy']:.4f}")