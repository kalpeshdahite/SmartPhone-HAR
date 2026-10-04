import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from models.cnn1d import CNN1D
from preprocessing.har_data import normalize   # same function used in training

SAMPLE_RATE_HZ = 50.0   # ASSUMPTION: verify against the dataset documentation
WINDOW = 150

# ---- Phone -> dataset conversion. These are GUESSES: calibrate on your phone. ----
ACC_TO_G = 1.0 / 9.80665          # phone accel in m/s^2 -> g
GYRO_TO_RADS = 1.0                # use 0.0174533 if the phone reports deg/s
ACC_SRC, ACC_SIGN = [1, 2, 0], [1, 1, 1]   # dataset (x,y,z) <- phone axis index
GYR_SRC, GYR_SIGN = [1, 2, 0], [1, 1, 1]


def phone_to_dataset(acc, gyr):
    """acc, gyr: (N,3) raw phone values -> (N,6) in dataset units/axes."""
    a = acc[:, ACC_SRC] * np.array(ACC_SIGN) * ACC_TO_G
    g = gyr[:, GYR_SRC] * np.array(GYR_SIGN) * GYRO_TO_RADS
    return np.concatenate([a, g], axis=1)


def resample_to_grid(t, v, rate=SAMPLE_RATE_HZ, n=WINDOW):
    """t: (N,) seconds, increasing; v: (N,6). Returns the latest (6,n) window on a
    uniform grid, or None if less than n samples' worth of time has arrived."""
    if len(t) < 2 or (t[-1] - t[0]) < (n - 1) / rate - 1e-6:
        return None
    grid = t[-1] - (n - 1 - np.arange(n)) / rate
    return np.stack([np.interp(grid, t, v[:, c]) for c in range(6)]).astype(np.float32)


def load_final_model(path):
    ck = torch.load(path, map_location="cpu")
    model = CNN1D()
    model.load_state_dict(ck["model_state"])
    model.eval()
    mean = np.array(ck["mean"], dtype=np.float32)
    std = np.array(ck["std"], dtype=np.float32)
    return model, mean, std, ck["class_names"], ck


@torch.no_grad()
def predict_window(model, win, mean, std):
    x = torch.from_numpy(normalize(win[None], mean, std)).float()
    return torch.softmax(model(x), dim=1)[0].numpy()