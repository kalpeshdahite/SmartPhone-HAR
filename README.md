# Smartphone-HAR: Self-Supervised Human Activity Recognition from Raw Sensor Data

A final-year project that recognises human activity from raw smartphone accelerometer and gyroscope signals, tests whether self-supervised learning (SSL) helps when labels are scarce, and runs the trained model on a live sensor stream.

Based on the benchmark idea in *"Benchmarking Encoders and Self-Supervised Learning for Smartphone-Based Human Activity Recognition"*, using the DAGHAR standardized UCI HAR data.

## Project status

| Part | Status |
|---|---|
| Dataset pipeline (load, normalize, label-fraction subsets) | Done, verified |
| Random Forest and 1D CNN baselines | Done, results below |
| SimCLR pretraining + label-efficiency study (3 seeds) | Done, results below |
| Live server, dashboard, replay test | Code written; replay verification result: **TBD** |
| Real phone streaming (phone in trouser pocket) | In progress |
| Second dataset, more encoders, statistical tests | Future work |

## Pipeline

```
Raw accel + gyro (6 channels)
   -> windows of 150 samples
   -> per-channel normalization (train statistics only)
   -> [SimCLR pretraining, no labels] -> CNN encoder
   -> classifier head -> activity + confidence
   -> live: phone -> Flask server -> sliding buffer -> same preprocessing -> model -> dashboard
```

## Data

DAGHAR standardized view of UCI HAR. Inspected, not assumed:

- Window: 6 channels x 150 time steps. Train 2420, validation 340, test 690 windows.
- 5 balanced classes. Labels are `standard activity code` 0-4. Names used: 0 sitting, 1 standing, 2 walking, 3 upstairs, 4 downstairs. **These names are inferred from the original UCI numbering and signal statistics; confirm against the DAGHAR documentation before publishing.**
- Subject-independent split: 21 train, 3 validation, 6 test users, no user or recording shared between splits.
- Accelerometer appears to be in g (gravity mostly on x), gyroscope in rad/s (unverified). Sampling rate assumed 50 Hz (**unverified**).
- No running class exists in this dataset.

## Results (test set, from actual runs)

Baselines (seed 42, 100% labels):

| Model | Accuracy | Macro F1 | Parameters | Inference (ms/window) |
|---|---:|---:|---:|---:|
| Random Forest (64 handcrafted features) | 0.8899 | 0.8901 | n/a | 0.1115 (CPU) |
| 1D CNN | 0.9362 | 0.9351 | 37,093 | 0.0109 (GPU, batched) |

The CNN's main error: 40 of 138 standing windows predicted as sitting.

Label efficiency (mean ± std over seeds 0, 1, 2):

| Labels | CNN from scratch | SSL (SimCLR) + CNN |
|---:|---:|---:|
| 100% | 0.934 ± 0.003 | 0.922 ± 0.015 |
| 10% | 0.926 ± 0.018 | 0.913 ± 0.013 |
| 5% | 0.927 ± 0.021 | 0.914 ± 0.009 |
| 1% | 0.760 ± 0.044 | 0.802 ± 0.032 |

Reading: SSL did not help at 100%, 10% or 5% labels. At 1% (25 windows) it was ahead in all 3 seeds, but 3 seeds with overlapping error bars is suggestive, not conclusive.

Caveats: the encoder was pretrained once (seed 42); the 340-window validation set selects the best epoch even in low-label runs; validation and test contain few users, so numbers are noisy.

## Repository structure

```
preprocessing/har_data.py     load, normalize, stratified label subsets, DataLoaders
models/cnn1d.py               1D CNN (encoder + classifier)
training/train_eval.py        training loop, metrics, plots
self_supervised/              SimCLR code (augmentations, NT-Xent live in scripts/run_ssl_expirement.py)
scripts/
  check_env.py                environment check
  inspect_raw_csv.py          dataset structure check
  inspect_labels_and_splits.py  labels, user leakage check, normalization stats
  phase1_demo.py              shapes, subsets, example windows
  run_rf_baseline.py          Random Forest baseline
  run_cnn_baseline.py         1D CNN baseline
  run_ssl_expirement.py       SimCLR pretraining + label-efficiency experiment
  train_final_model.py        trains the model used by the live demo
live/
  live_preprocess.py          phone->dataset conversion, resampling, prediction
  server.py                   Flask server (/ingest, /state, dashboard)
  dashboard.html              live display
  replay_test.py              streams real test windows into the server
results/                      metrics, confusion matrices, curves, tables, figures
checkpoints/                  saved models (not committed)
```

## How to run

```bash
python -m venv .venv && source .venv/Scripts/activate
python -m pip install -r requirements.txt   # includes flask
python scripts/check_env.py
python scripts/inspect_raw_csv.py
python scripts/inspect_labels_and_splits.py
python scripts/phase1_demo.py
python scripts/run_rf_baseline.py
python scripts/run_cnn_baseline.py
python scripts/run_ssl_expirement.py
python scripts/train_final_model.py        # model for the live demo
python live/server.py                      # open http://localhost:5000
python live/replay_test.py                 # in a second terminal
python live/replay_test.py --demo
```

## Live prototype

The phone streams timestamped accel and gyro samples to the server over the local network. The server keeps a sliding buffer, resamples to the training rate, applies the same normalization as training, and predicts about every 0.5 s with light smoothing over the last 3 predictions.

**The replay test** streams real test windows through the same server. Its accuracy should match the offline accuracy; this proves the live path is consistent with training. It is a simulation, not a phone result.

**Phone placement for this version: trouser pocket.** The training data was recorded with a waist-mounted phone, so expect a domain shift. Pocket orientation also changes how gravity falls on the axes, so the axis and unit mapping in `live/live_preprocess.py` are guesses that must be calibrated on the real phone. Keep the phone in a fixed orientation during demos. Sitting vs standing is expected to be the least reliable pair.

## Limitations

- One dataset, 5 classes, waist-mounted phone.
- Sampling rate and class names to be confirmed against dataset documentation.
- SSL evidence is limited (one pretraining run, 3 fine-tuning seeds).
- Live accuracy on a real phone in a pocket has not yet been measured.

## Future work

Dynamic placement (hand, bag, different pockets), a second dataset (e.g. WISDM or MotionSense for running), more SSL methods and encoders, 10+ seeds with statistical tests, and a small own-recorded fine-tuning set.

## References

- DAGHAR: https://github.com/H-IAAC/DAGHAR
- Benchmark implementation: https://github.com/H-IAAC/benchmarking-encoders-ssl-har
