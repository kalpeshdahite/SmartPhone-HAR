# Smartphone-HAR
## Installation Guide, Development Roadmap & GitHub Project Guide

> **Project:** Smartphone-Based Human Activity Recognition (HAR)  
> **Research basis:** *Benchmarking Encoders and Self-Supervised Learning for Smartphone-Based Human Activity Recognition*  
> **Primary implementation:** Python + PyTorch  
> **Recommended development environment:** Windows + VS Code + Python virtual environment  
> **Initial dataset:** UCI HAR through the standardized DAGHAR pipeline

---

## 1. What This Repository Is For

This repository contains the implementation of a smartphone-based Human Activity Recognition research project based on the supplied base paper.

The project is intentionally developed in stages rather than attempting the complete benchmark immediately.

### Development progression

```text
Base Paper
    ↓
DAGHAR / UCI HAR
    ↓
Dataset Analysis
    ↓
Preprocessing
    ↓
PyTorch Dataset + DataLoader
    ↓
ResNet-SE-5 Supervised Baseline
    ↓
Evaluation
    ↓
Self-Supervised Learning
    ↓
Fine-Tuning
    ↓
Few-Shot Learning
    ↓
Multiple Encoders
    ↓
Multiple Datasets
    ↓
Statistical Benchmarking
```

The source project plan specifically recommends reaching a working UCI HAR + ResNet-SE-5 supervised pipeline before moving to SSL. Do not skip this milestone.

---

# 2. Repository Philosophy

## Keep the GitHub repository clean

The GitHub repository should contain:

- Source code
- Configuration files
- Notebooks
- Documentation
- Requirements
- Reproducibility information
- Small example/sample data only when legally appropriate
- Experiment configuration files
- Results that are useful for demonstrating the work

The repository should **not** contain:

- `.venv/`
- Huge raw datasets
- Generated caches
- Model checkpoints unless intentionally released
- Personal files
- API keys
- `.env` secrets
- Temporary notebooks/checkpoints
- OS-specific junk files

Large datasets should be downloaded separately using the documented instructions.

---

# 3. Recommended Development Environment

For the current stage, use:

```text
Windows
│
├── VS Code
├── Git
├── Python
├── Virtual Environment
└── PyTorch
```

Docker is **not required initially**.

Recommended approach:

```text
Phase 1 → Native Windows + .venv
Phase 2 → Stable ML pipeline
Phase 3 → Optional Docker environment
Phase 4 → Reproducible training/deployment
```

The priority is to get the research pipeline working before adding infrastructure complexity.

---

# 4. Hardware Requirements

## Minimum

The project can be developed without a dedicated GPU.

A CPU is sufficient for:

- Dataset inspection
- Preprocessing
- Notebook development
- Debugging
- Small experiments
- Initial model verification

Training deep-learning models will generally be slower on CPU.

## Recommended

For larger experiments:

- NVIDIA GPU
- Sufficient system RAM
- SSD storage
- Updated NVIDIA driver

GPU availability should be checked before installing a CUDA-enabled PyTorch build.

---

# 5. Software Requirements

Recommended tools:

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| VS Code | Development environment |
| Git | Version control |
| Jupyter | Dataset/model experimentation |
| NumPy | Numerical processing |
| Pandas | Data processing |
| SciPy | Scientific/statistical operations |
| Scikit-learn | Metrics and utilities |
| Matplotlib | Visualization |
| Seaborn | Visualization |
| GitHub | Source control and collaboration |

---

# 6. Official Installation Resources

Use official resources whenever possible.

### Python

Python for Windows:

https://www.python.org/downloads/windows/

Python documentation:

https://docs.python.org/3/

### Git

Git for Windows:

https://git-scm.com/install/windows

Git documentation:

https://git-scm.com/doc

### VS Code

https://code.visualstudio.com/

### PyTorch

Official PyTorch installation selector:

https://pytorch.org/get-started/locally/

PyTorch documentation:

https://docs.pytorch.org/

PyTorch tutorials:

https://docs.pytorch.org/tutorials/

### DAGHAR

Official DAGHAR repository:

https://github.com/H-IAAC/DAGHAR

DAGHAR Zenodo dataset:

https://doi.org/10.5281/zenodo.11992126

### UCI HAR

UCI Machine Learning Repository:

https://archive.ics.uci.edu/

> Use the DAGHAR repository/documentation for the standardized benchmark workflow used by this project rather than manually modifying the raw datasets.

---

# 7. Step 1 — Install Python

Install a supported 64-bit Python version.

After installation, open PowerShell and verify:

```powershell
python --version
```

Also check:

```powershell
pip --version
```

If `python` is not recognized, check the Python installation and PATH configuration.

---

# 8. Step 2 — Install Git

Install Git for Windows.

Verify:

```powershell
git --version
```

Configure Git identity:

```powershell
git config --global user.name "YOUR NAME"
git config --global user.email "YOUR_EMAIL"
```

---

# 9. Step 3 — Install VS Code

Install VS Code and add these extensions:

### Required

- Python
- Pylance
- Jupyter

### Recommended

- GitLens
- Markdown All in One
- YAML
- GitHub Pull Requests and Issues

---

# 10. Step 4 — Create the Project

Create a project directory:

```powershell
mkdir Smartphone-HAR
cd Smartphone-HAR
```

Initialize Git:

```powershell
git init
```

---

# 11. Step 5 — Create the Python Virtual Environment

Create:

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

You should see something similar to:

```text
(.venv) PS C:\...\Smartphone-HAR>
```

If PowerShell blocks activation, you can either use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

or adjust the PowerShell execution policy according to your Windows configuration.

To deactivate:

```powershell
deactivate
```

---

# 12. Why We Use `.venv`

The virtual environment isolates this project's packages from the rest of the system.

This is important because:

```text
Project A
    ↓
its own dependencies

Project B
    ↓
different dependencies

Smartphone-HAR
    ↓
its own dependencies
```

Do not commit `.venv/` to GitHub.

---

# 13. Step 6 — Upgrade pip

With the environment activated:

```powershell
python -m pip install --upgrade pip
```

---

# 14. Step 7 — Install Basic Dependencies

Install the general scientific/ML dependencies:

```powershell
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter
```

---

# 15. Step 8 — Install PyTorch

Do not blindly copy a random PyTorch command from an old tutorial.

Use the official PyTorch installation selector:

https://pytorch.org/get-started/locally/

Select:

```text
Operating System → Windows
Package → Pip
Language → Python
Compute Platform → CPU / appropriate CUDA version
```

If you have an NVIDIA GPU, first check:

```powershell
nvidia-smi
```

Then select the appropriate supported CUDA build from the official PyTorch page.

---

# 16. Step 9 — Verify PyTorch

Run:

```powershell
python
```

Then:

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Expected CPU result:

```text
PyTorch version: ...
CUDA available: False
```

Expected GPU result:

```text
PyTorch version: ...
CUDA available: True
GPU: ...
```

Do not treat `CUDA available: False` as an error if you are intentionally using CPU.

---

# 17. Step 10 — Freeze the Environment

Once installation is working:

```powershell
pip freeze > requirements.txt
```

This file should be committed to GitHub.

Later, another developer can create the environment and install:

```powershell
pip install -r requirements.txt
```

> For a research repository, also record the Python version and PyTorch version because `pip freeze` alone does not explain the hardware/CUDA environment.

---

# 18. Step 11 — Create the Repository Structure

Use:

```text
Smartphone-HAR/
│
├── .github/
│   └── workflows/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── preprocessing/
│   ├── __init__.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── windowing.py
│   └── dataset.py
│
├── models/
│   ├── __init__.py
│   ├── resnet_se.py
│   ├── cnn_pff.py
│   ├── ts_tcc.py
│   ├── ts2vec.py
│   ├── imu_transformer.py
│   └── rnn.py
│
├── ssl/
│   ├── __init__.py
│   ├── tnc.py
│   ├── tfc.py
│   ├── lfr.py
│   └── diet.py
│
├── training/
│   ├── __init__.py
│   ├── supervised.py
│   ├── pretrain.py
│   └── finetune.py
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   ├── confusion_matrix.py
│   └── statistical_tests.py
│
├── experiments/
│   └── configs/
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_supervised_baseline.ipynb
│   └── 04_ssl_experiments.ipynb
│
├── results/
│   ├── metrics/
│   ├── confusion_matrices/
│   ├── learning_curves/
│   ├── figures/
│   └── tables/
│
├── scripts/
│
├── tests/
│
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE
```

---

# 19. `.gitignore`

Create `.gitignore` in the root:

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo

# Virtual environment
.venv/
venv/
env/

# Jupyter
.ipynb_checkpoints/

# VS Code
.vscode/

# Data
data/raw/*
data/processed/*
data/splits/*

# Generated results
results/checkpoints/
results/tmp/

# Model checkpoints
*.pth
*.pt
*.ckpt

# Logs
*.log

# Environment variables
.env

# OS files
.DS_Store
Thumbs.db
```

If you later decide to publish a small model checkpoint, remove that specific pattern or use Git LFS where appropriate.

---

# 20. Dataset Strategy

The source development plan recommends starting with:

```text
DAGHAR → UCI HAR
```

DAGHAR is useful because it provides standardized processing across smartphone HAR datasets.

The official DAGHAR repository describes scripts for downloading/processing the original datasets and generating standardized views.

Official repository:

https://github.com/H-IAAC/DAGHAR

---

# 21. DAGHAR Setup

Clone DAGHAR separately from your research repository:

```powershell
git clone https://github.com/H-IAAC/DAGHAR.git
```

Follow the official DAGHAR README for its dataset-generation workflow.

The repository documents a workflow based around:

```text
Original datasets
      ↓
DAGHAR processing
      ↓
Raw / standardized views
```

Do not copy the entire DAGHAR repository into your `Smartphone-HAR` repository unless there is a specific reason to do so.

Instead, document the required dataset version/location.

---

# 22. Data Directory in This Project

Your project should eventually have:

```text
data/
├── raw/
├── processed/
└── splits/
```

### `raw/`

Original/downloaded data.

### `processed/`

Standardized/windowed data.

### `splits/`

Train/validation/test information.

Large datasets should generally remain outside GitHub.

---

# 23. Dataset Analysis — FIRST CODING MILESTONE

Create:

```text
notebooks/01_dataset_analysis.ipynb
```

The notebook must answer:

1. What files are present?
2. What is the shape of the data?
3. How many activities?
4. How many subjects?
5. How many sensor channels?
6. What is the sampling frequency?
7. What is the class distribution?
8. What does one 3-second window look like?
9. How are train/validation/test samples represented?

Do not start with neural networks.

First understand the data.

---

# 24. Target Dataset Specifications

The source project plan specifies:

```text
Sampling rate = 20 Hz
Window duration = 3 seconds
Window type = Non-overlapping
Window length = 60 timesteps
Train / Validation / Test = 70 / 20 / 10
Split = Subject-independent
```

Therefore:

```text
20 samples/sec × 3 sec
=
60 timesteps/window
```

---

# 25. Sensor Channels

The standardized setup contains:

```text
Accelerometer X
Accelerometer Y
Accelerometer Z

Gyroscope X
Gyroscope Y
Gyroscope Z
```

Conceptually:

```text
6 channels × 60 timesteps
```

The exact tensor arrangement should be adapted to the encoder being used.

---

# 26. Activity Classes

Initial six-class setup:

| Label | Activity |
|---:|---|
| 0 | Sit |
| 1 | Stand |
| 2 | Walk |
| 3 | Walk Upstairs |
| 4 | Walk Downstairs |
| 5 | Run |

Always verify the actual label mapping in the prepared dataset before training.

Do not assume a mapping merely because it appears in documentation.

---

# 27. Preprocessing Pipeline

Implement:

```text
Raw Sensor Data
      ↓
20 Hz Standardization
      ↓
Gravity Removal / Check
      ↓
3-second Non-overlapping Windows
      ↓
Normalization
      ↓
Subject-independent Split
      ↓
Train / Validation / Test
```

---

# 28. Critical Data-Splitting Rule

Do NOT randomly split individual windows into train and test.

Bad:

```text
Same subject
 ├── Window 1 → Train
 ├── Window 2 → Test
 ├── Window 3 → Train
 └── Window 4 → Test
```

Preferred:

```text
Subjects
 ├── Train subjects
 ├── Validation subjects
 └── Test subjects
```

This prevents the model from simply learning person-specific movement patterns.

---

# 29. PyTorch Dataset

Create:

```text
preprocessing/dataset.py
```

The Dataset should return:

```python
{
    "signal": signal,
    "label": label,
    "subject": subject_id
}
```

Where:

```text
signal  → sensor window
label   → activity
subject → user
```

Then create a PyTorch DataLoader.

Initial batch size:

```text
64
```

---

# 30. First Machine Learning Model

The first model should be:

```text
ResNet-SE-5
```

Configuration:

| Parameter | Initial value |
|---|---|
| Dataset | UCI HAR |
| Encoder | ResNet-SE-5 |
| Training | Supervised |
| Batch size | 64 |
| Learning rate | 1e-4 |
| Number of classes | 6 |

The source project plan reports `1e-4` as the general encoder learning rate and notes a different learning rate for TS2Vec.

---

# 31. Classification Head

Use:

```text
ResNet-SE-5
      ↓
Encoder output
      ↓
ReLU
      ↓
128 units
      ↓
ReLU
      ↓
6 output units
```

---

# 32. Training Loop

Implement:

```text
Epoch
 │
 ├── Training
 │     ├── Forward pass
 │     ├── Loss
 │     ├── Backward pass
 │     └── Optimizer step
 │
 └── Validation
       ├── Validation loss
       └── Validation accuracy
```

Track:

- Train loss
- Validation loss
- Train accuracy
- Validation accuracy

Save the best checkpoint:

```text
best_model.pth
```

---

# 33. Evaluation

Do not report accuracy alone.

Calculate:

```text
Accuracy
Precision
Recall
F1-score
Confusion Matrix
```

Suggested files:

```text
evaluation/
├── metrics.py
└── confusion_matrix.py
```

---

# 34. First Project Milestone

The first milestone is:

```text
UCI HAR
    ↓
Preprocessing
    ↓
3-second windows
    ↓
Subject-independent split
    ↓
DataLoader
    ↓
ResNet-SE-5
    ↓
Supervised training
    ↓
Evaluation
```

Expected result:

```text
Model: ResNet-SE-5
Dataset: UCI HAR
Training: Supervised

Accuracy: XX.XX%
Precision: XX.XX%
Recall: XX.XX%
F1: XX.XX%

Confusion Matrix:
[generated matrix]
```

### STOP HERE

Do not implement SSL until:

- Dataset loading works
- Windowing works
- Subject split is correct
- DataLoader works
- Model trains
- Validation works
- Checkpointing works
- Metrics work
- Confusion matrix works

---

# 35. Self-Supervised Learning Stage

After the supervised baseline is stable, implement:

```text
LFR
TNC
DIET
TF-C
```

General pipeline:

```text
Unlabeled Sensor Data
        ↓
SSL Pretraining
        ↓
Learned Encoder
        ↓
Classification Head
        ↓
Fine-tuning
        ↓
Prediction
```

---

# 36. SSL Development Order

Recommended:

```text
Supervised baseline
        ↓
LFR
        ↓
TNC
        ↓
DIET
        ↓
TF-C
```

TF-C should be left until later because it is more complicated and uses time/frequency processing.

---

# 37. Fine-Tuning Experiments

Compare:

### Frozen encoder

```text
SSL pretrained encoder
        ↓
     FROZEN
        ↓
Classification head
        ↓
TRAIN
```

### Full fine-tuning

```text
SSL pretrained encoder
        ↓
     TRAIN
        ↓
Classification head
        ↓
TRAIN
```

The source project plan reports that full fine-tuning generally performs better, so it should eventually be the primary strategy while freezing is retained as a comparison.

---

# 38. Few-Shot Learning

After SSL works, reduce the amount of labeled data.

Start with:

```text
1 sample/class
5 samples/class
10 samples/class
25 samples/class
50 samples/class
```

Compare:

```text
Supervised
VS
SSL + Fine-tuning
```

The point is to test whether SSL provides better representations when labeled data is limited.

---

# 39. Reproducibility

Important experiments should be repeated with multiple seeds.

The source project plan uses:

```text
Seed 42
Seed 43
Seed 44
```

Record:

```text
Mean
Standard deviation
```

Example:

```text
Experiment: ResNet-SE-5 + TF-C

Seed 42 → 82.1
Seed 43 → 83.4
Seed 44 → 82.8

Mean → 82.77
Std  → ...
```

Do not report only the best seed.

---

# 40. Expand the Encoders

Once ResNet-SE-5 is stable, implement:

```text
ResNet-SE-5
CNN-PFF
TS-TCC
TS2Vec
IMU Transformer
RNN
```

Keep each model in its own file.

This allows experiments to reuse the same:

```text
Dataset
DataLoader
Training
Evaluation
```

without duplicating code.

---

# 41. Expand the Datasets

After UCI HAR works:

```text
UCI HAR
   ↓
MotionSense
   ↓
KuHAR
   ↓
RealWorld-Waist
   ↓
RealWorld-Thigh
   ↓
WISDM
```

Do not implement all datasets simultaneously.

Use UCI HAR as the reference implementation.

---

# 42. Final Benchmark Matrix

Eventually:

```text
6 Datasets
×
6 Encoders
×
Supervised + SSL Methods
×
Multiple Label Regimes
×
Multiple Random Seeds
```

This can become a large experiment matrix.

Use configuration files instead of hard-coding experiments.

Example:

```text
experiments/configs/
├── uci_supervised.yaml
├── uci_lfr.yaml
├── uci_tnc.yaml
├── uci_diet.yaml
└── uci_tfc.yaml
```

---

# 43. Results Organization

Use:

```text
results/
├── metrics/
├── confusion_matrices/
├── learning_curves/
├── figures/
└── tables/
```

Example:

```text
results/
├── metrics/
│   ├── supervised.csv
│   ├── lfr.csv
│   └── tfc.csv
│
├── confusion_matrices/
│   └── uci_resnet_se5.png
│
├── learning_curves/
│   └── resnet_se5_training.png
│
└── tables/
    └── benchmark_results.csv
```

---

# 44. Statistical Analysis

Eventually implement:

```text
evaluation/statistical_tests.py
```

The project plan calls for:

- Multiple random seeds
- Mean
- Standard deviation
- Paired Wilcoxon signed-rank test
- Bonferroni correction

Do this after the core pipeline works.

---

# 45. Final Research Outputs

The finished project should be capable of producing:

- Accuracy comparisons
- F1 comparisons
- Precision/Recall results
- Confusion matrices
- Learning curves
- Few-shot performance graphs
- Encoder comparisons
- SSL comparisons
- Ablation studies
- Statistical significance results
- Inference time
- Model-size comparisons

---

# 46. GitHub Workflow

After creating the repository locally:

```powershell
git status
```

Add files:

```powershell
git add .
```

Commit:

```powershell
git commit -m "Initial project setup"
```

Create the GitHub repository.

Then connect it:

```powershell
git remote add origin YOUR_REPOSITORY_URL
```

Rename the branch:

```powershell
git branch -M main
```

Push:

```powershell
git push -u origin main
```

---

# 47. Recommended Commit Strategy

Do not make one giant commit after finishing the entire project.

Use meaningful commits.

Examples:

```text
Initial repository structure
Set up Python environment documentation
Add dataset analysis notebook
Add UCI HAR loader
Implement preprocessing pipeline
Add subject independent splitting
Add PyTorch Dataset
Add ResNet-SE-5 baseline
Add supervised training loop
Add evaluation metrics
Add confusion matrix
Add LFR pretraining
Add TNC pretraining
Add few-shot evaluation
Add benchmark configuration
Add statistical analysis
Update README
```

This makes the development history useful.

---

# 48. Recommended Branch Strategy

For a small college research project:

```text
main
│
├── develop
│
├── feature/dataset-pipeline
├── feature/resnet-se5
├── feature/ssl-lfr
├── feature/ssl-tfc
└── feature/evaluation
```

If you are working alone, you can keep it simpler:

```text
main
feature/*
```

Do not create unnecessary branches just for the sake of it.

---

# 49. GitHub README Structure

Your root `README.md` should eventually contain:

```text
# Smartphone-HAR

## Overview

## Research Objective

## Base Paper

## Features

## Architecture

## Datasets

## Models

## SSL Methods

## Installation

## Dataset Preparation

## Project Structure

## Running the Project

## Experiments

## Results

## Reproducibility

## Contributors

## Citation

## License
```

This installation/development document can be linked from the README as:

```text
docs/DEVELOPMENT_GUIDE.md
```

---

# 50. Recommended Documentation Structure

As the project grows, use:

```text
docs/
├── INSTALLATION.md
├── DEVELOPMENT_GUIDE.md
├── DATASET.md
├── MODEL_ARCHITECTURE.md
├── EXPERIMENTS.md
└── RESULTS.md
```

For the beginning, keeping this entire guide as:

```text
DEVELOPMENT_GUIDE.md
```

is completely fine.

Later, split it when the repository becomes large.

---

# 51. Development Checklist

## Environment

- [ ] Python installed
- [ ] Git installed
- [ ] VS Code installed
- [ ] Virtual environment created
- [ ] PyTorch installed
- [ ] PyTorch verified
- [ ] CUDA verified if GPU is available
- [ ] requirements.txt created

## Repository

- [ ] Git initialized
- [ ] `.gitignore` created
- [ ] Folder structure created
- [ ] README created
- [ ] Initial commit created
- [ ] GitHub repository created
- [ ] Remote configured
- [ ] Code pushed

## Dataset

- [ ] DAGHAR reviewed
- [ ] UCI HAR prepared
- [ ] Dataset files understood
- [ ] Labels verified
- [ ] Subjects verified
- [ ] Sampling verified
- [ ] Sensor channels verified

## Preprocessing

- [ ] 20 Hz standardization
- [ ] Gravity handling/check
- [ ] 3-second windows
- [ ] 60 timestep windows
- [ ] Normalization
- [ ] Subject-independent splitting
- [ ] DataLoader

## Supervised Baseline

- [ ] ResNet-SE-5
- [ ] MLP classification head
- [ ] Training loop
- [ ] Validation
- [ ] Checkpointing
- [ ] Accuracy
- [ ] Precision
- [ ] Recall
- [ ] F1
- [ ] Confusion matrix

## SSL

- [ ] LFR
- [ ] TNC
- [ ] DIET
- [ ] TF-C
- [ ] Frozen encoder
- [ ] Full fine-tuning

## Few-Shot

- [ ] 1 sample/class
- [ ] 5 samples/class
- [ ] 10 samples/class
- [ ] 25 samples/class
- [ ] 50 samples/class
- [ ] Supervised comparison
- [ ] SSL comparison

## Benchmark

- [ ] CNN-PFF
- [ ] TS-TCC
- [ ] TS2Vec
- [ ] IMU Transformer
- [ ] RNN
- [ ] MotionSense
- [ ] KuHAR
- [ ] RealWorld-Waist
- [ ] RealWorld-Thigh
- [ ] WISDM

## Research Analysis

- [ ] Multiple seeds
- [ ] Mean
- [ ] Standard deviation
- [ ] Wilcoxon test
- [ ] Bonferroni correction
- [ ] Accuracy graphs
- [ ] F1 graphs
- [ ] Learning curves
- [ ] Confusion matrices
- [ ] Ablation studies
- [ ] Inference time
- [ ] Model size

---

# 52. Troubleshooting

## `python` is not recognized

Check Python installation:

```powershell
python --version
```

If necessary, reinstall Python and ensure the appropriate PATH option is enabled.

---

## `.venv\Scripts\Activate.ps1` is blocked

Try Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

Or review your PowerShell execution-policy configuration.

Do not randomly disable Windows security controls without understanding the change.

---

## `torch.cuda.is_available()` returns `False`

Check:

```powershell
nvidia-smi
```

Possible causes:

- No NVIDIA GPU
- NVIDIA driver issue
- CPU-only PyTorch installation
- Incompatible PyTorch/CUDA setup
- Environment mismatch

Use the official PyTorch installation selector to install the appropriate build.

---

## CUDA works on the machine but not in Python

Run:

```python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
```

Then inspect the GPU:

```python
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

---

## Dataset path errors

Avoid hard-coded absolute paths such as:

```text
C:\Users\YourName\Desktop\...
```

Prefer project-relative paths.

For example:

```text
Smartphone-HAR/
└── data/
```

Use Python's `pathlib` where possible:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
```

---

## Out-of-memory errors

Start smaller:

```text
batch_size = 16
```

Then:

```text
32
```

Then:

```text
64
```

Do not assume that the paper's batch size will fit every GPU.

---

## Training is too slow

Check:

```python
torch.cuda.is_available()
```

If CPU-only:

- Reduce batch size
- Run a small subset for debugging
- Reduce epochs during development
- Use a smaller experiment
- Use GPU/cloud resources for larger benchmark runs

Never use the full benchmark just to test whether your code works.

---

# 53. Development Rule: Debug Small

During development, use:

```text
Small dataset subset
        ↓
Small number of batches
        ↓
1–2 epochs
        ↓
Check shapes/loss
        ↓
Then scale up
```

Do not spend hours training a model before discovering that the labels are wrong.

---

# 54. Development Rule: Validate Every Stage

Use this progression:

```text
Dataset
  ↓
Can load?
  ↓
Preprocessing
  ↓
Correct shape?
  ↓
DataLoader
  ↓
Correct batch?
  ↓
Model
  ↓
Forward pass?
  ↓
Loss
  ↓
Backward pass?
  ↓
Training
  ↓
Validation
  ↓
Evaluation
```

Only move forward when the current stage is verified.

---

# 55. Reproducibility Checklist

For every important experiment record:

```text
Dataset
Dataset version
Preprocessing configuration
Encoder
SSL method
Training strategy
Batch size
Learning rate
Number of epochs
Optimizer
Random seed
Hardware
Python version
PyTorch version
CUDA version
Results
```

Store these in experiment configuration files and/or result CSV files.

---

# 56. Recommended Experiment Record

Example:

```yaml
experiment_name: uci_resnet_se5_supervised

dataset: UCI_HAR

encoder: ResNet-SE-5

training:
  strategy: supervised
  batch_size: 64
  learning_rate: 0.0001
  epochs: 50

preprocessing:
  sampling_rate: 20
  window_seconds: 3
  window_timesteps: 60
  split: subject_independent

seed: 42
```

The exact configuration should match the implementation actually used.

---

# 57. Important Research Warning

Do not claim that your implementation reproduces the paper exactly until you have verified:

- Dataset version
- Dataset preprocessing
- Windowing
- Subject split
- Model architecture
- Hyperparameters
- Training schedule
- SSL implementation
- Fine-tuning procedure
- Random seeds
- Evaluation procedure

A project can be **based on the paper** without being an exact reproduction.

Be precise in the final report.

---

# 58. What We Should NOT Do Yet

Do not start with:

```text
6 datasets
×
6 encoders
×
4 SSL methods
×
many label regimes
×
multiple seeds
```

That will make debugging unnecessarily difficult.

Start with:

```text
UCI HAR
    ↓
Preprocessing
    ↓
DataLoader
    ↓
ResNet-SE-5
    ↓
Supervised
    ↓
Evaluation
```

Then expand.

---

# 59. Immediate Next Steps

Follow this exact order:

### Step 1

Install:

```text
Python
Git
VS Code
```

### Step 2

Create:

```text
Smartphone-HAR
```

### Step 3

Create:

```text
.venv
```

### Step 4

Install:

```text
PyTorch
NumPy
Pandas
SciPy
Scikit-learn
Matplotlib
Seaborn
Jupyter
```

### Step 5

Verify PyTorch.

### Step 6

Create the repository structure.

### Step 7

Create `.gitignore`.

### Step 8

Create GitHub repository.

### Step 9

Prepare DAGHAR/UCI HAR.

### Step 10

Create:

```text
notebooks/01_dataset_analysis.ipynb
```

### Step 11

Verify:

```text
dataset
shape
channels
classes
subjects
sampling
distribution
```

### Step 12

Implement preprocessing.

### Step 13

Implement PyTorch Dataset/DataLoader.

### Step 14

Verify one batch.

### Step 15

Implement ResNet-SE-5.

### Step 16

Train a tiny debug experiment.

### Step 17

Train the real supervised baseline.

### Step 18

Evaluate.

### Step 19

Only then start SSL.

---

# 60. Final Project Roadmap

```text
PHASE 1
Environment
Dataset
Repository
        ↓
PHASE 2
Dataset Analysis
Preprocessing
DataLoader
        ↓
PHASE 3
ResNet-SE-5
Supervised Training
Evaluation
        ↓
PHASE 4
LFR
TNC
DIET
TF-C
        ↓
PHASE 5
Fine-Tuning
Frozen vs Full
        ↓
PHASE 6
Few-Shot Learning
1 / 5 / 10 / 25 / 50 samples
        ↓
PHASE 7
More Encoders
        ↓
PHASE 8
More Datasets
        ↓
PHASE 9
Multiple Seeds
Statistical Analysis
        ↓
PHASE 10
Final Benchmark
Graphs
Tables
Ablations
Inference Time
Model Size
```

---

# 61. Definition of Done

The project is complete when the repository can reproduce the documented experiments from a clean environment and produce:

```text
Dataset
   ↓
Preprocessing
   ↓
Training
   ↓
Evaluation
   ↓
Results
```

with documented configurations and reproducible experiment settings.

---

# 62. Useful Resources

## Core Development

- Python: https://www.python.org/downloads/windows/
- Python virtual environments: https://docs.python.org/3/library/venv.html
- Git for Windows: https://git-scm.com/install/windows
- VS Code: https://code.visualstudio.com/

## Machine Learning

- PyTorch installation: https://pytorch.org/get-started/locally/
- PyTorch documentation: https://docs.pytorch.org/
- PyTorch tutorials: https://docs.pytorch.org/tutorials/
- Scikit-learn: https://scikit-learn.org/
- NumPy: https://numpy.org/
- Pandas: https://pandas.pydata.org/
- SciPy: https://scipy.org/

## Dataset

- DAGHAR GitHub: https://github.com/H-IAAC/DAGHAR
- DAGHAR Zenodo: https://doi.org/10.5281/zenodo.11992126
- UCI Machine Learning Repository: https://archive.ics.uci.edu/

---

# 63. Base-Paper Development Plan

The original project planning material establishes the following overall sequence:

```text
Base paper
→ Dataset
→ Preprocessing
→ Supervised baseline
→ SSL
→ Fine-tuning
→ Few-shot learning
→ Multiple encoders
→ Multiple datasets
→ Statistical benchmarking
```

This document converts that plan into a repository-oriented development workflow.

---

# 64. Final Principle

**Do not optimize for the number of models implemented. Optimize for a pipeline that works, can be reproduced, and produces trustworthy comparisons.**

A smaller experiment with:

```text
correct data
+
correct subject split
+
correct preprocessing
+
reproducible training
+
proper evaluation
```

is far more valuable than a large benchmark with an unclear data pipeline.

---

## Project Status

Update this section as development progresses.

### Current stage

```text
[ ] Environment setup
[ ] GitHub repository
[ ] Dataset preparation
[ ] Dataset analysis
[ ] Preprocessing
[ ] DataLoader
[ ] ResNet-SE-5
[ ] Supervised baseline
[ ] Evaluation
[ ] SSL
[ ] Fine-tuning
[ ] Few-shot
[ ] Benchmark
[ ] Statistical analysis
```

### Last completed milestone

```text
Not started
```

### Next milestone

```text
Environment setup + repository creation
```
