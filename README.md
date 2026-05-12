# Macroinvertebrate Image Analysis System

A Python application that indexes a freshwater macroinvertebrate image
dataset, performs exploratory data analysis, trains a baseline image
classifier, and ships the result as a small Tkinter desktop app for
on-demand image prediction.

Built for **Software Technology 1 (4483 / 8995)** — Assignment 3, Group
Project (Postgraduate scope: all three stages implemented).

---

## Project goal

Given a folder of macroinvertebrate images organised by class, the system
can:

1. Scan the dataset and produce useful summary statistics (Stage 1).
2. Train a Random Forest classifier on flattened grayscale features
   (Stage 2).
3. Let a user select an image through a desktop GUI and see a predicted
   class with a confidence score (Stage 3).

The same `WorkflowService` powers the standalone training script and the
GUI, so the deployed application is a thin wrapper over the same logic
used during development.

---

## Main features

- **Dataset indexing** — recursive scan of `data/raw/`, treating each
  parent folder name as a class label.
- **Exploratory data analysis** — class distribution, image width/height
  distributions, and a 3×3 sample grid, all saved as PNGs in
  `outputs/eda/`.
- **Baseline classifier** — Random Forest trained on 128×128 grayscale,
  flattened features. Model is persisted to
  `outputs/models/macro_classifier.joblib`.
- **Desktop GUI** — three tabs: Predict, EDA Charts, Training.
- **Manual testing evidence** — see `MANUAL_TESTING.md`.

---

## Packages used

| Package         | Used for                                                          |
| --------------- | ----------------------------------------------------------------- |
| `pandas`        | Indexed image records and tabular EDA                             |
| `numpy`         | Feature arrays and numerical work                                 |
| `opencv-python` | Image loading, grayscale conversion, resizing                     |
| `matplotlib`    | EDA charts and confusion matrix rendering                         |
| `seaborn`       | Higher-level statistical plots used in EDA                        |
| `scikit-learn`  | Train/test split, Random Forest classifier, evaluation metrics    |
| `joblib`        | Persisting and loading the trained model artifact                 |
| `Pillow`        | Image previews inside the Tkinter GUI                             |
| `tkinter`       | Desktop application UI (ships with the Python standard library)   |

A pinned list lives in `requirements.txt`.

---

## Folder structure

```
macro-image-analysis/
├── data/
│   └── raw/                       # Kaggle dataset goes here
├── outputs/
│   ├── eda/                       # EDA charts (PNG)
│   ├── models/                    # Saved classifier (.joblib)
│   └── reports/                   # Classification reports + confusion matrix
├── src/
│   ├── __init__.py
│   ├── config.py                  # Paths, image size, model hyperparams
│   ├── main.py                    # GUI entry point: python -m src.main
│   ├── main_classifier.py         # Standalone training pipeline (Member 2)
│   ├── app.py                     # Tkinter GUI (Member 3)
│   ├── models/
│   │   └── __init__.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── dataset_indexer.py     # Member 1
│   │   ├── eda_service.py         # Member 1
│   │   ├── image_preprocessor.py  # Member 2
│   │   ├── classifier_service.py  # Member 2
│   │   ├── evaluation_utils.py    # Member 2
│   │   └── workflow_service.py    # Member 3
│   └── utils/
│       └── __init__.py
├── MANUAL_TESTING.md
├── README.md
└── requirements.txt
```

---

## Installation

The project targets Python 3.10+.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt
```

If your platform does not ship Tkinter with Python by default, install it
through your system package manager (for example
`sudo apt install python3-tk` on Debian/Ubuntu).

---

## Dataset

This project uses the
[Kaggle Stream Macroinvertebrates dataset](https://www.kaggle.com/datasets/kennethtm/stream-macroinvertebrates).

After downloading:

1. Extract the dataset into `data/raw/` so that each class has its own
   subfolder, e.g. `data/raw/Baetidae/img_001.jpg`.
2. Confirm that `data/raw/` exists and contains at least one supported
   image (`.jpg`, `.jpeg`, `.png`, `.bmp`).

The `data/raw/` folder is excluded from git in `.gitignore`.

---

## How to run

### 1. Training pipeline (Stage 1 + Stage 2, no GUI)

```bash
python -m src.main_classifier
```

Scans the dataset, prints a summary, trains the classifier, and writes
`outputs/models/macro_classifier.joblib` plus the report files under
`outputs/reports/`.

### 2. Deployed GUI (Stage 3)

```bash
python -m src.main
```

Opens the desktop application. The GUI has three tabs:

- **Predict** — browse for a single image, see the predicted class and
  confidence. Friendly errors fire if no model is found or the file is
  unreadable.
- **EDA Charts** — preview any PNG sitting in `outputs/eda/`. If the
  folder is empty, click *Generate EDA charts* to produce them from the
  raw dataset.
- **Training** — show a quick dataset summary (image counts per class,
  mean dimensions) and trigger a baseline training run that saves a model
  to `outputs/models/`. The training thread runs off the UI thread so the
  window stays responsive.

---

## Work division

| Member   | Main responsibility                                                                 |
| -------- | ----------------------------------------------------------------------------------- |
| Member 1 | Dataset indexer, image record model, Stage 1 EDA service and charts                  |
| Member 2 | Image preprocessor, classifier service, evaluation utilities, training pipeline      |
| Member 3 | `WorkflowService`, Tkinter GUI (`app.py`), entry point (`main.py`), README, testing  |

All members understand the full system. See *Implementation Summary* for
the per-member detail.

---

## Acknowledgements / reused code

- The folder structure, suggested class names, and the OpenCV
  preprocessing approach follow the layout described in *Assignment 3
  Full Guidance and Coding Examples* (course-provided materials). All
  code in this repository was written by the group; the guidance was
  used as a structural reference, not copy-pasted.
- The classifier service follows the standard
  `scikit-learn` Random Forest pattern from the unit's Week 5–7
  materials.
- Tkinter usage in `src/app.py` is based on the Python standard library
  documentation and the unit's GUI lab notes; no external GUI examples
  were copied.

---

## Testing

Manual test scenarios and their outcomes are recorded in
[`MANUAL_TESTING.md`](MANUAL_TESTING.md). The deployed GUI handles
missing dataset folders, missing model artifacts, unreadable image
files, and unsupported file types with user-friendly dialogs rather than
stack traces.

---

## Limitations and possible extensions

- The baseline preprocessor flattens 128×128 grayscale into a single
  feature vector. Convolutional features (e.g. MobileNetV2 transfer
  learning from Week 7/8) would likely lift accuracy meaningfully.
- The GUI loads charts as flat PNGs. A future revision could render
  Matplotlib figures directly into a Tkinter canvas for interactivity.
- Only single-image prediction is supported in the GUI. Batch prediction
  would be a small addition on top of `WorkflowService.predict_image`.
