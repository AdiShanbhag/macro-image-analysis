# Implementation Summary — Member 3 sections

This file is a drop-in for the group's combined Implementation Summary.
It covers the sections owned by Member 3 (Deployed Application). Member 1
and Member 2 paste their equivalent sections around these.

---

## Deployment design decisions

The deployed application is a Tkinter desktop app launched by
`python -m src.main`. The design follows three guiding decisions.

**1. The GUI is a thin layer over a single coordination service.**
`src/services/workflow_service.py` exposes one `WorkflowService` class
with all the verbs the app needs: `show_summary`, `generate_eda`,
`train_model`, `predict_image`, `list_eda_charts`,
`is_dataset_available`, and `is_model_available`. The Tkinter widgets in
`src/app.py` only know how to call these methods and render their
results. This means the same workflow can be reused by a console app or
a notebook with no extra code, and lets us unit test the logic without
needing a display.

**2. Self-contained indexer and preprocessor.**
`WorkflowService` ships its own small `_LocalDatasetIndexer` and
`_LocalImagePreprocessor` rather than importing Member 1's or Member 2's
service modules. This decoupling matters in practice: the deployed app
keeps working through every development phase, including the early phase
when Member 1's `EDAService` and `DatasetIndexer` had not yet been
pushed to the shared repository. The preprocessing pipeline is
deliberately identical to Member 2's (128×128 grayscale, normalised,
flattened), so a model trained by `main_classifier.py` is fully
compatible with `predict_image` in the GUI.

**3. The app degrades gracefully.**
The app must work on a fresh checkout where nothing is built yet. If
`outputs/eda/` is empty, the user can press *Generate EDA charts*. If
`outputs/models/` has no `.joblib` file, the Training tab can produce
one. If `data/raw/` is missing, every action that needs it surfaces a
clear dialog rather than a Python traceback. Conversely, if Member 1 has
already produced EDA PNGs, the EDA tab picks them up automatically by
scanning the directory.

## Interface design

The Tk window is built around a three-tab `ttk.Notebook`, chosen so a
marker can step through the entire project lifecycle in the live demo
without leaving the application.

* **Predict tab.** A *Choose Image…* / *Predict* / *Clear* control row,
  a large image preview frame, and a small results panel showing the
  predicted class, confidence percentage, and the original file path.
  This is the demo-critical screen, so it is intentionally simple.
* **EDA Charts tab.** A scrollable listbox of chart filenames on the
  left and a preview pane on the right. A *Generate EDA charts* button
  produces the standard chart set on the fly when the folder is empty,
  and *Refresh list* picks up charts created externally by Member 1's
  service.
* **Training tab.** *Show dataset summary* dumps per-class counts and
  mean dimensions into a read-only `Text` widget; *Train baseline model*
  confirms with the user, then runs Member 3's training routine on a
  worker thread so the UI stays responsive.

A persistent status bar at the bottom of the window reports the result
of the last action. Long-running operations (EDA generation and
training) execute on `threading.Thread` workers and marshal their UI
updates back to the Tk main loop with `self.after(0, …)` — this is the
standard pattern for keeping Tk responsive without resorting to async
frameworks.

Error paths are funnelled through `tkinter.messagebox` so the user
never sees a traceback during normal interaction. Specifically, the GUI
catches `FileNotFoundError` and `ValueError` from `WorkflowService`
explicitly, and falls back to a generic error dialog for anything
unexpected.

## Testing summary and evidence

Manual testing was performed against the GUI on a clean checkout. The
full scenario table lives in `MANUAL_TESTING.md`; the highlights are:

* **Happy path.** Launch → choose an image → predict → see class +
  confidence (scenarios 1, 2, 3 in `MANUAL_TESTING.md`).
* **Negative paths.** Predict before training (scenario 4), pick an
  unsupported file (5), delete the chosen file between selection and
  prediction (6), generate EDA with no dataset (11), train with no
  dataset (15). Every one of these surfaces a friendly dialog, not a
  traceback.
* **Concurrency.** Training and EDA generation were confirmed to keep
  the window responsive (drag-test during scenario 14).
* **Persistence.** After training inside the GUI and closing the
  window, the next launch loads the saved model automatically on first
  prediction (scenario 17).

Screenshots are stored under `docs/screenshots/` with filenames matching
the *Evidence* column of `MANUAL_TESTING.md`.

## Work division summary

| Member   | Files owned                                                                                                          |
| -------- | -------------------------------------------------------------------------------------------------------------------- |
| Member 1 | `src/services/dataset_indexer.py`, `src/services/eda_service.py`, `src/models/records.py`                            |
| Member 2 | `src/services/image_preprocessor.py`, `src/services/classifier_service.py`, `src/services/evaluation_utils.py`, `src/main_classifier.py` |
| Member 3 | `src/services/workflow_service.py`, `src/app.py`, `src/main.py`, `README.md`, `MANUAL_TESTING.md`                    |

The team agreed up front that each member would own a deployable slice
end-to-end:

* Member 1 produced Stage 1 EDA outputs that can be inspected on disk.
* Member 2 produced a trained model artifact that can be loaded by
  anything that imports `joblib`.
* Member 3 wrapped both artifact sets in a desktop application and made
  sure the app could *also* regenerate them itself if either side was
  not yet finished.

Co-ordination was handled by sharing the artifacts (PNG charts and
`.joblib` files), not by mutual code imports. That kept Member 3's part
unblocked during the early phase of the assignment.

## Code acknowledgements

* Folder layout, suggested class responsibilities, and the OpenCV +
  scikit-learn baseline pipeline follow the structural guidance in
  *Assignment 3 Full Guidance and Coding Examples* (course material).
  No code was copied; the document was used as a structural reference.
* The Tkinter usage in `src/app.py` follows the Python standard library
  documentation and the unit's GUI lab notes. Image previews use Pillow
  (`PIL.ImageTk`) as introduced in the unit's image-handling labs.
* The Random Forest pipeline used in `WorkflowService.train_model`
  mirrors the standard `scikit-learn` pattern from the unit's Week 5–7
  materials, kept self-contained so the deployed app does not depend on
  Member 2's service file.
* Threaded long-running operations use the standard
  `threading.Thread` + `self.after(0, …)` pattern documented in the
  Python Tkinter docs.
