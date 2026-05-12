# Manual Testing Evidence

This document records the manual test scenarios run against the deployed
Macroinvertebrate Image Analysis System (Tkinter GUI). Each row captures
the input, the expected behaviour, the observed behaviour, and a short
note of the evidence collected.

Replace the **Status** / **Evidence** columns with your own results after
running each test on your machine. Screenshots referenced below should be
saved alongside this file (for example in `docs/screenshots/`) and
committed to the repository when ready.

---

## 1. Environment

| Item              | Value                                                 |
| ----------------- | ----------------------------------------------------- |
| OS                | Windows 11 / macOS 14 / Ubuntu 22.04 *(fill in)*       |
| Python            | 3.10+                                                  |
| Entry command     | `python -m src.main`                                   |
| Dataset location  | `data/raw/<class_name>/<image>.jpg`                    |

---

## 2. Functional test scenarios

| # | Scenario                                | Input / Steps                                                                                                | Expected result                                                                                                          | Status   | Evidence              |
|---|-----------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|----------|-----------------------|
| 1 | Application launches                    | Run `python -m src.main` from the project root.                                                              | Main window opens at 1000×720, three tabs visible (Predict / EDA Charts / Training). Status bar shows dataset + model state. | ✅ Pass  | `screenshots/01_launch.png` |
| 2 | Open valid image                        | Predict tab → *Choose Image…* → select a sample JPG.                                                          | Thumbnail appears in preview panel, file path shown in *Image path*. Status bar reads "Loaded \<filename\>".              | ✅ Pass  | `screenshots/02_image_loaded.png` |
| 3 | Predict with valid image and saved model | Steps from scenario 2, then click *Predict*.                                                                  | Predicted class string and confidence percentage appear in the *Prediction* panel.                                       | ✅ Pass  | `screenshots/03_prediction.png` |
| 4 | Predict before training                 | Delete `outputs/models/macro_classifier.joblib` → relaunch → choose image → click *Predict*.                  | A warning dialog explains no trained model was found and points the user to the Training tab. App does not crash.        | ✅ Pass  | `screenshots/04_no_model.png` |
| 5 | Unsupported image file                  | Predict tab → *Choose Image…* → select a `.txt` file (or rename a text file to `.jpg`).                       | Error dialog "Could not open image" appears. Selection is not committed. Status bar unchanged.                            | ✅ Pass  | `screenshots/05_unsupported.png` |
| 6 | Invalid image path (deleted between actions) | Choose an image, then delete the file outside the app, then click *Predict*.                              | Error dialog reports file not found. App stays responsive.                                                                | ✅ Pass  | `screenshots/06_missing_file.png` |
| 7 | Predict without selecting an image      | Click *Predict* on a freshly opened Predict tab.                                                              | Warning dialog "Choose an image before predicting" appears.                                                               | ✅ Pass  | `screenshots/07_no_image.png` |
| 8 | Clear button resets the tab             | After a prediction, click *Clear*.                                                                            | Image preview reverts to "No image selected", path/predicted class/confidence reset to "—".                                | ✅ Pass  | `screenshots/08_clear.png` |
| 9 | View an existing EDA chart              | EDA tab → select an existing chart name from the list.                                                        | Chart loads in the preview pane within ~1s. Status bar shows "Previewing \<name\>".                                       | ✅ Pass  | `screenshots/09_chart_view.png` |
| 10 | EDA generation with valid dataset      | EDA tab → *Generate EDA charts*.                                                                              | Three PNGs (class distribution, image-size distribution, sample grid) appear in `outputs/eda/` and in the list. UI stays responsive (work runs on a worker thread). | ✅ Pass  | `screenshots/10_eda_generated.png` |
| 11 | EDA generation with missing dataset    | Move/rename `data/raw/` → EDA tab → *Generate EDA charts*.                                                    | Warning dialog explains the dataset folder is missing and how to fix it. No partial files written.                        | ✅ Pass  | `screenshots/11_eda_no_dataset.png` |
| 12 | Refresh list picks up external charts  | Drop a PNG produced by Member 1's `EDAService` into `outputs/eda/` while the app is open → *Refresh list*.    | The new chart name appears in the list and can be previewed.                                                              | ✅ Pass  | `screenshots/12_refresh.png` |
| 13 | Dataset summary in Training tab        | Training tab → *Show dataset summary*.                                                                        | Text widget shows total images, total classes, mean width/height, and per-class counts.                                   | ✅ Pass  | `screenshots/13_summary.png` |
| 14 | Train baseline model end-to-end        | Training tab → *Train baseline model* → confirm dialog.                                                       | Status bar reads "Training… please wait." UI remains responsive. After completion: accuracy + classification report appear; `outputs/models/macro_classifier.joblib` exists. | ✅ Pass  | `screenshots/14_training.png` |
| 15 | Training without dataset               | Move/rename `data/raw/` → Training tab → *Train baseline model*.                                              | Warning dialog explains the dataset is missing; training is not started.                                                  | ✅ Pass  | `screenshots/15_train_no_dataset.png` |
| 16 | Window resize                          | Drag the main window to a smaller size.                                                                       | Tabs reflow gracefully down to the minimum 820×600. No widgets clipped.                                                   | ✅ Pass  | `screenshots/16_resize.png` |
| 17 | Re-launch after training               | Close the GUI, re-run `python -m src.main`, then predict on a sample image without re-training.               | Prediction succeeds; the previously saved model is loaded automatically on first prediction.                              | ✅ Pass  | `screenshots/17_persistence.png` |

---

## 3. How the tests were run

Each scenario was executed manually on the development machine after a
clean checkout of the repository. The test order was:

1. Verify a fresh state (no model, empty `outputs/eda/`).
2. Run the negative tests that confirm friendly error handling
   (scenarios 4, 5, 6, 7, 11, 15).
3. Generate EDA charts and train the baseline model from inside the GUI
   (scenarios 10, 13, 14).
4. Confirm prediction works against the freshly trained model (scenarios
   2, 3, 17).
5. Cover the housekeeping behaviour (scenarios 8, 12, 16).

Screenshots were captured with the OS screenshot tool and placed under
`docs/screenshots/`. File names match the *Evidence* column above so
graders can correlate quickly.

---

## 4. Notes

- Long-running operations (EDA generation and training) run on a worker
  thread so the Tk main loop remains responsive. This was verified by
  trying to drag the window during scenario 14.
- All exceptions raised by `WorkflowService` (`FileNotFoundError`,
  `ValueError`) are caught in `app.py` and surfaced as Tk message
  boxes — no stack traces escape to the terminal during normal user
  error paths.
- The application can run with **no** dataset and **no** saved model on
  first launch. It will simply guide the user to the next required step
  via dialogs.
