# Week 13 Presentation — Member 3 Talking Points

Target time for Member 3's segments: roughly 3–4 minutes total, sitting
inside the group's overall 5–7 minute presentation. Adjust to your
tutor's exact instructions on the day.

---

## 1. Project introduction (~45 seconds)

Use this script verbatim if you want, or paraphrase. It opens the demo
and frames everything that follows.

> Our project is a Macroinvertebrate Image Analysis System built in
> Python. Macroinvertebrates are small aquatic organisms used in water
> quality studies, and the goal of the system is to take an image of one
> and tell the user which class it belongs to. We built it as three
> connected stages: dataset exploration, a baseline classifier, and a
> desktop application that wraps both. I'll walk through the deployed
> app today.

Beat to land: *"three connected stages — dataset exploration, a
classifier, and a desktop app — and I'll demo the deployed app."*

---

## 2. Live demonstration flow (~2 minutes)

Have these ready **before** the tutorial starts:

* `.venv` activated, dependencies installed.
* `data/raw/` populated.
* `outputs/models/macro_classifier.joblib` already produced (so the
  predict path is instant).
* One or two known-good sample images on the desktop for the demo.

Steps to run, narrating as you click:

1. **Open the app** with `python -m src.main`. Point out the status bar
   — it already tells you whether the dataset and model are present.
2. **Predict tab → Choose Image** → pick a sample. Note that the file
   path appears under the preview.
3. **Click Predict.** Call out the predicted class and the confidence
   percentage. Mention briefly that the model is a Random Forest trained
   on flattened 128×128 grayscale features and that the saved
   `.joblib` is loaded lazily on the first prediction.
4. **Switch to the EDA tab.** Click an existing chart (class distribution
   is the most informative one). One-line takeaway: *"this is the chart
   that informed our test/train split — the dataset is imbalanced, so
   we use `class_weight='balanced'` and a stratified split."*
5. **Switch to the Training tab.** Click *Show dataset summary*. Read
   off total images, total classes, mean dimensions. (Do **not** run a
   full training during the demo unless you've timed it; mention you
   ran it ahead of time and the result is the model the predict tab
   uses.)
6. **Resize the window** to show the UI reflows. Drag it during a
   non-training moment so judges can see it is genuinely responsive.

Safety tips during the demo:

* If something goes wrong on a click, *don't restart from scratch* —
  the GUI raises a dialog, click *OK*, and continue from the next step.
* Keep an unselected sample image ready in case the first one looks
  unclear.

---

## 3. Design and packages (~1 minute)

This is what you say while the GUI is on screen, not what you click. Hit
all four points:

* **Architecture.** "The GUI is a thin layer over a single
  `WorkflowService` class. The Tkinter app does not own any business
  logic — it only calls methods on the service and renders the
  results. That made the app easy to test and means our code can be
  reused by a console or a notebook."
* **OOP.** "Each responsibility is a separate class:
  `DatasetIndexer` for scanning, `ImagePreprocessor` for image-to-feature
  transformation, `ClassifierService` for training and prediction,
  `WorkflowService` for coordination, and `MacroApp` for the UI. That
  is the OO design we wanted to demonstrate — one class, one job."
* **Packages.** "OpenCV reads and resizes images, NumPy handles the
  feature arrays, Pandas powers the dataset index, Matplotlib and
  Seaborn produce the EDA charts, scikit-learn trains the Random
  Forest, joblib persists the model, Pillow handles the previews in
  Tkinter, and Tkinter ships the desktop application."
* **Why a self-contained workflow.** "The deployed app has its own
  small indexer and preprocessor inside `workflow_service.py`. That
  meant my part stayed runnable even while the rest of the team was
  still building theirs, and it makes the app robust on a clean
  checkout — if the EDA folder is empty, the app can generate the
  charts itself."

---

## 4. Testing approach (~30 seconds)

* "We used manual testing rather than a formal framework — appropriate
  for a single-user desktop app of this size."
* "Scenarios are recorded in `MANUAL_TESTING.md` with status and a
  screenshot reference for each."
* "We deliberately covered the failure paths: no dataset, no trained
  model, unsupported file type, predict before training, and missing
  files between selection and prediction. Each one shows a friendly
  dialog, not a stack trace."
* "Long-running operations (training and EDA generation) run on a
  worker thread so the UI stays interactive — we verified that by
  dragging the window during training."

---

## 5. Work division (~30 seconds)

> The work was split into three deployable slices. Member 1 owned the
> dataset indexer, the image record class, and the Stage 1 EDA outputs.
> Member 2 owned the preprocessor, the Random Forest classifier service,
> and the standalone training pipeline. I owned the `WorkflowService`,
> the Tkinter GUI, the entry point, the README, and the manual testing
> evidence. We coordinated by sharing the *artifacts* — the saved
> charts and the saved model — rather than by depending on each
> other's source files, so each member could work independently.

---

## 6. Likely tutor questions — quick answers

| Question                                                            | Answer cue                                                                                                                                                          |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Why a Random Forest and not a CNN?                                  | Baseline that's easy to explain and trains in seconds on a CPU; we mention transfer learning (MobileNetV2) as a documented extension in the README.                 |
| How is the model loaded by the GUI?                                 | `WorkflowService._ensure_model_loaded` lazy-loads `outputs/models/macro_classifier.joblib` via `joblib.load` on the first prediction.                                |
| What happens if the user predicts before training?                  | The GUI checks `workflow.is_model_available()` and shows a warning dialog that points the user to the Training tab — no crash, no traceback.                        |
| Why duplicate the indexer/preprocessor in `workflow_service.py`?    | Decoupling: the deployed app stays runnable even when Member 1's `EDAService` and `DatasetIndexer` are still in progress. The shape contract is the same.            |
| How did you avoid blocking the UI during training?                  | Training runs on a `threading.Thread`; result is marshalled back into the Tk main loop via `self.after(0, …)`.                                                       |
| Where is OOP demonstrated?                                          | Each class has one responsibility: `DatasetIndexer`, `EDAService`, `ImagePreprocessor`, `ClassifierService`, `WorkflowService`, `MacroApp` — single-responsibility throughout. |
| Can the app run with no dataset?                                    | Yes — predict still works if a saved model exists. The Training and EDA tabs show clear dialogs and refuse to start. The app never crashes.                          |

---

## 7. Backup plan

If the GUI fails to launch on the demo machine (display drivers /
remote desktop / Tk missing), fall back to:

* `python -m src.main_classifier` — runs Stage 1 + Stage 2 in the
  terminal and shows the saved model path.
* Open the saved EDA PNGs and the confusion matrix PNG directly in an
  image viewer to demonstrate the outputs.

Mention this only if Plan A fails — don't preempt yourself into a
weaker demo.
