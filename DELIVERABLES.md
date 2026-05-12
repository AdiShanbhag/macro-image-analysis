# Member 3 — Deliverables index

Everything in this folder is Member 3's work for Assignment 3. Drop the
files into your local clone of `AdiShanbhag/macro-image-analysis` at the
paths shown below and commit them on a feature branch (e.g.
`feature/member3-deployed-app`) per the team's git workflow guide.

| File in this folder                             | Goes in the repo at                                | Purpose                                                              |
| ----------------------------------------------- | -------------------------------------------------- | -------------------------------------------------------------------- |
| `src/services/workflow_service.py`              | `src/services/workflow_service.py`                 | Self-contained coordinator: indexer + preprocessor + EDA + train + predict |
| `src/app.py`                                    | `src/app.py`                                       | Tkinter GUI with Predict / EDA / Training tabs                       |
| `src/main.py`                                   | `src/main.py`                                      | Entry point: `python -m src.main`                                    |
| `README.md`                                     | `README.md` (overwrite current empty file)         | Full project README                                                  |
| `MANUAL_TESTING.md`                             | `MANUAL_TESTING.md` (overwrite current empty file) | Manual test scenarios + outcomes                                     |
| `docs/Implementation_Summary_Member3.md`        | Paste into the group's combined Implementation Summary | Member 3's sections (deployment design, interface, testing, work division, acknowledgements) |
| `docs/Presentation_Talking_Points_Member3.md`   | Keep locally — do not commit unless the group wants it | Cue card for Week 13 demo                                            |

## Verified

- All Python files compile (`py_compile`) and import cleanly.
- End-to-end smoke test against a synthetic 3-class dataset: EDA charts
  generated, model trained and saved, prediction succeeded, both
  `FileNotFoundError` and `ValueError` error paths handled gracefully.

## Notes on integration

- The deployed app does **not** import Member 1's `dataset_indexer.py`
  or `eda_service.py`, nor Member 2's `image_preprocessor.py` /
  `classifier_service.py`. It uses self-contained equivalents inside
  `workflow_service.py`. This was an explicit Member 3 design decision
  to keep the GUI runnable while the rest of the team's modules are in
  flight.
- It *does* read whatever artifacts those members produce in
  `outputs/eda/` and `outputs/models/`. So when Member 1 ships their
  charts and Member 2 ships their trained model, the GUI picks them up
  with no code change.
- The preprocessing pipeline inside `workflow_service.py` is
  intentionally identical to Member 2's `ImagePreprocessor` (128×128
  grayscale, normalised, flattened). Member 2's trained
  `.joblib` model is therefore predict-compatible.

## Suggested git workflow for committing this

```bash
cd path/to/your/local/macro-image-analysis
git checkout main
git pull origin main
git checkout -b feature/member3-deployed-app

# Copy the files in
# (adjust paths to wherever you've downloaded this folder)

git add src/services/workflow_service.py src/app.py src/main.py \
        README.md MANUAL_TESTING.md
git commit -m "Add deployed Tkinter app, workflow service, README, manual testing (M3)"
git push -u origin feature/member3-deployed-app
```

Then either merge to `main` directly or open a pull request, depending
on what the group agreed.

## requirements.txt addition

`Pillow` is already pinned in the current `requirements.txt`, so no
edits required. The Tkinter library ships with the Python standard
library on Windows/macOS; on Linux distros that split it out, install
`python3-tk` via the system package manager.
