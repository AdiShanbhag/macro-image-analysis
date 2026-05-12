"""
app.py

Member 3 — Deployed Application (Tkinter GUI).

A small desktop interface for the Macroinvertebrate Image Analysis System.
The GUI is organised as three tabs so a marker can step through the project
end-to-end during the Week 13 demo:

    1. Predict  — browse for an image, run the saved classifier, show class
                  and confidence.
    2. EDA      — view any chart Member 1 produced in outputs/eda/, or have
                  the deployed app generate them on the fly.
    3. Training — show a dataset summary and trigger a training run that
                  saves a model to outputs/models/.

All real work is delegated to ``WorkflowService`` — this module deliberately
stays UI-only so the logic can be tested independently.
"""

from __future__ import annotations

import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from src.services.workflow_service import WorkflowService


# Tk requires images to be kept alive by a reference; we attach them as
# attributes on widgets to avoid garbage collection.

class MacroApp(tk.Tk):
    """Tkinter desktop GUI for the Macroinvertebrate Image Analysis System."""

    APP_TITLE = "Macroinvertebrate Image Analysis System"
    WINDOW_SIZE = "1000x720"

    def __init__(self, workflow: WorkflowService | None = None) -> None:
        super().__init__()
        self.title(self.APP_TITLE)
        self.geometry(self.WINDOW_SIZE)
        self.minsize(820, 600)

        self.workflow = workflow or WorkflowService()
        self.selected_image_path: Path | None = None

        self._build_layout()
        self._refresh_status_bar()
        self._refresh_chart_list()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Compose the notebook of tabs and the persistent status bar."""
        style = ttk.Style(self)
        # Use whichever modern theme the OS provides; fall back silently.
        for theme in ("vista", "clam", "alt", "default"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break
        style.configure("Heading.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Result.TLabel", font=("Segoe UI", 13))
        style.configure("Mono.TLabel", font=("Consolas", 10))

        # Header
        header = ttk.Frame(self, padding=(16, 12, 16, 6))
        header.pack(fill="x")
        ttk.Label(header, text=self.APP_TITLE, style="Heading.TLabel").pack(
            side="left"
        )

        # Tabs
        notebook = ttk.Notebook(self)
        notebook.pack(expand=True, fill="both", padx=12, pady=(0, 6))

        self._build_predict_tab(notebook)
        self._build_eda_tab(notebook)
        self._build_training_tab(notebook)

        # Status bar
        self.status_var = tk.StringVar(value="Ready.")
        status_bar = ttk.Frame(self, relief="sunken", padding=(10, 4))
        status_bar.pack(fill="x", side="bottom")
        ttk.Label(status_bar, textvariable=self.status_var).pack(side="left")

    # ---------- Predict tab -------------------------------------------------

    def _build_predict_tab(self, parent: ttk.Notebook) -> None:
        tab = ttk.Frame(parent, padding=14)
        parent.add(tab, text="Predict")

        # Controls row
        controls = ttk.Frame(tab)
        controls.pack(fill="x", pady=(0, 10))

        ttk.Button(
            controls, text="Choose Image…", command=self._on_choose_image
        ).pack(side="left")
        ttk.Button(
            controls, text="Predict", command=self._on_predict_clicked
        ).pack(side="left", padx=8)
        ttk.Button(
            controls, text="Clear", command=self._on_clear_clicked
        ).pack(side="left")

        # Image preview
        preview_frame = ttk.LabelFrame(tab, text="Selected image", padding=10)
        preview_frame.pack(fill="both", expand=True, pady=(0, 10))
        self.image_label = ttk.Label(preview_frame, text="No image selected.")
        self.image_label.pack(expand=True)

        # Result panel
        result_frame = ttk.LabelFrame(tab, text="Prediction", padding=10)
        result_frame.pack(fill="x")
        self.predicted_class_var = tk.StringVar(value="—")
        self.confidence_var = tk.StringVar(value="—")
        self.path_var = tk.StringVar(value="—")

        for row, (label, var) in enumerate(
            [
                ("Predicted class:", self.predicted_class_var),
                ("Confidence:", self.confidence_var),
                ("Image path:", self.path_var),
            ]
        ):
            ttk.Label(result_frame, text=label, width=18).grid(
                row=row, column=0, sticky="w", padx=(0, 8), pady=2
            )
            ttk.Label(
                result_frame, textvariable=var, style="Result.TLabel"
            ).grid(row=row, column=1, sticky="w", pady=2)

    # ---------- EDA tab -----------------------------------------------------

    def _build_eda_tab(self, parent: ttk.Notebook) -> None:
        tab = ttk.Frame(parent, padding=14)
        parent.add(tab, text="EDA Charts")

        controls = ttk.Frame(tab)
        controls.pack(fill="x", pady=(0, 10))

        ttk.Button(
            controls,
            text="Generate EDA charts",
            command=self._on_generate_eda_clicked,
        ).pack(side="left")
        ttk.Button(
            controls, text="Refresh list", command=self._refresh_chart_list
        ).pack(side="left", padx=8)

        # Chart picker on the left, preview on the right.
        body = ttk.Frame(tab)
        body.pack(fill="both", expand=True)

        list_frame = ttk.LabelFrame(body, text="Available charts", padding=8)
        list_frame.pack(side="left", fill="y", padx=(0, 10))

        self.chart_listbox = tk.Listbox(list_frame, width=28, height=18)
        self.chart_listbox.pack(side="left", fill="y")
        scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.chart_listbox.yview
        )
        scrollbar.pack(side="right", fill="y")
        self.chart_listbox.configure(yscrollcommand=scrollbar.set)
        self.chart_listbox.bind("<<ListboxSelect>>", self._on_chart_selected)

        preview_frame = ttk.LabelFrame(body, text="Preview", padding=8)
        preview_frame.pack(side="right", fill="both", expand=True)
        self.chart_preview = ttk.Label(
            preview_frame,
            text=(
                "No chart selected.\n\n"
                "Click ‘Generate EDA charts’ if outputs/eda/ is empty."
            ),
            justify="center",
        )
        self.chart_preview.pack(expand=True)

    # ---------- Training tab -----------------------------------------------

    def _build_training_tab(self, parent: ttk.Notebook) -> None:
        tab = ttk.Frame(parent, padding=14)
        parent.add(tab, text="Training")

        controls = ttk.Frame(tab)
        controls.pack(fill="x", pady=(0, 10))

        ttk.Button(
            controls,
            text="Show dataset summary",
            command=self._on_show_summary_clicked,
        ).pack(side="left")
        ttk.Button(
            controls,
            text="Train baseline model",
            command=self._on_train_clicked,
        ).pack(side="left", padx=8)

        self.training_output = tk.Text(
            tab, wrap="word", height=24, state="disabled", font=("Consolas", 10)
        )
        self.training_output.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Event handlers — Predict
    # ------------------------------------------------------------------

    def _on_choose_image(self) -> None:
        """Open a file dialog and show a thumbnail of the chosen image."""
        file_path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            pil_image = Image.open(file_path)
            pil_image.thumbnail((420, 420))
            photo = ImageTk.PhotoImage(pil_image)
        except Exception as exc:
            messagebox.showerror(
                "Could not open image",
                f"The selected file could not be opened as an image.\n\n{exc}",
            )
            return

        self.selected_image_path = Path(file_path)
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo  # keep a reference
        self.path_var.set(str(self.selected_image_path))
        self.predicted_class_var.set("—")
        self.confidence_var.set("—")
        self._set_status(f"Loaded {self.selected_image_path.name}")

    def _on_predict_clicked(self) -> None:
        """Run the saved classifier on the currently selected image."""
        if not self.selected_image_path:
            messagebox.showwarning(
                "No image selected", "Choose an image before predicting."
            )
            return

        if not self.workflow.is_model_available():
            messagebox.showwarning(
                "No trained model",
                (
                    "No saved model was found at\n"
                    f"{self.workflow.model_path}\n\n"
                    "Go to the Training tab and click ‘Train baseline model’ "
                    "first, or copy a model artifact into outputs/models/."
                ),
            )
            return

        self._set_status("Predicting…")
        try:
            result = self.workflow.predict_image(self.selected_image_path)
        except FileNotFoundError as exc:
            messagebox.showerror("File not found", str(exc))
            self._set_status("Prediction failed.")
            return
        except ValueError as exc:
            messagebox.showerror("Unsupported image", str(exc))
            self._set_status("Prediction failed.")
            return
        except Exception as exc:
            messagebox.showerror(
                "Prediction error",
                f"Something went wrong while predicting:\n\n{exc}",
            )
            self._set_status("Prediction failed.")
            return

        self.predicted_class_var.set(result["predicted_class"])
        self.confidence_var.set(f"{result['confidence']:.1%}")
        self._set_status("Prediction complete.")

    def _on_clear_clicked(self) -> None:
        """Reset the predict tab to its initial state."""
        self.selected_image_path = None
        self.image_label.configure(image="", text="No image selected.")
        self.image_label.image = None
        self.predicted_class_var.set("—")
        self.confidence_var.set("—")
        self.path_var.set("—")
        self._set_status("Cleared.")

    # ------------------------------------------------------------------
    # Event handlers — EDA
    # ------------------------------------------------------------------

    def _on_generate_eda_clicked(self) -> None:
        """Kick off EDA generation on a worker thread so the UI stays responsive."""
        if not self.workflow.is_dataset_available():
            messagebox.showwarning(
                "Dataset not available",
                (
                    "No images were found in data/raw/.\n\n"
                    "Place the extracted Kaggle dataset there so each class "
                    "has its own subfolder, then try again."
                ),
            )
            return

        self._set_status("Generating EDA charts… (this may take a moment)")
        thread = threading.Thread(target=self._run_eda_generation, daemon=True)
        thread.start()

    def _run_eda_generation(self) -> None:
        try:
            produced = self.workflow.generate_eda()
        except (FileNotFoundError, ValueError) as exc:
            self.after(0, lambda: messagebox.showerror("EDA failed", str(exc)))
            self.after(0, lambda: self._set_status("EDA failed."))
            return
        except Exception as exc:
            tb = traceback.format_exc()
            self.after(
                0,
                lambda: messagebox.showerror(
                    "EDA failed",
                    f"Unexpected error while generating EDA charts:\n\n{exc}\n\n{tb}",
                ),
            )
            self.after(0, lambda: self._set_status("EDA failed."))
            return

        def on_done() -> None:
            self._refresh_chart_list()
            self._set_status(f"Generated {len(produced)} EDA chart(s).")

        self.after(0, on_done)

    def _refresh_chart_list(self) -> None:
        """Reload the chart list from outputs/eda/."""
        self.chart_listbox.delete(0, tk.END)
        for path in self.workflow.list_eda_charts():
            self.chart_listbox.insert(tk.END, path.name)

    def _on_chart_selected(self, _event) -> None:
        """Display the selected chart in the preview pane."""
        selection = self.chart_listbox.curselection()
        if not selection:
            return
        name = self.chart_listbox.get(selection[0])
        chart_path = self.workflow.eda_output_dir / name

        if not chart_path.exists():
            messagebox.showwarning(
                "Chart missing",
                f"The chart file {chart_path} no longer exists.",
            )
            self._refresh_chart_list()
            return

        try:
            pil_image = Image.open(chart_path)
            pil_image.thumbnail((640, 520))
            photo = ImageTk.PhotoImage(pil_image)
        except Exception as exc:
            messagebox.showerror(
                "Could not open chart", f"Could not display chart:\n\n{exc}"
            )
            return

        self.chart_preview.configure(image=photo, text="")
        self.chart_preview.image = photo
        self._set_status(f"Previewing {name}.")

    # ------------------------------------------------------------------
    # Event handlers — Training
    # ------------------------------------------------------------------

    def _on_show_summary_clicked(self) -> None:
        try:
            summary = self.workflow.show_summary()
        except (FileNotFoundError, ValueError) as exc:
            messagebox.showerror("Dataset error", str(exc))
            return

        lines = [
            f"Total images : {summary['total_images']}",
            f"Total classes: {summary['total_classes']}",
            f"Mean width   : {summary['mean_width']:.1f}",
            f"Mean height  : {summary['mean_height']:.1f}",
            "",
            "Images per class:",
        ]
        for label, count in summary["images_per_class"].items():
            lines.append(f"  {label:<30s} {count}")
        self._write_training_output("\n".join(lines))
        self._set_status("Dataset summary loaded.")

    def _on_train_clicked(self) -> None:
        if not self.workflow.is_dataset_available():
            messagebox.showwarning(
                "Dataset not available",
                (
                    "No images were found in data/raw/.\n\n"
                    "Add the Kaggle dataset before training."
                ),
            )
            return

        if not messagebox.askyesno(
            "Confirm training",
            (
                "Training the baseline model can take a minute or two "
                "depending on dataset size. Continue?"
            ),
        ):
            return

        self._set_status("Training… please wait.")
        self._write_training_output(
            "Training in progress — the window will stay responsive.\n"
            "Results will appear here when training completes.\n"
        )
        thread = threading.Thread(target=self._run_training, daemon=True)
        thread.start()

    def _run_training(self) -> None:
        try:
            results = self.workflow.train_model()
        except Exception as exc:
            tb = traceback.format_exc()
            self.after(
                0,
                lambda: messagebox.showerror(
                    "Training failed",
                    f"Training could not complete:\n\n{exc}\n\n{tb}",
                ),
            )
            self.after(0, lambda: self._set_status("Training failed."))
            return

        summary_lines = [
            f"Accuracy        : {results['accuracy']:.4f}",
            f"Training samples: {results['training_samples']}",
            f"Test samples    : {results['test_samples']}",
            f"Skipped images  : {results['skipped_images']}",
            f"Model saved to  : {results['model_path']}",
            "",
            "Classification report:",
            results["report"],
        ]

        def on_done() -> None:
            self._write_training_output("\n".join(summary_lines))
            self._set_status(
                f"Training complete. Accuracy {results['accuracy']:.1%}."
            )

        self.after(0, on_done)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_training_output(self, text: str) -> None:
        self.training_output.configure(state="normal")
        self.training_output.delete("1.0", tk.END)
        self.training_output.insert("1.0", text)
        self.training_output.configure(state="disabled")

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _refresh_status_bar(self) -> None:
        dataset_ok = self.workflow.is_dataset_available()
        model_ok = self.workflow.is_model_available()
        parts = [
            f"Dataset: {'found' if dataset_ok else 'missing'}",
            f"Model: {'found' if model_ok else 'not trained'}",
        ]
        self.status_var.set("  |  ".join(parts))


def launch() -> None:
    """Create the GUI and start the Tk main loop."""
    app = MacroApp()
    app.mainloop()


if __name__ == "__main__":
    launch()
