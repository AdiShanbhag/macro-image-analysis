"""
app.py

Member 3 — Deployed Application (Tkinter GUI).

A desktop interface for the Macroinvertebrate Image Analysis System.
Organised as three tabs:

    1. Predict  — browse for an image, run the saved classifier, show class
                  and confidence.
    2. EDA      — view any chart in outputs/eda/, or generate them on the fly.
    3. Training — select which class folders to train on, train the model,
                  view the confusion matrix and classification report inside
                  the GUI.

All real work is delegated to WorkflowService.
"""

from __future__ import annotations

import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from src.services.workflow_service import WorkflowService


class MacroApp(tk.Tk):
    """Tkinter desktop GUI for the Macroinvertebrate Image Analysis System."""

    APP_TITLE = "Macroinvertebrate Image Analysis System"
    WINDOW_SIZE = "1100x780"

    def __init__(self, workflow: WorkflowService | None = None) -> None:
        super().__init__()
        self.title(self.APP_TITLE)
        self.geometry(self.WINDOW_SIZE)
        self.minsize(900, 650)

        self.workflow = workflow or WorkflowService()
        self.selected_image_path: Path | None = None
        self._confusion_matrix_photo = None

        self._build_layout()
        self._refresh_status_bar()
        self._refresh_chart_list()
        self._populate_class_list()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        style = ttk.Style(self)
        for theme in ("vista", "clam", "alt", "default"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break
        style.configure("Heading.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Result.TLabel", font=("Segoe UI", 13))
        style.configure("Mono.TLabel", font=("Consolas", 10))

        header = ttk.Frame(self, padding=(16, 12, 16, 6))
        header.pack(fill="x")
        ttk.Label(header, text=self.APP_TITLE, style="Heading.TLabel").pack(side="left")

        notebook = ttk.Notebook(self)
        notebook.pack(expand=True, fill="both", padx=12, pady=(0, 6))

        self._build_predict_tab(notebook)
        self._build_eda_tab(notebook)
        self._build_training_tab(notebook)

        self.status_var = tk.StringVar(value="Ready.")
        status_bar = ttk.Frame(self, relief="sunken", padding=(10, 4))
        status_bar.pack(fill="x", side="bottom")
        ttk.Label(status_bar, textvariable=self.status_var).pack(side="left")

    # ---------- Predict tab -------------------------------------------------

    def _build_predict_tab(self, parent: ttk.Notebook) -> None:
        tab = ttk.Frame(parent, padding=14)
        parent.add(tab, text="Predict")

        controls = ttk.Frame(tab)
        controls.pack(fill="x", pady=(0, 10))
        ttk.Button(controls, text="Choose Image…", command=self._on_choose_image).pack(side="left")
        ttk.Button(controls, text="Predict", command=self._on_predict_clicked).pack(side="left", padx=8)
        ttk.Button(controls, text="Clear", command=self._on_clear_clicked).pack(side="left")

        preview_frame = ttk.LabelFrame(tab, text="Selected image", padding=10)
        preview_frame.pack(fill="both", expand=True, pady=(0, 10))
        self.image_label = ttk.Label(preview_frame, text="No image selected.")
        self.image_label.pack(expand=True)

        result_frame = ttk.LabelFrame(tab, text="Prediction", padding=10)
        result_frame.pack(fill="x")
        self.predicted_class_var = tk.StringVar(value="—")
        self.confidence_var = tk.StringVar(value="—")
        self.path_var = tk.StringVar(value="—")

        for row, (label, var) in enumerate([
            ("Predicted class:", self.predicted_class_var),
            ("Confidence:", self.confidence_var),
            ("Image path:", self.path_var),
        ]):
            ttk.Label(result_frame, text=label, width=18).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            ttk.Label(result_frame, textvariable=var, style="Result.TLabel").grid(row=row, column=1, sticky="w", pady=2)

    # ---------- EDA tab -----------------------------------------------------

    def _build_eda_tab(self, parent: ttk.Notebook) -> None:
        tab = ttk.Frame(parent, padding=14)
        parent.add(tab, text="EDA Charts")

        controls = ttk.Frame(tab)
        controls.pack(fill="x", pady=(0, 10))
        ttk.Button(controls, text="Generate EDA charts", command=self._on_generate_eda_clicked).pack(side="left")
        ttk.Button(controls, text="Refresh list", command=self._refresh_chart_list).pack(side="left", padx=8)

        body = ttk.Frame(tab)
        body.pack(fill="both", expand=True)

        list_frame = ttk.LabelFrame(body, text="Available charts", padding=8)
        list_frame.pack(side="left", fill="y", padx=(0, 10))
        self.chart_listbox = tk.Listbox(list_frame, width=28, height=18)
        self.chart_listbox.pack(side="left", fill="y")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.chart_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.chart_listbox.configure(yscrollcommand=scrollbar.set)
        self.chart_listbox.bind("<<ListboxSelect>>", self._on_chart_selected)

        preview_frame = ttk.LabelFrame(body, text="Preview", padding=8)
        preview_frame.pack(side="right", fill="both", expand=True)
        self.chart_preview = ttk.Label(
            preview_frame,
            text="No chart selected.\n\nClick 'Generate EDA charts' if outputs/eda/ is empty.",
            justify="center",
        )
        self.chart_preview.pack(expand=True)

    # ---------- Training tab -----------------------------------------------

    def _build_training_tab(self, parent: ttk.Notebook) -> None:
        tab = ttk.Frame(parent, padding=14)
        parent.add(tab, text="Training")

        # Top: class selector on left, controls on right
        top_frame = ttk.Frame(tab)
        top_frame.pack(fill="x", pady=(0, 10))

        # Class folder selector
        selector_frame = ttk.LabelFrame(top_frame, text="Select class folders to train on", padding=8)
        selector_frame.pack(side="left", fill="y", padx=(0, 12))

        list_controls = ttk.Frame(selector_frame)
        list_controls.pack(fill="x", pady=(0, 4))
        ttk.Button(list_controls, text="Select All", command=self._on_select_all_classes).pack(side="left")
        ttk.Button(list_controls, text="Clear All", command=self._on_clear_all_classes).pack(side="left", padx=4)

        self.class_listbox = tk.Listbox(
            selector_frame,
            selectmode=tk.MULTIPLE,
            width=30,
            height=12,
            exportselection=False,
        )
        self.class_listbox.pack(side="left", fill="y")
        class_scroll = ttk.Scrollbar(selector_frame, orient="vertical", command=self.class_listbox.yview)
        class_scroll.pack(side="right", fill="y")
        self.class_listbox.configure(yscrollcommand=class_scroll.set)

        # Controls on right
        right_frame = ttk.Frame(top_frame)
        right_frame.pack(side="left", fill="both", expand=True)

        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill="x", pady=(0, 8))
        ttk.Button(btn_frame, text="Show dataset summary", command=self._on_show_summary_clicked).pack(side="left")
        ttk.Button(btn_frame, text="Train on selected classes", command=self._on_train_clicked).pack(side="left", padx=8)

        # Summary output
        summary_frame = ttk.LabelFrame(right_frame, text="Dataset summary", padding=6)
        summary_frame.pack(fill="both", expand=True)
        self.summary_output = tk.Text(
            summary_frame, wrap="word", height=10, state="disabled", font=("Consolas", 9)
        )
        self.summary_output.pack(fill="both", expand=True)

        # Bottom: training results and confusion matrix side by side
        bottom_frame = ttk.Frame(tab)
        bottom_frame.pack(fill="both", expand=True)

        # Training results text
        results_frame = ttk.LabelFrame(bottom_frame, text="Training results", padding=6)
        results_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.training_output = tk.Text(
            results_frame, wrap="word", height=14, state="disabled", font=("Consolas", 9)
        )
        self.training_output.pack(fill="both", expand=True)

        # Confusion matrix preview
        cm_frame = ttk.LabelFrame(bottom_frame, text="Confusion matrix", padding=6)
        cm_frame.pack(side="right", fill="both", expand=True)
        self.cm_label = ttk.Label(
            cm_frame,
            text="Confusion matrix will appear here after training.",
            justify="center",
        )
        self.cm_label.pack(expand=True)

    # ------------------------------------------------------------------
    # Predict handlers
    # ------------------------------------------------------------------

    def _on_choose_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            pil_image = Image.open(file_path)
            pil_image.thumbnail((420, 420))
            photo = ImageTk.PhotoImage(pil_image)
        except Exception as exc:
            messagebox.showerror("Could not open image", f"The file could not be opened as an image.\n\n{exc}")
            return

        self.selected_image_path = Path(file_path)
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo
        self.path_var.set(str(self.selected_image_path))
        self.predicted_class_var.set("—")
        self.confidence_var.set("—")
        self._set_status(f"Loaded {self.selected_image_path.name}")

    def _on_predict_clicked(self) -> None:
        if not self.selected_image_path:
            messagebox.showwarning("No image selected", "Choose an image before predicting.")
            return
        if not self.workflow.is_model_available():
            messagebox.showwarning(
                "No trained model",
                f"No saved model found at\n{self.workflow.model_path}\n\n"
                "Go to the Training tab and train the model first.",
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
            messagebox.showerror("Prediction error", f"Something went wrong:\n\n{exc}")
            self._set_status("Prediction failed.")
            return

        self.predicted_class_var.set(result["predicted_class"])
        self.confidence_var.set(f"{result['confidence']:.1%}")
        self._set_status("Prediction complete.")

    def _on_clear_clicked(self) -> None:
        self.selected_image_path = None
        self.image_label.configure(image="", text="No image selected.")
        self.image_label.image = None
        self.predicted_class_var.set("—")
        self.confidence_var.set("—")
        self.path_var.set("—")
        self._set_status("Cleared.")

    # ------------------------------------------------------------------
    # EDA handlers
    # ------------------------------------------------------------------

    def _on_generate_eda_clicked(self) -> None:
        if not self.workflow.is_dataset_available():
            messagebox.showwarning(
                "Dataset not available",
                "No images found in data/raw/.\nPlace the dataset there and try again.",
            )
            return
        self._set_status("Generating EDA charts… (this may take a moment)")
        threading.Thread(target=self._run_eda_generation, daemon=True).start()

    def _run_eda_generation(self) -> None:
        try:
            produced = self.workflow.generate_eda()
        except Exception as exc:
            tb = traceback.format_exc()
            self.after(0, lambda: messagebox.showerror("EDA failed", f"{exc}\n\n{tb}"))
            self.after(0, lambda: self._set_status("EDA failed."))
            return

        def on_done() -> None:
            self._refresh_chart_list()
            self._set_status(f"Generated {len(produced)} EDA chart(s).")

        self.after(0, on_done)

    def _refresh_chart_list(self) -> None:
        self.chart_listbox.delete(0, tk.END)
        for path in self.workflow.list_eda_charts():
            self.chart_listbox.insert(tk.END, path.name)

    def _on_chart_selected(self, _event) -> None:
        selection = self.chart_listbox.curselection()
        if not selection:
            return
        name = self.chart_listbox.get(selection[0])
        chart_path = self.workflow.eda_output_dir / name
        if not chart_path.exists():
            messagebox.showwarning("Chart missing", f"{chart_path} no longer exists.")
            self._refresh_chart_list()
            return
        try:
            pil_image = Image.open(chart_path)
            pil_image.thumbnail((640, 520))
            photo = ImageTk.PhotoImage(pil_image)
        except Exception as exc:
            messagebox.showerror("Could not open chart", str(exc))
            return
        self.chart_preview.configure(image=photo, text="")
        self.chart_preview.image = photo
        self._set_status(f"Previewing {name}.")

    # ------------------------------------------------------------------
    # Training handlers
    # ------------------------------------------------------------------

    def _populate_class_list(self) -> None:
        """Fill the class selector listbox with available folder names."""
        self.class_listbox.delete(0, tk.END)
        try:
            classes = self.workflow.list_available_classes()
        except Exception:
            return
        for cls in sorted(classes):
            self.class_listbox.insert(tk.END, cls)
        # Select all by default
        self.class_listbox.select_set(0, tk.END)

    def _on_select_all_classes(self) -> None:
        self.class_listbox.select_set(0, tk.END)

    def _on_clear_all_classes(self) -> None:
        self.class_listbox.selection_clear(0, tk.END)

    def _get_selected_classes(self) -> list[str]:
        indices = self.class_listbox.curselection()
        return [self.class_listbox.get(i) for i in indices]

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
        self._write_text(self.summary_output, "\n".join(lines))
        self._set_status("Dataset summary loaded.")

    def _on_train_clicked(self) -> None:
        selected_classes = self._get_selected_classes()
        if not selected_classes:
            messagebox.showwarning("No classes selected", "Select at least one class folder before training.")
            return
        if not self.workflow.is_dataset_available():
            messagebox.showwarning("Dataset not available", "No images found in data/raw/.")
            return
        if not messagebox.askyesno(
            "Confirm training",
            f"Train on {len(selected_classes)} selected class(es)?\n\n"
            "This may take a minute or two.",
        ):
            return

        self._set_status("Training… please wait.")
        self._write_text(
            self.training_output,
            f"Training on {len(selected_classes)} class(es):\n"
            + "\n".join(f"  - {c}" for c in selected_classes)
            + "\n\nPlease wait…",
        )
        threading.Thread(
            target=self._run_training,
            args=(selected_classes,),
            daemon=True,
        ).start()

    def _run_training(self, selected_classes: list[str]) -> None:
        try:
            results = self.workflow.train_model(selected_classes=selected_classes)
        except Exception as exc:
            tb = traceback.format_exc()
            self.after(0, lambda: messagebox.showerror("Training failed", f"{exc}\n\n{tb}"))
            self.after(0, lambda: self._set_status("Training failed."))
            return

        summary_lines = [
            f"Classes trained : {len(selected_classes)}",
            f"Accuracy        : {results['accuracy']:.4f}  ({results['accuracy']:.1%})",
            f"Training samples: {results['training_samples']}",
            f"Test samples    : {results['test_samples']}",
            f"Skipped images  : {results['skipped_images']}",
            f"Model saved to  : {results['model_path']}",
            "",
            "Classification report:",
            results["report"],
        ]

        def on_done() -> None:
            self._write_text(self.training_output, "\n".join(summary_lines))
            self._set_status(f"Training complete. Accuracy {results['accuracy']:.1%}.")
            self._show_confusion_matrix(results)

        self.after(0, on_done)

    def _show_confusion_matrix(self, results: dict) -> None:
        """Load and display the saved confusion matrix PNG in the GUI."""
        cm_path = self.workflow.model_output_dir.parent / "reports" / "confusion_matrix.png"
        if not cm_path.exists():
            self.cm_label.configure(text="Confusion matrix image not found.", image="")
            return
        try:
            pil_image = Image.open(cm_path)
            pil_image.thumbnail((480, 380))
            photo = ImageTk.PhotoImage(pil_image)
            self._confusion_matrix_photo = photo
            self.cm_label.configure(image=photo, text="")
            self.cm_label.image = photo
        except Exception as exc:
            self.cm_label.configure(text=f"Could not display confusion matrix:\n{exc}", image="")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state="disabled")

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
    app = MacroApp()
    app.mainloop()


if __name__ == "__main__":
    launch()