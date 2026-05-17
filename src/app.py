"""
app.py

Macroinvertebrate Image Analysis System — Deployed GUI.

One scrollable screen with four sequential sections:
    1. Select Classes   — pick which folders to include
    2. EDA              — generate and view charts for selected classes
    3. Training         — train the model, view report and confusion matrix
    4. Predict          — select an image and see the predicted class

Each section unlocks after the previous one completes.
All outputs are displayed inline and also saved to disk.
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
    """Single scrollable screen GUI for the Macroinvertebrate Analysis System."""

    APP_TITLE = "Macroinvertebrate Image Analysis System"
    WINDOW_SIZE = "1000x800"

    def __init__(self, workflow: WorkflowService | None = None) -> None:
        super().__init__()
        self.title(self.APP_TITLE)
        self.geometry(self.WINDOW_SIZE)
        self.minsize(900, 650)

        self.workflow = workflow or WorkflowService()
        self.selected_image_path: Path | None = None

        self._eda_done = False
        self._training_done = False

        self._eda_photos: list[ImageTk.PhotoImage] = []
        self._cm_photo: ImageTk.PhotoImage | None = None
        self._predict_photo: ImageTk.PhotoImage | None = None

        self._build_ui()
        self._populate_class_list()
        self._update_button_states()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=(16, 12))
        header.pack(fill="x")
        ttk.Label(
            header,
            text=self.APP_TITLE,
            font=("Segoe UI", 15, "bold"),
        ).pack(side="left")

        self.status_var = tk.StringVar(value="Ready. Start by selecting class folders.")
        status_bar = ttk.Frame(self, relief="sunken", padding=(10, 4))
        status_bar.pack(fill="x", side="bottom")
        ttk.Label(status_bar, textvariable=self.status_var).pack(side="left")

        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.inner = ttk.Frame(canvas, padding=(20, 10))
        self._canvas_window = canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        ))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(
            self._canvas_window, width=e.width
        ))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(
            int(-1 * (e.delta / 120)), "units"
        ))

        self._build_section_1()
        ttk.Separator(self.inner, orient="horizontal").pack(fill="x", pady=16)
        self._build_section_2()
        ttk.Separator(self.inner, orient="horizontal").pack(fill="x", pady=16)
        self._build_section_3()
        ttk.Separator(self.inner, orient="horizontal").pack(fill="x", pady=16)
        self._build_section_4()

    # ------------------------------------------------------------------
    # Section 1 — Select Classes
    # ------------------------------------------------------------------

    def _build_section_1(self) -> None:
        frame = ttk.LabelFrame(self.inner, text="Step 1 — Select Class Folders", padding=12)
        frame.pack(fill="x", pady=(0, 4))

        ttk.Label(
            frame,
            text="Select which insect class folders to include. EDA and training will use only these classes.",
            wraplength=860,
        ).pack(anchor="w", pady=(0, 8))

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill="x", pady=(0, 6))
        ttk.Button(btn_row, text="Select All", command=self._on_select_all).pack(side="left")
        ttk.Button(btn_row, text="Clear All", command=self._on_clear_all).pack(side="left", padx=6)
        self.class_count_var = tk.StringVar(value="0 classes selected")
        ttk.Label(btn_row, textvariable=self.class_count_var).pack(side="left", padx=12)

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill="x")
        self.class_listbox = tk.Listbox(
            list_frame,
            selectmode=tk.MULTIPLE,
            height=10,
            exportselection=False,
            font=("Consolas", 10),
        )
        self.class_listbox.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self.class_listbox.yview)
        sb.pack(side="right", fill="y")
        self.class_listbox.configure(yscrollcommand=sb.set)
        self.class_listbox.bind("<<ListboxSelect>>", self._on_class_selection_changed)

    # ------------------------------------------------------------------
    # Section 2 — EDA
    # ------------------------------------------------------------------

    def _build_section_2(self) -> None:
        frame = ttk.LabelFrame(self.inner, text="Step 2 — Exploratory Data Analysis", padding=12)
        frame.pack(fill="x", pady=(0, 4))

        ttk.Label(
            frame,
            text="Generate EDA charts for the selected classes. Charts are shown below and saved to outputs/eda/.",
            wraplength=860,
        ).pack(anchor="w", pady=(0, 8))

        self.eda_btn = ttk.Button(frame, text="Generate EDA Charts", command=self._on_generate_eda)
        self.eda_btn.pack(anchor="w")

        self.eda_charts_frame = ttk.Frame(frame)
        self.eda_charts_frame.pack(fill="x", pady=(10, 0))

    # ------------------------------------------------------------------
    # Section 3 — Training
    # ------------------------------------------------------------------

    def _build_section_3(self) -> None:
        frame = ttk.LabelFrame(self.inner, text="Step 3 — Train the Classifier", padding=12)
        frame.pack(fill="x", pady=(0, 4))

        ttk.Label(
            frame,
            text="Train the Random Forest classifier on the selected classes. Results are shown below and saved to outputs/reports/.",
            wraplength=860,
        ).pack(anchor="w", pady=(0, 8))

        self.train_btn = ttk.Button(frame, text="Train Baseline Model", command=self._on_train)
        self.train_btn.pack(anchor="w")

        report_frame = ttk.LabelFrame(frame, text="Classification Report", padding=8)
        report_frame.pack(fill="x", pady=(12, 0))
        self.report_text = tk.Text(
            report_frame, wrap="word", height=14, state="disabled", font=("Consolas", 9)
        )
        self.report_text.pack(fill="both", expand=True)

        cm_frame = ttk.LabelFrame(frame, text="Confusion Matrix", padding=8)
        cm_frame.pack(fill="x", pady=(12, 0))
        self.cm_label = ttk.Label(
            cm_frame,
            text="Confusion matrix will appear here after training.",
            justify="center",
        )
        self.cm_label.pack(pady=20)

    # ------------------------------------------------------------------
    # Section 4 — Predict
    # ------------------------------------------------------------------

    def _build_section_4(self) -> None:
        frame = ttk.LabelFrame(self.inner, text="Step 4 — Predict Image Class", padding=12)
        frame.pack(fill="x", pady=(0, 4))

        ttk.Label(
            frame,
            text="Select an image to classify using the trained model.",
            wraplength=860,
        ).pack(anchor="w", pady=(0, 8))

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill="x", pady=(0, 8))
        self.choose_btn = ttk.Button(btn_row, text="Choose Image…", command=self._on_choose_image)
        self.choose_btn.pack(side="left")
        self.predict_btn = ttk.Button(btn_row, text="Predict", command=self._on_predict)
        self.predict_btn.pack(side="left", padx=8)
        ttk.Button(btn_row, text="Clear", command=self._on_clear_predict).pack(side="left")

        preview_frame = ttk.LabelFrame(frame, text="Selected Image", padding=8)
        preview_frame.pack(fill="x", pady=(0, 8))
        self.image_label = ttk.Label(preview_frame, text="No image selected.")
        self.image_label.pack(pady=20)

        result_frame = ttk.LabelFrame(frame, text="Prediction Result", padding=10)
        result_frame.pack(fill="x")
        self.predicted_class_var = tk.StringVar(value="—")
        self.confidence_var = tk.StringVar(value="—")
        self.image_path_var = tk.StringVar(value="—")

        for row, (label, var) in enumerate([
            ("Predicted class:", self.predicted_class_var),
            ("Confidence:", self.confidence_var),
            ("Image path:", self.image_path_var),
        ]):
            ttk.Label(result_frame, text=label, width=18).grid(
                row=row, column=0, sticky="w", padx=(0, 8), pady=3
            )
            ttk.Label(result_frame, textvariable=var, font=("Segoe UI", 12)).grid(
                row=row, column=1, sticky="w", pady=3
            )

    # ------------------------------------------------------------------
    # Section 1 handlers
    # ------------------------------------------------------------------

    def _populate_class_list(self) -> None:
        self.class_listbox.delete(0, tk.END)
        try:
            classes = self.workflow.list_available_classes()
        except Exception:
            return
        for cls in classes:
            self.class_listbox.insert(tk.END, cls)
        self.class_listbox.select_set(0, tk.END)
        self._update_class_count()

    def _on_select_all(self) -> None:
        self.class_listbox.select_set(0, tk.END)
        self._update_class_count()
        self._update_button_states()

    def _on_clear_all(self) -> None:
        self.class_listbox.selection_clear(0, tk.END)
        self._update_class_count()
        self._update_button_states()

    def _on_class_selection_changed(self, _event=None) -> None:
        self._update_class_count()
        self._update_button_states()

    def _update_class_count(self) -> None:
        count = len(self.class_listbox.curselection())
        self.class_count_var.set(f"{count} class(es) selected")

    def _get_selected_classes(self) -> list[str]:
        return [self.class_listbox.get(i) for i in self.class_listbox.curselection()]

    # ------------------------------------------------------------------
    # Section 2 handlers
    # ------------------------------------------------------------------

    def _on_generate_eda(self) -> None:
        selected = self._get_selected_classes()
        if not selected:
            messagebox.showwarning("No classes selected", "Select at least one class folder first.")
            return
        self._set_status("Generating EDA charts… please wait.")
        self.eda_btn.configure(state="disabled")
        threading.Thread(target=self._run_eda, args=(selected,), daemon=True).start()

    def _run_eda(self, selected_classes: list[str]) -> None:
        try:
            chart_paths = self.workflow.generate_eda(selected_classes=selected_classes)
        except Exception as exc:
            tb = traceback.format_exc()
            self.after(0, lambda: messagebox.showerror("EDA failed", f"{exc}\n\n{tb}"))
            self.after(0, lambda: self._set_status("EDA failed."))
            self.after(0, lambda: self.eda_btn.configure(state="normal"))
            return

        def on_done() -> None:
            self._display_eda_charts(chart_paths)
            self._eda_done = True
            self._update_button_states()
            self._set_status(f"EDA complete. {len(chart_paths)} chart(s) generated and saved to outputs/eda/.")

        self.after(0, on_done)

    def _display_eda_charts(self, chart_paths: list[Path]) -> None:
        for widget in self.eda_charts_frame.winfo_children():
            widget.destroy()
        self._eda_photos.clear()

        for path in chart_paths:
            if not path.exists():
                continue
            try:
                pil_image = Image.open(path)
                pil_image.thumbnail((860, 500))
                photo = ImageTk.PhotoImage(pil_image)
                self._eda_photos.append(photo)

                lf = ttk.LabelFrame(self.eda_charts_frame, text=path.name, padding=6)
                lf.pack(fill="x", pady=(0, 10))
                ttk.Label(lf, image=photo).pack()
            except Exception:
                continue

    # ------------------------------------------------------------------
    # Section 3 handlers
    # ------------------------------------------------------------------

    def _on_train(self) -> None:
        selected = self._get_selected_classes()
        if not selected:
            messagebox.showwarning("No classes selected", "Select at least one class folder first.")
            return
        if not messagebox.askyesno(
            "Confirm training",
            f"Train on {len(selected)} selected class(es)?\nThis may take a minute or two.",
        ):
            return

        self._set_status("Training… please wait.")
        self.train_btn.configure(state="disabled")
        self._write_text(self.report_text, "Training in progress…\n\nPlease wait.")
        threading.Thread(target=self._run_training, args=(selected,), daemon=True).start()

    def _run_training(self, selected_classes: list[str]) -> None:
        try:
            results = self.workflow.train_model(selected_classes=selected_classes)
        except Exception as exc:
            tb = traceback.format_exc()
            self.after(0, lambda: messagebox.showerror("Training failed", f"{exc}\n\n{tb}"))
            self.after(0, lambda: self._set_status("Training failed."))
            self.after(0, lambda: self.train_btn.configure(state="normal"))
            return

        def on_done() -> None:
            lines = [
                f"Classes trained : {len(selected_classes)}",
                f"Accuracy        : {results['accuracy']:.4f}  ({results['accuracy']:.1%})",
                f"Training samples: {results['training_samples']}",
                f"Test samples    : {results['test_samples']}",
                f"Skipped images  : {results['skipped_images']}",
                f"Model saved to  : {results['model_path']}",
                "",
                "Classification Report:",
                "=" * 60,
                results["report"],
            ]
            self._write_text(self.report_text, "\n".join(lines))
            self._display_confusion_matrix()
            self._training_done = True
            self._update_button_states()
            self._set_status(f"Training complete. Accuracy: {results['accuracy']:.1%}. Model saved.")

        self.after(0, on_done)

    def _display_confusion_matrix(self) -> None:
        cm_path = self.workflow.reports_output_dir / "confusion_matrix.png"
        if not cm_path.exists():
            self.cm_label.configure(text="Confusion matrix image not found.", image="")
            return
        try:
            pil_image = Image.open(cm_path)
            pil_image.thumbnail((860, 600))
            photo = ImageTk.PhotoImage(pil_image)
            self._cm_photo = photo
            self.cm_label.configure(image=photo, text="")
            self.cm_label.image = photo
        except Exception as exc:
            self.cm_label.configure(text=f"Could not display confusion matrix:\n{exc}", image="")

    # ------------------------------------------------------------------
    # Section 4 handlers
    # ------------------------------------------------------------------

    def _on_choose_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Choose an image to classify",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            pil_image = Image.open(file_path)
            pil_image.thumbnail((500, 400))
            photo = ImageTk.PhotoImage(pil_image)
            self._predict_photo = photo
        except Exception as exc:
            messagebox.showerror("Could not open image", str(exc))
            return

        self.selected_image_path = Path(file_path)
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo
        self.image_path_var.set(str(self.selected_image_path))
        self.predicted_class_var.set("—")
        self.confidence_var.set("—")
        self._set_status(f"Image loaded: {self.selected_image_path.name}")

    def _on_predict(self) -> None:
        if not self.selected_image_path:
            messagebox.showwarning("No image", "Choose an image first.")
            return
        if not self.workflow.is_model_available():
            messagebox.showwarning("No model", "Train the model first in Step 3.")
            return
        self._set_status("Predicting…")
        try:
            result = self.workflow.predict_image(self.selected_image_path)
        except Exception as exc:
            messagebox.showerror("Prediction error", str(exc))
            self._set_status("Prediction failed.")
            return

        self.predicted_class_var.set(result["predicted_class"])
        self.confidence_var.set(f"{result['confidence']:.1%}")
        self._set_status(
            f"Predicted: {result['predicted_class']} ({result['confidence']:.1%} confidence)"
        )

    def _on_clear_predict(self) -> None:
        self.selected_image_path = None
        self.image_label.configure(image="", text="No image selected.")
        self.image_label.image = None
        self.predicted_class_var.set("—")
        self.confidence_var.set("—")
        self.image_path_var.set("—")
        self._set_status("Cleared.")

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _update_button_states(self) -> None:
        has_selection = len(self.class_listbox.curselection()) > 0
        self.eda_btn.configure(state="normal" if has_selection else "disabled")
        self.train_btn.configure(state="normal" if self._eda_done else "disabled")
        self.choose_btn.configure(state="normal" if self._training_done else "disabled")
        self.predict_btn.configure(state="normal" if self._training_done else "disabled")

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


def launch() -> None:
    app = MacroApp()
    app.mainloop()


if __name__ == "__main__":
    launch()