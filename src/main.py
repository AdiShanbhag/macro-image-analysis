"""
main.py

Member 3 — Entry point for the deployed application.

Running this module starts the Tkinter GUI defined in ``src/app.py``. It
ensures the output directories used by the rest of the system exist before
the window opens so the GUI never has to apologise for a missing folder
on a fresh checkout.

Usage:
    python -m src.main
"""

from __future__ import annotations

from src.config import EDA_OUTPUT_DIR, MODEL_OUTPUT_DIR, REPORTS_OUTPUT_DIR
from src.app import launch


def _prepare_output_directories() -> None:
    """Create the expected output folders if they are not already there."""
    for directory in (EDA_OUTPUT_DIR, MODEL_OUTPUT_DIR, REPORTS_OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Start the deployed Tkinter application."""
    _prepare_output_directories()
    launch()


if __name__ == "__main__":
    main()
