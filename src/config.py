
"""Shared paths and helpers for the railway ticket project."""

import sys
from pathlib import Path


# Project root directory
ROOT = Path(__file__).resolve().parent.parent

# Dataset paths
RAW_CSV = ROOT / "data" / "raw" / "railway_data.csv"
CLEAN_CSV = ROOT / "data" / "processed" / "cleaned_data.csv"

# Output directories
PLOTS_DIR = ROOT / "plots"
MODEL_DIR = ROOT / "models"


def setup_console():
    """Enable UTF-8 output where supported, including on Windows."""

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass


if __name__ == "__main__":
    setup_console()

    print("Project root:", ROOT)
    print("Raw CSV:", RAW_CSV)
    print("Clean CSV:", CLEAN_CSV)
    print("Plots directory:", PLOTS_DIR)
    print("Models directory:", MODEL_DIR)
