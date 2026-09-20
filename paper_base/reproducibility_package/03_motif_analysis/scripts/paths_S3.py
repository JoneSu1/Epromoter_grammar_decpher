"""Local paths for the DeepSTARR proximal/distal motif supplement."""

from __future__ import annotations

import os
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = MODULE_ROOT / "data"
RAW_DIR = DATA_DIR / "S3_raw"
CLEAN_DIR = DATA_DIR / "S3_clean"
_OUTPUT_ROOT = Path(os.environ["REPRO_OUTPUT_ROOT"]).resolve() / "figure4" if os.environ.get("REPRO_OUTPUT_ROOT") else MODULE_ROOT
PLOT_DIR = _OUTPUT_ROOT / "figures" / "supplement"
QA_DIR = _OUTPUT_ROOT / "qa" / "S3"


def panel_dirs(panel: str) -> dict[str, Path]:
    dirs = {
        "svg": PLOT_DIR / "svg",
        "pdf": PLOT_DIR / "pdf",
        "png": PLOT_DIR / "png",
        "tiff": PLOT_DIR / "tiff",
        "qa": QA_DIR / panel,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
