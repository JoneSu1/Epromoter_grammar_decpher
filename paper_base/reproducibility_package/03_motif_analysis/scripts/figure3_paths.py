"""Shared local paths for formal motif-analysis Figure 3 scripts.

The formal_script/motif_analysis module is intended to be portable: scripts,
clean inputs, raw audit inputs, generated plots and QA records all live under
the same module root.
"""

from __future__ import annotations

import os
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = MODULE_ROOT / "data" / "clean"
RAW_DIR = MODULE_ROOT / "data" / "raw"
_OUTPUT_ROOT = Path(os.environ["REPRO_OUTPUT_ROOT"]).resolve() / "figure4" if os.environ.get("REPRO_OUTPUT_ROOT") else MODULE_ROOT
POLISH_OUTPUT = _OUTPUT_ROOT / "plot"
POLISH_QA = _OUTPUT_ROOT / "qa"


def panel_dirs(panel: str) -> dict[str, Path]:
    dirs = {
        "png": POLISH_OUTPUT / "png" / panel,
        "svg": POLISH_OUTPUT / "svg" / panel,
        "pdf": POLISH_OUTPUT / "pdf" / panel,
        "tiff": POLISH_OUTPUT / "tiff" / panel,
        "qa": POLISH_QA / panel,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
