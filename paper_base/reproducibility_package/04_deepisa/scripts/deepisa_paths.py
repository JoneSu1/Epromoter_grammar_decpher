from __future__ import annotations

import os
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
DATA_ROOT = BASE / "data"
OLD_PLOT_ROOT = BASE / "figures"
OUTPUT_ROOT = Path(os.environ["REPRO_OUTPUT_ROOT"]).resolve() / "figure5" if os.environ.get("REPRO_OUTPUT_ROOT") else BASE / "figures" / "main"


def panel_dirs(panel: str) -> dict[str, Path]:
    dirs = {
        "png": OUTPUT_ROOT / "png" / panel,
        "svg": OUTPUT_ROOT / "svg" / panel,
        "pdf": OUTPUT_ROOT / "pdf" / panel,
        "tiff": OUTPUT_ROOT / "tiff" / panel,
        "qa": OUTPUT_ROOT / "qa" / panel,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
