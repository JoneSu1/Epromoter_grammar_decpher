"""Shared package-local paths for the DeepCAGE Figure 2 scripts."""

from __future__ import annotations

import os
from pathlib import Path


def find_deepcage_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "data" / "DATA").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise FileNotFoundError("Expected a parent directory containing data/DATA and scripts.")


REPO_ROOT = find_deepcage_root()
DATA_DIR = REPO_ROOT / "data" / "DATA"
MODISCO_DIR = REPO_ROOT / "data" / "modisco"
_OUTPUT_ROOT = Path(os.environ["REPRO_OUTPUT_ROOT"]).resolve() / "figure2" if os.environ.get("REPRO_OUTPUT_ROOT") else REPO_ROOT
POLISH_OUTPUT = _OUTPUT_ROOT / "figures"
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
