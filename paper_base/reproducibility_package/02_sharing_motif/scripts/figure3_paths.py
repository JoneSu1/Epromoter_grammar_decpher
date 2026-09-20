"""Shared paths for polished Shared_motif Figure 3 scripts."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CLEAN_DIR = REPO_ROOT / "plot" / "data" / "clean"
RAW_DIR = REPO_ROOT / "plot" / "data" / "raw"
POLISH_OUTPUT = REPO_ROOT / "plot" / "output" / "polished_figure_code"
POLISH_QA = REPO_ROOT / "plot" / "qa" / "polished_figure_code"


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
