#!/usr/bin/env python
# coding: utf-8
"""Shared package-local utilities for Figure 1 plotting scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt


PANEL_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path | None = None) -> Path:
    """Find the package-local DeepSTARR module root."""
    current = (start or PANEL_DIR).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate package-local DeepSTARR root from {current}")


REPO_ROOT = find_repo_root()
DATA_ROOT = REPO_ROOT / "data"
OUTPUT_ROOT = Path(os.environ["REPRO_OUTPUT_ROOT"]).resolve() / "figure1" if os.environ.get("REPRO_OUTPUT_ROOT") else REPO_ROOT / "figures"


PALETTE = {
    "core_promoter": "#7B3294",
    "proximal_promoter": "#2166AC",
    "utr5": "#D9A441",
    "CDS + UTR3": "#1B9E77",
    "CDS": "#1B9E77",
    "utr3": "#1B9E77",
    "intron": "#8C6D31",
    "intergenic": "#B2182B",
    "unannotated": "#B2182B",
    "Dev_only": "#4C78A8",
    "HK_only": "#54A24B",
    "Both": "#D64F4F",
}


def apply_pub_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def panel_output_dir(panel: str) -> Path:
    out_dir = OUTPUT_ROOT / "_stems" / panel
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_figure(fig: plt.Figure, stem: Path, *, dpi: int = 600) -> list[Path]:
    """Save editable and high-resolution figure outputs."""
    for fmt in ("svg", "pdf", "png", "tiff"):
        (OUTPUT_ROOT / fmt).mkdir(parents=True, exist_ok=True)
    outputs = [
        OUTPUT_ROOT / "svg" / f"{stem.name}.svg",
        OUTPUT_ROOT / "pdf" / f"{stem.name}.pdf",
        OUTPUT_ROOT / "png" / f"{stem.name}.png",
        OUTPUT_ROOT / "tiff" / f"{stem.name}.tiff",
    ]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], dpi=dpi, bbox_inches="tight")
    fig.savefig(outputs[3], dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return outputs


def require_columns(frame, columns: Iterable[str], *, label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def pearson_corr(x, y) -> float:
    """Dependency-light Pearson r with explicit finite-value filtering."""
    import numpy as np

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(x_arr[mask], y_arr[mask])[0, 1])


apply_pub_style()
