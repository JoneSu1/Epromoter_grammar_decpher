"""Publication style helpers for polished Shared_motif Figure 3."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl


MM_PER_INCH = 25.4
SINGLE_COL_MM = 89
DOUBLE_COL_MM = 183

INK = "#222222"
MID = "#666666"
GRID = "#D9D9D9"
TASK_COLORS = {
    "CAGE": "#D64F4F",
    "CAGE_NEW": "#D64F4F",
    "DEV": "#4C78A8",
    "HK": "#54A24B",
}


def mm_to_in(mm: float) -> float:
    return mm / MM_PER_INCH


def set_pub_style(font_size: float = 12.7) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": font_size,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.85,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def save_all(fig, dirs: dict[str, Path], stem: str, dpi: int = 600) -> dict[str, Path]:
    paths = {
        "svg": dirs["svg"] / f"{stem}.svg",
        "pdf": dirs["pdf"] / f"{stem}.pdf",
        "tiff": dirs["tiff"] / f"{stem}.tiff",
        "png": dirs["png"] / f"{stem}.png",
    }
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["tiff"], dpi=dpi, bbox_inches="tight")
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    return paths
