from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
DATA_IN = BASE / "data" / "position_aware_summary"
PLOT_OUT = BASE / "figures" / "supplement"
DATA_OUT = DATA_IN / "supporting"

MM_PER_INCH = 25.4
EXPORT_DPI = 600
TASK_ORDER = ["CAGE", "DEV", "HK"]
TASK_COLORS = {"CAGE": "#C76A1D", "DEV": "#239B7A", "HK": "#2A7FAE"}


def mm_to_in(value: float) -> float:
    return value / MM_PER_INCH


def set_style() -> None:
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
            "legend.frameon": False,
            "savefig.dpi": EXPORT_DPI,
        }
    )


def save_all(fig: plt.Figure, stem: Path) -> None:
    paths = {
        "svg": PLOT_OUT / "svg" / f"{stem.name}.svg",
        "pdf": PLOT_OUT / "pdf" / f"{stem.name}.pdf",
        "png": PLOT_OUT / "png" / f"{stem.name}.png",
        "tiff": PLOT_OUT / "tiff" / f"{stem.name}.tiff",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["png"], dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(paths["tiff"], dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def deterministic_subset(values: np.ndarray, limit: int) -> np.ndarray:
    if len(values) <= limit:
        return values
    idx = np.linspace(0, len(values) - 1, limit).round().astype(int)
    return np.sort(values)[idx]


def deterministic_offsets(n: int, width: float = 0.11) -> np.ndarray:
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0.0])
    return np.linspace(-width / 2, width / 2, n)


def annotate_median(ax: plt.Axes, x: float, y: float, ymax: float, fmt: str = "{:.2f}") -> None:
    offset = ymax * 0.025
    va = "bottom"
    y_text = y + offset
    if y_text > ymax * 0.93:
        y_text = y - offset * 1.35
        va = "top"
    ax.text(x, y_text, fmt.format(y), ha="center", va=va, fontsize=5.8, color="#222222")


def draw_region_strength(ax: plt.Axes, region_summary: pd.DataFrame) -> None:
    data = [
        region_summary.loc[
            (region_summary["task"] == task) & (region_summary["eligible"]),
            "mean_abs_interaction",
        ].dropna().to_numpy()
        for task in TASK_ORDER
    ]
    parts = ax.violinplot(data, positions=np.arange(len(TASK_ORDER)), widths=0.68, showextrema=False)
    for body, task in zip(parts["bodies"], TASK_ORDER):
        body.set_facecolor(TASK_COLORS[task])
        body.set_edgecolor(TASK_COLORS[task])
        body.set_alpha(0.22)
        body.set_linewidth(0.9)
    for i, (task, values) in enumerate(zip(TASK_ORDER, data)):
        values_plot = deterministic_subset(values, 260)
        x = i + deterministic_offsets(len(values_plot), 0.13)
        ax.scatter(x, values_plot, s=4.5, color=TASK_COLORS[task], alpha=0.25, linewidths=0)
        median = np.median(values)
        ax.plot([i - 0.17, i + 0.17], [median, median], color="#222222", lw=1.2)
        annotate_median(ax, i, median, 0.69)
    ax.set_xticks(np.arange(len(TASK_ORDER)))
    ax.set_xticklabels([f"{task}\nn={len(values)}" for task, values in zip(TASK_ORDER, data)], fontsize=6.6)
    ax.set_ylabel("Mean |interaction| per region", fontsize=7)
    ax.set_title("Interaction strength\nincreases from CAGE to HK", fontsize=8, pad=5)
    ax.grid(axis="y", color="#E9E9E9", lw=0.5)
    ax.tick_params(axis="both", labelsize=6.5, length=2.5)
    ax.set_ylim(-0.02, 0.69)


def draw_similarity(ax: plt.Axes, similarity: pd.DataFrame, column: str, ylabel: str, title: str) -> None:
    pairs = ["CAGE-DEV", "CAGE-HK", "DEV-HK"]
    data = [similarity.loc[similarity["task_pair"] == pair, column].dropna().to_numpy() for pair in pairs]
    parts = ax.violinplot(data, positions=np.arange(len(pairs)), widths=0.68, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#DDDDDD")
        body.set_edgecolor("#8A8A8A")
        body.set_alpha(0.62)
        body.set_linewidth(0.9)
    for i, values in enumerate(data):
        values_plot = deterministic_subset(values, 240)
        x = i + deterministic_offsets(len(values_plot), 0.12)
        ax.scatter(x, values_plot, s=4.2, color="#777777", alpha=0.28, linewidths=0)
        median = np.median(values)
        ax.plot([i - 0.17, i + 0.17], [median, median], color="#222222", lw=1.2)
        annotate_median(ax, i, median, 1.0 if "correlation" not in column else 1.0)
    ax.set_xticks(np.arange(len(pairs)))
    ax.set_xticklabels([f"{pair}\nn={len(values)}" for pair, values in zip(pairs, data)], rotation=0, ha="center", fontsize=6.3)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_title(title, fontsize=8, pad=5)
    if "correlation" in column:
        ax.axhline(0, color="#9B9B9B", lw=0.8, ls=":")
        ax.set_ylim(-1.03, 1.03)
    else:
        ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", color="#E9E9E9", lw=0.5)
    ax.tick_params(axis="both", labelsize=6.5, length=2.5)


def main() -> None:
    set_style()
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    region_summary = pd.read_csv(DATA_IN / "region_task_architecture_summary.csv")
    similarity = pd.read_csv(DATA_IN / "cross_task_architecture_similarity.csv")
    region_summary.to_csv(DATA_OUT / "Supp_Fig_S6e_cde_region_task_architecture_summary.csv", index=False)
    similarity.to_csv(DATA_OUT / "Supp_Fig_S6e_cde_cross_task_architecture_similarity.csv", index=False)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(mm_to_in(183), mm_to_in(58)),
        dpi=300,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.05]},
    )
    draw_region_strength(axes[0], region_summary)
    draw_similarity(
        axes[1],
        similarity,
        "tf_set_jaccard",
        "Selected TF-set Jaccard",
        "Highest selected-TF overlap\nin DEV-HK",
    )
    draw_similarity(
        axes[2],
        similarity,
        "position_profile_correlation",
        "Position-bin profile correlation",
        "Most similar profiles\nin DEV-HK",
    )
    fig.suptitle("Position-aware architectures are task-dependent and most conserved between DEV and HK", fontsize=9.0, y=0.97)
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.24, top=0.76, wspace=0.34)
    save_all(fig, PLOT_OUT / "Supp_Fig_S6e_cde_position_aware_architecture_summary")
    print(PLOT_OUT / "svg" / "Supp_Fig_S6e_cde_position_aware_architecture_summary.svg")


if __name__ == "__main__":
    main()
