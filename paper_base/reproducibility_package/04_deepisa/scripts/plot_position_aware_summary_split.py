from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
DATA_IN = BASE / "data" / "position_aware_summary"
PLOT_OUT = BASE / "figures" / "supplement"
DATA_OUT = DATA_IN / "supporting_split_panels"

MM_PER_INCH = 25.4
EXPORT_DPI = 600
TASK_ORDER = ["CAGE", "DEV", "HK"]
TASK_COLORS = {"CAGE": "#C76A1D", "DEV": "#239B7A", "HK": "#2A7FAE"}
SIGN_COLORS = {"Negative": "#5B7FCB", "Positive": "#D95F59", "Not retained": "#D7D7D7"}


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


def add_cell_grid(ax: plt.Axes, n_rows: int, n_cols: int) -> None:
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)


def plot_workflow() -> None:
    fig, ax = plt.subplots(figsize=(mm_to_in(92), mm_to_in(48)), dpi=300)
    ax.set_axis_off()
    y0 = 0.60
    ax.plot([0.08, 0.92], [y0, y0], color="#BFBFBF", lw=3, solid_capstyle="round")
    x_positions = np.linspace(0.20, 0.80, 6)
    for i, x in enumerate(x_positions, start=1):
        ax.add_patch(plt.Rectangle((x - 0.018, y0 - 0.11), 0.036, 0.22, facecolor="#6B6B6B", edgecolor="#404040", lw=0.7))
        ax.text(x, y0 + 0.16, f"M{i}", ha="center", va="bottom", fontsize=7)
    ax.annotate("", xy=(0.73, y0 + 0.26), xytext=(0.27, y0 + 0.26), arrowprops=dict(arrowstyle="<->", lw=1.0, color="#3F6F9F"))
    ax.text(0.50, y0 + 0.34, "Top non-overlapping motifs", ha="center", va="bottom", fontsize=8)
    ax.text(0.50, 0.22, "Retain motif position and pair interaction", ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(0.50, 0.34), xytext=(0.50, 0.47), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#7A7A7A"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_all(fig, PLOT_OUT / "position_aware_a_selected_motif_workflow")


def plot_position_bin_heatmaps(summary: pd.DataFrame) -> None:
    labels = ["0-50", "50-100", "100-150", "150-200", "200-250"]
    vlim = np.nanpercentile(np.abs(summary["median_interaction"]), 98)
    vlim = max(float(vlim), 0.04)
    for task in TASK_ORDER:
        matrix = np.full((5, 5), np.nan)
        counts = np.zeros((5, 5))
        sub = summary[summary["task"] == task]
        for row in sub.itertuples(index=False):
            matrix[int(row.bin1), int(row.bin2)] = float(row.median_interaction)
            counts[int(row.bin1), int(row.bin2)] = int(row.n_pairs)
        fig, ax = plt.subplots(figsize=(mm_to_in(70), mm_to_in(64)), dpi=300)
        im = ax.imshow(matrix[::-1, :], cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        add_cell_grid(ax, 5, 5)
        ax.set_title(f"{task} position-bin interactions", fontsize=8.5, color=TASK_COLORS[task], pad=5)
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
        ax.set_yticks(np.arange(5))
        ax.set_yticklabels(labels[::-1], fontsize=6.5)
        ax.set_xlabel("Position bin B (bp)", fontsize=7)
        ax.set_ylabel("Position bin A (bp)", fontsize=7)
        for i in range(5):
            for j in range(5):
                value = matrix[::-1, :][i, j]
                count = counts[::-1, :][i, j]
                color = "white" if abs(value) > vlim * 0.62 else "#1F1F1F"
                ax.text(j, i, f"{int(count)}", ha="center", va="center", fontsize=5.8, color=color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
        cbar.set_label("Median interaction", fontsize=6.5)
        cbar.ax.tick_params(labelsize=6)
        save_all(fig, PLOT_OUT / f"position_aware_b_{task.lower()}_position_bin_heatmap")


def plot_region_strength(region_summary: pd.DataFrame) -> None:
    data = [region_summary.loc[(region_summary["task"] == task) & (region_summary["eligible"]), "mean_abs_interaction"].dropna().to_numpy() for task in TASK_ORDER]
    fig, ax = plt.subplots(figsize=(mm_to_in(78), mm_to_in(62)), dpi=300)
    parts = ax.violinplot(data, positions=np.arange(len(TASK_ORDER)), widths=0.72, showextrema=False)
    for body, task in zip(parts["bodies"], TASK_ORDER):
        body.set_facecolor(TASK_COLORS[task])
        body.set_edgecolor(TASK_COLORS[task])
        body.set_alpha(0.24)
    for i, (task, values) in enumerate(zip(TASK_ORDER, data)):
        if len(values) > 350:
            idx = np.linspace(0, len(values) - 1, 350).round().astype(int)
            values_plot = np.sort(values)[idx]
        else:
            values_plot = values
        offsets = np.linspace(-0.075, 0.075, len(values_plot)) if len(values_plot) else np.array([])
        x = i + offsets
        ax.scatter(x, values_plot, s=6, color=TASK_COLORS[task], alpha=0.28, linewidths=0)
        ax.plot([i - 0.18, i + 0.18], [np.median(values), np.median(values)], color="#222222", lw=1.3)
    ax.set_xticks(np.arange(len(TASK_ORDER)))
    ax.set_xticklabels([f"{task}\nn={len(values)}" for task, values in zip(TASK_ORDER, data)], fontsize=7)
    ax.set_ylabel("Mean |interaction| per region", fontsize=7)
    ax.set_title("Region-level interaction strength", fontsize=8.5, pad=5)
    ax.grid(axis="y", color="#EAEAEA", lw=0.5)
    save_all(fig, PLOT_OUT / "position_aware_c_region_level_interaction_strength")


def plot_similarity(similarity: pd.DataFrame, column: str, ylabel: str, title: str, stem: str) -> None:
    pairs = ["CAGE-DEV", "CAGE-HK", "DEV-HK"]
    data = [similarity.loc[similarity["task_pair"] == pair, column].dropna().to_numpy() for pair in pairs]
    fig, ax = plt.subplots(figsize=(mm_to_in(82), mm_to_in(62)), dpi=300)
    parts = ax.violinplot(data, positions=np.arange(len(pairs)), widths=0.74, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#D9D9D9")
        body.set_edgecolor("#7A7A7A")
        body.set_alpha(0.55)
    for i, values in enumerate(data):
        if len(values) > 300:
            idx = np.linspace(0, len(values) - 1, 300).round().astype(int)
            values_plot = np.sort(values)[idx]
        else:
            values_plot = values
        offsets = np.linspace(-0.075, 0.075, len(values_plot)) if len(values_plot) else np.array([])
        x = i + offsets
        ax.scatter(x, values_plot, s=6, color="#777777", alpha=0.30, linewidths=0)
        ax.plot([i - 0.18, i + 0.18], [np.median(values), np.median(values)], color="#222222", lw=1.3)
    ax.set_xticks(np.arange(len(pairs)))
    ax.set_xticklabels([f"{pair}\nn={len(values)}" for pair, values in zip(pairs, data)], rotation=0, ha="center", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_title(title, fontsize=8.5, pad=5)
    if "correlation" in column:
        ax.axhline(0, color="#A0A0A0", lw=0.8, ls=":")
        ax.set_ylim(-1.03, 1.03)
    else:
        ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", color="#EAEAEA", lw=0.5)
    save_all(fig, PLOT_OUT / stem)


def main() -> None:
    set_style()
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(DATA_IN / "global_position_bin_interaction_summary.csv")
    region_summary = pd.read_csv(DATA_IN / "region_task_architecture_summary.csv")
    similarity = pd.read_csv(DATA_IN / "cross_task_architecture_similarity.csv")
    for src in DATA_IN.glob("*.csv"):
        pd.read_csv(src).to_csv(DATA_OUT / src.name, index=False)
    plot_workflow()
    plot_position_bin_heatmaps(summary)
    plot_region_strength(region_summary)
    plot_similarity(
        similarity,
        "tf_set_jaccard",
        "Selected TF-set Jaccard",
        "Cross-task selected-TF similarity",
        "position_aware_d_selected_tf_set_similarity",
    )
    plot_similarity(
        similarity,
        "position_profile_correlation",
        "Position-bin profile correlation",
        "Interaction profile similarity",
        "position_aware_e_interaction_profile_similarity",
    )
    print(PLOT_OUT)


if __name__ == "__main__":
    main()
