#!/usr/bin/env python
"""Fig. 2f: measured and predicted CAGE/DEV/HK pairwise activity correlations."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 12.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

import matplotlib.pyplot as plt
import pandas as pd

from figure_paths import DATA_DIR, panel_dirs
from figure_stats import pearson
from figure_style import DOUBLE_COL_MM, mm_to_in, save_all, set_pub_style


TSV = DATA_DIR / "PROMOTER_Dominant_True_Pred_Verify.tsv"
COLS = ["True_CAGE", "Pred_CAGE", "True_DEV", "Pred_DEV", "True_HK", "Pred_HK"]
width_mm = 183
EXPORT_TARGETS = ("figure.svg", "figure.pdf", "figure.tiff", "figure.png")
EXPORT_DPI = 600


def load_data() -> pd.DataFrame:
    df = pd.read_csv(TSV, sep="\t", usecols=COLS)
    before_count = len(df)
    for col in COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    clean = df.dropna(subset=COLS).copy()
    after_count = len(clean)
    clean.attrs["missing_data_counts"] = {
        "rows_before": before_count,
        "rows_after": after_count,
        "rows_dropped": before_count - after_count,
    }
    return clean


def add_scatter(ax, x, y, xlabel: str, ylabel: str, title: str):
    ax.scatter(x, y, s=3.2, color="#555555", alpha=0.12, edgecolors="none", rasterized=True)
    pcc = pearson(x, y)
    ax.text(0.04, 0.96, f"r = {pcc:.2f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=11.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", labelsize=12.0)
    ax.grid(True, color="#D9D9D9", linewidth=0.4, linestyle="--", alpha=0.55)
    ax.set_axisbelow(True)
    ax.set_box_aspect(1)
    return pcc


def draw(df: pd.DataFrame):
    set_pub_style(font_size=12.7)
    fig, axes = plt.subplots(2, 3, figsize=(mm_to_in(DOUBLE_COL_MM * 1.25), mm_to_in(170)))
    specs = [
        (0, 0, "True_CAGE", "True_HK", "Measured CAGE", "Measured HK", "Measured: HK vs CAGE"),
        (0, 1, "True_CAGE", "True_DEV", "Measured CAGE", "Measured DEV", "Measured: DEV vs CAGE"),
        (0, 2, "True_DEV", "True_HK", "Measured DEV", "Measured HK", "Measured: HK vs DEV"),
        (1, 0, "Pred_CAGE", "Pred_HK", "Predicted CAGE", "Predicted HK", "Pred: HK vs CAGE"),
        (1, 1, "Pred_CAGE", "Pred_DEV", "Predicted CAGE", "Predicted DEV", "Pred: DEV vs CAGE"),
        (1, 2, "Pred_DEV", "Pred_HK", "Predicted DEV", "Predicted HK", "Pred: HK vs DEV"),
    ]
    rows = []
    for row, col, xcol, ycol, xlabel, ylabel, title in specs:
        pcc = add_scatter(axes[row, col], df[xcol], df[ycol], xlabel, ylabel, title)
        rows.append({"comparison": title, "x": xcol, "y": ycol, "pcc": pcc, "n": len(df)})

    fig.subplots_adjust(left=0.13, right=0.99, wspace=0.52, hspace=0.72)
    return fig, rows


def draw_without_hk_dev(df: pd.DataFrame):
    set_pub_style(font_size=12.7)
    fig, axes = plt.subplots(2, 2, figsize=(mm_to_in(128), mm_to_in(170)))
    specs = [
        (0, 0, "True_CAGE", "True_HK", "Measured CAGE", "Measured HK", "Measured: HK vs CAGE"),
        (0, 1, "True_CAGE", "True_DEV", "Measured CAGE", "Measured DEV", "Measured: DEV vs CAGE"),
        (1, 0, "Pred_CAGE", "Pred_HK", "Predicted CAGE", "Predicted HK", "Pred: HK vs CAGE"),
        (1, 1, "Pred_CAGE", "Pred_DEV", "Predicted CAGE", "Predicted DEV", "Pred: DEV vs CAGE"),
    ]
    rows = []
    for row, col, xcol, ycol, xlabel, ylabel, title in specs:
        pcc = add_scatter(axes[row, col], df[xcol], df[ycol], xlabel, ylabel, title)
        rows.append({"comparison": title, "x": xcol, "y": ycol, "pcc": pcc, "n": len(df)})

    fig.subplots_adjust(left=0.17, right=0.99, wspace=0.50, hspace=0.72)
    return fig, rows


def main() -> None:
    dirs = panel_dirs("figure2f")
    df = load_data()
    fig, rows = draw(df)
    save_all(fig, dirs, "fig2f_signal_correlation")
    plt.close(fig)

    fig_without, rows_without = draw_without_hk_dev(df)
    save_all(fig_without, dirs, "fig2f_signal_correlation_without_hk_dev")
    plt.close(fig_without)

    pd.DataFrame(rows).to_csv(dirs["qa"] / "fig2f_pairwise_correlations.csv", index=False)
    pd.DataFrame(rows_without).to_csv(dirs["qa"] / "fig2f_pairwise_correlations_without_hk_dev.csv", index=False)
    print("saved:", dirs["png"] / "fig2f_signal_correlation.png")
    print("saved:", dirs["png"] / "fig2f_signal_correlation_without_hk_dev.png")


if __name__ == "__main__":
    main()
