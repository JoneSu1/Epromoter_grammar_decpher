#!/usr/bin/env python
"""Fig. 2c: merged-data model prediction and PROMOTER_Dominant performance."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

import matplotlib.pyplot as plt
import pandas as pd

from figure_paths import DATA_DIR, panel_dirs
from figure_stats import regression_metrics
from figure_style import mm_to_in, save_all, set_pub_style


TSV_MERGED = DATA_DIR / "Merged_All_Data_Predictions_Splits.tsv"
TSV_PROMOTER = DATA_DIR / "PROMOTER_Dominant_Predictions_Splits.tsv"
width_mm = 183
EXPORT_TARGETS = ("figure.svg", "figure.pdf", "figure.tiff", "figure.png")
EXPORT_DPI = 600


def load_table(path, usecols) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", usecols=usecols)
    before_count = len(df)
    for col in usecols:
        if col != "Split":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    clean = df.dropna(subset=usecols).copy()
    after_count = len(clean)
    clean.attrs["missing_data_counts"] = {
        "rows_before": before_count,
        "rows_after": after_count,
        "rows_dropped": before_count - after_count,
    }
    return clean


def add_original_density_panel(ax, truth, pred):
    stat = regression_metrics(truth, pred)
    hb = ax.hexbin(truth, pred, gridsize=50, cmap="viridis", mincnt=1, bins="log")
    low = min(float(truth.min()), float(pred.min()))
    high = max(float(truth.max()), float(pred.max()))
    lims = [low, high]
    ax.plot(lims, lims, color="#D62728", linestyle="--", linewidth=2.0, alpha=0.85)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Measured CAGE log2(TPM + 1)")
    ax.set_ylabel("Predicted CAGE log2(TPM + 1)")
    ax.tick_params(axis="both", labelsize=9.7)
    ax.text(
        0.04,
        0.96,
        f"r = {stat['pcc']:.2f}\nn = {stat['n']:,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        zorder=4,
    )
    return hb, stat


def draw_test_only(df: pd.DataFrame):
    set_pub_style(font_size=9.7)
    sub = df[df["Split"] == "Test"]
    fig, ax = plt.subplots(figsize=(mm_to_in(100), mm_to_in(100)))
    hb, stat = add_original_density_panel(ax, sub["y"], sub["Predicted_CAGE"])
    cbar = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Counts (log scale)", fontsize=9.7)
    cbar.ax.tick_params(labelsize=9.0)
    return fig, [{"panel": "test_only", "set": "Test", **stat}]


def draw_promoter_all(df: pd.DataFrame):
    set_pub_style(font_size=9.7)
    fig, ax = plt.subplots(figsize=(mm_to_in(100), mm_to_in(100)))
    hb, stat = add_original_density_panel(ax, df["y"], df["Predicted_CAGE"])
    cbar = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("Counts (log scale)", fontsize=9.7)
    cbar.ax.tick_params(labelsize=9.0)
    return fig, [{"panel": "promoter_all", "set": "PROMOTER_Dominant", **stat}]


def main() -> None:
    dirs = panel_dirs("figure2c")
    merged = load_table(TSV_MERGED, ["y", "Predicted_CAGE", "Split"])
    promoter = load_table(TSV_PROMOTER, ["y", "Predicted_CAGE"])

    rows = []
    fig, stats_rows = draw_test_only(merged)
    save_all(fig, dirs, "fig2c_prediction_scatter")
    plt.close(fig)
    rows.extend(stats_rows)

    fig, stats_rows = draw_promoter_all(promoter)
    save_all(fig, dirs, "fig2c_promoter_all")
    plt.close(fig)
    rows.extend(stats_rows)

    pd.DataFrame(rows).to_csv(dirs["qa"] / "fig2c_metrics.csv", index=False)
    print("saved:", dirs["png"])


if __name__ == "__main__":
    main()
