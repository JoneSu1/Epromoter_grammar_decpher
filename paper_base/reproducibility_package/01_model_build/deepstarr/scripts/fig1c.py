#!/usr/bin/env python
# coding: utf-8
"""Fig. 1C: predicted DEV versus HK activity by enhancer sharing class."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl

from common import PALETTE, panel_output_dir, require_columns, save_figure
from data import load_fig1c_data

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
# save_figure exports .svg, .pdf, .png, and .tiff with dpi=600.


SOURCE_ORDER = ["Dev_only", "HK_only", "Both"]
SOURCE_LABELS = {
    "Dev_only": "DEV only",
    "HK_only": "HK only",
    "Both": "Both",
}
AXIS_LIMITS = (-2, 8)


def _plot_overlap_axis(ax, df, x_col: str, y_col: str, title: str):
    require_columns(df, [x_col, y_col, "Source"], label=f"Fig. 1C {title}")
    for source in SOURCE_ORDER:
        subset = df[df["Source"] == source]
        if subset.empty:
            continue
        ax.scatter(
            subset[x_col],
            subset[y_col],
            s=2.2,
            alpha=0.28 if source == "Both" else 0.32,
            linewidths=0,
            edgecolors="none",
            rasterized=True,
            color=PALETTE[source],
            label=f"{SOURCE_LABELS[source]} (n={len(subset):,})",
        )
    ax.plot(AXIS_LIMITS, AXIS_LIMITS, linestyle="--", linewidth=0.7, color="0.35")
    ax.set_title(title)
    ax.set_xlabel("Predicted DEV activity (log2)")
    ax.set_xlim(*AXIS_LIMITS)
    ax.set_ylim(*AXIS_LIMITS)


def plot_predicted(df, promoter_title: str, stem: str):
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    _plot_overlap_axis(ax, df, "Dev_pred", "Hk_pred", promoter_title)
    ax.set_ylabel("Predicted HK activity (log2)")
    fig.tight_layout()
    return save_figure(fig, panel_output_dir("fig1c") / stem)


def main() -> None:
    frame = load_fig1c_data()
    print("Source counts:")
    print(frame["Source"].value_counts().to_string())
    outputs = []
    for promoter_group, promoter_title in [
        ("proximal_promoter", "Proximal region"),
        ("distal_promoter", "Distal region"),
    ]:
        subset = frame[frame["Promoter_group"] == promoter_group]
        outputs.extend(plot_predicted(subset, promoter_title, f"{promoter_group.replace('_promoter', '_region')}_predicted"))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
