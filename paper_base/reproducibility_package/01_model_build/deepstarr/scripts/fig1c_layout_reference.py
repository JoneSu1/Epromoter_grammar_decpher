#!/usr/bin/env python
# coding: utf-8
"""PPT layout reference for Fig. 1C two-panel activity-class scatter."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import OUTPUT_ROOT, PALETTE, require_columns, save_figure
from data import load_fig1c_data


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


SOURCE_ORDER = ["Dev_only", "HK_only", "Both"]
LEGEND_ORDER = ["HK_only", "Both", "Dev_only"]
SOURCE_LABELS = {
    "Dev_only": "DEV only",
    "HK_only": "HK only",
    "Both": "Both",
}
AXIS_LIMITS = (-2, 8)


def plot_axis(ax, df, title: str, *, show_ylabel: bool) -> None:
    require_columns(df, ["Dev_pred", "Hk_pred", "Source"], label=title)
    for source in SOURCE_ORDER:
        subset = df[df["Source"] == source]
        if subset.empty:
            continue
        ax.scatter(
            subset["Dev_pred"],
            subset["Hk_pred"],
            s=1.8,
            alpha=0.28 if source == "Both" else 0.30,
            linewidths=0,
            edgecolors="none",
            rasterized=True,
            color=PALETTE[source],
        )
    ax.plot(AXIS_LIMITS, AXIS_LIMITS, linestyle="--", linewidth=0.65, color="0.35")
    ax.set_xlim(*AXIS_LIMITS)
    ax.set_ylim(*AXIS_LIMITS)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, pad=3)
    ax.set_xlabel("Predicted DEV activity (log2)")
    ax.set_ylabel("Predicted HK activity (log2)" if show_ylabel else "")


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=PALETTE[source],
            label=SOURCE_LABELS[source],
            markersize=4.2,
        )
        for source in LEGEND_ORDER
    ]


def main() -> None:
    frame = load_fig1c_data()
    fig = plt.figure(figsize=(6.6, 2.65))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.22, 1.0], wspace=0.13)
    left_ax = fig.add_subplot(grid[0, 0])
    legend_ax = fig.add_subplot(grid[0, 1])
    right_ax = fig.add_subplot(grid[0, 2], sharex=left_ax, sharey=left_ax)
    legend_ax.set_axis_off()

    plot_axis(
        left_ax,
        frame[frame["Promoter_group"] == "proximal_promoter"],
        "Proximal region",
        show_ylabel=True,
    )
    plot_axis(
        right_ax,
        frame[frame["Promoter_group"] == "distal_promoter"],
        "Distal region",
        show_ylabel=True,
    )
    legend_ax.legend(
        handles=legend_handles(),
        loc="center",
        fontsize=8,
        handletextpad=0.45,
        labelspacing=0.65,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.22, top=0.87)
    outputs = save_figure(
        fig,
        OUTPUT_ROOT / "layout_reference" / "fig1c_ppt_layout_reference",
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
