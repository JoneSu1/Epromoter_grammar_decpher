#!/usr/bin/env python
# coding: utf-8
"""DeepSTARR proximal/distal motif density and complexity supplement."""

from __future__ import annotations

from math import erf, sqrt

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from paths_S3 import CLEAN_DIR, panel_dirs
from style_S3 import SINGLE_COL_MM, LOCATION_COLORS, TASK_COLORS, mm_to_in, save_all, set_pub_style


DATA = CLEAN_DIR / "motif_density_complexity_by_sequence.csv"
ORDER = ["Proximal", "Distal"]
width_mm = 183
EXPORT_EXTS = (".svg", ".pdf", ".tiff", ".png")
EXPORT_DPI = 600


def star(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def mannwhitneyu_two_sided(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan
    values = np.concatenate([x, y])
    groups = np.concatenate([np.zeros(n1, dtype=bool), np.ones(n2, dtype=bool)])
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks_sorted = np.empty_like(sorted_values, dtype=float)
    tie_counts = []
    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks_sorted[start:end] = (start + 1 + end) / 2.0
        tie_counts.append(end - start)
        start = end
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    r1 = ranks[~groups].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u = min(u1, n1 * n2 - u1)
    n = n1 + n2
    tie_term = sum(t**3 - t for t in tie_counts)
    variance = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0
    if variance <= 0:
        return u, 1.0
    z = (u - n1 * n2 / 2.0) / sqrt(variance)
    p = 2.0 * (0.5 * (1.0 - erf(abs(z) / sqrt(2.0))))
    return u, p


def add_pair_stat(ax, df, value_col):
    g1 = df[df["location"].eq("Proximal")][value_col]
    g2 = df[df["location"].eq("Distal")][value_col]
    _, p = mannwhitneyu_two_sided(g1, g2)
    ymax = max(float(df[value_col].max()), 1.0)
    y = ymax * 1.06
    h = ymax * 0.045
    ax.plot([0, 0, 1, 1], [y, y + h, y + h, y], color="#222222", lw=0.85, clip_on=False)
    ax.text(0.5, y + h * 1.28, star(p), ha="center", va="bottom", fontsize=10.5)
    ax.set_ylim(0, y + h * 2.3)
    return p


def draw_metric(ax, df, task, value_col, ylabel):
    sub = df[df["task"].eq(task)].copy()
    sns.violinplot(
        data=sub,
        x="location",
        y=value_col,
        hue="location",
        order=ORDER,
        hue_order=ORDER,
        palette=LOCATION_COLORS,
        cut=0,
        density_norm="width",
        inner=None,
        linewidth=0.9,
        legend=False,
        ax=ax,
    )
    for poly in ax.collections:
        poly.set_alpha(0.78)
        poly.set_edgecolor("#555555")
    sns.boxplot(
        data=sub,
        x="location",
        y=value_col,
        order=ORDER,
        width=0.18,
        boxprops={"facecolor": "#303030", "edgecolor": "#303030", "alpha": 0.55, "linewidth": 0.8},
        whiskerprops={"color": "#303030", "linewidth": 0.8},
        capprops={"color": "#303030", "linewidth": 0.8},
        medianprops={"color": "white", "linewidth": 1.4},
        fliersize=0,
        ax=ax,
    )
    for x, loc in enumerate(ORDER):
        mean_value = sub.loc[sub["location"].eq(loc), value_col].mean()
        ax.hlines(mean_value, x - 0.22, x + 0.22, color="black", linewidth=2.2, zorder=7)
    p = add_pair_stat(ax, sub, value_col)
    ax.set_title(task, color=TASK_COLORS[task], fontweight="bold", fontsize=13.0, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=0)
    return {"task": task, "metric": value_col, "p_mannwhitney": p}


def plot_single_metric(df, task, metric, ylabel, stem, dirs):
    fig, ax = plt.subplots(figsize=(mm_to_in(SINGLE_COL_MM * 0.78), mm_to_in(72)))
    row = draw_metric(ax, df, task, metric, ylabel)
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.18, top=0.88)
    paths = save_all(fig, dirs, stem)
    plt.close(fig)
    print(f"saved: {paths['png']}")
    return row


def main() -> None:
    set_pub_style(font_size=12.7)
    dirs = panel_dirs("density_complexity")
    df = pd.read_csv(DATA)

    rows = []
    for task in ["HK", "DEV"]:
        rows.append(plot_single_metric(df, task, "motif_density", "Hits/peak", f"density_{task.lower()}_proximal_vs_distal", dirs))
        rows.append(
            plot_single_metric(
                df,
                task,
                "motif_complexity",
                "Unique motifs/peak",
                f"complexity_{task.lower()}_proximal_vs_distal",
                dirs,
            )
        )
    pd.DataFrame(rows).to_csv(dirs["qa"] / "density_complexity_stats.csv", index=False)


if __name__ == "__main__":
    main()
