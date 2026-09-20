#!/usr/bin/env python
# coding: utf-8
"""Fig. 1B: true versus predicted activity split by promoter class."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl

from common import panel_output_dir, pearson_corr, require_columns, save_figure
from data import load_fig1b_data

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


def plot_true_vs_pred_region(
    df,
    true_col: str,
    pred_col: str,
    region_col: str,
    promoter_col: str,
    label: str,
    promoter_class: str,
    title: str,
    stem: str,
):
    require_columns(df, [true_col, pred_col, region_col, promoter_col], label=f"Fig. 1B {label}")
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    sub_df = df[df[promoter_col] == promoter_class]
    ax.scatter(
        sub_df[true_col],
        sub_df[pred_col],
        s=4,
        alpha=0.25,
        linewidths=0,
        rasterized=True,
        color="0.28",
    )
    r_value = pearson_corr(sub_df[true_col], sub_df[pred_col])
    ax.plot([-2, 8], [-2, 8], linestyle="--", linewidth=0.7, color="0.25")
    ax.text(0.04, 0.94, f"r = {r_value:.2f}\nn = {len(sub_df):,}", transform=ax.transAxes, va="top")
    ax.set_title(title)
    ax.set_xlim(-2, 8)
    ax.set_ylim(-2, 8)
    ax.set_xlabel(f"Observed {label} activity (log2)")
    ax.set_ylabel(f"Predicted {label} activity (log2)")
    fig.tight_layout()
    return save_figure(fig, panel_output_dir("fig1b") / stem)


def main() -> None:
    dev_df, hk_df = load_fig1b_data()
    outputs = []
    for promoter_class, title, stem_part in [
        ("proximal_promoter", "Proximal region", "proximal_region"),
        ("distal_promoter", "Distal region", "distal_region"),
    ]:
        outputs.extend(
            plot_true_vs_pred_region(
                hk_df,
                "Hk_true",
                "Hk_pred",
                "Hk_region",
                "Hk_promoter_type",
                "HK",
                promoter_class,
                title,
                f"hk_{stem_part}",
            )
        )
        outputs.extend(
            plot_true_vs_pred_region(
                dev_df,
                "Dev_true",
                "Dev_pred",
                "Dev_region",
                "Dev_promoter_type",
                "DEV",
                promoter_class,
                title,
                f"dev_{stem_part}",
            )
        )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
