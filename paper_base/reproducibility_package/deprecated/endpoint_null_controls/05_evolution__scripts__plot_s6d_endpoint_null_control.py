#!/usr/bin/env python
"""Plot mutation-count-matched endpoint-null controls for Supplementary Fig. S5f."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = MODULE_ROOT / "data" / "processed"
FIG_ROOT = MODULE_ROOT / "figures" / "supplement"
DETAIL = DATA_PROCESSED / "endpoint_null_mutation_count_matched_detail.tsv"
SUMMARY = DATA_PROCESSED / "endpoint_null_mutation_count_matched_summary.tsv"
TASK_COLORS = {"HK": "#2F8F46", "DEV": "#3E73B8"}


def mm_to_in(mm: float) -> float:
    return mm / 25.4


def add_violin_with_guided(ax, detail: pd.DataFrame, metric: str, ylabel: str) -> None:
    order = ["HK", "DEV"]
    positions = np.arange(len(order))
    values = [detail.loc[detail["task"] == task, metric].dropna().to_numpy() for task in order]
    parts = ax.violinplot(values, positions=positions, widths=0.68, showmeans=False, showextrema=False, showmedians=False)
    for body, task in zip(parts["bodies"], order):
        body.set_facecolor("#B8B8B8")
        body.set_edgecolor("#555555")
        body.set_alpha(0.75)
        body.set_linewidth(0.8)
    box = ax.boxplot(values, positions=positions, widths=0.18, patch_artist=True, showfliers=False, whis=(5, 95))
    for patch in box["boxes"]:
        patch.set_facecolor("#333333")
        patch.set_edgecolor("#111111")
        patch.set_alpha(0.78)
        patch.set_linewidth(0.8)
    for median in box["medians"]:
        median.set_color("white")
        median.set_linewidth(1.4)
    for element in ["whiskers", "caps"]:
        for artist in box[element]:
            artist.set_color("#111111")
            artist.set_linewidth(0.8)

    guided_col = "primary_target_delta_vs_init" if metric == "null_target_delta_vs_init" else "primary_cage_delta_vs_init"
    guided = detail[["task", "ID", guided_col]].drop_duplicates()
    for i, task in enumerate(order):
        y = guided.loc[guided["task"] == task, guided_col].median()
        ax.scatter(i, y, s=36, color=TASK_COLORS[task], edgecolor="black", linewidth=0.4, zorder=6)
        x_text = i + 0.10 if i == 0 else i - 0.10
        ha = "left" if i == 0 else "right"
        y_text = y
        label = f"guided med. {y:.2f}"
        va = "center"
        if metric == "null_cage_delta_vs_init":
            label = f"med. {y:.2f}"
            if abs(y) < 1:
                y_text = y + 0.55
                va = "bottom"
        ax.text(x_text, y_text, label, va=va, ha=ha, fontsize=6.8, color=TASK_COLORS[task])

    ax.axhline(0, color="#777777", lw=0.8, ls=":")
    ax.set_xticks(positions)
    ax.set_xticklabels(order, fontsize=8.0)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel(ylabel, fontsize=8.2)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    detail = pd.read_csv(DETAIL, sep="\t")
    summary = pd.read_csv(SUMMARY, sep="\t")
    for fmt in ["png", "svg", "pdf"]:
        (FIG_ROOT / fmt).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(mm_to_in(170), mm_to_in(62)))
    add_violin_with_guided(axes[0], detail, "null_target_delta_vs_init", "Δ prediction vs round0")
    add_violin_with_guided(axes[1], detail, "null_cage_delta_vs_init", "Δ prediction vs round0")
    axes[0].set_title("Target score change", fontsize=9.0, pad=7)
    axes[1].set_title("DeepCAGE score change", fontsize=9.0, pad=7)

    hit_text = []
    for task in ["HK", "DEV"]:
        row = summary.loc[summary["task"] == task].iloc[0]
        hit_text.append(f"{task} 0/{int(row['null_rows']):,}")
    fig.text(0.5, 0.02, "Random hit6: " + "; ".join(hit_text), ha="center", va="bottom", fontsize=7.4)
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#B8B8B8", markeredgecolor="#555555", label="Random endpoints"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#444444", markeredgecolor="black", label="Guided median"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=7.4,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.78, wspace=0.35)

    stem = "S5f_endpoint_null_control"
    for fmt in ["png", "svg", "pdf"]:
        path = FIG_ROOT / fmt / f"{stem}.{fmt}"
        fig.savefig(path, dpi=600 if fmt == "png" else None)
        print(f"saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
