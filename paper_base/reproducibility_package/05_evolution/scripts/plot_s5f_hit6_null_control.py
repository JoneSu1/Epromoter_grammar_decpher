#!/usr/bin/env python
"""Plot hit6 mutation-count-matched controls for Supplementary Fig. S5f."""

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
DETAIL = DATA_PROCESSED / "hit6_null_mutation_count_matched_detail.tsv"
SUMMARY = DATA_PROCESSED / "hit6_null_mutation_count_matched_summary.tsv"
TASK_COLORS = {"HK": "#2F8F46", "DEV": "#3E73B8"}


def mm_to_in(mm: float) -> float:
    return mm / 25.4


def add_grouped_violin(ax, detail: pd.DataFrame, metric: str, ylabel: str) -> None:
    order = ["HK", "DEV"]
    guided_col = "hit6_target_delta_vs_round0" if metric == "null_target_delta_vs_round0" else "hit6_cage_delta_vs_round0"
    guided = detail[["task", "ID", guided_col]].drop_duplicates()
    positions = [-0.18, 0.18, 0.82, 1.18]
    values = []
    colors = []
    for task in order:
        values.append(detail.loc[detail["task"] == task, metric].dropna().to_numpy())
        colors.append("#B8B8B8")
        values.append(guided.loc[guided["task"] == task, guided_col].dropna().to_numpy())
        colors.append(TASK_COLORS[task])

    parts = ax.violinplot(values, positions=positions, widths=0.28, showmeans=False, showextrema=False, showmedians=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#555555")
        body.set_alpha(0.75)
        body.set_linewidth(0.8)
    box = ax.boxplot(values, positions=positions, widths=0.09, patch_artist=True, showfliers=False, whis=(5, 95))
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#111111")
        patch.set_alpha(0.88)
        patch.set_linewidth(0.8)
    for median in box["medians"]:
        median.set_color("white")
        median.set_linewidth(1.4)
    for element in ["whiskers", "caps"]:
        for artist in box[element]:
            artist.set_color("#111111")
            artist.set_linewidth(0.8)

    ax.axhline(0, color="#777777", lw=0.8, ls=":")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, fontsize=8.0)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel(ylabel, fontsize=8.2)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    if not DETAIL.exists() or not SUMMARY.exists():
        raise FileNotFoundError(
            "Missing hit6 null tables. Run prepare_s5f_hit6_null_control.py in an environment with TensorFlow/tf_keras first."
        )
    detail = pd.read_csv(DETAIL, sep="\t")
    summary = pd.read_csv(SUMMARY, sep="\t")
    for fmt in ["png", "svg", "pdf"]:
        (FIG_ROOT / fmt).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(mm_to_in(170), mm_to_in(62)))
    add_grouped_violin(axes[0], detail, "null_target_delta_vs_round0", "Δ prediction vs round0")
    add_grouped_violin(axes[1], detail, "null_cage_delta_vs_round0", "Δ prediction vs round0")
    axes[0].set_title("Target score change at hit6", fontsize=9.0, pad=7)
    axes[1].set_title("DeepCAGE score change at hit6", fontsize=9.0, pad=7)

    hit_text = []
    for task in ["HK", "DEV"]:
        row = summary.loc[summary["task"] == task].iloc[0]
        hit_ids = detail.loc[(detail["task"] == task) & (detail["reached_hit6"].astype(bool)), "ID"].nunique()
        hit_text.append(f"{task} {hit_ids}/{int(row['null_ids'])} promoters")
    fig.text(
        0.5,
        0.02,
        "Random hit6-positive promoters: " + "; ".join(hit_text) + " (20 controls/promoter)",
        ha="center",
        va="bottom",
        fontsize=7.4,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], marker="s", color="none", markerfacecolor="#B8B8B8", markeredgecolor="#555555", label="Random controls"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor=TASK_COLORS["HK"], markeredgecolor="#555555", label="HK-guided hit6"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor=TASK_COLORS["DEV"], markeredgecolor="#555555", label="DEV-guided hit6"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=7.4,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.78, wspace=0.35)

    stem = "S5f_hit6_null_control"
    for fmt in ["png", "svg", "pdf"]:
        path = FIG_ROOT / fmt / f"{stem}.{fmt}"
        fig.savefig(path, dpi=600 if fmt == "png" else None)
        print(f"saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
