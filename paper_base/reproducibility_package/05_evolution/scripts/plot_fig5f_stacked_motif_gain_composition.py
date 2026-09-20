#!/usr/bin/env python
"""Plot 100% stacked motif-gain composition for Evolution Fig. 5f.

Each horizontal bar is normalized independently to 100%:
motif share = motif gained hit6 hits / all gained hit6 motif hits in the same series.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
FIG_DIRS = {
    ".png": ROOT / "figures" / "main" / "png",
    ".svg": ROOT / "figures" / "main" / "svg",
    ".pdf": ROOT / "figures" / "main" / "pdf",
    ".tiff": ROOT / "figures" / "main" / "tiff",
}
WORKING_COPY_DIR = (
    Path(r"F:\phd\Drophila\Draft_paper\脚本\Evolution\script_整理")
    / "plot_v2"
    / "polish2"
    / "figures"
    / "main"
)

INPUT = DATA_DIR / "motif_new_by_motif.tsv"
OUTPUT_TABLE = DATA_DIR / "fig5f_stacked_motif_gain_composition.tsv"
OUTPUT_STEM = "Fig5f_sub_stacked_motif_gain_composition"

BRANCHES = ["HK", "DEV"]
SERIES_BY_BRANCH = {
    "HK": ["HK target", "HK-CAGE"],
    "DEV": ["DEV target", "DEV-CAGE"],
}
SERIES_LABELS = {
    "HK target": "HK target",
    "HK-CAGE": "HK-CAGE",
    "DEV target": "DEV target",
    "DEV-CAGE": "DEV-CAGE",
}

MOTIF_COLORS = {
    "MC_000 DRE/3": "#4E79A7",
    "MC_001 Ohler1": "#A0CBE8",
    "MC_002 ATA": "#F28E2B",
    "MC_003 MAF/2": "#59A14F",
    "MC_004 Ohler7": "#8CD17D",
    "MC_005 SREBP/2": "#B6992D",
    "MC_006 kni/1": "#F1CE63",
    "MC_007 CREB/ATF/3": "#499894",
    "MC_008 Ebox/CAGCTG/CACCTG": "#E15759",
    "MC_009 HD/16": "#B07AA1",
}


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 9.2,
        "axes.linewidth": 0.85,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save_outputs(fig: mpl.figure.Figure, stem: str) -> None:
    for suffix, out_dir in FIG_DIRS.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{stem}{suffix}"
        if suffix in {".png", ".tiff"}:
            fig.savefig(path, bbox_inches="tight", dpi=600)
        else:
            fig.savefig(path, bbox_inches="tight")

    WORKING_COPY_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".svg", ".pdf", ".tiff"]:
        path = WORKING_COPY_DIR / f"{stem}{suffix}"
        if suffix in {".png", ".tiff"}:
            fig.savefig(path, bbox_inches="tight", dpi=600)
        else:
            fig.savefig(path, bbox_inches="tight")


def build_composition_table(data: pd.DataFrame) -> pd.DataFrame:
    required = {"series", "motif_label", "new_hit6_count", "ID"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns in {INPUT}: {sorted(missing)}")

    gains = data.copy()
    gains["new_hit6_count"] = pd.to_numeric(gains["new_hit6_count"], errors="coerce").fillna(0)
    gains["positive_gain"] = gains["new_hit6_count"].clip(lower=0)
    grouped = (
        gains.groupby(["series", "motif_label"], observed=True, as_index=False)
        .agg(
            gained_hits=("positive_gain", "sum"),
            n_sequences_with_gain=("positive_gain", lambda x: int((x > 0).sum())),
            n_sequences=("ID", "nunique"),
        )
    )
    grouped["series_total_gained_hits"] = grouped.groupby("series", observed=True)["gained_hits"].transform("sum")
    grouped["share_percent"] = np.where(
        grouped["series_total_gained_hits"] > 0,
        grouped["gained_hits"] / grouped["series_total_gained_hits"] * 100,
        0.0,
    )

    motif_order = (
        grouped.groupby("motif_label", observed=True)["gained_hits"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    grouped["motif_label"] = pd.Categorical(grouped["motif_label"], categories=motif_order, ordered=True)
    return grouped.sort_values(["series", "motif_label"]).reset_index(drop=True)


def text_color(hex_color: str) -> str:
    rgb = mpl.colors.to_rgb(hex_color)
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "black" if luminance > 0.62 else "white"


def draw_stacked_branch(ax: mpl.axes.Axes, table: pd.DataFrame, branch: str, motif_order: list[str]) -> None:
    series_order = SERIES_BY_BRANCH[branch]
    y_positions = np.array([1.0, 0.0])
    for y, series in zip(y_positions, series_order):
        sub = table.loc[table["series"] == series].set_index("motif_label").reindex(motif_order)
        left = 0.0
        total = int(sub["series_total_gained_hits"].max())
        for motif, row in sub.iterrows():
            width = float(row["share_percent"])
            if width <= 0:
                continue
            color = MOTIF_COLORS.get(str(motif), "#BDBDBD")
            ax.barh(
                y,
                width,
                left=left,
                height=0.46,
                color=color,
                edgecolor="white",
                linewidth=0.65,
            )
            if width >= 7.0:
                short = str(motif).split()[0].replace("MC_", "")
                ax.text(
                    left + width / 2,
                    y,
                    short,
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color=text_color(color),
                    fontweight="bold",
                )
            left += width
        ax.text(
            101.5,
            y,
            f"n={total}",
            ha="left",
            va="center",
            fontsize=9.5,
            color="0.35",
        )

    ax.set_xlim(0, 112)
    ax.set_ylim(-0.55, 1.55)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([SERIES_LABELS[s] for s in series_order])
    ax.set_xlabel("Share of gained hit6 motif hits (%)")
    ax.set_title(f"{branch}-guided evolution", pad=8)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(25))
    ax.grid(axis="x", color="0.88", linestyle=":", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def plot_combined(table: pd.DataFrame) -> mpl.figure.Figure:
    motif_order = list(table["motif_label"].cat.categories)
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.6), dpi=300, sharex=True)
    for ax, branch in zip(axes, BRANCHES):
        draw_stacked_branch(ax, table, branch, motif_order)
    axes[0].set_xlabel("")

    handles = [Patch(facecolor=MOTIF_COLORS[m], edgecolor="white", label=m) for m in motif_order]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.035),
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    fig.suptitle("Motif composition of hit6 gains", y=0.99, fontsize=15)
    fig.subplots_adjust(left=0.17, right=0.92, top=0.86, bottom=0.24, hspace=0.58)
    return fig


def plot_single_branch(table: pd.DataFrame, branch: str) -> mpl.figure.Figure:
    motif_order = list(table["motif_label"].cat.categories)
    fig, ax = plt.subplots(figsize=(7.0, 2.75), dpi=300)
    draw_stacked_branch(ax, table, branch, motif_order)
    handles = [Patch(facecolor=MOTIF_COLORS[m], edgecolor="white", label=m) for m in motif_order]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.18, right=0.91, top=0.78, bottom=0.36)
    return fig


def main() -> None:
    data = pd.read_csv(INPUT, sep="\t")
    table = build_composition_table(data)
    table.to_csv(OUTPUT_TABLE, sep="\t", index=False)

    fig = plot_combined(table)
    save_outputs(fig, OUTPUT_STEM)
    plt.close(fig)

    for branch in BRANCHES:
        branch_fig = plot_single_branch(table, branch)
        save_outputs(branch_fig, f"{OUTPUT_STEM}_{branch}")
        plt.close(branch_fig)

    sums = table.groupby("series", observed=True)["share_percent"].sum().round(8).to_dict()
    print(f"Wrote {OUTPUT_TABLE}")
    print(f"Percent sums by series: {sums}")
    print(f"Wrote {OUTPUT_STEM} combined and split outputs")


if __name__ == "__main__":
    main()
