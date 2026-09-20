#!/usr/bin/env python
"""Plot paired target-vs-CAGE motif-share dumbbell plots for Evolution Fig. 5f.

Each branch is normalized internally:
target share = motif hit6 gains / all target hit6 gains;
CAGE share = motif hit6 gains / all CAGE hit6 gains on the same branch.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


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
OUTPUT_TABLE = DATA_DIR / "fig5f_paired_slope_motif_share.tsv"
OUTPUT_STEM = "Fig5f_sub_paired_dumbbell_motif_share"

BRANCHES = ["HK", "DEV"]
TARGET_SERIES = {"HK": "HK target", "DEV": "DEV target"}
CAGE_SERIES = {"HK": "HK-CAGE", "DEV": "DEV-CAGE"}
TARGET_COLORS = {"HK": "#1F77A5", "DEV": "#17956F"}
CAGE_COLORS = {"HK": "#B8D7EA", "DEV": "#BEE3CF"}
LINE_COLORS = {"HK": "#68A9CB", "DEV": "#62B896"}
CAGE_TEXT_COLORS = {"HK": "#4F96BD", "DEV": "#3E9B73"}
FONT_FAMILY = "Arial"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 11.5,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11.0,
        "axes.linewidth": 0.85,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def display_motif_label(label: str) -> str:
    label = str(label)
    if label.startswith("MC_") and " " in label:
        label = label.split(" ", 1)[1]
    replacements = {
        "Ebox/CAGCTG/CACCTG": "Ebox/CAGCTG",
        "CREB/ATF/3": "CREB/ATF",
        "SREBP/2": "SREBP",
    }
    return replacements.get(label, label)


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


def build_share_table(data: pd.DataFrame) -> pd.DataFrame:
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
    return grouped


def build_paired_table(data: pd.DataFrame) -> pd.DataFrame:
    share = build_share_table(data)
    rows = []
    for branch in BRANCHES:
        target = (
            share.loc[share["series"] == TARGET_SERIES[branch]]
            .set_index("motif_label")
            .add_prefix("target_")
        )
        cage = (
            share.loc[share["series"] == CAGE_SERIES[branch]]
            .set_index("motif_label")
            .add_prefix("cage_")
        )
        paired = target.join(cage, how="outer").fillna(0).reset_index()
        paired["branch"] = branch
        paired["target_series"] = TARGET_SERIES[branch]
        paired["cage_series"] = CAGE_SERIES[branch]
        paired["delta_cage_minus_target_percent"] = paired["cage_share_percent"] - paired["target_share_percent"]
        rows.append(paired)

    out = pd.concat(rows, ignore_index=True)
    motif_order = (
        out.groupby("motif_label", observed=True)["delta_cage_minus_target_percent"]
        .apply(lambda x: x.abs().max())
        .sort_values(ascending=False)
        .index.tolist()
    )
    out["motif_label"] = pd.Categorical(out["motif_label"], categories=motif_order, ordered=True)
    return out.sort_values(["motif_label", "branch"]).reset_index(drop=True)


def draw_branch(ax: mpl.axes.Axes, table: pd.DataFrame, branch: str, show_ylabels: bool = True) -> None:
    sub = table.loc[table["branch"] == branch].copy()
    sub = sub.sort_values("motif_label")
    y = np.arange(len(sub))
    target_values = sub["target_share_percent"].to_numpy()
    cage_values = sub["cage_share_percent"].to_numpy()
    deltas = sub["delta_cage_minus_target_percent"].to_numpy()

    for i, (_, row) in enumerate(sub.iterrows()):
        ax.plot(
            [target_values[i], cage_values[i]],
            [y[i], y[i]],
            color=LINE_COLORS[branch],
            alpha=0.62,
            linewidth=1.2,
            solid_capstyle="round",
            zorder=1,
        )
        _ = deltas[i]

    ax.scatter(
        target_values,
        y,
        s=28,
        facecolors=TARGET_COLORS[branch],
        edgecolors="white",
        linewidth=0.45,
        zorder=4,
    )
    ax.scatter(
        cage_values,
        y,
        s=28,
        facecolors=CAGE_COLORS[branch],
        edgecolors=TARGET_COLORS[branch],
        linewidth=0.65,
        zorder=5,
    )

    x_max = max(10.0, float(max(target_values.max(), cage_values.max())) + 7.0)
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.7, len(sub) - 0.3)
    ax.set_yticks(y)
    ax.set_yticklabels([display_motif_label(label) for label in sub["motif_label"]])
    if not show_ylabels:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_xlabel("Gained hit6 motif hits (%)")
    ax.set_ylabel("")
    ax.set_title(f"{branch}-guided evolution", pad=8)
    ax.grid(axis="x", color="0.88", linestyle=":", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)

    ax.text(
        0.98,
        0.98,
        f"Target total={int(sub['target_series_total_gained_hits'].max())}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.0,
        color=TARGET_COLORS[branch],
    )
    ax.text(
        0.98,
        0.91,
        f"CAGE total={int(sub['cage_series_total_gained_hits'].max())}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.0,
        color=CAGE_TEXT_COLORS[branch],
    )



def plot_combined(table: pd.DataFrame) -> mpl.figure.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), dpi=300, sharey=True)
    draw_branch(axes[0], table, "HK", show_ylabels=True)
    draw_branch(axes[1], table, "DEV", show_ylabels=False)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TARGET_COLORS["HK"], markeredgecolor="white", markersize=5, label=TARGET_SERIES["HK"]),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CAGE_COLORS["HK"], markeredgecolor=TARGET_COLORS["HK"], markersize=5, label=CAGE_SERIES["HK"]),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TARGET_COLORS["DEV"], markeredgecolor="white", markersize=5, label=TARGET_SERIES["DEV"]),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CAGE_COLORS["DEV"], markeredgecolor=TARGET_COLORS["DEV"], markersize=5, label=CAGE_SERIES["DEV"]),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.6425, 0.0),
        ncol=4,
        handletextpad=0.35,
        columnspacing=0.9,
        prop={"family": FONT_FAMILY, "size": 11.0},
    )
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.27, top=0.88, wspace=0.12)
    return fig


def plot_single_branch(table: pd.DataFrame, branch: str) -> mpl.figure.Figure:
    fig, ax = plt.subplots(figsize=(3.8, 3.2), dpi=300)
    draw_branch(ax, table, branch, show_ylabels=True)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TARGET_COLORS[branch], markeredgecolor="white", markersize=5, label=TARGET_SERIES[branch]),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CAGE_COLORS[branch], markeredgecolor=TARGET_COLORS[branch], markersize=5, label=CAGE_SERIES[branch]),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.705, 0.0),
        ncol=2,
        handletextpad=0.35,
        columnspacing=0.9,
        prop={"family": FONT_FAMILY, "size": 11.0},
    )
    fig.subplots_adjust(left=0.43, right=0.98, bottom=0.24, top=0.83)
    return fig


def main() -> None:
    data = pd.read_csv(INPUT, sep="\t")
    table = build_paired_table(data)
    table.to_csv(OUTPUT_TABLE, sep="\t", index=False)

    fig = plot_combined(table)
    save_outputs(fig, OUTPUT_STEM)
    plt.close(fig)

    for branch in BRANCHES:
        branch_fig = plot_single_branch(table, branch)
        save_outputs(branch_fig, f"{OUTPUT_STEM}_{branch}")
        plt.close(branch_fig)

    sums = {
        f"{branch} target": round(float(table.loc[table["branch"] == branch, "target_share_percent"].sum()), 8)
        for branch in BRANCHES
    }
    sums.update(
        {
            f"{branch} CAGE": round(float(table.loc[table["branch"] == branch, "cage_share_percent"].sum()), 8)
            for branch in BRANCHES
        }
    )
    print(f"Wrote {OUTPUT_TABLE}")
    print(f"Percent sums: {sums}")
    print(f"Wrote {OUTPUT_STEM} combined and split outputs")


if __name__ == "__main__":
    main()
