#!/usr/bin/env python
"""Plot paired CAGE-vs-target motif enrichment for Evolution Fig. 5f.

For each motif and branch:
delta = CAGE share of gained hit6 motif hits - target share of gained hit6 motif hits.

Negative values indicate target/enhancer enrichment; positive values indicate
CAGE/promoter enrichment.
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
OUTPUT_TABLE = DATA_DIR / "fig5f_cage_vs_target_motif_enrichment.tsv"
OUTPUT_STEM = "Fig5f_sub_cage_vs_target_motif_enrichment"

BRANCHES = ["HK", "DEV"]
TARGET_SERIES = {"HK": "HK target", "DEV": "DEV target"}
CAGE_SERIES = {"HK": "HK-CAGE", "DEV": "DEV-CAGE"}
TARGET_COLORS = {"HK": "#1F77A5", "DEV": "#17956F"}
CAGE_COLORS = {"HK": "#79B7D8", "DEV": "#C99A22"}


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 14,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10.5,
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
        0,
    )
    return grouped


def build_delta_table(data: pd.DataFrame) -> pd.DataFrame:
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
        .sort_values(ascending=True)
        .index.tolist()
    )
    out["motif_label"] = pd.Categorical(out["motif_label"], categories=motif_order, ordered=True)
    return out.sort_values(["motif_label", "branch"]).reset_index(drop=True)


def plot_delta(table: pd.DataFrame) -> mpl.figure.Figure:
    motif_order = list(table["motif_label"].cat.categories)
    y = np.arange(len(motif_order))
    x_abs = float(np.nanmax(np.abs(table["delta_cage_minus_target_percent"])))
    x_lim = max(8.0, np.ceil((x_abs + 1.0) / 5.0) * 5.0)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 5.1), dpi=300, sharey=True)
    for ax, branch in zip(axes, BRANCHES):
        sub = table.loc[table["branch"] == branch].set_index("motif_label").reindex(motif_order)
        values = sub["delta_cage_minus_target_percent"].to_numpy()
        colors = [CAGE_COLORS[branch] if v >= 0 else TARGET_COLORS[branch] for v in values]

        ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.55, height=0.68)
        ax.axvline(0, color="0.15", linewidth=1.0)
        ax.grid(axis="x", color="0.86", linestyle=":", linewidth=0.75)
        ax.set_axisbelow(True)
        ax.set_xlim(-x_lim, x_lim)
        ax.set_title(f"{branch}-guided evolution", pad=20)
        ax.set_xlabel("")
        ax.tick_params(axis="both", length=3)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))

        ax.text(
            0.02,
            1.005,
            "Target-enriched",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.6,
            color=TARGET_COLORS[branch],
            fontweight="bold",
        )
        ax.text(
            0.98,
            1.005,
            "CAGE-enriched",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.6,
            color=CAGE_COLORS[branch],
            fontweight="bold",
        )

        handles = [
            Patch(facecolor=TARGET_COLORS[branch], edgecolor="white", label=TARGET_SERIES[branch]),
            Patch(facecolor=CAGE_COLORS[branch], edgecolor="white", label=CAGE_SERIES[branch]),
        ]
        ax.legend(
            handles=handles,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            borderaxespad=0,
            ncol=2,
            handlelength=1.25,
            handletextpad=0.4,
            columnspacing=0.8,
        )

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(motif_order)
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    fig.suptitle("Relative motif enrichment in CAGE versus target readouts", y=0.985, fontsize=15.5)
    fig.supxlabel("CAGE share - target share of gained hit6 motif hits (%)", y=0.055, fontsize=13.5)
    fig.subplots_adjust(left=0.25, right=0.985, bottom=0.24, top=0.76, wspace=0.15)
    return fig


def plot_single_branch(table: pd.DataFrame, branch: str) -> mpl.figure.Figure:
    motif_order = list(table["motif_label"].cat.categories)
    y = np.arange(len(motif_order))
    sub = table.loc[table["branch"] == branch].set_index("motif_label").reindex(motif_order)
    values = sub["delta_cage_minus_target_percent"].to_numpy()
    x_abs = float(np.nanmax(np.abs(values)))
    x_lim = max(8.0, np.ceil((x_abs + 1.0) / 5.0) * 5.0)
    colors = [CAGE_COLORS[branch] if v >= 0 else TARGET_COLORS[branch] for v in values]

    fig, ax = plt.subplots(figsize=(5.0, 5.15), dpi=300)
    ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.55, height=0.68)
    ax.axvline(0, color="0.15", linewidth=1.0)
    ax.grid(axis="x", color="0.86", linestyle=":", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.set_xlim(-x_lim, x_lim)
    ax.set_yticks(y)
    ax.set_yticklabels(motif_order)
    ax.set_title(f"{branch}-guided: CAGE versus target motif gains", pad=18)
    ax.set_xlabel("CAGE share - target share (%)", labelpad=8)
    ax.text(
        0.02,
        1.005,
        "Target-enriched",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.6,
        color=TARGET_COLORS[branch],
        fontweight="bold",
    )
    ax.text(
        0.98,
        1.005,
        "CAGE-enriched",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.6,
        color=CAGE_COLORS[branch],
        fontweight="bold",
    )
    handles = [
        Patch(facecolor=TARGET_COLORS[branch], edgecolor="white", label=TARGET_SERIES[branch]),
        Patch(facecolor=CAGE_COLORS[branch], edgecolor="white", label=CAGE_SERIES[branch]),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        borderaxespad=0,
        ncol=2,
        handlelength=1.25,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.38, right=0.98, bottom=0.30, top=0.82)
    return fig


def main() -> None:
    data = pd.read_csv(INPUT, sep="\t")
    table = build_delta_table(data)
    table.to_csv(OUTPUT_TABLE, sep="\t", index=False)
    fig = plot_delta(table)
    save_outputs(fig, OUTPUT_STEM)
    plt.close(fig)
    for branch in BRANCHES:
        branch_fig = plot_single_branch(table, branch)
        save_outputs(branch_fig, f"{OUTPUT_STEM}_{branch}")
        plt.close(branch_fig)
    print(f"Wrote {OUTPUT_TABLE}")
    print(f"Wrote {OUTPUT_STEM} to reproducibility_package and plot_v2 working figure folder")


if __name__ == "__main__":
    main()
