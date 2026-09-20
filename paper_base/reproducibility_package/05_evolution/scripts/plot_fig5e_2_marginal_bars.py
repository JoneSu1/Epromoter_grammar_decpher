#!/usr/bin/env python
"""Alternative Fig. 5e: marginal motif-gain distributions on one plane.

The original Fig. 5e keeps the joint target-versus-CAGE motif-gain density.
This companion view collapses the two marginal barplots onto the same x-y
plane so the magnitude difference between the target readout and the matched
CAGE readout is easier to compare.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import plot_fig5_polish2 as base


OUT_STEM = base.FIG_MAIN / "Fig5e_2_polished_new_target_vs_cage_motif_bars"
COUNTS_TSV = base.LOG_DIR / "Fig5e_2_new_target_vs_cage_motif_bar_counts.tsv"
MANIFEST_JSON = base.LOG_DIR / "Fig5e_2_new_target_vs_cage_motif_bars_manifest.json"

SERIES_BY_BRANCH = {
    "HK": [("HK target", "target_new_motifs"), ("HK-CAGE", "cage_new_motifs")],
    "DEV": [("DEV target", "target_new_motifs"), ("DEV-CAGE", "cage_new_motifs")],
}
BRANCH_TITLES = {
    "HK": "HK-guided",
    "DEV": "DEV-guided",
}
TITLE = "Target readouts gain more motifs\nthan matched CAGE readouts"


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 12.3,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save_pub_local(fig: mpl.figure.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=600)
    fig.savefig(stem.with_suffix(".tiff"), bbox_inches="tight", dpi=600)
    print(f"Saved {stem.name}")


def build_counts(motif_scatter: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for branch, series_specs in SERIES_BY_BRANCH.items():
        sub = motif_scatter.loc[motif_scatter["branch_task"].eq(branch)].copy()
        n_sequences = int(sub["ID"].nunique())
        max_hits = int(max(sub[col].max() for _, col in series_specs))
        for series, col in series_specs:
            counts = sub.groupby(col, observed=True)["ID"].nunique()
            for hit_count in range(max_hits + 1):
                n = int(counts.get(hit_count, 0))
                records.append(
                    {
                        "branch_task": branch,
                        "series": series,
                        "new_motif_hits": hit_count,
                        "n_sequences": n,
                        "fraction_sequences": n / n_sequences if n_sequences else np.nan,
                    }
                )
            summaries.append(
                {
                    "branch_task": branch,
                    "series": series,
                    "n_sequences": n_sequences,
                    "mean_new_motif_hits": float(sub[col].mean()),
                    "median_new_motif_hits": float(sub[col].median()),
                    "max_new_motif_hits": int(sub[col].max()),
                }
            )
    return pd.DataFrame(records), pd.DataFrame(summaries)


def plot_5e_2(motif_scatter: pd.DataFrame) -> None:
    counts, summaries = build_counts(motif_scatter)
    COUNTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(COUNTS_TSV, sep="\t", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.5), dpi=300, sharey=True)
    bar_width = 0.38
    offsets = [-bar_width / 2, bar_width / 2]
    for ax, branch in zip(axes, ["HK", "DEV"]):
        branch_counts = counts.loc[counts["branch_task"].eq(branch)]
        max_x = int(branch_counts["new_motif_hits"].max())
        x = np.arange(max_x + 1)
        for offset, (series, _) in zip(offsets, SERIES_BY_BRANCH[branch]):
            series_counts = (
                branch_counts.loc[branch_counts["series"].eq(series)]
                .set_index("new_motif_hits")
                .reindex(x, fill_value=0)
            )
            ax.bar(
                x + offset,
                series_counts["n_sequences"].to_numpy(),
                width=bar_width,
                color=base.SERIES_COLORS[series],
                edgecolor="white",
                linewidth=0.35,
            )

        branch_summary = summaries.loc[summaries["branch_task"].eq(branch)]
        mean_by_series = branch_summary.set_index("series")["mean_new_motif_hits"].to_dict()
        delta_mean = float(mean_by_series[SERIES_BY_BRANCH[branch][0][0]] - mean_by_series[SERIES_BY_BRANCH[branch][1][0]])
        for row in branch_summary.itertuples(index=False):
            color = base.SERIES_COLORS[row.series]
            ax.axvline(row.mean_new_motif_hits, color=color, linestyle=(0, (2.2, 1.6)), linewidth=1.0, alpha=0.85)
        target_series = SERIES_BY_BRANCH[branch][0][0]
        cage_series = SERIES_BY_BRANCH[branch][1][0]
        n_sequences = int(branch_summary["n_sequences"].iloc[0])
        ax.legend(
            handles=[
                Patch(facecolor=base.SERIES_COLORS[series], edgecolor="none", label=series)
                for series, _ in SERIES_BY_BRANCH[branch]
            ],
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
            frameon=False,
            fontsize=9.8,
            handlelength=0.9,
            handleheight=0.7,
            borderaxespad=0.2,
            labelspacing=0.24,
        )
        ax.set_title(f"{BRANCH_TITLES[branch]} (N={n_sequences}, Δmean={delta_mean:.1f})", fontsize=14.2, pad=16)
        ax.set_xlabel("New motif hits at hit6\nrelative to round0", fontsize=13.2)
        ax.set_xlim(-0.65, max_x + 0.65)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.grid(axis="y", color="0.88", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=3, width=0.8)
    axes[0].set_ylabel("Sequences", fontsize=13.2)
    fig.suptitle(TITLE, fontsize=14.1, y=0.985, linespacing=1.05)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.73, wspace=0.25)
    save_pub_local(fig, OUT_STEM)
    plt.close(fig)

    manifest = {
        "core_conclusion": (
            "Target readouts gain more annotated motifs than their matched branch-specific CAGE readouts "
            "after constrained greedy optimization."
        ),
        "relationship_to_fig5e": (
            "Fig5e keeps the joint target-versus-CAGE distribution; Fig5e_2 collapses the x- and y-axis "
            "marginal barplots onto a shared count axis for direct readout-to-readout comparison."
        ),
        "n_rows": int(len(motif_scatter)),
        "n_sequences_by_branch": {
            branch: int(motif_scatter.loc[motif_scatter["branch_task"].eq(branch), "ID"].nunique())
            for branch in ["HK", "DEV"]
        },
        "series_color_policy": {series: base.SERIES_COLORS[series] for specs in SERIES_BY_BRANCH.values() for series, _ in specs},
        "count_table": str(COUNTS_TSV),
        "summary": summaries.to_dict(orient="records"),
        "delta_mean_target_minus_cage": {
            branch: float(
                summaries.loc[
                    summaries["branch_task"].eq(branch)
                    & summaries["series"].eq(SERIES_BY_BRANCH[branch][0][0]),
                    "mean_new_motif_hits",
                ].iloc[0]
                - summaries.loc[
                    summaries["branch_task"].eq(branch)
                    & summaries["series"].eq(SERIES_BY_BRANCH[branch][1][0]),
                    "mean_new_motif_hits",
                ].iloc[0]
            )
            for branch in ["HK", "DEV"]
        },
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    motif_scatter = pd.read_csv(base.DATA / "motif_new_scatter.tsv", sep="\t")
    plot_5e_2(motif_scatter)


if __name__ == "__main__":
    main()
