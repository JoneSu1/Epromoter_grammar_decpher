#!/usr/bin/env python
"""Plot percentage composition of hit6 motif gains for Evolution Fig. 5f.

Each series is normalized independently:
percent = sum(new hit6 motif gains for a motif) / sum(all new hit6 motif gains in that series) * 100.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
OUTPUT_TABLE = DATA_DIR / "fig5f_percentage_motif_gain.tsv"
OUTPUT_STEM = "Fig5f_sub_percentage_motif_gain"

SERIES_ORDER = ["HK target", "HK-CAGE", "DEV target", "DEV-CAGE"]
SERIES_COLORS = {
    "HK target": "#1F77A5",
    "HK-CAGE": "#79B7D8",
    "DEV target": "#17956F",
    "DEV-CAGE": "#C99A22",
}


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 15,
        "axes.labelsize": 16,
        "axes.titlesize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "axes.linewidth": 0.9,
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
    fig.savefig(WORKING_COPY_DIR / f"{stem}.png", bbox_inches="tight", dpi=600)
    fig.savefig(WORKING_COPY_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(WORKING_COPY_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(WORKING_COPY_DIR / f"{stem}.tiff", bbox_inches="tight", dpi=600)


def build_percentage_table(data: pd.DataFrame) -> pd.DataFrame:
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
    totals = grouped.groupby("series", observed=True)["gained_hits"].sum().rename("series_total_gained_hits")
    grouped = grouped.merge(totals, on="series", how="left")
    grouped["percent_of_series_gains"] = np.where(
        grouped["series_total_gained_hits"] > 0,
        grouped["gained_hits"] / grouped["series_total_gained_hits"] * 100,
        0,
    )

    grouped["series"] = pd.Categorical(grouped["series"], categories=SERIES_ORDER, ordered=True)
    motif_order = (
        grouped.groupby("motif_label", observed=True)["percent_of_series_gains"]
        .max()
        .sort_values(ascending=True)
        .index.tolist()
    )
    grouped["motif_label"] = pd.Categorical(grouped["motif_label"], categories=motif_order, ordered=True)
    return grouped.sort_values(["motif_label", "series"]).reset_index(drop=True)


def plot(table: pd.DataFrame) -> mpl.figure.Figure:
    motif_order = list(table["motif_label"].cat.categories)
    y = np.arange(len(motif_order), dtype=float)
    bar_height = 0.17
    offsets = np.linspace(-1.5 * bar_height, 1.5 * bar_height, len(SERIES_ORDER))

    fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=300)
    for offset, series in zip(offsets, SERIES_ORDER):
        sub = table.loc[table["series"] == series].set_index("motif_label").reindex(motif_order)
        values = sub["percent_of_series_gains"].fillna(0).to_numpy()
        ax.barh(
            y + offset,
            values,
            height=bar_height * 0.88,
            color=SERIES_COLORS[series],
            edgecolor="white",
            linewidth=0.45,
            label=series,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(motif_order)
    ax.set_xlabel("Share of gained motif hits at hit6 (%)")
    ax.set_ylabel("")
    ax.set_title("Motif composition of hit6 gains", pad=10)
    ax.grid(axis="x", color="0.85", linestyle=":", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(5, np.ceil(table["percent_of_series_gains"].max() / 5) * 5 + 2))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6))

    totals = table.drop_duplicates("series")[["series", "series_total_gained_hits"]]
    total_values = {row.series: int(row.series_total_gained_hits) for row in totals.itertuples(index=False)}
    total_text = (
        "Total gained hits: "
        f"HK target={total_values.get('HK target', 0)}, HK-CAGE={total_values.get('HK-CAGE', 0)}\n"
        f"DEV target={total_values.get('DEV target', 0)}, DEV-CAGE={total_values.get('DEV-CAGE', 0)}"
    )
    ax.text(
        0.0,
        -0.16,
        total_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="0.35",
    )
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.99, 0.02),
        frameon=False,
        ncol=1,
        handlelength=1.05,
        handletextpad=0.35,
        labelspacing=0.25,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.35, right=0.98, top=0.88, bottom=0.23)
    return fig


def main() -> None:
    data = pd.read_csv(INPUT, sep="\t")
    table = build_percentage_table(data)
    table.to_csv(OUTPUT_TABLE, sep="\t", index=False)
    fig = plot(table)
    save_outputs(fig, OUTPUT_STEM)
    plt.close(fig)
    print(f"Wrote {OUTPUT_TABLE}")
    print(f"Wrote {OUTPUT_STEM} to reproducibility_package and plot_v2 working figure folder")


if __name__ == "__main__":
    main()
