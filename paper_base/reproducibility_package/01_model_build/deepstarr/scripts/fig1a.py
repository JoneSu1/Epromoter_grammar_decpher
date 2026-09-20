#!/usr/bin/env python
# coding: utf-8
"""Fig. 1A: summit annotation distribution for HK and DEV enhancers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl

from common import PALETTE, panel_output_dir, save_figure

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
# save_figure exports .svg, .pdf, .png, and .tiff with dpi=600.


COUNTS = {
    "HK": {
        "core_promoter": 3448,
        "proximal_promoter": 827,
        "utr5": 198,
        "utr3": 136,
        "CDS": 84,
        "intron": 1417,
        "intergenic": 952,
    },
    "DEV": {
        "core_promoter": 1939,
        "proximal_promoter": 557,
        "utr5": 542,
        "utr3": 390,
        "CDS": 381,
        "intron": 5525,
        "intergenic": 2324,
    },
}

LABELS = ["core_promoter", "proximal_promoter", "utr5", "CDS + UTR3", "intron", "intergenic"]


def _collapsed_counts(raw_counts: dict[str, int]) -> list[int]:
    return [
        raw_counts["core_promoter"],
        raw_counts["proximal_promoter"],
        raw_counts["utr5"],
        raw_counts["CDS"] + raw_counts["utr3"],
        raw_counts["intron"],
        raw_counts["intergenic"],
    ]


def plot_panel(dataset: str) -> list:
    values = _collapsed_counts(COUNTS[dataset])
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    wedges, _texts, autotexts = ax.pie(
        values,
        colors=[PALETTE[label] for label in LABELS],
        startangle=120,
        counterclock=False,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 4 else "",
        pctdistance=0.72,
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        textprops={"fontsize": 6},
    )
    for text in autotexts:
        text.set_color("white")
    ax.legend(wedges, LABELS, loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=1.0)
    ax.set_title(f"{dataset} summit annotation")
    ax.set(aspect="equal")
    return save_figure(fig, panel_output_dir("fig1a") / f"{dataset.lower()}_annotation")


def main() -> None:
    for dataset in ("HK", "DEV"):
        for path in plot_panel(dataset):
            print(path)


if __name__ == "__main__":
    main()
