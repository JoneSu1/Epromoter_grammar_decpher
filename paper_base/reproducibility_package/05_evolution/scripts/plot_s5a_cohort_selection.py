#!/usr/bin/env python
"""Supplementary Fig. S5a: cohort selection and CAGE-stratum checks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures" / "supplement"
LOG_DIR = ROOT / "logs"
OUT_STEM = FIG_DIR / "FigS5a_polished_cohort_composition"
MANIFEST = LOG_DIR / "FigS5a_cohort_selection_manifest.json"

GROUP_FULL = ["Low (<2)", "Medium (2-4)", "High (>4)"]
GROUP_COLORS = {
    "Low (<2)": "#4F83BF",
    "Medium (2-4)": "#E6A400",
    "High (>4)": "#D55E00",
}
PARTITION_COLORS = {
    "Dev-specific": "#5B8CC0",
    "Dual function": "#9B74B1",
    "Hk-specific": "#D95F02",
    "Unclassified": "#D9D9D9",
}


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.2,
        "axes.labelsize": 8.4,
        "axes.titlesize": 9.6,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save_pub(fig: mpl.figure.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def draw_funnel(ax: plt.Axes, counts: pd.DataFrame) -> None:
    ax.set_axis_off()

    counts_by_stage = counts.set_index("stage")["n_sequences"].astype(int).to_dict()
    n_core = counts_by_stage["Core promoters with CAGE prediction"]
    n_intersection = counts_by_stage["CAGE+DEV+HK super-consensus"]
    n_greedy = counts_by_stage["Reviewer-grade greedy cohort"]

    def add_box(
        center: tuple[float, float],
        width: float,
        height: float,
        label: str,
        fontsize: float = 7.1,
        facecolor: str = "#F7F9FA",
        edgecolor: str = "#0B2D3A",
        linewidth: float = 1.05,
    ) -> None:
        x, y = center
        rect = plt.Rectangle(
            (x - width / 2, y - height / 2),
            width,
            height,
            transform=ax.transAxes,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            joinstyle="miter",
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#111111",
            linespacing=0.95,
        )

    def add_arrow(start: tuple[float, float], end: tuple[float, float], color: str = "#D6D8DA") -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.2,
                color=color,
                shrinkA=0,
                shrinkB=0,
            )
        )

    def add_check(center: tuple[float, float], color: str) -> None:
        x, y = center
        ax.plot(
            [x - 0.012, x - 0.003, x + 0.016],
            [y - 0.002, y - 0.013, y + 0.014],
            transform=ax.transAxes,
            color=color,
            lw=1.25,
            solid_capstyle="round",
        )

    # Compact editable TSS sketch: enough context to anchor the cohort without competing with the data panels.
    ax.plot([0.42, 0.58], [0.905, 0.905], transform=ax.transAxes, color="#155A73", lw=1.3)
    ax.plot([0.50], [0.905], transform=ax.transAxes, marker="o", ms=4.0, color="#C80000")
    ax.text(0.50, 0.878, "TSS", transform=ax.transAxes, ha="center", va="top", fontsize=5.9)
    ax.plot([0.45, 0.45], [0.805, 0.975], transform=ax.transAxes, color="#7A7A7A", lw=0.7, ls=(0, (3, 3)))
    ax.plot([0.55, 0.55], [0.805, 0.975], transform=ax.transAxes, color="#7A7A7A", lw=0.7, ls=(0, (3, 3)))
    ax.plot([0.45, 0.55], [0.97, 0.97], transform=ax.transAxes, color="#111111", lw=0.7)
    ax.plot([0.45, 0.45], [0.955, 0.97], transform=ax.transAxes, color="#111111", lw=0.7)
    ax.plot([0.55, 0.55], [0.955, 0.97], transform=ax.transAxes, color="#111111", lw=0.7)
    ax.text(0.50, 0.985, "±50 bp", transform=ax.transAxes, ha="center", va="bottom", fontsize=5.6)
    ax.arrow(0.50, 0.935, 0.045, 0, transform=ax.transAxes, color="#C80000", lw=1.0, head_width=0.020, head_length=0.020, length_includes_head=True)
    for i, y in enumerate([0.835, 0.815, 0.795]):
        x0 = 0.40 + i * 0.028
        ax.plot([x0, x0 + 0.13], [y, y], transform=ax.transAxes, color="#C80000", lw=0.8)
        ax.plot([x0], [y], transform=ax.transAxes, marker="o", ms=2.3, color="#C80000")

    add_box((0.50, 0.680), 0.39, 0.105, f"Core promoters with CAGE prediction\nn = {n_core:,}", fontsize=6.45)
    add_arrow((0.50, 0.628), (0.50, 0.565))

    readout_y = 0.500
    readouts = [
        (0.22, "CAGE high-confidence", "#E15759"),
        (0.50, "HK high-confidence", "#59A14F"),
        (0.78, "DEV high-confidence", "#4E79A7"),
    ]
    for x, label, color in readouts:
        add_box((x, readout_y), 0.262, 0.074, label, fontsize=5.9, facecolor="#FBFCFC")
        add_check((x + 0.108, readout_y), color)

    bracket_y_top = 0.445
    bracket_y_bottom = 0.385
    ax.plot([0.22, 0.22, 0.78, 0.78], [bracket_y_top, bracket_y_bottom, bracket_y_bottom, bracket_y_top], transform=ax.transAxes, color="#D6D8DA", lw=1.1)
    ax.text(0.54, 0.363, "intersection", transform=ax.transAxes, ha="center", va="center", fontsize=5.8, color="#111111")
    add_arrow((0.50, bracket_y_bottom), (0.50, 0.325))

    add_box(
        (0.50, 0.265),
        0.44,
        0.09,
        f"CAGE+HK+DEV high-confidence\nn = {n_intersection:,} promoters",
        fontsize=6.15,
    )
    add_arrow((0.50, 0.22), (0.50, 0.15))
    add_box(
        (0.50, 0.095),
        0.35,
        0.095,
        f"HK/DEV-inactive\npromoters\nn = {n_greedy}",
        fontsize=5.95,
    )


def draw_partition(ax: plt.Axes, partition: pd.DataFrame) -> None:
    data = partition.copy()
    data["fraction"] = data["fraction"].astype(float)
    data = data.sort_values("fraction", ascending=False)
    total = int(data["n_sequences"].sum())

    x0 = 0
    for _, row in data.iterrows():
        label = str(row["category"])
        width = float(row["fraction"])
        ax.barh(
            [0],
            [width],
            left=x0,
            height=0.42,
            color=PARTITION_COLORS.get(label, "#BDBDBD"),
            edgecolor="white",
            linewidth=0.8,
        )
        if width > 0.07:
            if label == "Dual function":
                text = f"Dual\n{width * 100:.0f}%"
            elif label == "Hk-specific":
                text = f"HK\n{width * 100:.0f}%"
            else:
                text = f"{width * 100:.0f}%"
            ax.text(
                x0 + width / 2,
                0,
                text,
                ha="center",
                va="center",
                color="white",
                fontsize=9.8,
                linespacing=0.85,
            )
        x0 += width

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xlabel("Fraction of CAGE high-confidence promoters")
    ax.set_title("Functional composition", fontsize=7.6, pad=12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.0"])
    ax.text(
        0.5,
        0.47,
        f"CAGE high-confidence set, n={total:,}",
        ha="center",
        va="center",
        fontsize=7.4,
        color="#4B5563",
    )
    small = data.loc[data["fraction"].le(0.07)].sort_values("category")
    if not small.empty:
        small_text = "; ".join(
            f"{row['category']} {row['fraction'] * 100:.1f}%" for _, row in small.iterrows()
        )
        ax.text(
            0.02,
            -0.47,
            f"Minor classes: {small_text}",
            ha="left",
            va="center",
            fontsize=5.9,
            color="#6B7280",
        )


def draw_cage_groups(ax: plt.Axes, group_comp: pd.DataFrame) -> None:
    sources = ["Super-consensus", "Greedy cohort"]
    x = np.arange(len(sources))
    bottom = np.zeros(len(sources))

    for group in GROUP_FULL:
        sub = group_comp.loc[group_comp["CAGE_Group"].eq(group)].set_index("source").reindex(sources)
        vals = sub["fraction"].fillna(0).to_numpy(dtype=float)
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=GROUP_COLORS[group],
            width=0.58,
            edgecolor="white",
            linewidth=0.8,
            label=group,
        )
        for j, val in enumerate(vals):
            if val >= 0.08:
                if j == 0:
                    center = bottom[j] + val / 2
                    text_color = "white" if group != "Medium (2-4)" else "#2C2C2C"
                    offset = min(0.040, val * 0.18)
                    ax.text(
                        x[j],
                        center + offset,
                        group.split()[0],
                        ha="center",
                        va="center",
                        fontsize=8.9,
                        color=text_color,
                    )
                    ax.text(
                        x[j],
                        center - offset,
                        f"{val * 100:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=8.9,
                        color=text_color,
                    )
                else:
                    ax.text(
                        x[j],
                        bottom[j] + val / 2,
                        f"{val * 100:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=8.9,
                        color="white" if group != "Medium (2-4)" else "#2C2C2C",
                    )
        bottom += vals

    counts = group_comp.groupby("source")["n_sequences"].sum().reindex(sources).astype(int)
    labels = [f"Super-\nconsensus\nn={counts.iloc[0]:,}", f"Greedy\ncohort\nn={counts.iloc[1]:,}"]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction")
    ax.set_title("CAGE-stratum representation", fontsize=7.6, pad=12)


def main() -> None:
    cohort = pd.read_csv(DATA / "cohort_filter_counts.tsv", sep="\t")
    partition = pd.read_csv(DATA / "cage_highconf_partition.tsv", sep="\t")
    group_comp = pd.read_csv(DATA / "cage_group_composition.tsv", sep="\t")

    fig = plt.figure(figsize=(7.25, 2.45), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.38, 1.0, 0.95], wspace=0.48)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    draw_funnel(axes[0], cohort)
    draw_partition(axes[1], partition)
    draw_cage_groups(axes[2], group_comp)

    fig.subplots_adjust(left=0.045, right=0.99, top=0.80, bottom=0.20)
    save_pub(fig, OUT_STEM)
    plt.close(fig)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "core_conclusion": (
            "The Fig. 5 greedy analysis uses a traceable 415-sequence cohort that retains "
            "CAGE-stratum structure after high-confidence promoter filtering."
        ),
        "source_data": {
            "cohort_filter_counts": str(DATA / "cohort_filter_counts.tsv"),
            "cage_highconf_partition": str(DATA / "cage_highconf_partition.tsv"),
            "cage_group_composition": str(DATA / "cage_group_composition.tsv"),
        },
        "main_cohort_n": int(cohort.loc[cohort["stage"].eq("Reviewer-grade greedy cohort"), "n_sequences"].iloc[0]),
        "quad_negative_note": "Strict quad-negative subset is not used as the main Fig. 5 cohort.",
        "outputs": [str(OUT_STEM.with_suffix(ext)) for ext in [".svg", ".pdf", ".png", ".tiff"]],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
