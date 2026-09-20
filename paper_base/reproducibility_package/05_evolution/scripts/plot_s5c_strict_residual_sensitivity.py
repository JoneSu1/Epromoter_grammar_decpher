#!/usr/bin/env python
"""Supplementary strict STARR-silent CAGE residual sensitivity panel."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures" / "supplement"
LOG_DIR = ROOT / "logs"
OUT_STEM = FIG_DIR / "FigS5c_strict_starr_silent_residual_sensitivity"
MANIFEST = LOG_DIR / "FigS5c_strict_starr_silent_residual_sensitivity_manifest.json"

GROUP_ORDER = ["Low (<2)", "Medium (2-4)", "High (>4)"]
GROUP_COLORS = {
    "Low (<2)": "#4F83BF",
    "Medium (2-4)": "#E6A400",
    "High (>4)": "#D55E00",
}
ACCURACY_ORDER = ["Accurate (<0.5)", "Acceptable (0.5-1.0)", "Divergent (>1.0)"]
ACCURACY_COLORS = {
    "Accurate (<0.5)": "#5E9F6E",
    "Acceptable (0.5-1.0)": "#D4A72C",
    "Divergent (>1.0)": "#B36A3C",
}
ACCURACY_DISPLAY = {
    "Accurate (<0.5)": "Accurate (<0.5)",
    "Acceptable (0.5-1.0)": "Moderate (0.5-1.0)",
    "Divergent (>1.0)": "High residual (1.0-1.911)",
}
CAGE_RESIDUAL_MAX = 1.911


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.8,
        "axes.labelsize": 9.0,
        "axes.titlesize": 10.2,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 8.2,
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


def plot_residual_box(ax: plt.Axes, strict: pd.DataFrame) -> None:
    sns.boxplot(
        data=strict,
        x="CAGE_Group",
        y="CAGE_Residual",
        hue="CAGE_Group",
        order=GROUP_ORDER,
        hue_order=GROUP_ORDER,
        palette=GROUP_COLORS,
        showfliers=False,
        width=0.56,
        linewidth=0.75,
        ax=ax,
    )
    sns.stripplot(
        data=strict,
        x="CAGE_Group",
        y="CAGE_Residual",
        order=GROUP_ORDER,
        color="0.18",
        alpha=0.16,
        size=1.8,
        jitter=0.20,
        ax=ax,
        rasterized=True,
    )
    for y, label, style in [(0.5, "0.5", "--"), (1.0, "1.0", ":"), (CAGE_RESIDUAL_MAX, "1.911", "-.")]:
        ax.axhline(y, color="0.38", linewidth=0.75, linestyle=style)
        ax.text(2.48, y + 0.025, label, ha="right", va="bottom", fontsize=7.0, color="0.35")
    counts = strict["CAGE_Group"].value_counts().reindex(GROUP_ORDER).fillna(0).astype(int)
    ax.set_xticks(np.arange(len(GROUP_ORDER)))
    ax.set_xticklabels([f"{g}\nn={counts.loc[g]}" for g in GROUP_ORDER])
    ax.set_xlabel("")
    ax.set_ylabel("|Measured - predicted CAGE|")
    ax.set_title("CAGE residuals in strict STARR-silent promoters", fontsize=8.2)
    ax.set_ylim(0, max(2.05, strict["CAGE_Residual"].max() * 1.08))


def plot_accuracy_stack(ax: plt.Axes, strict: pd.DataFrame) -> None:
    counts = (
        strict.groupby(["CAGE_Group", "Accuracy_Level"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    totals = strict["CAGE_Group"].value_counts().reindex(GROUP_ORDER).fillna(0).astype(int)
    x = np.arange(len(GROUP_ORDER))
    bottom = np.zeros(len(GROUP_ORDER))

    for level in ACCURACY_ORDER:
        vals = []
        for group in GROUP_ORDER:
            n = counts.loc[
                counts["CAGE_Group"].eq(group) & counts["Accuracy_Level"].eq(level),
                "n",
            ].sum()
            denom = totals.loc[group]
            vals.append(float(n / denom) if denom else 0.0)
        vals_arr = np.array(vals)
        ax.bar(
            x,
            vals_arr,
            bottom=bottom,
            color=ACCURACY_COLORS[level],
            edgecolor="white",
            linewidth=0.7,
            width=0.58,
            label=ACCURACY_DISPLAY[level],
        )
        for xi, val, base in zip(x, vals_arr, bottom):
            if val >= 0.10:
                ax.text(
                    xi,
                    base + val / 2,
                    f"{val * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="white" if level != "Acceptable (0.5-1.0)" else "#252525",
                )
        bottom += vals_arr

    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace(" ", "\n", 1) for g in GROUP_ORDER])
    ax.set_ylabel("Fraction of strict subset")
    ax.set_title("Residual accuracy classes", fontsize=8.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False)


def main() -> None:
    strict = pd.read_csv(DATA / "quad_negative_promoters.tsv", sep="\t").copy()
    strict = strict.loc[strict["CAGE_Group"].isin(GROUP_ORDER)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.05), dpi=300, gridspec_kw={"width_ratios": [1.05, 1.0]})
    plot_residual_box(axes[0], strict)
    plot_accuracy_stack(axes[1], strict)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.30, top=0.80, wspace=0.36)
    save_pub(fig, OUT_STEM)
    plt.close(fig)

    summary = {
        "core_conclusion": (
            "A stricter STARR-silent promoter subset is used to characterize CAGE residuals under the "
            "pre-defined CAGE cutoff; this is a sensitivity/QC check and not the main 415-sequence cohort definition."
        ),
        "source_data": str(DATA / "quad_negative_promoters.tsv"),
        "n_strict_subset": int(len(strict)),
        "n_by_cage_group": {
            group: int((strict["CAGE_Group"] == group).sum())
            for group in GROUP_ORDER
        },
        "cage_residual_max": CAGE_RESIDUAL_MAX,
        "outputs": [str(OUT_STEM.with_suffix(ext)) for ext in [".svg", ".pdf", ".png", ".tiff"]],
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
