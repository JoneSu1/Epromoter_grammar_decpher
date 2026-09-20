#!/usr/bin/env python
"""Supplementary three-readout high-confidence filtering and activity distributions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures" / "supplement"
LOG_DIR = ROOT / "logs"
OUT_STEM = FIG_DIR / "FigS5b_highconf_filter_and_activity_distribution"
MANIFEST = LOG_DIR / "FigS5b_highconf_filter_and_activity_distribution_manifest.json"

RESIDUAL_THRESHOLDS = {
    "CAGE": 1.911,
    "DEV": 1.113,
    "HK": 2.861,
}
GROUP_ORDER = ["Low (<2)", "Medium (2-4)", "High (>4)"]
GROUP_COLORS = {
    "Low (<2)": "#4F83BF",
    "Medium (2-4)": "#E6A400",
    "High (>4)": "#D55E00",
}
READOUT_COLORS = {
    "CAGE measured": "#E15759",
    "DEV measured": "#4E79A7",
    "HK measured": "#59A14F",
}
READOUT_SIGNAL_COLORS = {
    "CAGE": "#E15759",
    "DEV": "#4E79A7",
    "HK": "#59A14F",
}


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.6,
        "axes.labelsize": 8.8,
        "axes.titlesize": 9.8,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
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


def corr(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    return float(np.corrcoef(x.loc[valid], y.loc[valid])[0, 1])


def paired_valid_count(df: pd.DataFrame, x_col: str, y_col: str) -> int:
    return int((df[x_col].notna() & df[y_col].notna()).sum())


def plot_highconf_scatter(
    ax: plt.Axes,
    dual: pd.DataFrame,
    *,
    readout: str,
    measured_col: str,
    predicted_col: str,
    color: str,
) -> None:
    threshold = RESIDUAL_THRESHOLDS[readout]
    x = dual[measured_col]
    y = dual[predicted_col]
    residual = (x - y).abs()
    in_band = residual <= threshold

    ax.scatter(
        x.loc[in_band],
        y.loc[in_band],
        s=2.5,
        alpha=0.22,
        color=color,
        linewidth=0,
        rasterized=True,
        label=f"|residual| <= {threshold}",
    )
    if (~in_band).any():
        ax.scatter(
            x.loc[~in_band],
            y.loc[~in_band],
            s=3.0,
            alpha=0.16,
            color="#9CA3AF",
            linewidth=0,
            rasterized=True,
            label="outside band",
        )

    lim_min = min(x.min(), y.min()) - 0.35
    lim_max = max(x.max(), y.max()) + 0.35
    grid = np.linspace(lim_min, lim_max, 200)
    ax.plot(grid, grid, color="0.35", linewidth=0.8, linestyle="--")
    ax.fill_between(
        grid,
        grid - threshold,
        grid + threshold,
        color=color,
        alpha=0.10,
        linewidth=0,
        label="high-confidence band",
    )
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    ax.set_aspect("equal", adjustable="box")
    if readout == "CAGE":
        measured_label = "Measured CAGE log2(TPM + 1)"
        predicted_label = "Predicted CAGE log2(TPM + 1)"
        title = "CAGE"
    else:
        measured_label = f"Measured {readout} log2TPM"
        predicted_label = f"Predicted {readout} log2TPM"
        title = f"{readout} STARR-seq"
    ax.set_xlabel(measured_label)
    ax.set_ylabel(predicted_label)
    ax.set_title(title, fontsize=6.8)
    ax.text(
        0.04,
        0.96,
        f"PCC={corr(x, y):.3f}\nN={len(dual):,}\n|residual| <= {threshold}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
    )

def plot_activity_distribution(ax: plt.Axes, cohort: pd.DataFrame) -> None:
    unique = cohort.drop_duplicates("ID").copy()
    long_rows = []
    specs = [
        ("CAGE measured", "Real_CAGE_log2TPM"),
        ("DEV measured", "Dev_true"),
        ("HK measured", "Hk_true"),
    ]
    for label, col in specs:
        for value in unique.loc[unique[col].notna(), col]:
            long_rows.append({"readout": label, "activity": float(value)})
    long = pd.DataFrame(long_rows)

    sns.violinplot(
        data=long,
        x="readout",
        y="activity",
        hue="readout",
        order=[s[0] for s in specs],
        hue_order=[s[0] for s in specs],
        palette=[READOUT_COLORS[s[0]] for s in specs],
        legend=False,
        cut=0,
        inner=None,
        linewidth=0.7,
        saturation=0.85,
        ax=ax,
    )
    sns.boxplot(
        data=long,
        x="readout",
        y="activity",
        order=[s[0] for s in specs],
        width=0.18,
        showcaps=True,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "0.25", "linewidth": 0.7},
        medianprops={"color": "0.1", "linewidth": 0.8},
        whiskerprops={"color": "0.25", "linewidth": 0.7},
        capprops={"color": "0.25", "linewidth": 0.7},
        ax=ax,
    )
    ax.axhline(0, color="0.35", linewidth=0.8, linestyle="--")
    ax.axhline(2, color="0.55", linewidth=0.65, linestyle=":")
    ax.axhline(4, color="0.55", linewidth=0.65, linestyle=":")
    ax.set_xlabel("")
    ax.set_ylabel("Measured activity")
    ax.set_title("Measured baseline activity", fontsize=7.8)
    ax.tick_params(axis="x", rotation=18)
    ax.text(
        0.5,
        0.98,
        f"Unique promoters, n={unique['ID'].nunique():,}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.6,
    )


def main() -> None:
    dual = pd.read_csv(DATA / "dual_function_performance.tsv", sep="\t")
    cohort = pd.read_csv(DATA / "fig5b_greedy_round0_cohort.tsv", sep="\t")

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.85), dpi=300)
    plot_highconf_scatter(
        axes[0, 0],
        dual,
        readout="CAGE",
        measured_col="Real_CAGE_log2TPM",
        predicted_col="Pred_CAGE_log2TPM",
        color=READOUT_SIGNAL_COLORS["CAGE"],
    )
    plot_highconf_scatter(
        axes[0, 1],
        dual,
        readout="DEV",
        measured_col="Dev_true",
        predicted_col="Dev_pred",
        color=READOUT_SIGNAL_COLORS["DEV"],
    )
    plot_highconf_scatter(
        axes[1, 0],
        dual,
        readout="HK",
        measured_col="Hk_true",
        predicted_col="Hk_pred",
        color=READOUT_SIGNAL_COLORS["HK"],
    )
    plot_activity_distribution(axes[1, 1], cohort)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.17, top=0.92, hspace=0.43, wspace=0.35)
    save_pub(fig, OUT_STEM)
    plt.close(fig)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "core_conclusion": (
            "The CAGE+DEV+HK super-consensus is supported by readout-specific measured-versus-predicted "
            "agreement under the three residual thresholds, and the final 415 promoters retain baseline "
            "activity structure."
        ),
        "source_data": {
            "dual_function_performance": str(DATA / "dual_function_performance.tsv"),
            "fig5b_greedy_round0_cohort": str(DATA / "fig5b_greedy_round0_cohort.tsv"),
        },
        "residual_thresholds": RESIDUAL_THRESHOLDS,
        "n_dual_function_rows": int(len(dual)),
        "paired_non_missing_counts": {
            "CAGE": paired_valid_count(dual, "Real_CAGE_log2TPM", "Pred_CAGE_log2TPM"),
            "DEV": paired_valid_count(dual, "Dev_true", "Dev_pred"),
            "HK": paired_valid_count(dual, "Hk_true", "Hk_pred"),
        },
        "baseline_non_missing_counts": {
            "CAGE measured": int(cohort["Real_CAGE_log2TPM"].notna().sum()),
            "DEV measured": int(cohort["Dev_true"].notna().sum()),
            "HK measured": int(cohort["Hk_true"].notna().sum()),
        },
        "n_unique_final_cohort": int(cohort["ID"].nunique()),
        "outputs": [str(OUT_STEM.with_suffix(ext)) for ext in [".svg", ".pdf", ".png", ".tiff"]],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
