#!/usr/bin/env python
"""Supplementary Fig. S6: hit6 sequence-realism constraints for greedy evolution."""

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
SINGLE_DIR = ROOT / "figures" / "supplement_single"
LOG_DIR = ROOT / "logs"
OUT_STEM = FIG_DIR / "FigS6_sequence_realism_constraints"
MANIFEST = LOG_DIR / "FigS6_sequence_realism_constraints_manifest.json"

SUMMARY_INPUT = DATA / "primary_greedy_summary.tsv"
TRAJECTORY_INPUT = DATA / "greedy_trajectories_all.tsv"
QC_DATA = DATA / "sequence_realism_qc.tsv"
QC_SUMMARY = DATA / "sequence_realism_qc_summary.tsv"

TASK_ORDER = ["HK", "DEV"]
TASK_COLORS = {"HK": "#2B83BA", "DEV": "#009E73"}

MAX_TOTAL_MUTS = 60
MAX_10BP_MUTS = 3
MAX_ABS_GC_CHANGE = 0.05


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.2,
        "axes.labelsize": 8.3,
        "axes.titlesize": 8.4,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 7.7,
        "legend.fontsize": 7.4,
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


def load_qc_data() -> pd.DataFrame:
    summary = pd.read_csv(SUMMARY_INPUT, sep="\t")
    trajectories = pd.read_csv(TRAJECTORY_INPUT, sep="\t")

    summary_required = ["task", "seed", "ID", "step_to_hit6", "reached_hit6"]
    trajectory_required = ["task", "seed", "ID", "Round", "total_muts", "max_10bp_muts", "gc_delta"]
    missing = [col for col in summary_required if col not in summary.columns]
    if missing:
        raise ValueError(f"Missing required columns in {SUMMARY_INPUT}: {missing}")
    missing = [col for col in trajectory_required if col not in trajectories.columns]
    if missing:
        raise ValueError(f"Missing required columns in {TRAJECTORY_INPUT}: {missing}")

    summary = summary.loc[summary["reached_hit6"].astype(str).str.lower().eq("true")].copy()
    summary["Round"] = pd.to_numeric(summary["step_to_hit6"], errors="raise").astype(int)
    trajectories["Round"] = pd.to_numeric(trajectories["Round"], errors="raise").astype(int)

    qc = summary.loc[:, ["task", "seed", "ID", "Round"]].merge(
        trajectories.loc[:, trajectory_required],
        on=["task", "seed", "ID", "Round"],
        how="left",
        validate="one_to_one",
    )
    if qc[["total_muts", "max_10bp_muts", "gc_delta"]].isna().any().any():
        missing_hits = qc.loc[qc[["total_muts", "max_10bp_muts", "gc_delta"]].isna().any(axis=1), ["task", "seed", "ID", "Round"]]
        raise ValueError(f"Missing hit6 trajectory rows for {len(missing_hits)} runs")

    qc = qc.rename(
        columns={
            "Round": "hit6_round",
            "total_muts": "hit6_total_muts",
            "max_10bp_muts": "hit6_max_10bp_muts",
            "gc_delta": "hit6_gc_delta",
        }
    )
    qc["task"] = pd.Categorical(qc["task"], categories=TASK_ORDER, ordered=True)
    qc["hit6_total_muts"] = pd.to_numeric(qc["hit6_total_muts"], errors="raise")
    qc["hit6_max_10bp_muts"] = pd.to_numeric(qc["hit6_max_10bp_muts"], errors="raise")
    qc["hit6_gc_delta"] = pd.to_numeric(qc["hit6_gc_delta"], errors="raise")
    qc["abs_gc_delta"] = qc["hit6_gc_delta"].abs()
    qc["passes_total_mut_constraint"] = qc["hit6_total_muts"].le(MAX_TOTAL_MUTS)
    qc["passes_local_density_constraint"] = qc["hit6_max_10bp_muts"].le(MAX_10BP_MUTS)
    qc["passes_gc_constraint"] = qc["abs_gc_delta"].le(MAX_ABS_GC_CHANGE + 1e-12)
    qc["passes_all_constraints"] = (
        qc["passes_total_mut_constraint"]
        & qc["passes_local_density_constraint"]
        & qc["passes_gc_constraint"]
    )
    return qc.sort_values(["task", "ID"]).reset_index(drop=True)


def write_qc_tables(qc: pd.DataFrame) -> pd.DataFrame:
    QC_DATA.parent.mkdir(parents=True, exist_ok=True)
    qc.to_csv(QC_DATA, sep="\t", index=False)

    summary_rows = []
    for task, sub in qc.groupby("task", observed=True):
        summary_rows.append(
            {
                "task": task,
                "n_runs": int(len(sub)),
                "n_unique_promoters": int(sub["ID"].nunique()),
                "median_hit6_round": float(sub["hit6_round"].median()),
                "max_hit6_round": float(sub["hit6_round"].max()),
                "median_total_muts": float(sub["hit6_total_muts"].median()),
                "max_total_muts": float(sub["hit6_total_muts"].max()),
                "median_max_10bp_muts": float(sub["hit6_max_10bp_muts"].median()),
                "max_max_10bp_muts": float(sub["hit6_max_10bp_muts"].max()),
                "median_abs_gc_delta": float(sub["abs_gc_delta"].median()),
                "max_abs_gc_delta": float(sub["abs_gc_delta"].max()),
                "n_total_mut_violations": int((~sub["passes_total_mut_constraint"]).sum()),
                "n_local_density_violations": int((~sub["passes_local_density_constraint"]).sum()),
                "n_gc_violations": int((~sub["passes_gc_constraint"]).sum()),
                "n_any_constraint_violations": int((~sub["passes_all_constraints"]).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(QC_SUMMARY, sep="\t", index=False)
    return summary


def style_box(ax: plt.Axes, data: pd.DataFrame, y: str, ylabel: str) -> None:
    sns.violinplot(
        data=data,
        x="task",
        y=y,
        order=TASK_ORDER,
        hue="task",
        hue_order=TASK_ORDER,
        palette=[TASK_COLORS[t] for t in TASK_ORDER],
        cut=0,
        inner=None,
        linewidth=0.6,
        saturation=0.85,
        legend=False,
        ax=ax,
    )
    sns.boxplot(
        data=data,
        x="task",
        y=y,
        order=TASK_ORDER,
        width=0.18,
        showcaps=True,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "0.25", "linewidth": 0.7},
        medianprops={"color": "0.1", "linewidth": 0.8},
        whiskerprops={"color": "0.25", "linewidth": 0.7},
        capprops={"color": "0.25", "linewidth": 0.7},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="0.90", linewidth=0.55)


def plot_total_mutations(ax: plt.Axes, qc: pd.DataFrame) -> None:
    style_box(ax, qc, "hit6_total_muts", "Total mutations at hit6")
    ax.axhline(MAX_TOTAL_MUTS, color="0.35", linestyle="--", linewidth=0.85)
    ax.text(1.48, MAX_TOTAL_MUTS, "limit=60", ha="right", va="bottom", fontsize=6.6, color="0.35")
    ax.set_title("Hit6 mutation burden")
    ax.set_ylim(0, MAX_TOTAL_MUTS * 1.10)


def plot_local_density(ax: plt.Axes, qc: pd.DataFrame) -> None:
    counts = (
        qc.groupby(["task", "hit6_max_10bp_muts"], observed=True)
        .size()
        .reset_index(name="n")
    )
    x_values = np.arange(0, MAX_10BP_MUTS + 1)
    width = 0.34
    offsets = {"HK": -width / 2, "DEV": width / 2}
    for task in TASK_ORDER:
        sub = counts.loc[counts["task"].eq(task)].set_index("hit6_max_10bp_muts")
        vals = [int(sub.loc[x, "n"]) if x in sub.index else 0 for x in x_values]
        ax.bar(
            x_values + offsets[task],
            vals,
            width=width,
            color=TASK_COLORS[task],
            edgecolor="white",
            linewidth=0.5,
            label=task,
        )
    ax.set_xlim(-0.6, MAX_10BP_MUTS + 0.72)
    ax.axvline(MAX_10BP_MUTS + 0.5, color="0.35", linestyle="--", linewidth=0.85)
    ax.text(
        MAX_10BP_MUTS + 0.59,
        ax.get_ylim()[1] * 0.82,
        "limit=3",
        ha="left",
        va="center",
        rotation=90,
        fontsize=6.6,
        color="0.35",
    )
    ax.set_xticks(x_values)
    ax.set_xlabel("Max mutations per 10-bp window")
    ax.set_ylabel("Greedy runs")
    ax.set_title("Local mutation density")
    ax.grid(axis="y", color="0.90", linewidth=0.55)
    ax.legend(frameon=False, loc="upper left", handlelength=1.0)


def plot_gc_delta(ax: plt.Axes, qc: pd.DataFrame) -> None:
    style_box(ax, qc, "hit6_gc_delta", "GC change from round0 at hit6")
    ax.axhline(MAX_ABS_GC_CHANGE, color="0.35", linestyle="--", linewidth=0.85)
    ax.axhline(-MAX_ABS_GC_CHANGE, color="0.35", linestyle="--", linewidth=0.85)
    ax.text(1.48, MAX_ABS_GC_CHANGE, "+0.05", ha="right", va="bottom", fontsize=6.6, color="0.35")
    ax.text(1.48, -MAX_ABS_GC_CHANGE, "-0.05", ha="right", va="top", fontsize=6.6, color="0.35")
    ax.set_title("GC-content stability")
    ax.set_ylim(-0.065, 0.065)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.20,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
    )


def normalized_constraint_data(qc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("Total mutations\n(limit 60)", "hit6_total_muts", MAX_TOTAL_MUTS),
        ("10-bp density\n(limit 3)", "hit6_max_10bp_muts", MAX_10BP_MUTS),
        ("|GC change|\n(limit 0.05)", "abs_gc_delta", MAX_ABS_GC_CHANGE),
    ]
    for metric, col, limit in specs:
        for _, row in qc.iterrows():
            rows.append(
                {
                    "task": row["task"],
                    "metric": metric,
                    "fraction_of_limit": float(row[col]) / float(limit),
                }
            )
    return pd.DataFrame(rows)


def plot_normalized_constraints(ax: plt.Axes, qc: pd.DataFrame) -> None:
    norm = normalized_constraint_data(qc)
    metric_order = ["Total mutations\n(limit 60)", "10-bp density\n(limit 3)", "|GC change|\n(limit 0.05)"]
    sns.violinplot(
        data=norm,
        x="metric",
        y="fraction_of_limit",
        hue="task",
        order=metric_order,
        hue_order=TASK_ORDER,
        palette=[TASK_COLORS[t] for t in TASK_ORDER],
        split=False,
        cut=0,
        inner=None,
        linewidth=0.6,
        saturation=0.85,
        ax=ax,
    )
    sns.boxplot(
        data=norm,
        x="metric",
        y="fraction_of_limit",
        hue="task",
        order=metric_order,
        hue_order=TASK_ORDER,
        width=0.18,
        dodge=True,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "0.25", "linewidth": 0.65},
        medianprops={"color": "0.1", "linewidth": 0.75},
        whiskerprops={"color": "0.25", "linewidth": 0.65},
        capprops={"color": "0.25", "linewidth": 0.65},
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[: len(TASK_ORDER)], labels[: len(TASK_ORDER)], frameon=False, loc="upper left", ncol=2)
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=0.85)
    ax.text(2.48, 1.0, "constraint limit", ha="right", va="bottom", fontsize=6.8, color="0.35")
    ax.set_xlabel("")
    ax.set_ylabel("Fraction of constraint limit at hit6")
    ax.set_title("Sequence-realism constraints at hit6")
    ax.set_ylim(0, 1.14)
    ax.grid(axis="y", color="0.90", linewidth=0.55)


def make_figure(qc: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.45), dpi=300)
    plot_total_mutations(axes[0], qc)
    plot_local_density(axes[1], qc)
    plot_gc_delta(axes[2], qc)
    for ax, label in zip(axes, ["a", "b", "c"]):
        add_panel_label(ax, label)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.24, top=0.82, wspace=0.48)
    save_pub(fig, OUT_STEM)
    plt.close(fig)

    single_specs = [
        ("S6a_total_mutation_burden", plot_total_mutations, (2.25, 2.35)),
        ("S6b_local_mutation_density", plot_local_density, (2.55, 2.35)),
        ("S6c_gc_content_stability", plot_gc_delta, (2.25, 2.35)),
    ]
    for name, func, size in single_specs:
        fig, ax = plt.subplots(figsize=size, dpi=300)
        func(ax, qc)
        fig.subplots_adjust(left=0.26, right=0.98, bottom=0.26, top=0.82)
        save_pub(fig, SINGLE_DIR / name)
        plt.close(fig)


def main() -> None:
    qc = load_qc_data()
    summary = write_qc_tables(qc)
    make_figure(qc)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "core_conclusion": (
            "Constrained greedy evolution reaches hit6 while retaining the pre-specified "
            "sequence-realism constraints on mutation burden, local mutation density, and GC content."
        ),
        "summary_input": str(SUMMARY_INPUT),
        "trajectory_input": str(TRAJECTORY_INPUT),
        "processed_data": str(QC_DATA),
        "summary": str(QC_SUMMARY),
        "constraints": {
            "max_total_mutations": MAX_TOTAL_MUTS,
            "max_mutations_per_10bp_window": MAX_10BP_MUTS,
            "max_absolute_gc_change": MAX_ABS_GC_CHANGE,
        },
        "n_runs": int(len(qc)),
        "n_unique_promoters": int(qc["ID"].nunique()),
        "constraint_violations_total": int((~qc["passes_all_constraints"]).sum()),
        "summary_by_task": summary.to_dict(orient="records"),
        "outputs": [str(OUT_STEM.with_suffix(ext)) for ext in [".svg", ".pdf", ".png", ".tiff"]],
        "single_panel_outputs_without_extension": [
            str(SINGLE_DIR / name)
            for name in ["S6a_total_mutation_burden", "S6b_local_mutation_density", "S6c_gc_content_stability"]
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
