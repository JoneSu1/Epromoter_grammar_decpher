from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from deepisa_paths import DATA_ROOT, panel_dirs
from deepisa_style import DOUBLE_COL_MM, INK, TASK_COLORS, mm_to_in, save_all, set_pub_style


@dataclass(frozen=True)
class TaskSpec:
    label: str
    folder: str
    track: int


PANEL = "fig2_screening_null_distribution"
RESULT_ROOT = DATA_ROOT.parents[1] / "frozen_data" / "deepisa_raw"
TASKS = [
    TaskSpec("CAGE", "results_cage_newisa", 0),
    TaskSpec("DEV", "results_dev_newisa", 0),
    TaskSpec("HK", "results_hk_newisa", 1),
]
NULL_COLOR = "#767676"
REAL_COLOR = "#272727"
EXPORT_DPI = 600
SUMMARY_CSV = DATA_ROOT / "fig2" / "fig2_real_vs_null_distribution_summary.csv"
width_mm = 183
EXPORT_TARGETS = ("figure.svg", "figure.pdf", "figure.tiff", "figure.png")


def data_dir(task: TaskSpec) -> Path:
    return RESULT_ROOT / task.folder


def isa_col(task: TaskSpec) -> str:
    return f"isa_t{task.track}"


def interaction_col(task: TaskSpec) -> str:
    return f"interaction_t{task.track}"


def read_task_table(task: TaskSpec, filename: str) -> pd.DataFrame:
    path = data_dir(task) / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_raw() -> dict[str, dict[str, pd.DataFrame]]:
    files = ["motif_single_isa.csv", "motif_combi_isa.csv", "null_isa.csv", "null_interaction.csv"]
    return {
        task.label: {name.replace(".csv", ""): read_task_table(task, name) for name in files}
        for task in TASKS
    }


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.08, 1.08, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=14, fontweight="bold", color="black")


def annotate_panel(ax, summary: pd.Series) -> None:
    text = (
        f"n={int(summary.real_n):,}/{int(summary.null_n):,}\n"
        f"P={summary.p_label}"
    )
    ax.text(
        0.03, 0.96, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=9.4, linespacing=0.95, color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.96, "pad": 1.2},
    )


def finite_numeric(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric[numeric.notna()].astype(float)


def plot_density(ax, real: pd.Series, null: pd.Series, summary: pd.Series, xlabel: str) -> None:
    real = finite_numeric(real)
    null = finite_numeric(null)
    sns.kdeplot(null, ax=ax, color=NULL_COLOR, lw=1.5, label="Null", cut=0, warn_singular=False)
    sns.kdeplot(real, ax=ax, color=REAL_COLOR, lw=1.8, label="Motif-called", cut=0, warn_singular=False)
    ax.axvline(0, color="#999999", lw=1.0, ls=":")
    annotate_panel(ax, summary)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=3)
    ax.tick_params(axis="both", labelsize=12, width=0.9, length=4.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.95)


def build_qa_counts(raw: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        specs = [
            ("single_motif_isa", "motif_single_isa", isa_col(task), "null_isa", isa_col(task)),
            ("motif_pair_interaction", "motif_combi_isa", interaction_col(task),
             "null_interaction", interaction_col(task)),
        ]
        for comparison, real_table, real_col, null_table, null_col in specs:
            real = raw[task.label][real_table][real_col]
            null = raw[task.label][null_table][null_col]
            rows.append({
                "task": task.label,
                "comparison": comparison,
                "real_table": real_table,
                "real_column": real_col,
                "real_rows_before_dropna": len(real),
                "real_rows_plotted": int(real.notna().sum()),
                "null_table": null_table,
                "null_column": null_col,
                "null_rows_before_dropna": len(null),
                "null_rows_plotted": int(null.notna().sum()),
                "dropna_reason": "KDE requires finite numeric scores; source tables are otherwise unchanged.",
            })
    return pd.DataFrame(rows)


def main() -> None:
    set_pub_style(font_size=12)
    dirs = panel_dirs(PANEL)
    raw = load_raw()
    summary = pd.read_csv(SUMMARY_CSV)

    fig, axes = plt.subplots(
        2, 3,
        figsize=(mm_to_in(DOUBLE_COL_MM), mm_to_in(128)),
        gridspec_kw={"wspace": 0.56, "hspace": 0.82},
        dpi=300,
    )

    for col, task in enumerate(TASKS):
        panels = [
            (
                0,
                raw[task.label]["motif_single_isa"][isa_col(task)],
                raw[task.label]["null_isa"][isa_col(task)],
                "single_motif_isa",
                "Single-motif ISA\nscore",
            ),
            (
                1,
                raw[task.label]["motif_combi_isa"][interaction_col(task)],
                raw[task.label]["null_interaction"][interaction_col(task)],
                "motif_pair_interaction",
                "Motif-pair interaction\nscore",
            ),
        ]
        for row, real, null, comparison, xlabel in panels:
            ax = axes[row, col]
            panel_summary = summary[(summary.task == task.label) & (summary.comparison == comparison)].iloc[0]
            plot_density(ax, real, null, panel_summary, xlabel)
            ax.set_title(task.label if row == 0 else "", fontsize=13.4, color=TASK_COLORS[task.label], pad=8)
            ax.set_ylabel("Density" if col == 0 else "", fontsize=12)

    handles, labels = axes[0, 2].get_legend_handles_labels()
    for ax in axes.flat:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.985),
               ncol=2, columnspacing=1.2, handlelength=1.8, frameon=False, fontsize=11)

    for ax in axes.flat:
        ax.grid(False)
        ax.tick_params(colors=INK)

    fig.subplots_adjust(left=0.085, right=0.965, bottom=0.13, top=0.84)
    paths = save_all(fig, dirs, PANEL, dpi=EXPORT_DPI)
    plt.close(fig)
    summary.to_csv(dirs["qa"] / "source_summary.csv", index=False)
    build_qa_counts(raw).to_csv(dirs["qa"] / "dropna_counts.csv", index=False)
    build_qa_counts(raw).to_csv(DATA_ROOT / "fig2" / "fig2_total_and_plotted_counts.csv", index=False)
    print(paths["png"])


if __name__ == "__main__":
    main()
