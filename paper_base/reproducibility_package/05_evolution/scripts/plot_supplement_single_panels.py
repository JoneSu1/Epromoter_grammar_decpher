#!/usr/bin/env python
"""Export each Fig. S5 evidence panel as a standalone publication figure."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
DATA = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures" / "supplement_single"
LOG_DIR = ROOT / "logs"
MANIFEST = LOG_DIR / "supplement_single_panels_manifest.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import plotting module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s5a = load_module("plot_s5a_cohort_selection", SCRIPT_DIR / "plot_s5a_cohort_selection.py")
s5b = load_module("plot_s5b_highconf_activity", SCRIPT_DIR / "plot_s5b_highconf_activity.py")
s5c = load_module("plot_s5c_strict_residual_sensitivity", SCRIPT_DIR / "plot_s5c_strict_residual_sensitivity.py")


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
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
    plt.close(fig)


def export_s5a_panels(outputs: list[str]) -> None:
    cohort = pd.read_csv(DATA / "cohort_filter_counts.tsv", sep="\t")
    partition = pd.read_csv(DATA / "cage_highconf_partition.tsv", sep="\t")
    group_comp = pd.read_csv(DATA / "cage_group_composition.tsv", sep="\t")

    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)
    s5a.draw_funnel(ax, cohort)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.96, bottom=0.06)
    stem = FIG_DIR / "S5a_01_cohort_selection_funnel"
    save_pub(fig, stem)
    outputs.append(str(stem))

    fig, ax = plt.subplots(figsize=(3.0, 2.25), dpi=300)
    s5a.draw_partition(ax, partition)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.80, bottom=0.26)
    stem = FIG_DIR / "S5a_02_functional_composition"
    save_pub(fig, stem)
    outputs.append(str(stem))

    fig, ax = plt.subplots(figsize=(2.25, 2.55), dpi=300)
    s5a.draw_cage_groups(ax, group_comp)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.82, bottom=0.24)
    stem = FIG_DIR / "S5a_03_cage_stratum_representation"
    save_pub(fig, stem)
    outputs.append(str(stem))


def export_s5b_panels(outputs: list[str]) -> None:
    dual = pd.read_csv(DATA / "dual_function_performance.tsv", sep="\t")
    cohort = pd.read_csv(DATA / "fig5b_greedy_round0_cohort.tsv", sep="\t")

    scatter_specs = [
        (
            "S5b_01_cage_highconf_filtering",
            "CAGE",
            "Real_CAGE_log2TPM",
            "Pred_CAGE_log2TPM",
            s5b.READOUT_SIGNAL_COLORS["CAGE"],
        ),
        ("S5b_02_dev_highconf_filtering", "DEV", "Dev_true", "Dev_pred", s5b.READOUT_SIGNAL_COLORS["DEV"]),
        ("S5b_03_hk_highconf_filtering", "HK", "Hk_true", "Hk_pred", s5b.READOUT_SIGNAL_COLORS["HK"]),
    ]
    for filename, readout, measured_col, predicted_col, color in scatter_specs:
        fig, ax = plt.subplots(figsize=(2.75, 2.65), dpi=300)
        s5b.plot_highconf_scatter(
            ax,
            dual,
            readout=readout,
            measured_col=measured_col,
            predicted_col=predicted_col,
            color=color,
        )
        fig.subplots_adjust(left=0.31, right=0.97, top=0.88, bottom=0.29)
        stem = FIG_DIR / filename
        save_pub(fig, stem)
        outputs.append(str(stem))

    fig, ax = plt.subplots(figsize=(3.2, 2.65), dpi=300)
    s5b.plot_activity_distribution(ax, cohort)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.85, bottom=0.34)
    stem = FIG_DIR / "S5b_04_415_cohort_baseline_activity"
    save_pub(fig, stem)
    outputs.append(str(stem))


def export_s5c_panels(outputs: list[str]) -> None:
    strict = pd.read_csv(DATA / "quad_negative_promoters.tsv", sep="\t").copy()
    strict = strict.loc[strict["CAGE_Group"].isin(s5c.GROUP_ORDER)].copy()
    strict["CAGE_Group"] = pd.Categorical(strict["CAGE_Group"], categories=s5c.GROUP_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(3.15, 2.35), dpi=300)
    s5c.plot_residual_box(ax, strict)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.83, bottom=0.22)
    stem = FIG_DIR / "S5c_01_strict_starr_silent_residuals"
    save_pub(fig, stem)
    outputs.append(str(stem))

    fig, ax = plt.subplots(figsize=(2.85, 2.55), dpi=300)
    s5c.plot_accuracy_stack(ax, strict)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.83, bottom=0.30)
    stem = FIG_DIR / "S5c_02_residual_accuracy_classes"
    save_pub(fig, stem)
    outputs.append(str(stem))


def main() -> None:
    outputs: list[str] = []
    export_s5a_panels(outputs)
    export_s5b_panels(outputs)
    export_s5c_panels(outputs)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "Standalone Supplementary Fig. S5 panels for flexible manuscript layout.",
        "main_fig5_cohort": "415 reviewer-grade greedy cohort",
        "strict_subset_note": "Strict STARR-silent residual panels are sensitivity/QC only, not the main Fig. 5 cohort definition.",
        "cage_scale": "log2(TPM + 1)",
        "starr_seq_scale": "log2TPM",
        "outputs_without_extension": outputs,
        "formats": [".svg", ".pdf", ".png", ".tiff"],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
