#!/usr/bin/env python
# coding: utf-8
"""Supplement to Fig. 3e: activity-contrast examples for shared motifs.

Series 2 contrasts:
    HK+CAGE-, HK+CAGE+, DEV+CAGE-, DEV+CAGE+, HK-DEV-CAGE+

Definitions are quantile based on all 19,777 core-promoter predictions:
    "+" = task prediction >= task Q75
    "-" = task prediction <= task Q25

Within each contrast class, one representative sequence is selected by combining
task-contrast strength and summed shared-motif Fi-NeMo hit importance.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import logomaker
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
BASE_SCRIPT = SCRIPT.with_name("fig3e_core_promoter_shared_motif_logos_polished.py")
ROOT = SCRIPT.parents[4]
PLOT = ROOT / "plot"
OUT = PLOT / "output" / "polished_figure_code"
QA = OUT / "qa" / "figure3e_series2"
TRACE = OUT / "data" / "figure3e_series2_activity_contrast_trace.csv"
STEM_A = "figure3e_series2a_activity_contrast_3x3_shared_motif_logos"
STEM_B = "figure3e_series2b_activity_contrast_3x2_shared_motif_logos"

spec = importlib.util.spec_from_file_location("fig3e_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
sys.modules["fig3e_base"] = base
assert spec.loader is not None
spec.loader.exec_module(base)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.75,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


@dataclass(frozen=True)
class ContrastExample:
    contrast: str
    peak_id: int
    genomic_id: str
    pred_hk: float
    pred_dev: float
    pred_cage: float
    contrast_score: float
    motif_score: float
    n_candidates: int
    n_shared_motif_hits_total: int


CONTRASTS = [
    "HK+CAGE-",
    "HK+CAGE+",
    "DEV+CAGE-",
    "DEV+CAGE+",
    "HK-DEV-CAGE+",
]
FIGURE_SPLITS = [
    ("3x3", STEM_A, CONTRASTS[:3], (6.35, 4.1)),
    ("3x2", STEM_B, CONTRASTS[3:], (4.35, 4.1)),
]


def ensure_dirs() -> None:
    for suffix in ["png", "svg", "pdf", "tiff"]:
        (OUT / suffix / "figure3e_series2").mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    TRACE.parent.mkdir(parents=True, exist_ok=True)


def load_all_predictions() -> pd.DataFrame:
    arrays = {}
    n = None
    for task, path in base.PRED_H5.items():
        with h5py.File(path, "r") as h5:
            pred = h5[task]["pred"][:].astype(float)
        arrays[task] = pred
        n = len(pred) if n is None else n
    assert n is not None
    return pd.DataFrame(
        {
            "peak_id": np.arange(n, dtype=int),
            "HK": arrays["HK"],
            "DEV": arrays["DEV"],
            "CAGE": arrays["CAGE"],
        }
    )


def thresholds(preds: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        task: {"low": float(preds[task].quantile(0.25)), "high": float(preds[task].quantile(0.75))}
        for task in base.TASK_ORDER
    }


def contrast_mask(preds: pd.DataFrame, thr: dict[str, dict[str, float]], contrast: str) -> pd.Series:
    if contrast == "HK+CAGE-":
        return (preds["HK"] >= thr["HK"]["high"]) & (preds["CAGE"] <= thr["CAGE"]["low"])
    if contrast == "HK+CAGE+":
        return (preds["HK"] >= thr["HK"]["high"]) & (preds["CAGE"] >= thr["CAGE"]["high"])
    if contrast == "DEV+CAGE-":
        return (preds["DEV"] >= thr["DEV"]["high"]) & (preds["CAGE"] <= thr["CAGE"]["low"])
    if contrast == "DEV+CAGE+":
        return (preds["DEV"] >= thr["DEV"]["high"]) & (preds["CAGE"] >= thr["CAGE"]["high"])
    if contrast == "HK-DEV-CAGE+":
        return (
            (preds["HK"] <= thr["HK"]["low"])
            & (preds["DEV"] <= thr["DEV"]["low"])
            & (preds["CAGE"] >= thr["CAGE"]["high"])
        )
    raise ValueError(f"Unknown contrast: {contrast}")


def shared_motif_score(hits_by_task: dict[str, pd.DataFrame], peak_id: int, top_n: int = 5) -> tuple[float, int]:
    total = 0.0
    n_hits = 0
    for task in base.TASK_ORDER:
        sub = hits_by_task[task][hits_by_task[task]["peak_id"] == int(peak_id)]
        n_hits += len(sub)
        if len(sub):
            total += float(sub.nlargest(min(top_n, len(sub)), "hit_importance")["hit_importance"].sum())
    return total, n_hits


def draw_motif_markers_compact(ax, motifs, y_top: float, y_span: float) -> None:
    occupied: list[float] = []
    base_y = y_top - y_span * 0.25
    step = y_span * 0.11
    for start, end, label, _score in motifs:
        ax.axvspan(start, end, facecolor="#9ECAE1", alpha=0.24, edgecolor="none", zorder=-10)
        center = (start + end) / 2.0
        half_width = max(12.0, len(label) * 3.25)
        left = center - half_width
        right = center + half_width
        level = 0
        while level < len(occupied) and left <= occupied[level] + 2:
            level += 1
        if level == len(occupied):
            occupied.append(right)
        else:
            occupied[level] = right
        y = base_y - step * level
        ax.plot([start, end], [y - step * 0.12, y - step * 0.12], color="#2166AC", lw=1.45, solid_capstyle="butt")
        ax.text(center, y, label, ha="center", va="bottom", fontsize=6.5, color="black")


def draw_track_compact(
    ax,
    df: pd.DataFrame,
    motifs,
    task: str,
    pred_value: float,
    panel_label: str,
    show_ylabel: bool,
    show_xlabel: bool,
    ylim: tuple[float, float],
) -> None:
    logo = logomaker.Logo(
        df,
        ax=ax,
        color_scheme=base.BASE_COLORS,
        vpad=0.01,
        width=0.92,
        baseline_width=0.42,
    )
    logo.style_spines(visible=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.52)
    ax.axhline(0, color="0.25", lw=0.50, zorder=0)
    draw_motif_markers_compact(ax, motifs, ylim[1], ylim[1] - ylim[0])
    ax.text(
        0.01,
        1.58,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="black",
        linespacing=1.05,
    )
    ax.text(
        0.99,
        1.055,
        f"{task} pred={pred_value:.2f}",
        transform=ax.transAxes,
        color=base.TASK_COLORS[task],
        ha="right",
        va="bottom",
        fontsize=8.7,
        fontweight="bold",
    )
    ax.set_xlim(0, 248)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_yticks([0, round(ylim[1] * 0.55, 2)])
    ax.tick_params(axis="both", labelsize=7.0, length=2)
    ax.set_ylabel("Contribution" if show_ylabel else "", fontsize=7.5)
    ax.set_xlabel("Position in region" if show_xlabel else "", fontsize=7.5)


def score_contrast(row: pd.Series, thr: dict[str, dict[str, float]], contrast: str) -> float:
    if contrast == "HK+CAGE-":
        return float((row["HK"] - thr["HK"]["high"]) + (thr["CAGE"]["low"] - row["CAGE"]))
    if contrast == "HK+CAGE+":
        return float((row["HK"] - thr["HK"]["high"]) + (row["CAGE"] - thr["CAGE"]["high"]))
    if contrast == "DEV+CAGE-":
        return float((row["DEV"] - thr["DEV"]["high"]) + (thr["CAGE"]["low"] - row["CAGE"]))
    if contrast == "DEV+CAGE+":
        return float((row["DEV"] - thr["DEV"]["high"]) + (row["CAGE"] - thr["CAGE"]["high"]))
    if contrast == "HK-DEV-CAGE+":
        return float(
            (thr["HK"]["low"] - row["HK"])
            + (thr["DEV"]["low"] - row["DEV"])
            + (row["CAGE"] - thr["CAGE"]["high"])
        )
    raise ValueError(contrast)


def select_examples(
    preds: pd.DataFrame,
    thr: dict[str, dict[str, float]],
    hits_by_task: dict[str, pd.DataFrame],
    genomic_ids: dict[int, str],
) -> list[ContrastExample]:
    examples = []
    used_peak_ids: set[int] = set()
    for contrast in CONTRASTS:
        candidates = preds.loc[contrast_mask(preds, thr, contrast)].copy()
        if candidates.empty:
            raise RuntimeError(f"No candidates for {contrast}.")
        ranked = []
        for _, row in candidates.iterrows():
            peak_id = int(row["peak_id"])
            if peak_id in used_peak_ids:
                continue
            motif_score, n_hits = shared_motif_score(hits_by_task, peak_id)
            if n_hits < 2:
                continue
            contrast_score = score_contrast(row, thr, contrast)
            # Activity contrast is the primary message for this supplement;
            # motif score is the secondary criterion to keep labels visible.
            ranked.append((contrast_score, motif_score, n_hits, peak_id, row))
        if not ranked:
            raise RuntimeError(f"No shared-motif-hit candidates for {contrast}.")
        contrast_score, motif_score, n_hits, peak_id, row = sorted(ranked, reverse=True)[0]
        used_peak_ids.add(int(peak_id))
        examples.append(
            ContrastExample(
                contrast=contrast,
                peak_id=int(peak_id),
                genomic_id=genomic_ids.get(int(peak_id), f"prom_{peak_id}"),
                pred_hk=float(row["HK"]),
                pred_dev=float(row["DEV"]),
                pred_cage=float(row["CAGE"]),
                contrast_score=float(contrast_score),
                motif_score=float(motif_score),
                n_candidates=int(len(candidates)),
                n_shared_motif_hits_total=int(n_hits),
            )
        )
    return examples


def save_all(fig: plt.Figure, stem: str) -> list[Path]:
    paths = [
        OUT / "png" / "figure3e_series2" / f"{stem}.png",
        OUT / "svg" / "figure3e_series2" / f"{stem}.svg",
        OUT / "pdf" / "figure3e_series2" / f"{stem}.pdf",
        OUT / "tiff" / "figure3e_series2" / f"{stem}.tiff",
        QA / f"{stem}_preview.png",
    ]
    fig.savefig(paths[0], bbox_inches="tight", dpi=600)
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], bbox_inches="tight")
    fig.savefig(paths[3], bbox_inches="tight", dpi=600)
    fig.savefig(paths[4], bbox_inches="tight", dpi=600)
    return paths


def draw_split_figure(
    split_name: str,
    stem: str,
    split_contrasts: list[str],
    figsize: tuple[float, float],
    examples_by_contrast: dict[str, ContrastExample],
    hits_by_task: dict[str, pd.DataFrame],
    arrays_by_task: dict[str, dict[str, np.ndarray]],
    ylim: tuple[float, float],
) -> tuple[list[Path], list[dict]]:
    fig, axes = plt.subplots(len(base.TASK_ORDER), len(split_contrasts), figsize=figsize, sharex=True, sharey=False)
    if len(split_contrasts) == 1:
        axes = np.asarray(axes).reshape(len(base.TASK_ORDER), 1)
    trace_rows = []
    for col_idx, contrast in enumerate(split_contrasts):
        ex = examples_by_contrast[contrast]
        column_label = f"{ex.contrast}\n{base.genomic_id_for_display(ex.genomic_id)}"
        for row_idx, task in enumerate(base.TASK_ORDER):
            motifs = base.top_motifs(hits_by_task[task], ex.peak_id, max_n=3)
            pred_value = {"HK": ex.pred_hk, "DEV": ex.pred_dev, "CAGE": ex.pred_cage}[task]
            draw_track_compact(
                axes[row_idx, col_idx],
                base.logo_df(arrays_by_task[task], ex.peak_id),
                motifs,
                task,
                pred_value,
                panel_label=column_label if row_idx == 0 else "",
                show_ylabel=col_idx == 0,
                show_xlabel=row_idx == len(base.TASK_ORDER) - 1,
                ylim=ylim,
            )
            if row_idx < len(base.TASK_ORDER) - 1:
                axes[row_idx, col_idx].tick_params(labelbottom=False)
            trace_rows.append(
                {
                    "figure_split": split_name,
                    "contrast": ex.contrast,
                    "task": task,
                    "peak_id": ex.peak_id,
                    "genomic_id": ex.genomic_id,
                    "pred_HK": ex.pred_hk,
                    "pred_DEV": ex.pred_dev,
                    "pred_CAGE": ex.pred_cage,
                    "contrast_score": ex.contrast_score,
                    "motif_score": ex.motif_score,
                    "n_candidates": ex.n_candidates,
                    "n_shared_motif_hits_total": ex.n_shared_motif_hits_total,
                    "n_motifs_displayed": len(motifs),
                    "motifs_displayed": ";".join(m[2] for m in motifs),
                }
            )

    fig.subplots_adjust(left=0.080 if len(split_contrasts) == 3 else 0.115, right=0.995, top=0.80, bottom=0.13, wspace=0.18, hspace=0.62)
    paths = save_all(fig, stem)
    plt.close(fig)
    return paths, trace_rows


def main() -> None:
    ensure_dirs()
    base.check_inputs()
    preds = load_all_predictions()
    thr = thresholds(preds)
    hits_by_task = base.load_hits()
    arrays_by_task = base.load_logo_arrays()
    genomic_ids = base.load_genomic_ids()
    examples = select_examples(preds, thr, hits_by_task, genomic_ids)

    ylim = base.compute_ylim(
        [
            base.Example(
                class_label=ex.contrast,
                peak_id=ex.peak_id,
                region_name=f"prom_{ex.peak_id}",
                genomic_id=ex.genomic_id,
                pred_hk=ex.pred_hk,
                pred_dev=ex.pred_dev,
                pred_cage=ex.pred_cage,
                selection_score=ex.motif_score,
                n_hits_total=ex.n_shared_motif_hits_total,
            )
            for ex in examples
        ],
        arrays_by_task,
    )

    examples_by_contrast = {ex.contrast: ex for ex in examples}
    paths = []
    trace_rows = []
    for split_name, stem, split_contrasts, figsize in FIGURE_SPLITS:
        split_paths, split_trace = draw_split_figure(
            split_name,
            stem,
            split_contrasts,
            figsize,
            examples_by_contrast,
            hits_by_task,
            arrays_by_task,
            ylim,
        )
        paths.extend(split_paths)
        for row in split_trace:
            row.update(
                {
                    "threshold_HK_low_Q25": thr["HK"]["low"],
                    "threshold_HK_high_Q75": thr["HK"]["high"],
                    "threshold_DEV_low_Q25": thr["DEV"]["low"],
                    "threshold_DEV_high_Q75": thr["DEV"]["high"],
                    "threshold_CAGE_low_Q25": thr["CAGE"]["low"],
                    "threshold_CAGE_high_Q75": thr["CAGE"]["high"],
                }
            )
        trace_rows.extend(split_trace)

    pd.DataFrame(trace_rows).to_csv(TRACE, index=False)
    manifest = {
        "panel": "Fig3e supplement series 2",
        "archetype": "quantitative grid",
        "contrast_definitions": {
            "+": "task prediction >= task Q75 across all 19,777 core-promoter predictions",
            "-": "task prediction <= task Q25 across all 19,777 core-promoter predictions",
        },
        "thresholds": thr,
        "outputs": [str(path) for path in paths],
        "trace": str(TRACE),
    }
    (TRACE.parent / "figure3e_series2_activity_contrast_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Wrote trace:", TRACE)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
