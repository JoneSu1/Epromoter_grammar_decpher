#!/usr/bin/env python
"""Split Fig. 5d full-length contribution logos into branch-paired plates.

Each selected sequence is exported as two 249 bp logo plates:

1. HK target with its HK-CAGE readout.
2. DEV target with its DEV-CAGE readout.

The color semantics match Fig. 5f: HK target, HK-CAGE, DEV target and
DEV-CAGE are treated as four named series. Contributions are actual
base-level values computed upstream as hypothetical contribution multiplied by
the one-hot sequence, including CAGE.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_fig5_polished as base
import plot_fig5d_fig1e_style_full249 as full249


OUT_DIR = base.POLISH_ROOT / "figures" / "Fig5d_branch_pairs_full249_FINAL"
ALL10_DIR = OUT_DIR / "all10"
LOW_HIGH_DIR = OUT_DIR / "low_high_only"
FORCE_OVERWRITE = os.environ.get("FIG5D_FORCE_OVERWRITE", "0") == "1"
SERIES_COLORS = {
    "HK target": base.SERIES_COLORS["HK target"],
    "HK-CAGE": base.SERIES_COLORS["HK-CAGE"],
    "DEV target": base.SERIES_COLORS["DEV target"],
    "DEV-CAGE": base.SERIES_COLORS["DEV-CAGE"],
}
PAIR_SPECS = {
    "HK": [
        ("HK target", "HK", "HK", "round0"),
        ("HK target", "HK", "HK", "hit6"),
        ("HK-CAGE", "HK", "CAGE", "round0"),
        ("HK-CAGE", "HK", "CAGE", "hit6"),
    ],
    "DEV": [
        ("DEV target", "DEV", "DEV", "round0"),
        ("DEV target", "DEV", "DEV", "hit6"),
        ("DEV-CAGE", "DEV", "CAGE", "round0"),
        ("DEV-CAGE", "DEV", "CAGE", "hit6"),
    ],
}


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 20,
        "axes.labelsize": 21,
        "axes.titlesize": 22,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
        "axes.linewidth": 0.9,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save_pair(fig: mpl.figure.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [stem.with_suffix(".svg"), stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], bbox_inches="tight", dpi=600)
    print(f"Saved {stem}")
    return outputs


def copy_pair_exports(source_files: list[Path], target_stem: Path) -> list[Path]:
    target_stem.parent.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source in source_files:
        target = target_stem.with_suffix(source.suffix)
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def expected_exports(stem: Path) -> list[Path]:
    return [stem.with_suffix(".svg"), stem.with_suffix(".pdf"), stem.with_suffix(".png")]


def exports_complete(stem: Path) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in expected_exports(stem))


def row_title(series: str, milestone: str, track: str, row: pd.Series) -> tuple[str, str]:
    pred_value = full249.pred_for_track(track, row)
    if milestone == "round0":
        return f"{series} | initial", f"round0 | pred={pred_value:.2f}"
    round_value = int(row["Round"]) if pd.notna(row["Round"]) else 0
    return f"{series} | hit6", f"R{round_value} | pred={pred_value:.2f}"


def render_pair(
    *,
    seq_no: int,
    seq_id: str,
    cage_group: str,
    pair_name: str,
    attr_by_track: dict[str, dict[str, object]],
    annotated_hits: pd.DataFrame,
    score_max: float,
    y_lim: tuple[float, float],
    pad: float,
) -> mpl.figure.Figure:
    fig, axes = plt.subplots(4, 1, figsize=(14.2, 8.2), dpi=300, sharex=False, sharey=True)
    for axis_index, (ax, (series, branch, track, milestone)) in enumerate(zip(axes, PAIR_SPECS[pair_name])):
        meta_row = full249.select_meta_row(attr_by_track, track, branch, seq_id, milestone)
        muts = full249.mutation_positions(attr_by_track, branch, seq_id)
        left_title, right_title = row_title(series, milestone, track, meta_row)
        full249.render_axis(
            ax,
            arr=full249.arr_for(attr_by_track, track, meta_row),
            motifs=full249.top_motifs(full249.hits_for(annotated_hits, track, branch, seq_id, milestone), 0, 249),
            muts=muts,
            left_title=left_title,
            right_title=right_title,
            left_title_color=SERIES_COLORS[series],
            y_lim=y_lim,
            y_ref_max=score_max,
            pad=pad,
            show_xlabel=(axis_index == len(PAIR_SPECS[pair_name]) - 1),
        )

    header = f"{base.short_id(seq_id)} | DeepCAGE {cage_group}"
    axes[0].text(0.0, 1.54, header, transform=axes[0].transAxes, ha="left", va="bottom", fontsize=22.0)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.075, top=0.895, hspace=1.04)
    return fig


def plot_branch_pairs_full249() -> None:
    ALL10_DIR.mkdir(parents=True, exist_ok=True)
    LOW_HIGH_DIR.mkdir(parents=True, exist_ok=True)

    selected10 = pd.read_csv(base.PLOT_V1_DATA / "legacy_selected_10_metadata.tsv", sep="\t")
    selected10["ID"] = selected10["ID"].astype(str)
    selected_ids = set(selected10["ID"])
    attr_by_track, annotated_hits = full249.load_inputs(selected_ids)

    all_attr = np.concatenate([np.asarray(v["actual"]).reshape(-1) for v in attr_by_track.values()])
    score_min = float(np.nanpercentile(all_attr, 0.25))
    score_max = float(np.nanpercentile(all_attr, 99.75))
    span = max(score_max - score_min, 0.08)
    pad = max(span * 0.18, 0.025)
    y_lim = (min(score_min - pad * 0.55, -0.03), score_max + pad * 4.8)

    exported_all: list[str] = []
    exported_low_high: list[str] = []
    missing: list[str] = []
    for seq_no, row in enumerate(selected10.itertuples(index=False), start=1):
        seq_id = str(row.ID)
        try:
            for branch in ["HK", "DEV"]:
                _ = full249.select_meta_row(attr_by_track, branch, branch, seq_id, "hit6")
                _ = full249.select_meta_row(attr_by_track, "CAGE", branch, seq_id, "hit6")
        except KeyError:
            missing.append(seq_id)
            continue

        for pair_name in ["HK", "DEV"]:
            stem = ALL10_DIR / f"Fig5d_pair_seq{seq_no:02d}_{pair_name}_{base.short_id(seq_id)}"
            low_high_stem = LOW_HIGH_DIR / stem.name
            needs_low_high = str(row.CAGE_Group) != "Medium (2-4)"
            if exports_complete(stem) and not FORCE_OVERWRITE:
                exported_all.append(str(stem))
                if needs_low_high:
                    if not exports_complete(low_high_stem):
                        copy_pair_exports(expected_exports(stem), low_high_stem)
                    exported_low_high.append(str(low_high_stem))
                print(f"Checkpoint skip {stem.name}")
                continue

            fig = render_pair(
                seq_no=seq_no,
                seq_id=seq_id,
                cage_group=str(row.CAGE_Group),
                pair_name=pair_name,
                attr_by_track=attr_by_track,
                annotated_hits=annotated_hits,
                score_max=score_max,
                y_lim=y_lim,
                pad=pad,
            )
            source_files = save_pair(fig, stem)
            exported_all.append(str(stem))
            if needs_low_high:
                copy_pair_exports(source_files, low_high_stem)
                exported_low_high.append(str(low_high_stem))
            plt.close(fig)

    group_counts = selected10["CAGE_Group"].value_counts().reindex(base.GROUP_FULL, fill_value=0).to_dict()
    manifest = {
        "n_sequences_requested": int(len(selected10)),
        "n_pair_figures_exported_all10": int(len(exported_all)),
        "n_pair_figures_exported_low_high_only": int(len(exported_low_high)),
        "cage_group_counts_in_selected10": {str(k): int(v) for k, v in group_counts.items()},
        "missing_ids": missing,
        "layout": "Each original selected sequence is split into two full 249 bp paired logo plates: HK/HK-CAGE and DEV/DEV-CAGE.",
        "panel_label_policy": "No per-sequence panel labels are drawn inside the plates.",
        "low_high_only_rule": "Baseline DeepCAGE Medium (2-4) sequences are excluded only from the low_high_only folder; all original selected sequences remain in all10.",
        "normalization": "Raw actual base-level contributions are shown; no median normalization is applied.",
        "series_color_policy": SERIES_COLORS,
        "source": (
            "reviewer_greedy_20260724_115905 attribution_finemo_24bp_annotated; "
            "actual contribution computed explicitly as hyp_contrib multiplied by one_hot for HK, DEV and CAGE"
        ),
    }
    (OUT_DIR / "fig5d_branch_pairs_full249_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    plot_branch_pairs_full249()
