#!/usr/bin/env python
"""Fig. 5d full-length contribution logos in the DeepSTARR Fig. 1e style.

The renderer keeps the complete 249 bp axis and draws base-level contribution
letters on a wide panel. Actual contribution is computed explicitly as
hypothetical contribution multiplied by one-hot sequence for every track,
including CAGE.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd

import plot_fig5_polished as base


OUT_DIR = base.POLISH_ROOT / "figures" / "Fig5d_fig1e_style_full249_FINAL"
CAGE_TITLE_COLOR = "#D55E00"
ROW_SPECS = [
    ("HK", "HK", "round0"),
    ("HK", "HK", "hit6"),
    ("HK", "CAGE", "round0"),
    ("HK", "CAGE", "hit6"),
    ("DEV", "DEV", "round0"),
    ("DEV", "DEV", "hit6"),
    ("DEV", "CAGE", "round0"),
    ("DEV", "CAGE", "hit6"),
]

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


def save_fig1e_style(fig: mpl.figure.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=600)
    print(f"Saved {stem.name}")


def load_inputs(selected_ids: set[str]) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    attr_by_track: dict[str, dict[str, object]] = {}
    for track in ["HK", "DEV", "CAGE"]:
        z = np.load(base.ANNOTATED_ATTR_DIR / track / "attribution_arrays.npz")
        meta = pd.read_csv(base.ANNOTATED_ATTR_DIR / track / "metadata.tsv", sep="\t")
        meta["ID"] = meta["ID"].astype(str)
        actual = z["hyp_contrib"] * z["one_hot"]
        attr_by_track[track] = {
            "actual": actual,
            "one_hot": z["one_hot"],
            "meta": meta,
        }

    hit_cols = [
        "track",
        "branch_task",
        "ID",
        "milestone",
        "start",
        "end",
        "hit_importance",
        "hit_coefficient",
        "MC_ID",
        "Match_1",
    ]
    hits = pd.read_csv(base.ANNOTATED_HITS_LONG, sep="\t", usecols=hit_cols)
    hits["ID"] = hits["ID"].astype(str)
    hits = hits.loc[
        hits["ID"].isin(selected_ids)
        & hits["track"].isin(["HK", "DEV", "CAGE"])
        & hits["branch_task"].isin(["HK", "DEV"])
        & hits["milestone"].isin(["round0", "hit6"])
    ].copy()
    return attr_by_track, hits


def select_meta_row(
    attr_by_track: dict[str, dict[str, object]],
    track: str,
    branch: str,
    seq_id: str,
    milestone: str,
) -> pd.Series:
    meta = attr_by_track[track]["meta"]
    assert isinstance(meta, pd.DataFrame)
    sub = meta.loc[
        meta["ID"].eq(seq_id)
        & meta["branch_task"].eq(branch)
        & meta["milestone"].eq(milestone)
    ]
    if sub.empty and milestone == "round0":
        sub = meta.loc[meta["ID"].eq(seq_id) & meta["milestone"].eq("round0")]
    if sub.empty:
        raise KeyError(f"Missing attribution row for {track}/{branch}/{milestone}: {seq_id}")
    return sub.sort_values(["sequence_index"]).iloc[0]


def arr_for(attr_by_track: dict[str, dict[str, object]], track: str, row: pd.Series) -> np.ndarray:
    actual = attr_by_track[track]["actual"]
    assert isinstance(actual, np.ndarray)
    return actual[int(row["sequence_index"])]


def pred_for_track(track: str, row: pd.Series) -> float:
    if track == "HK":
        return float(row["Hk_pred"])
    if track == "DEV":
        return float(row["Dev_pred"])
    return float(row["CAGE_pred"])


def display_track_name(track: str, branch: str) -> str:
    if track != "CAGE":
        return track
    return "HCAGE" if branch == "HK" else "DCAGE"


def title_parts_for_row(track: str, branch: str, milestone: str, row: pd.Series) -> tuple[str, str]:
    display = display_track_name(track, branch)
    pred_value = pred_for_track(track, row)
    if milestone == "round0":
        return f"{display} | initial", f"round0 | pred={pred_value:.2f}"
    round_value = int(row["Round"]) if pd.notna(row["Round"]) else 0
    return f"{display} | hit6", f"R{round_value} | pred={pred_value:.2f}"


def title_color(branch: str) -> str:
    return base.TASK_COLORS[branch]


def left_title_color(track: str, branch: str) -> str:
    if track == "CAGE":
        return CAGE_TITLE_COLOR
    return title_color(branch)


def sequence_for(attr_by_track: dict[str, dict[str, object]], track: str, row: pd.Series) -> str:
    one_hot = attr_by_track[track]["one_hot"]
    assert isinstance(one_hot, np.ndarray)
    return base.onehot_to_seq(one_hot[int(row["sequence_index"])])


def mutation_positions(attr_by_track: dict[str, dict[str, object]], branch: str, seq_id: str) -> list[int]:
    row0 = select_meta_row(attr_by_track, branch, branch, seq_id, "round0")
    row1 = select_meta_row(attr_by_track, branch, branch, seq_id, "hit6")
    seq0 = sequence_for(attr_by_track, branch, row0)
    seq1 = sequence_for(attr_by_track, branch, row1)
    return [i + 1 for i, (a, b) in enumerate(zip(seq0, seq1)) if a != b]


def hits_for(hits: pd.DataFrame, track: str, branch: str, seq_id: str, milestone: str) -> pd.DataFrame:
    sub = hits.loc[
        hits["track"].eq(track)
        & hits["branch_task"].eq(branch)
        & hits["ID"].eq(seq_id)
        & hits["milestone"].eq(milestone)
    ].copy()
    if sub.empty and milestone == "round0":
        sub = hits.loc[
            hits["track"].eq(track)
            & hits["ID"].eq(seq_id)
            & hits["milestone"].eq("round0")
        ].copy()
    return sub


def top_motifs(hits: pd.DataFrame, start: int, end: int, max_n: int = 2) -> list[tuple[int, int, str, float]]:
    if hits.empty:
        return []
    sub = hits.copy()
    sub["clip_start"] = sub["start"].clip(lower=start, upper=end)
    sub["clip_end"] = sub["end"].clip(lower=start, upper=end)
    sub = sub.loc[sub["clip_end"] > sub["clip_start"]].copy()
    sub = sub.sort_values(["hit_importance", "hit_coefficient"], ascending=False)
    kept: list[tuple[int, int, str, float]] = []
    for row in sub.itertuples(index=False):
        s = int(row.clip_start)
        e = int(row.clip_end)
        if any(base.interval_overlap_bp((s, e), (ks, ke)) > 2 for ks, ke, _, _ in kept):
            continue
        label = f"{row.MC_ID} {row.Match_1}"
        confidence = float(row.hit_importance) if pd.notna(row.hit_importance) else float(row.hit_coefficient)
        kept.append((s, e, label, confidence))
        if len(kept) >= max_n:
            break
    return sorted(kept, key=lambda item: (item[0], item[1]))


def draw_full_letter_logo(ax: mpl.axes.Axes, arr: np.ndarray, start: int = 0, end: int = 249) -> None:
    for pos in range(start, end):
        vals = arr[pos, :]
        x = pos + 1
        pos_bottom = 0.0
        neg_bottom = 0.0
        for base_name, value in sorted(zip(base.BASES, vals), key=lambda item: item[1]):
            if value < 0:
                draw_base_letter(ax, base_name, x, neg_bottom, float(value))
                neg_bottom += float(value)
        for base_name, value in sorted(zip(base.BASES, vals), key=lambda item: item[1]):
            if value > 0:
                draw_base_letter(ax, base_name, x, pos_bottom, float(value))
                pos_bottom += float(value)


def draw_base_letter(ax: mpl.axes.Axes, base_name: str, x: float, y0: float, height: float) -> None:
    if abs(height) < 1e-4:
        return
    font_prop = FontProperties(family="Arial", weight="bold")
    path = TextPath((0, 0), base_name, size=1, prop=font_prop)
    bbox = path.get_extents()
    sx = 0.78 / max(bbox.width, 1e-6)
    sy = abs(height) / max(bbox.height, 1e-6)
    y = y0 if height > 0 else y0 + height
    transform = Affine2D().scale(sx, sy).translate(x - 0.39, y)
    patch = PathPatch(path, transform=transform + ax.transData, color=base.BASE_COLORS[base_name], lw=0)
    ax.add_patch(patch)


def draw_fig1e_motifs(
    ax: mpl.axes.Axes,
    motifs: list[tuple[int, int, str, float]],
    *,
    y_ref_max: float,
    pad: float,
) -> None:
    label_levels: list[float] = []
    for start, end, label, _ in motifs:
        x0, x1 = start + 1, end + 1
        ax.axvspan(x0, x1, facecolor="#9ECAE1", alpha=0.34, edgecolor="none", linewidth=0, zorder=0)
        label_x = (x0 + x1) / 2
        half_width = max(13.0, len(label) * 1.35)
        left = label_x - half_width
        level = 0
        while level < len(label_levels) and left <= label_levels[level] + 3:
            level += 1
        right = label_x + half_width
        if level == len(label_levels):
            label_levels.append(right)
        else:
            label_levels[level] = right
        y_text = max(y_ref_max, 0.0) + pad * (0.55 + 1.55 * level)
        y_bar = y_text - pad * 0.18
        ax.plot([x0, x1], [y_bar, y_bar], color="#2166AC", lw=2.0, solid_capstyle="butt", zorder=4)
        ax.text(label_x, y_text, label, ha="center", va="bottom", fontsize=18.8, color="black")


def render_axis(
    ax: mpl.axes.Axes,
    *,
    arr: np.ndarray,
    motifs: list[tuple[int, int, str, float]],
    muts: list[int],
    left_title: str,
    right_title: str,
    left_title_color: str,
    y_lim: tuple[float, float],
    y_ref_max: float,
    pad: float,
    show_xlabel: bool,
) -> None:
    draw_full_letter_logo(ax, arr)
    draw_fig1e_motifs(ax, motifs, y_ref_max=y_ref_max, pad=pad)
    for pos in muts:
        ax.plot([pos, pos], [y_lim[0], y_lim[0] + pad * 0.42], color="#B2182B", lw=0.75, solid_capstyle="butt")
    ax.axhline(0, color="0.25", lw=0.55, zorder=0)
    ax.set_xlim(0, 249)
    ax.set_ylim(*y_lim)
    ax.text(
        0.0,
        1.03,
        left_title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=22,
        fontweight="bold",
        color=left_title_color,
    )
    ax.text(
        1.0,
        1.03,
        right_title,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=20,
        fontweight="bold",
        color="#2B2B2B",
    )
    ax.set_ylabel("")
    ax.set_xlabel("Position in region" if show_xlabel else "")
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.tick_params(axis="both", length=3, width=0.8)


def plot_fig1e_style_full249() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected10 = pd.read_csv(base.PLOT_V1_DATA / "legacy_selected_10_metadata.tsv", sep="\t")
    selected_ids = set(selected10["ID"].astype(str))
    attr_by_track, annotated_hits = load_inputs(selected_ids)

    all_attr = np.concatenate([np.asarray(v["actual"]).reshape(-1) for v in attr_by_track.values()])
    score_min = float(np.nanpercentile(all_attr, 0.25))
    score_max = float(np.nanpercentile(all_attr, 99.75))
    span = max(score_max - score_min, 0.08)
    pad = max(span * 0.18, 0.025)
    y_lim = (min(score_min - pad * 0.55, -0.03), score_max + pad * 4.8)

    exported: list[str] = []
    missing: list[str] = []
    for seq_no, row in enumerate(selected10.itertuples(index=False), start=1):
        seq_id = str(row.ID)
        try:
            for branch in ["HK", "DEV"]:
                _ = select_meta_row(attr_by_track, branch, branch, seq_id, "hit6")
                _ = select_meta_row(attr_by_track, "CAGE", branch, seq_id, "hit6")
        except KeyError:
            missing.append(seq_id)
            continue

        fig, axes = plt.subplots(8, 1, figsize=(14.2, 15.0), dpi=300, sharex=False, sharey=True)
        for axis_index, (ax, (branch, track, milestone)) in enumerate(zip(axes, ROW_SPECS)):
            meta_row = select_meta_row(attr_by_track, track, branch, seq_id, milestone)
            muts = mutation_positions(attr_by_track, branch, seq_id)
            left_title, right_title = title_parts_for_row(track, branch, milestone, meta_row)
            render_axis(
                ax,
                arr=arr_for(attr_by_track, track, meta_row),
                motifs=top_motifs(hits_for(annotated_hits, track, branch, seq_id, milestone), 0, 249),
                muts=muts,
                left_title=left_title,
                right_title=right_title,
                left_title_color=left_title_color(track, branch),
                y_lim=y_lim,
                y_ref_max=score_max,
                pad=pad,
                show_xlabel=(axis_index == len(ROW_SPECS) - 1),
            )

        header = f"{base.short_id(seq_id)} | DeepCAGE {row.CAGE_Group}"
        axes[0].text(0.0, 1.54, header, transform=axes[0].transAxes, ha="left", va="bottom", fontsize=22.0)
        axes[0].text(
            -0.045,
            1.54,
            "d" if seq_no == 1 else f"d{seq_no}",
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
            fontsize=19,
            fontweight="bold",
        )
        fig.subplots_adjust(left=0.09, right=0.99, bottom=0.045, top=0.935, hspace=1.02)
        stem = OUT_DIR / f"Fig5d_fig1e_style_seq{seq_no:02d}_{base.short_id(seq_id)}"
        save_fig1e_style(fig, stem)
        exported.append(str(stem))
        plt.close(fig)

    manifest = {
        "n_requested": int(len(selected10)),
        "n_exported": int(len(exported)),
        "missing_ids": missing,
        "layout": "DeepSTARR Fig. 1e-inspired full 249 bp single-axis letter logos, one axis per track/milestone.",
        "normalization": "Raw actual contributions are shown; no median normalization is applied.",
        "title_color_policy": "Left title colour denotes readout track: HK blue, DEV green, CAGE orange; HCAGE/DCAGE text denotes greedy branch.",
        "source": (
            "reviewer_greedy_20260724_115905 attribution_finemo_24bp_annotated; "
            "actual contribution computed explicitly as hyp_contrib multiplied by one_hot for HK, DEV and CAGE"
        ),
    }
    (OUT_DIR / "fig5d_fig1e_style_full249_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    plot_fig1e_style_full249()
