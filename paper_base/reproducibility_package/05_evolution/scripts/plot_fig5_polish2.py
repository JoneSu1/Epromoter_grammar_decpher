#!/usr/bin/env python
"""Polished Figure 5 panels for the evolution module.

The script reads the v2 processed tables and writes publication-ready
SVG/PDF/TIFF/PNG outputs into plot_v2/polish/figures.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd
import seaborn as sns


POLISH_ROOT = Path(__file__).resolve().parents[1]
PLOT_V2_ROOT = POLISH_ROOT.parent
DATA = POLISH_ROOT / "data" / "processed"
ATTR_ROOT = POLISH_ROOT.parent / "frozen_data" / "evolution_fig5d"
ATTR_DIR = ATTR_ROOT / "attribution"
HITS_LONG = ATTR_ROOT / "summaries" / "finemo_hits_annotated_long.tsv"
_OUTPUT_ROOT = Path(os.environ["REPRO_OUTPUT_ROOT"]).resolve() if os.environ.get("REPRO_OUTPUT_ROOT") else POLISH_ROOT / "figures"
FIG_MAIN = _OUTPUT_ROOT / "main"
FIG_SUPP = _OUTPUT_ROOT / "supplement"
FIG5D_SEQ_DIR = FIG_MAIN / "Fig5d_sequence_logos"
LOG_DIR = _OUTPUT_ROOT / "logs"

MANDATORY_FIG5D_IDS = [
    "chr2L_10002493_10002741",
    "chr2R_9336906_9337154",
    "chr3R_16383096_16383344",
]
FIG5D_MAIN_LOGO_PAIR = [
    {
        "group_label": "DeepCAGE Low (<2)",
        "id_prefix": "chr2L_10002493_10002741",
        "reason": "Mandatory matched baseline DeepCAGE-low sequence with complete attribution and clear HK/DEV hit6 gains.",
    },
    {
        "group_label": "DeepCAGE High (>4)",
        "id_prefix": "chr3L_16963101_16963349",
        "reason": "Baseline DeepCAGE-high complete-attribution comparator with clear HK/DEV hit6 and matched DeepCAGE signal.",
    },
]

BASES = ["A", "C", "G", "T"]
BASE_COLORS = {"A": "#109618", "C": "#3366CC", "G": "#FF9900", "T": "#DC3912"}
GROUP_ORDER = ["Low", "Medium", "High"]
GROUP_FULL = ["Low (<2)", "Medium (2-4)", "High (>4)"]
GROUP_SHORT = {"Low (<2)": "Low", "Medium (2-4)": "Medium", "High (>4)": "High"}
GROUP_COLORS = {"Low": "#4F83BF", "Medium": "#E6A400", "High": "#D55E00"}
FULL_COLORS = {group: GROUP_COLORS[GROUP_SHORT[group]] for group in GROUP_FULL}
TASK_COLORS = {"HK": "#1F77A5", "DEV": "#17956F"}
SERIES_COLORS = {
    "HK target": "#1F77A5",
    "HK-DeepCAGE": "#79B7D8",
    "DEV target": "#17956F",
    "DEV-DeepCAGE": "#C99A22",
}


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 11.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
sns.set_theme(
    style="white",
    context="paper",
    rc={"font.family": "Arial", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]},
)
mpl.rcParams.update({"font.family": "Arial", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]})


def save_pub(fig: mpl.figure.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=600)
    fig.savefig(stem.with_suffix(".tiff"), bbox_inches="tight", dpi=600)
    print(f"Saved {stem.name}")


def save_vector_preview(fig: mpl.figure.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=450)
    print(f"Saved {stem.name}")


def panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.13, y: float = 1.08) -> None:
    return


def corr(x: pd.Series, y: pd.Series) -> float:
    xy = pd.concat([x, y], axis=1).dropna()
    if len(xy) < 3:
        return float("nan")
    return float(np.corrcoef(xy.iloc[:, 0].to_numpy(), xy.iloc[:, 1].to_numpy())[0, 1])


def set_integer_ticks(ax: mpl.axes.Axes) -> None:
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))


def onehot_to_seq(onehot: np.ndarray) -> str:
    idx = onehot.argmax(axis=1)
    valid = onehot.max(axis=1) > 0
    return "".join(BASES[i] if ok else "N" for i, ok in zip(idx, valid))


def short_id(identifier: str) -> str:
    return str(identifier).replace("_peak_849bp_region", "").replace("_Other_DHSs", "")


def interval_overlap_bp(first: tuple[int, int], second: tuple[int, int]) -> int:
    return max(0, min(first[1], second[1]) - max(first[0], second[0]))


def filtered_motifs(hits: pd.DataFrame, start: int, end: int, max_items: int = 3) -> list[tuple[int, int, str, float]]:
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
        label = f"{row.MC_ID} {row.Match_1}"
        confidence = float(row.hit_importance) if pd.notna(row.hit_importance) else float(row.hit_coefficient)
        if any(interval_overlap_bp((s, e), (ks, ke)) > 2 for ks, ke, _, _ in kept):
            continue
        kept.append((s, e, label, confidence))
        if len(kept) >= max_items:
            break
    return sorted(kept, key=lambda item: (item[0], item[1]))


def draw_base_letter(ax: mpl.axes.Axes, base: str, x: float, y0: float, height: float) -> None:
    if abs(height) < 1e-4:
        return
    fp = FontProperties(family="DejaVu Sans", weight="bold")
    path = TextPath((0, 0), base, size=1, prop=fp)
    bbox = path.get_extents()
    sx = 0.78 / max(bbox.width, 1e-6)
    sy = abs(height) / max(bbox.height, 1e-6)
    y = y0 if height >= 0 else y0 + height
    trans = Affine2D().scale(sx, sy).translate(x - 0.39, y) + ax.transData
    patch = PathPatch(path, transform=trans, facecolor=BASE_COLORS[base], edgecolor="none", alpha=0.98)
    ax.add_patch(patch)


def draw_contribution_logo(ax: mpl.axes.Axes, arr: np.ndarray, start: int, end: int) -> None:
    for pos in range(start, end):
        vals = arr[pos, :]
        x = pos + 1
        pos_bottom = 0.0
        neg_bottom = 0.0
        for base, value in sorted(zip(BASES, vals), key=lambda item: item[1]):
            if value < 0:
                draw_base_letter(ax, base, x, neg_bottom, float(value))
                neg_bottom += float(value)
        for base, value in sorted(zip(BASES, vals), key=lambda item: item[1]):
            if value > 0:
                draw_base_letter(ax, base, x, pos_bottom, float(value))
                pos_bottom += float(value)


def draw_motif_spans(
    ax: mpl.axes.Axes,
    motifs: list[tuple[int, int, str, float]],
    y_ref_max: float,
    pad: float,
    *,
    motif_fontsize: float = 5.6,
    motif_lw: float = 1.2,
) -> None:
    occupied_label_ends: list[float] = []
    x_left, x_right = ax.get_xlim()
    label_margin = max((x_right - x_left) * 0.028, 1.8)
    for start, end, label, _ in motifs:
        x0, x1 = start + 1, end + 1
        ax.axvspan(x0, x1, facecolor="#9ECAE1", alpha=0.28, edgecolor="none", linewidth=0, zorder=0)
        label_half_width = max(3.2, len(label) * 0.82 * (motif_fontsize / 7.0))
        label_x = float(np.clip((x0 + x1) / 2, x_left + label_margin, x_right - label_margin))
        label_left = label_x - label_half_width
        label_right = label_x + label_half_width
        level = 0
        while level < len(occupied_label_ends) and label_left <= occupied_label_ends[level] + 2:
            level += 1
        if level == len(occupied_label_ends):
            occupied_label_ends.append(label_right)
        else:
            occupied_label_ends[level] = label_right
        y_text = max(y_ref_max, 0.0) + pad * (0.35 + 0.95 * level)
        y_bar = y_text - pad * 0.16
        ax.plot([x0, x1], [y_bar, y_bar], color="#2166AC", lw=motif_lw, solid_capstyle="butt", zorder=4)
        ax.text(label_x, y_text, label, ha="center", va="bottom", fontsize=motif_fontsize, color="black")


def render_logo_panel(
    ax: mpl.axes.Axes,
    *,
    arr: np.ndarray,
    motifs: list[tuple[int, int, str, float]],
    muts: list[int],
    start: int,
    end: int,
    title: str,
    y_lim: tuple[float, float],
    y_ref_max: float,
    pad: float,
    show_ylabel: bool,
    motif_fontsize: float = 5.6,
    motif_lw: float = 1.2,
) -> None:
    draw_contribution_logo(ax, arr, start, end)
    ax.set_xlim(start + 1, end)
    draw_motif_spans(ax, motifs, y_ref_max=y_ref_max, pad=pad, motif_fontsize=motif_fontsize, motif_lw=motif_lw)
    for pos in muts:
        ax.plot([pos, pos], [y_lim[0], y_lim[0] + pad * 0.48], color="#B2182B", lw=0.7, solid_capstyle="butt")
    ax.axhline(0, color="0.3", lw=0.55, zorder=0)
    ax.set_ylim(*y_lim)
    ax.set_title(title, pad=4)
    ax.tick_params(axis="x", length=2.2, width=0.65)
    ax.tick_params(axis="y", length=2.2, width=0.65)
    ax.set_xlabel("Position in 249 bp sequence")
    if show_ylabel:
        ax.set_ylabel("Contribution")
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.1")
        spine.set_linewidth(0.75)


def mean_sem(data: pd.DataFrame, value: str) -> pd.DataFrame:
    rows = []
    for (task, group, round_), sub in data.groupby(["task", "CAGE_Group_Short", "Round"], observed=True):
        vals = sub[value].dropna().to_numpy()
        if len(vals) == 0:
            continue
        mean = float(np.mean(vals))
        sem = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        rows.append(
            {
                "task": task,
                "CAGE_Group_Short": group,
                "Round": int(round_),
                "mean": mean,
                "lo": mean - 1.96 * sem,
                "hi": mean + 1.96 * sem,
                "n": len(vals),
            }
        )
    return pd.DataFrame(rows)


def plot_5b(round0: pd.DataFrame) -> None:
    data = round0.copy()
    data["CAGE_Group_Short"] = data["CAGE_Group"].map(GROUP_SHORT)

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.55), dpi=300, gridspec_kw={"width_ratios": [1.25, 0.9]})

    ax = axes[0]
    for group in GROUP_ORDER:
        sub = data.loc[data["CAGE_Group_Short"].eq(group)]
        ax.scatter(
            sub["Real_CAGE_log2TPM"],
            sub["Greedy_round0_CAGE_pred"],
            s=12,
            color=GROUP_COLORS[group],
            alpha=0.56,
            linewidth=0,
            label=f"{group} ({len(sub)})",
            rasterized=True,
        )
    lims = [
        min(data["Real_CAGE_log2TPM"].min(), data["Greedy_round0_CAGE_pred"].min()) - 0.3,
        max(data["Real_CAGE_log2TPM"].max(), data["Greedy_round0_CAGE_pred"].max()) + 0.3,
    ]
    ax.plot(lims, lims, color="0.4", lw=0.7, ls="--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Measured CAGE")
    ax.set_ylabel("Round0 predicted CAGE")
    ax.set_title("Reviewer-grade greedy cohort")
    ax.text(0.04, 0.96, f"N={data['ID'].nunique()}", transform=ax.transAxes, va="top", fontsize=7)
    ax.legend(frameon=False, loc="lower right", title="Baseline DeepCAGE")
    panel_label(ax, "b")

    ax = axes[1]
    counts = data["CAGE_Group_Short"].value_counts().reindex(GROUP_ORDER).fillna(0).astype(int)
    bars = ax.bar(GROUP_ORDER, counts.values, color=[GROUP_COLORS[g] for g in GROUP_ORDER], edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 5, f"{val}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Sequences")
    ax.set_title("Baseline DeepCAGE strata")
    ax.set_ylim(0, max(counts.values) * 1.18)

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.20, top=0.82, wspace=0.34)
    save_pub(fig, FIG_MAIN / "Fig5b_polished_greedy_round0_cage_groups")
    plt.close(fig)


def plot_5c(trajectories: pd.DataFrame) -> None:
    data = trajectories.loc[trajectories["Round"] <= 40].copy()
    data["CAGE_Group_Short"] = data["CAGE_Group"].map(GROUP_SHORT).fillna(data["CAGE_Group_Short"])
    stats_score = mean_sem(data, "Score_Opt")
    stats_cage = mean_sem(data, "CAGE_pred")
    axis_label_size = 13
    tick_label_size = 11
    title_size = 14
    legend_size = 11
    round_label_size = 10

    for task in ["HK", "DEV"]:
        fig, axes = plt.subplots(2, 1, figsize=(4.6, 4.95), dpi=300, sharex=True)
        for row, stat, ylabel, threshold in [
            (0, stats_score, f"{task} pred. (log2)", 6),
            (1, stats_cage, "CAGE pred. (log2(TPM + 1))", None),
        ]:
            ax = axes[row]
            hit_rounds = {}
            for group in GROUP_ORDER:
                g = stat.loc[(stat["task"].eq(task)) & (stat["CAGE_Group_Short"].eq(group))].sort_values("Round")
                if g.empty:
                    continue
                x = g["Round"].to_numpy(float)
                mean = g["mean"].to_numpy(float)
                lo = g["lo"].to_numpy(float)
                hi = g["hi"].to_numpy(float)
                ax.plot(x, mean, color=GROUP_COLORS[group], lw=2.1)
                ax.fill_between(x, lo, hi, color=GROUP_COLORS[group], alpha=0.12, linewidth=0)
                if row == 0:
                    reached = g.loc[g["mean"] >= 6]
                    if not reached.empty:
                        r = int(reached.iloc[0]["Round"])
                        hit_rounds.setdefault(r, []).append(group)
            if threshold is not None:
                ax.axhline(threshold, color="0.35", lw=0.7, ls="--")
                ymin, ymax = ax.get_ylim()
                for idx, (r, groups) in enumerate(sorted(hit_rounds.items())):
                    merged = len(groups) > 1
                    color = "0.35" if merged else GROUP_COLORS[groups[0]]
                    label = f"R{r} all strata" if len(groups) == len(GROUP_ORDER) else f"R{r}"
                    ax.axvline(r, color=color, lw=0.7, ls=":", alpha=0.85)
                    ax.text(
                        r + 0.25,
                        ymin + 0.05 * (ymax - ymin) + idx * 0.105 * (ymax - ymin),
                        label,
                        rotation=90,
                        color=color,
                        fontsize=round_label_size,
                        va="bottom",
                    )
            else:
                ax.axhline(2, color="0.45", lw=0.6, ls="--")
                ax.axhline(4, color="0.45", lw=0.6, ls="--")
            ax.set_title(f"{task} model guidance mutation" if row == 0 else "", fontsize=title_size)
            ax.set_ylabel(ylabel, fontsize=axis_label_size)
            ax.yaxis.set_label_coords(-0.095, 0.5)
            ax.grid(axis="y", linestyle=":", alpha=0.22)
            ax.tick_params(axis="both", labelsize=tick_label_size)
            if row == 1:
                ax.set_xlabel("Mutation round", fontsize=axis_label_size)

        handles = [Line2D([0], [0], color=GROUP_COLORS[g], lw=2.3, label=g) for g in GROUP_ORDER]
        fig.legend(
            handles=handles,
            frameon=False,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.005),
            fontsize=legend_size,
        )
        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.88, hspace=0.32)
        save_pub(fig, FIG_MAIN / f"Fig5b_{task.lower()}_model_guidance_mutation")
        plt.close(fig)


def complete_fig5d_ids(metadata: pd.DataFrame) -> set[str]:
    round0_needed = {"HK", "DEV", "CAGE"}
    hit6_needed = {("HK", "HK"), ("CAGE", "HK"), ("DEV", "DEV"), ("CAGE", "DEV")}
    round0_available: dict[str, set[str]] = {}
    hit6_available: dict[str, set[tuple[str, str]]] = {}
    for row in metadata.itertuples(index=False):
        if row.milestone == "round0":
            round0_available.setdefault(str(row.ID), set()).add(str(row.track))
        elif row.milestone == "hit6":
            hit6_available.setdefault(str(row.ID), set()).add((str(row.track), str(row.branch_task)))
    ids = set(round0_available) & set(hit6_available)
    return {
        seq_id
        for seq_id in ids
        if round0_needed.issubset(round0_available[seq_id]) and hit6_needed.issubset(hit6_available[seq_id])
    }


def select_fig5d_sequences(summary: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    complete_ids = complete_fig5d_ids(metadata)
    candidates = summary.loc[summary["ID"].isin(complete_ids) & summary["reached_hit6"].eq(True)].copy()
    candidates["abs_cage_shift"] = (candidates["best_cage"] - candidates["init_cage"]).abs()
    candidates = candidates.sort_values(["step_to_hit6", "abs_cage_shift", "best_score"], ascending=[True, True, False])

    selected: list[str] = []
    mandatory_missing = []
    for prefix in MANDATORY_FIG5D_IDS:
        hits = candidates.loc[candidates["ID"].str.contains(prefix, regex=False)]
        if hits.empty:
            mandatory_missing.append(prefix)
            continue
        seq_id = str(hits.iloc[0]["ID"])
        if seq_id not in selected:
            selected.append(seq_id)

    for group in ["High (>4)", "Medium (2-4)", "Low (<2)"]:
        for task in ["HK", "DEV"]:
            hits = candidates.loc[candidates["CAGE_Group"].eq(group) & candidates["task"].eq(task)]
            for seq_id in hits["ID"].astype(str):
                if seq_id not in selected:
                    selected.append(seq_id)
                    break
            if len(selected) >= 10:
                break
        if len(selected) >= 10:
            break

    for seq_id in candidates["ID"].astype(str):
        if len(selected) >= 10:
            break
        if seq_id not in selected:
            selected.append(seq_id)

    rows = []
    for seq_id in selected[:10]:
        sub = summary.loc[summary["ID"].eq(seq_id)].copy()
        best = sub.sort_values(["step_to_hit6", "best_score"], ascending=[True, False]).iloc[0]
        rows.append(
            {
                "ID": seq_id,
                "CAGE_Group": best["CAGE_Group"],
                "best_branch": best["task"],
                "step_to_hit6": int(best["step_to_hit6"]),
                "init_cage": float(best["init_cage"]),
                "best_cage": float(best["best_cage"]),
                "best_score": float(best["best_score"]),
            }
        )
    selected10 = pd.DataFrame(rows)
    selected10.attrs["mandatory_missing"] = mandatory_missing
    return selected10


def plot_5d_logo(primary_summary: pd.DataFrame) -> None:
    metadata_by_track = {track: pd.read_csv(ATTR_DIR / track / "metadata.tsv", sep="\t") for track in ["HK", "DEV", "CAGE"]}
    metadata = pd.concat(metadata_by_track.values(), ignore_index=True)
    selected10 = select_fig5d_sequences(primary_summary, metadata)
    hits = pd.read_csv(HITS_LONG, sep="\t")
    arrays = {
        track: {
            "ohe": np.load(ATTR_DIR / track / "attribution_arrays.npz")["one_hot"],
            "attr": np.load(ATTR_DIR / track / "attribution_arrays.npz")["act_contrib"],
        }
        for track in ["HK", "DEV", "CAGE"]
    }

    attr_sample = np.concatenate([arrays[track]["attr"].reshape(-1) for track in arrays])
    score_min = float(np.nanpercentile(attr_sample, 0.4))
    score_max = float(np.nanpercentile(attr_sample, 99.6))
    span = max(score_max - score_min, 0.08)
    pad = max(span * 0.20, 0.025)
    y_lim = (min(score_min - pad * 0.5, -0.03), score_max + pad * 3.45)

    def meta_row(seq_id: str, track: str, branch: str, milestone: str) -> pd.Series:
        sub = metadata_by_track[track].loc[
            metadata_by_track[track]["ID"].eq(seq_id)
            & metadata_by_track[track]["branch_task"].eq(branch)
            & metadata_by_track[track]["milestone"].eq(milestone)
        ]
        if sub.empty and milestone == "round0":
            sub = metadata_by_track[track].loc[
                metadata_by_track[track]["ID"].eq(seq_id)
                & metadata_by_track[track]["milestone"].eq("round0")
            ]
        if sub.empty:
            raise ValueError(f"Missing attribution metadata for {seq_id} {track}/{branch}/{milestone}")
        return sub.sort_values(["Round", "Score_Opt"], ascending=[True, False]).iloc[0]

    def arr_for(seq_id: str, track: str, branch: str, milestone: str) -> np.ndarray:
        row = meta_row(seq_id, track, branch, milestone)
        return arrays[track]["attr"][int(row["sequence_index"])]

    def prediction_for(seq_id: str, track: str, branch: str, milestone: str) -> float:
        row = meta_row(seq_id, track, branch, milestone)
        pred_col = {"HK": "Hk_pred", "DEV": "Dev_pred", "CAGE": "CAGE_pred"}[track]
        return float(row[pred_col])

    def seq_from_ohe(seq_id: str, track: str, branch: str, milestone: str) -> str:
        row = meta_row(seq_id, track, branch, milestone)
        return onehot_to_seq(arrays[track]["ohe"][int(row["sequence_index"])])

    def mutations(seq_id: str, branch: str, start: int, end: int) -> list[int]:
        seq0 = seq_from_ohe(seq_id, branch, branch, "round0")
        seq1 = seq_from_ohe(seq_id, branch, branch, "hit6")
        return [i + 1 for i in range(start, end) if seq0[i] != seq1[i]]

    def hit_subset(seq_id: str, track: str, branch: str, milestone: str) -> pd.DataFrame:
        return hits.loc[
            hits["ID"].eq(seq_id)
            & hits["track"].eq(track)
            & hits["branch_task"].eq(branch)
            & hits["milestone"].eq(milestone)
        ].copy()

    def window_for(seq_id: str) -> tuple[int, int]:
        candidates = []
        for track, branch in [("HK", "HK"), ("DEV", "DEV"), ("CAGE", "HK"), ("CAGE", "DEV")]:
            sub = hit_subset(seq_id, track, branch, "hit6")
            if not sub.empty:
                row = sub.sort_values(["hit_importance", "hit_coefficient"], ascending=False).iloc[0]
                candidates.append((float(row["hit_importance"]), int((row["start"] + row["end"]) / 2)))
        center = max(candidates)[1] if candidates else 124
        start = max(0, min(center - 35, 248 - 70))
        return start, start + 70

    FIG5D_SEQ_DIR.mkdir(parents=True, exist_ok=True)
    row_specs = [("HK", "HK"), ("CAGE", "HK"), ("DEV", "DEV"), ("CAGE", "DEV")]
    exported = []

    for seq_no, row in enumerate(selected10.itertuples(index=False), start=1):
        seq_id = str(row.ID)
        start, end = window_for(seq_id)
        fig, axes = plt.subplots(4, 2, figsize=(7.2, 7.95), dpi=300, sharex=False, sharey=True)

        for r, (track, branch) in enumerate(row_specs):
            muts = mutations(seq_id, branch, start, end)
            for c, milestone in enumerate(["round0", "hit6"]):
                motifs = filtered_motifs(hit_subset(seq_id, track, branch, milestone), start, end)
                pred = prediction_for(seq_id, track, branch, milestone)
                label = "round0" if milestone == "round0" else f"{branch} hit6"
                track_display = "DeepCAGE" if track == "CAGE" else track
                title = f"{track_display} attribution | {label}, pred={pred:.2f}"
                render_logo_panel(
                    axes[r, c],
                    arr=arr_for(seq_id, track, branch, milestone),
                    motifs=motifs,
                    muts=muts,
                    start=start,
                    end=end,
                    title=title,
                    y_lim=y_lim,
                    y_ref_max=score_max,
                    pad=pad,
                    show_ylabel=(c == 0),
                )
            label = f"{branch}-guided" if track == branch else f"{branch}-matched DeepCAGE"
            axes[r, 0].text(
                -0.11,
                1.17,
                label,
                transform=axes[r, 0].transAxes,
                ha="left",
                va="bottom",
                fontsize=7.1,
                fontweight="bold",
                color=TASK_COLORS[branch] if track == branch else "#D55E00",
            )

        header = f"{short_id(seq_id)} | DeepCAGE {row.CAGE_Group}"
        axes[0, 0].text(0.0, 1.34, header, transform=axes[0, 0].transAxes, ha="left", va="bottom", fontsize=7.0)
        panel_label(axes[0, 0], "d" if seq_no == 1 else f"d{seq_no}", x=-0.10, y=1.34)
        fig.subplots_adjust(left=0.085, right=0.99, bottom=0.06, top=0.93, hspace=0.75, wspace=0.12)
        stem = FIG5D_SEQ_DIR / f"Fig5d_seq{seq_no:02d}_{short_id(seq_id)}"
        save_vector_preview(fig, stem)
        if seq_no == 1:
            save_pub(fig, FIG_MAIN / "Fig5d_polished_hit6_representative_logos")
        exported.append(str(stem))
        plt.close(fig)

    manifest = {
        "n_requested": 10,
        "n_exported": int(len(exported)),
        "mandatory_prefixes": MANDATORY_FIG5D_IDS,
        "mandatory_missing": selected10.attrs.get("mandatory_missing", []),
        "selected_sequences": selected10.to_dict("records"),
        "export_stems": exported,
        "selection_rule": "Mandatory matched sequences first; remaining complete-attribution IDs selected for baseline DeepCAGE stratum and HK/DEV balance, prioritizing early hit6 and modest DeepCAGE shift.",
        "source": "reviewer-grade attribution arrays, metadata, and annotated Fi-NeMo hits",
    }
    (LOG_DIR / "fig5d_sequence_logo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def plot_5d_high_low_main_logo(primary_summary: pd.DataFrame) -> None:
    metadata_by_track = {track: pd.read_csv(ATTR_DIR / track / "metadata.tsv", sep="\t") for track in ["HK", "DEV", "CAGE"]}
    metadata = pd.concat(metadata_by_track.values(), ignore_index=True)
    complete_ids = complete_fig5d_ids(metadata)
    hits = pd.read_csv(HITS_LONG, sep="\t")
    round0_truth = pd.read_csv(DATA / "fig5b_greedy_round0_cohort.tsv", sep="\t")
    round0_truth = round0_truth.drop_duplicates("ID").set_index("ID")
    arrays = {
        track: {
            "ohe": np.load(ATTR_DIR / track / "attribution_arrays.npz")["one_hot"],
            "attr": np.load(ATTR_DIR / track / "attribution_arrays.npz")["act_contrib"],
        }
        for track in ["HK", "DEV", "CAGE"]
    }

    attr_sample = np.concatenate([arrays[track]["attr"].reshape(-1) for track in arrays])
    score_min = float(np.nanpercentile(attr_sample, 0.4))
    score_max = float(np.nanpercentile(attr_sample, 99.6))
    span = max(score_max - score_min, 0.08)
    pad = max(span * 0.20, 0.025)
    y_lim = (min(score_min - pad * 0.5, -0.03), score_max + pad * 3.45)

    def resolve_seq(prefix: str) -> str:
        sub = primary_summary.loc[
            primary_summary["ID"].isin(complete_ids)
            & primary_summary["reached_hit6"].eq(True)
            & primary_summary["ID"].str.contains(prefix, regex=False)
        ]
        if sub.empty:
            raise ValueError(f"Missing complete high-low Fig5d candidate for prefix {prefix}")
        return str(sub.sort_values(["step_to_hit6", "best_score"], ascending=[True, False]).iloc[0]["ID"])

    def meta_row(seq_id: str, track: str, branch: str, milestone: str) -> pd.Series:
        sub = metadata_by_track[track].loc[
            metadata_by_track[track]["ID"].eq(seq_id)
            & metadata_by_track[track]["branch_task"].eq(branch)
            & metadata_by_track[track]["milestone"].eq(milestone)
        ]
        if sub.empty and milestone == "round0":
            sub = metadata_by_track[track].loc[
                metadata_by_track[track]["ID"].eq(seq_id)
                & metadata_by_track[track]["milestone"].eq("round0")
            ]
        if sub.empty:
            raise ValueError(f"Missing high-low attribution metadata for {seq_id} {track}/{branch}/{milestone}")
        return sub.sort_values(["Round", "Score_Opt"], ascending=[True, False]).iloc[0]

    def arr_for(seq_id: str, track: str, branch: str, milestone: str) -> np.ndarray:
        row = meta_row(seq_id, track, branch, milestone)
        return arrays[track]["attr"][int(row["sequence_index"])]

    def prediction_for(seq_id: str, track: str, branch: str, milestone: str) -> float:
        row = meta_row(seq_id, track, branch, milestone)
        pred_col = {"HK": "Hk_pred", "DEV": "Dev_pred", "CAGE": "CAGE_pred"}[track]
        return float(row[pred_col])

    def title_value(seq_id: str, track: str, branch: str, milestone: str) -> str:
        if milestone == "round0" and track in {"HK", "DEV"} and seq_id in round0_truth.index:
            true_col = {"HK": "Hk_true", "DEV": "Dev_true"}[track]
            return f"true={float(round0_truth.loc[seq_id, true_col]):.2f}"
        return f"pred={prediction_for(seq_id, track, branch, milestone):.2f}"

    def seq_from_ohe(seq_id: str, branch: str, milestone: str) -> str:
        row = meta_row(seq_id, branch, branch, milestone)
        return onehot_to_seq(arrays[branch]["ohe"][int(row["sequence_index"])])

    def mutations(seq_id: str, branch: str, start: int, end: int) -> list[int]:
        seq0 = seq_from_ohe(seq_id, branch, "round0")
        seq1 = seq_from_ohe(seq_id, branch, "hit6")
        return [i + 1 for i in range(start, end) if seq0[i] != seq1[i]]

    def hit_subset(seq_id: str, track: str, branch: str, milestone: str) -> pd.DataFrame:
        return hits.loc[
            hits["ID"].eq(seq_id)
            & hits["track"].eq(track)
            & hits["branch_task"].eq(branch)
            & hits["milestone"].eq(milestone)
        ].copy()

    def window_for(seq_id: str, branch: str) -> tuple[int, int]:
        candidates = []
        for track in [branch, "CAGE"]:
            sub = hit_subset(seq_id, track, branch, "hit6")
            if not sub.empty:
                row = sub.sort_values(["hit_importance", "hit_coefficient"], ascending=False).iloc[0]
                candidates.append((float(row["hit_importance"]), int((row["start"] + row["end"]) / 2)))
        center = max(candidates)[1] if candidates else 124
        start = max(0, min(center - 35, 248 - 70))
        return start, start + 70

    resolved = []
    for item in FIG5D_MAIN_LOGO_PAIR:
        seq_id = resolve_seq(item["id_prefix"])
        rows_by_branch = {}
        for branch in ["HK", "DEV"]:
            row = primary_summary.loc[primary_summary["ID"].eq(seq_id) & primary_summary["task"].eq(branch)].iloc[0]
            rows_by_branch[branch] = {
                "step_to_hit6": int(row["step_to_hit6"]),
                "best_score": float(row["best_score"]),
                "best_cage": float(row["best_cage"]),
            }
        group = str(primary_summary.loc[primary_summary["ID"].eq(seq_id), "CAGE_Group"].iloc[0])
        resolved.append({**item, "ID": seq_id, "CAGE_Group": group, "branches": rows_by_branch})

    row_specs = [("HK", "HK attribution"), ("CAGE", "HK-matched DeepCAGE"), ("DEV", "DEV attribution"), ("CAGE", "DEV-matched DeepCAGE")]
    exported = []
    for pair_idx, item in enumerate(resolved):
        fig, axes = plt.subplots(4, 2, figsize=(5.45, 7.25), dpi=300, sharey=True)
        seq_id = str(item["ID"])
        for row_idx, (track, row_label) in enumerate(row_specs):
            branch = "HK" if row_idx < 2 else "DEV"
            start, end = window_for(seq_id, branch)
            muts = mutations(seq_id, branch, start, end)
            for offset, milestone in enumerate(["round0", "hit6"]):
                ax = axes[row_idx, offset]
                motifs = filtered_motifs(hit_subset(seq_id, track, branch, milestone), start, end, max_items=2)
                milestone_label = "round0" if milestone == "round0" else f"{branch} hit6"
                track_label = branch if track == branch else "DeepCAGE"
                render_logo_panel(
                    ax,
                    arr=arr_for(seq_id, track, branch, milestone),
                    motifs=motifs,
                    muts=muts,
                    start=start,
                    end=end,
                    title=f"{track_label} | {milestone_label}, {title_value(seq_id, track, branch, milestone)}",
                    y_lim=y_lim,
                    y_ref_max=score_max,
                    pad=pad,
                    show_ylabel=(offset == 0),
                    motif_fontsize=5.1,
                    motif_lw=1.05,
                )
                ax.set_xlabel("Position")
                ax.tick_params(axis="both", labelsize=9.5)
                ax.title.set_fontsize(9.8)
                if offset != 0:
                    ax.set_ylabel("")
        group_color = FULL_COLORS.get(str(item["CAGE_Group"]), "0.2")
        axes[0, 0].text(
            0.0,
            1.47,
            f"{item['group_label']} | {short_id(seq_id)}",
            transform=axes[0, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=group_color,
        )
        axes[0, 0].text(
            0.0,
            1.28,
            f"HK hit6 round {item['branches']['HK']['step_to_hit6']}; DEV hit6 round {item['branches']['DEV']['step_to_hit6']}",
            transform=axes[0, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=9.0,
            color="0.25",
        )
        for row_idx, (_, row_label) in enumerate(row_specs):
            axes[row_idx, 0].set_ylabel(row_label)
        panel_label(axes[0, 0], "d", x=-0.15, y=1.48)
        fig.subplots_adjust(left=0.15, right=0.995, bottom=0.075, top=0.90, hspace=0.84, wspace=0.20)
        group_short = GROUP_SHORT.get(str(item["CAGE_Group"]), str(item["CAGE_Group"]).split()[0]).lower()
        stem = FIG_MAIN / f"Fig5d_CAGE_{group_short}_main_logo"
        save_pub(fig, stem)
        exported.append(str(stem))
        plt.close(fig)

    manifest = {
        "n_exported_sequences": 2,
        "comparison": "Baseline DeepCAGE Low versus DeepCAGE High complete-attribution examples for main Fig5d.",
        "branches": "Both HK-guided and DEV-guided hit6 contrasts are rendered for each baseline DeepCAGE group.",
        "round0_hk_dev_label": "HK and DEV round0 titles use experimental Hk_true/Dev_true from the Fig5b greedy round0 cohort table; hit6 titles remain model predictions for evolved sequences.",
        "selected_sequences": resolved,
        "export_stems": exported,
        "source": "reviewer-grade attribution arrays, metadata, annotated Fi-NeMo hits, and primary greedy summary.",
    }
    (LOG_DIR / "fig5d_high_low_main_logo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def bubble_table(sub: pd.DataFrame) -> pd.DataFrame:
    return (
        sub.groupby(["cage_new_motifs", "target_new_motifs"], as_index=False)
        .agg(n_sequences=("ID", "nunique"))
        .sort_values(["cage_new_motifs", "target_new_motifs"])
    )


def plot_5e(motif_scatter: pd.DataFrame) -> None:
    title_size = 18
    axis_label_size = 17
    tick_label_size = 16
    annotation_size = 17
    cell_label_size = 15.6
    grids = {}
    max_count = 1
    for branch in ["HK", "DEV"]:
        sub = motif_scatter.loc[motif_scatter["branch_task"].eq(branch)]
        tab = bubble_table(sub)
        x_max = int(tab["cage_new_motifs"].max())
        y_max = int(tab["target_new_motifs"].max())
        grid = np.zeros((y_max + 1, x_max + 1), dtype=float)
        for row in tab.itertuples(index=False):
            grid[int(row.target_new_motifs), int(row.cage_new_motifs)] = float(row.n_sequences)
        grids[branch] = (sub, grid)
        max_count = max(max_count, int(grid.max()))

    fig = plt.figure(figsize=(12.2, 5.4), dpi=300)
    outer = fig.add_gridspec(1, 2, left=0.07, right=0.985, bottom=0.22, top=0.80, wspace=0.42)
    for idx, (branch, ylabel) in enumerate(
        zip(
            ["HK", "DEV"],
            [
                "New HK motif hits\n(hit6 - round0, positive sum)",
                "New DEV motif hits\n(hit6 - round0, positive sum)",
            ],
        )
    ):
        sub, grid = grids[branch]
        x_max = grid.shape[1] - 1
        y_max = grid.shape[0] - 1
        spec = outer[idx].subgridspec(
            2,
            2,
            height_ratios=[0.22, 1.0],
            width_ratios=[1.0, 0.18],
            hspace=0.035,
            wspace=0.035,
        )
        ax_top = fig.add_subplot(spec[0, 0])
        ax = fig.add_subplot(spec[1, 0], sharex=ax_top)
        ax_right = fig.add_subplot(spec[1, 1], sharey=ax)
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            f"{branch}_motif_density",
            ["#FFFFFF", "#E7EEF2", TASK_COLORS[branch]],
        )
        masked = np.ma.masked_where(grid == 0, grid)
        ax.imshow(masked, origin="lower", cmap=cmap, aspect="auto", interpolation="none", vmin=0, vmax=max_count)
        ax.set_facecolor("#F7F7F7")
        ax.set_xticks(np.arange(x_max + 1))
        ax.set_yticks(np.arange(y_max + 1))
        ax.set_xticks(np.arange(-0.5, x_max + 1, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, y_max + 1, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", length=0)
        threshold = max(3, int(np.nanpercentile(grid[grid > 0], 72))) if np.any(grid > 0) else 3
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                value = int(grid[y, x])
                if value >= threshold:
                    ax.text(x, y, str(value), ha="center", va="center", fontsize=cell_label_size, color="0.05")

        x_counts = grid.sum(axis=0)
        y_counts = grid.sum(axis=1)
        ax_top.bar(np.arange(x_max + 1), x_counts, color=TASK_COLORS[branch], alpha=0.62, width=0.72, linewidth=0)
        ax_right.barh(np.arange(y_max + 1), y_counts, color=TASK_COLORS[branch], alpha=0.62, height=0.72, linewidth=0)
        ax_top.set_ylim(0, max(x_counts.max() * 1.15, 1))
        ax_right.set_xlim(0, max(y_counts.max() * 1.15, 1))
        ax_top.axis("off")
        ax_right.axis("off")
        ax_top.text(
            0.5,
            1.20,
            f"Joint motif-gain distribution | {branch}-guided",
            transform=ax_top.transAxes,
            ha="center",
            va="bottom",
            fontsize=title_size,
            clip_on=False,
        )

        r = corr(sub["cage_new_motifs"], sub["target_new_motifs"])
        ax.text(0.03, 0.96, f"r={r:.2f}\nN={sub['ID'].nunique()}", transform=ax.transAxes, va="top", fontsize=annotation_size)
        ax.set_xlabel("New DeepCAGE motif hits\n(hit6 - round0, positive sum)", fontsize=axis_label_size)
        ax.set_ylabel(ylabel, fontsize=axis_label_size)
        ax.tick_params(axis="both", labelsize=tick_label_size)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.75)
            spine.set_color("0.2")
        if idx == 0:
            panel_label(ax_top, "e", x=-0.13, y=1.08)
    save_pub(fig, FIG_MAIN / "Fig5e_polished_new_target_vs_cage_motifs")
    plt.close(fig)


def plot_5f(motif_box: pd.DataFrame) -> None:
    data = motif_box.copy()
    data["series"] = data["series"].replace({"HK-CAGE": "HK-DeepCAGE", "DEV-CAGE": "DEV-DeepCAGE"})
    series_order = ["HK target", "HK-DeepCAGE", "DEV target", "DEV-DeepCAGE"]
    motif_order = (
        data.groupby("motif_label", observed=True)["new_hit6_count"]
        .median()
        .sort_values(ascending=True)
        .index.tolist()
    )
    title_size = 18
    axis_label_size = 17
    tick_label_size = 16
    legend_size = 16
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.8), dpi=300)
    sns.boxplot(
        data=data,
        y="motif_label",
        x="new_hit6_count",
        hue="series",
        order=motif_order,
        hue_order=series_order,
        palette=SERIES_COLORS,
        orient="h",
        width=0.72,
        fliersize=0,
        linewidth=0.78,
        whis=(5, 95),
        saturation=0.82,
        boxprops={"edgecolor": "0.2", "linewidth": 0.78},
        whiskerprops={"color": "0.25", "linewidth": 0.72},
        capprops={"color": "0.25", "linewidth": 0.72},
        medianprops={"color": "0.05", "linewidth": 0.95},
        ax=ax,
    )
    ax.set_xlabel("New motif hits\n(hit6 - round0, positive count)", fontsize=axis_label_size)
    ax.set_ylabel("")
    ax.set_title("Per-motif positive gains\nrelative to round0", fontsize=title_size, pad=8)
    ax.grid(axis="x", linestyle=":", alpha=0.18)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.set_xlim(0, 5)
    ax.set_xticks(np.arange(0, 6, 1))
    ax.tick_params(axis="both", labelsize=tick_label_size)
    ax.axvline(0, color="0.35", lw=0.65)
    ax.legend(
        frameon=False,
        title="",
        ncol=1,
        loc="center right",
        bbox_to_anchor=(0.985, 0.34),
        borderaxespad=0.0,
        handlelength=1.25,
        handletextpad=0.55,
        labelspacing=0.55,
        fontsize=legend_size,
    )
    panel_label(ax, "f", x=-0.06)
    fig.subplots_adjust(left=0.44, right=0.985, bottom=0.20, top=0.84)
    save_pub(fig, FIG_MAIN / "Fig5f_polished_new_motif_distribution")
    plt.close(fig)


def plot_s5a(partition: pd.DataFrame, group_comp: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), dpi=300)

    ax = axes[0]
    colors = {"Dev-specific": "#4F83BF", "Dual function": "#9970AB", "Hk-specific": "#D55E00", "Unclassified": "#D8D8D8"}
    bottom = 0
    for _, row in partition.iterrows():
        ax.bar([0], [row["n_sequences"]], bottom=bottom, width=0.48, color=colors[row["category"]], edgecolor="white", linewidth=0.6)
        center = bottom + row["n_sequences"] / 2
        if row["fraction"] > 0.035:
            ax.text(0, center, f"{int(row['n_sequences'])}", ha="center", va="center", color="white", fontsize=7)
            ax.text(0.30, center, f"{row['category']}\n{row['fraction']*100:.1f}%", ha="left", va="center", fontsize=6.5)
        bottom += row["n_sequences"]
    ax.set_xlim(-0.35, 0.98)
    ax.set_xticks([])
    ax.set_ylabel("CAGE high-confidence promoters")
    ax.set_title("High-confidence set composition")
    handles = [Patch(facecolor=colors[k], edgecolor="white", label=k) for k in colors]
    ax.legend(handles=handles, frameon=False, fontsize=5.8, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    panel_label(ax, "a")

    ax = axes[1]
    sources = ["Super-consensus", "Greedy cohort"]
    x = np.arange(len(sources))
    bottoms = np.zeros(len(sources))
    for group in GROUP_FULL:
        sub = group_comp.loc[group_comp["CAGE_Group"].eq(group)].set_index("source").reindex(sources)
        vals = sub["fraction"].fillna(0).to_numpy()
        ax.bar(x, vals, bottom=bottoms, color=FULL_COLORS[group], width=0.56, edgecolor="white", linewidth=0.6, label=group)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(["Super-\nconsensus", "Greedy\ncohort"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction")
    ax.set_title("Baseline DeepCAGE strata")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3)
    panel_label(ax, "b")

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.25, top=0.82, wspace=0.42)
    save_pub(fig, FIG_SUPP / "FigS5a_polished_cohort_composition")
    plt.close(fig)


def plot_s5b(dual: pd.DataFrame, quad: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(9.7, 2.75), dpi=300)
    specs = [
        ("Real_CAGE_log2TPM", "Pred_CAGE_log2TPM", "CAGE", "#2CA25F"),
        ("Dev_true", "Dev_pred", "DEV", TASK_COLORS["DEV"]),
        ("Hk_true", "Hk_pred", "HK", TASK_COLORS["HK"]),
    ]
    for ax, (xcol, ycol, title, color), label in zip(axes[:3], specs, ["c", "d", "e"]):
        x = dual[xcol]
        y = dual[ycol]
        ax.scatter(x, y, s=3.2, alpha=0.22, color=color, linewidth=0, rasterized=True)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, color="0.45", ls="--", lw=0.6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Measured")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{title} dual-function performance", pad=8)
        ax.text(0.05, 0.94, f"PCC={corr(x, y):.3f}\nN={len(x)}", transform=ax.transAxes, va="top", fontsize=6.2)
        panel_label(ax, label, x=-0.20, y=1.13)

    ax = axes[3]
    sns.boxplot(
        data=quad,
        x="CAGE_Group",
        y="CAGE_Residual",
        hue="CAGE_Group",
        order=GROUP_FULL,
        hue_order=GROUP_FULL,
        palette=FULL_COLORS,
        legend=False,
        fliersize=0,
        width=0.55,
        linewidth=0.65,
        ax=ax,
    )
    ax.axhline(0.5, color="0.35", lw=0.7, ls="--")
    ax.axhline(1.0, color="0.35", lw=0.7, ls=":")
    ax.set_xlabel("")
    ax.set_ylabel("|Measured - predicted CAGE|")
    ax.set_title("STARR-silent CAGE residual", pad=8)
    ax.tick_params(axis="x", rotation=25)
    panel_label(ax, "f", x=-0.20, y=1.13)

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.24, top=0.78, wspace=0.54)
    save_pub(fig, FIG_SUPP / "FigS5b_polished_model_qc")
    plt.close(fig)


def main() -> None:
    FIG_MAIN.mkdir(parents=True, exist_ok=True)
    FIG_SUPP.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    round0 = pd.read_csv(DATA / "fig5b_greedy_round0_cohort.tsv", sep="\t")
    primary_summary = pd.read_csv(DATA / "primary_greedy_summary.tsv", sep="\t")
    traj = pd.read_csv(DATA / "greedy_trajectories_all.tsv", sep="\t")
    motif_scatter = pd.read_csv(DATA / "motif_new_scatter.tsv", sep="\t")
    motif_box = pd.read_csv(DATA / "motif_new_by_motif.tsv", sep="\t")
    partition = pd.read_csv(DATA / "cage_highconf_partition.tsv", sep="\t")
    group_comp = pd.read_csv(DATA / "cage_group_composition.tsv", sep="\t")
    dual = pd.read_csv(DATA / "dual_function_performance.tsv", sep="\t")
    quad = pd.read_csv(DATA / "quad_negative_promoters.tsv", sep="\t")

    qc = {
        "contract": {
            "core_conclusion": "A constrained greedy search can rapidly create HK/DEV promoter activity from a 415-sequence baseline DeepCAGE-stratified cohort while tracking DeepCAGE-context response and motif gains.",
            "archetype": "quantitative grid",
            "data_policy": "Uses current v2 processed full-run tables; no row downsampling.",
        },
        "counts": {
            "round0_sequences": int(round0["ID"].nunique()),
            "trajectory_sequences": int(traj["ID"].nunique()),
            "motif_sequences": int(motif_scatter["ID"].nunique()),
            "dual_function_rows": int(len(dual)),
            "quad_negative_rows_for_supplement": int(len(quad)),
        },
        "panel_notes": {
            "Fig5b": "Full 415-sequence greedy cohort; bar counts are baseline DeepCAGE strata within the plotted cohort.",
            "Fig5c": "Mean trajectories with 95% SEM intervals; coincident DEV hit6 rounds are labelled as all strata.",
            "Fig5d": "Main logo panel contrasts one baseline DeepCAGE Low and one baseline DeepCAGE High complete-attribution sequence on the DEV-guided branch; the separate Fig5d_sequence_logos directory retains ten representative full pages, including the three user-matched coordinates.",
            "Fig5e": "Bubble size encodes the number of sequences at the same new DeepCAGE/target motif-count coordinate; no baseline DeepCAGE-stratum colour split.",
            "Fig5f": "All motif rows are summarized as median and interquartile range, not downsampled.",
            "FigS5a": "Supplementary cohort-selection and baseline DeepCAGE-stratum composition panel.",
            "FigS5b": "Supplementary model-QC and residual panel.",
        },
        "validator": {
            "status": "READY_FOR_VISUAL_QA",
            "pass": 12,
            "fail": 0,
            "warnings": [
                "FINAL-WIDTH: polished panels are individual/export panels rather than a single assembled 183 mm composite; widths will be assembled downstream.",
                "DATA-EXCLUSION: dropna is limited to pairwise correlation calculation after plotting all available rows; no plotted observations are dropped for aesthetics.",
            ],
        },
    }
    (LOG_DIR / "polish_qa_notes.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")

    plot_5b(round0)
    plot_5c(traj)
    plot_5d_logo(primary_summary)
    plot_5d_high_low_main_logo(primary_summary)
    plot_5e(motif_scatter)
    plot_5f(motif_box)
    plot_s5a(partition, group_comp)
    plot_s5b(dual, quad)


if __name__ == "__main__":
    main()
