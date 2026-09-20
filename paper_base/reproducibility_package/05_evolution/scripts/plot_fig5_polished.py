#!/usr/bin/env python
"""Polished Figure 5 panels for the evolution module.

The script reads the v2 processed tables and writes publication-ready
SVG/PDF/TIFF/PNG outputs into plot_v2/polish/figures.
"""

from __future__ import annotations

import json
import math
import re
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
DATA = PLOT_V2_ROOT / "data" / "processed"
PLOT_V1_DATA = PLOT_V2_ROOT.parent / "plot_v1" / "data" / "processed"
PLOT_V1_RAW = PLOT_V2_ROOT.parent / "plot_v1" / "data" / "raw"
DEEPMUT_ROOT = Path(r"G:\我的云端硬盘\DeepEpromote\Drosophila\DeepSTARR\promoter_mut\results_context_dependent\DeepMut_Refactored")
DEEPMUT_INPUTS = DEEPMUT_ROOT / "finemo_inputs"
DEEPMUT_HITS = DEEPMUT_ROOT / "finemo_hits"
ANNOTATED_RUN_ROOT = Path(
    r"G:\我的云端硬盘\DeepEpromote\Drosophila\DeepSTARR\promoter_mut\results_context_dependent"
    r"\reviewer_grade_runs\reviewer_greedy_20260724_115905\attribution_finemo_24bp_annotated"
)
ANNOTATED_MOTIF_MAP = ANNOTATED_RUN_ROOT / "motif_atlas" / "finemo_pattern_to_annotated_mc.tsv"
ANNOTATED_ATTR_DIR = ANNOTATED_RUN_ROOT / "attribution"
ANNOTATED_HITS_LONG = ANNOTATED_RUN_ROOT / "summaries" / "finemo_hits_annotated_long.tsv"
FIG_MAIN = POLISH_ROOT / "figures" / "main"
FIG_SUPP = POLISH_ROOT / "figures" / "supplement"
FIG5D_SEQ_DIR = FIG_MAIN / "Fig5d_sequence_logos_full249"
LOG_DIR = POLISH_ROOT / "logs"

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
    "HK-CAGE": "#79B7D8",
    "DEV target": "#17956F",
    "DEV-CAGE": "#C99A22",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
sns.set_theme(style="white", context="paper")


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
    ax.text(x, y, label, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold")


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


def filtered_motifs(hits: pd.DataFrame, start: int, end: int) -> list[tuple[int, int, str, float]]:
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
        if len(kept) >= 2:
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
    if end - start > 120:
        x = np.arange(start + 1, end + 1)
        for base_i, base in enumerate(BASES):
            vals = arr[start:end, base_i]
            keep = np.abs(vals) > 1e-8
            if np.any(keep):
                ax.bar(
                    x[keep],
                    vals[keep],
                    width=0.82,
                    color=BASE_COLORS[base],
                    edgecolor="none",
                    alpha=0.95,
                    rasterized=True,
                )
        return
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
        label_half_width = max(4.2, len(label) * 0.56 * (motif_fontsize / 7.0))
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
        y_text = max(y_ref_max, 0.0) + pad * (0.35 + 1.45 * level)
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
) -> None:
    draw_contribution_logo(ax, arr, start, end)
    ax.set_xlim(start + 1, end)
    draw_motif_spans(ax, motifs, y_ref_max=y_ref_max, pad=pad)
    for pos in muts:
        ax.plot([pos, pos], [y_lim[0], y_lim[0] + pad * 0.48], color="#B2182B", lw=0.7, solid_capstyle="butt")
    ax.axhline(0, color="0.3", lw=0.55, zorder=0)
    ax.set_ylim(*y_lim)
    ax.set_title(title, pad=4)
    ax.tick_params(axis="x", length=2.2, width=0.65)
    ax.tick_params(axis="y", length=2.2, width=0.65)
    ax.set_xlabel("Position in sequence (bp)")
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
    ax.legend(frameon=False, loc="lower right", title="Initial CAGE")
    panel_label(ax, "b")

    ax = axes[1]
    counts = data["CAGE_Group_Short"].value_counts().reindex(GROUP_ORDER).fillna(0).astype(int)
    bars = ax.bar(GROUP_ORDER, counts.values, color=[GROUP_COLORS[g] for g in GROUP_ORDER], edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 5, f"{val}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Sequences")
    ax.set_title("Initial CAGE strata")
    ax.set_ylim(0, max(counts.values) * 1.18)

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.20, top=0.82, wspace=0.34)
    save_pub(fig, FIG_MAIN / "Fig5b_polished_greedy_round0_cage_groups")
    plt.close(fig)


def plot_5c(trajectories: pd.DataFrame) -> None:
    data = trajectories.loc[trajectories["Round"] <= 40].copy()
    data["CAGE_Group_Short"] = data["CAGE_Group"].map(GROUP_SHORT).fillna(data["CAGE_Group_Short"])
    stats_score = mean_sem(data, "Score_Opt")
    stats_cage = mean_sem(data, "CAGE_pred")

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0), dpi=300, sharex=True)
    for col, task in enumerate(["HK", "DEV"]):
        for row, stat, ylabel, threshold in [
            (0, stats_score, f"{task} prediction", 6),
            (1, stats_cage, "CAGE prediction", None),
        ]:
            ax = axes[row, col]
            hit_rounds = {}
            for group in GROUP_ORDER:
                g = stat.loc[(stat["task"].eq(task)) & (stat["CAGE_Group_Short"].eq(group))].sort_values("Round")
                if g.empty:
                    continue
                x = g["Round"].to_numpy(float)
                mean = g["mean"].to_numpy(float)
                lo = g["lo"].to_numpy(float)
                hi = g["hi"].to_numpy(float)
                ax.plot(x, mean, color=GROUP_COLORS[group], lw=1.7)
                ax.fill_between(x, lo, hi, color=GROUP_COLORS[group], alpha=0.13, linewidth=0)
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
                        ymin + 0.06 * (ymax - ymin) + idx * 0.08 * (ymax - ymin),
                        label,
                        rotation=90,
                        color=color,
                        fontsize=6,
                        va="bottom",
                    )
            else:
                ax.axhline(2, color="0.45", lw=0.6, ls="--")
                ax.axhline(4, color="0.45", lw=0.6, ls="--")
            ax.set_title(f"{task} optimization" if row == 0 else "")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", linestyle=":", alpha=0.22)
            if row == 1:
                ax.set_xlabel("Mutation round")
            if row == 0 and col == 0:
                panel_label(ax, "c")

    handles = [Line2D([0], [0], color=GROUP_COLORS[g], lw=1.8, label=g) for g in GROUP_ORDER]
    fig.legend(handles=handles, title="Initial CAGE", frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.88, hspace=0.20, wspace=0.30)
    save_pub(fig, FIG_MAIN / "Fig5c_polished_greedy_trajectories")
    plt.close(fig)


def plot_5d_logo() -> None:
    selected10 = pd.read_csv(PLOT_V1_DATA / "legacy_selected_10_metadata.tsv", sep="\t")
    selected_ids = set(selected10["ID"].astype(str))
    attr_by_track: dict[str, dict[str, object]] = {}
    for track in ["HK", "DEV", "CAGE"]:
        z = np.load(ANNOTATED_ATTR_DIR / track / "attribution_arrays.npz")
        meta = pd.read_csv(ANNOTATED_ATTR_DIR / track / "metadata.tsv", sep="\t")
        meta["ID"] = meta["ID"].astype(str)
        attr_by_track[track] = {
            "act": z["act_contrib"],
            "ohe": z["one_hot"],
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
    annotated_hits = pd.read_csv(ANNOTATED_HITS_LONG, sep="\t", usecols=hit_cols)
    annotated_hits["ID"] = annotated_hits["ID"].astype(str)
    annotated_hits = annotated_hits.loc[
        annotated_hits["ID"].isin(selected_ids)
        & annotated_hits["track"].isin(["HK", "DEV", "CAGE"])
        & annotated_hits["branch_task"].isin(["HK", "DEV"])
        & annotated_hits["milestone"].isin(["round0", "hit6"])
    ].copy()

    def select_meta_row(track: str, branch: str, seq_id: str, milestone: str) -> pd.Series:
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
            raise KeyError(f"Missing annotated attribution row for {track}/{branch}/{milestone}: {seq_id}")
        return sub.sort_values(["sequence_index"]).iloc[0]

    def hits_for(track: str, branch: str, seq_id: str, milestone: str) -> pd.DataFrame:
        sub = annotated_hits.loc[
            annotated_hits["track"].eq(track)
            & annotated_hits["branch_task"].eq(branch)
            & annotated_hits["ID"].eq(seq_id)
            & annotated_hits["milestone"].eq(milestone)
        ].copy()
        if sub.empty and milestone == "round0":
            sub = annotated_hits.loc[
                annotated_hits["track"].eq(track)
                & annotated_hits["ID"].eq(seq_id)
                & annotated_hits["milestone"].eq("round0")
            ].copy()
        return sub

    all_attr = np.concatenate([np.asarray(v["act"]).reshape(-1) for v in attr_by_track.values()])
    score_min = float(np.nanpercentile(all_attr, 0.4))
    score_max = float(np.nanpercentile(all_attr, 99.6))
    span = max(score_max - score_min, 0.08)
    pad = max(span * 0.20, 0.025)
    y_lim = (min(score_min - pad * 0.5, -0.03), score_max + pad * 3.45)

    def seq_from_row(track: str, row: pd.Series) -> str:
        ohe = attr_by_track[track]["ohe"]
        assert isinstance(ohe, np.ndarray)
        return onehot_to_seq(ohe[int(row["sequence_index"])])

    def arr_for(track: str, row: pd.Series) -> np.ndarray:
        act = attr_by_track[track]["act"]
        assert isinstance(act, np.ndarray)
        return act[int(row["sequence_index"])]

    def mutations(branch: str, seq_id: str, start: int, end: int) -> list[int]:
        row0 = select_meta_row(branch, branch, seq_id, "round0")
        row1 = select_meta_row(branch, branch, seq_id, "hit6")
        seq0 = seq_from_row(branch, row0)
        seq1 = seq_from_row(branch, row1)
        return [i + 1 for i in range(start, end) if seq0[i] != seq1[i]]

    FIG5D_SEQ_DIR.mkdir(parents=True, exist_ok=True)
    row_specs = [("HK", "HK"), ("HK", "CAGE"), ("DEV", "DEV"), ("DEV", "CAGE")]
    exported = []
    missing = []

    for seq_no, row in enumerate(selected10.itertuples(index=False), start=1):
        seq_id = str(row.ID)
        try:
            seq_len = int(np.asarray(attr_by_track["HK"]["act"]).shape[1])
            _ = select_meta_row("HK", "HK", seq_id, "hit6")
            _ = select_meta_row("DEV", "DEV", seq_id, "hit6")
            _ = select_meta_row("CAGE", "HK", seq_id, "hit6")
            _ = select_meta_row("CAGE", "DEV", seq_id, "hit6")
        except KeyError:
            missing.append(seq_id)
            continue
        start, end = 0, seq_len
        fig, axes = plt.subplots(4, 2, figsize=(7.2, 7.95), dpi=300, sharex=False, sharey=True)

        for r, (branch, track) in enumerate(row_specs):
            muts = mutations(branch, seq_id, start, end)
            for c, milestone in enumerate(["round0", "hit6"]):
                meta_row = select_meta_row(track, branch, seq_id, milestone)
                hsub = hits_for(track, branch, seq_id, milestone)
                motifs = filtered_motifs(hsub, start, end)
                title = f"{track} attribution | round0" if milestone == "round0" else f"{track} attribution | {branch} hit6"
                render_logo_panel(
                    axes[r, c],
                    arr=arr_for(track, meta_row),
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
            axes[r, 0].text(
                -0.11,
                1.17,
                f"{branch}-guided" if track == branch else f"{branch}-matched CAGE",
                transform=axes[r, 0].transAxes,
                ha="left",
                va="bottom",
                fontsize=7.1,
                fontweight="bold",
                color=TASK_COLORS[branch] if track == branch else "#D55E00",
            )

        header = (
            f"seq {seq_no:02d} | {short_id(seq_id)} | CAGE {row.CAGE_Group}; "
            f"measured={float(row.Real_CAGE_log2TPM):.2f}, predicted={float(row.Pred_CAGE_log2TPM):.2f}"
        )
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
        "n_requested": int(len(selected10)),
        "n_exported": int(len(exported)),
        "missing_ids": missing,
        "export_stems": exported,
        "source": (
            "legacy_selected_10_metadata plus reviewer_greedy_20260724_115905 "
            "attribution_finemo_24bp_annotated attribution arrays and annotated FiNeMo hits; "
            "contribution logos use act_contrib, equivalent to attribution multiplied by one-hot"
        ),
    }
    (LOG_DIR / "fig5d_sequence_logo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def bubble_table(sub: pd.DataFrame) -> pd.DataFrame:
    return (
        sub.groupby(["cage_new_motifs", "target_new_motifs"], as_index=False)
        .agg(n_sequences=("ID", "nunique"))
        .sort_values(["cage_new_motifs", "target_new_motifs"])
    )


def plot_5e(motif_scatter: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.75), dpi=300)
    for ax, branch, ylabel in zip(axes, ["HK", "DEV"], ["New HK motif count", "New DEV motif count"]):
        sub = motif_scatter.loc[motif_scatter["branch_task"].eq(branch)]
        tab = bubble_table(sub)
        sizes = 18 + 9 * np.sqrt(tab["n_sequences"].to_numpy())
        ax.scatter(
            tab["cage_new_motifs"],
            tab["target_new_motifs"],
            s=sizes,
            color=TASK_COLORS[branch],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.45,
        )
        r = corr(sub["cage_new_motifs"], sub["target_new_motifs"])
        ax.text(0.04, 0.95, f"r={r:.2f}\nN={sub['ID'].nunique()}", transform=ax.transAxes, va="top", fontsize=7)
        ax.set_xlabel("New CAGE motif count")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{branch}-guided sequences")
        set_integer_ticks(ax)
        max_n = int(tab["n_sequences"].max())
        legend_counts = sorted(set([1, 5, 10, max_n]))
        legend_counts = [n for n in legend_counts if n <= max_n]
        handles = [
            ax.scatter([], [], s=18 + 9 * math.sqrt(n), color=TASK_COLORS[branch], alpha=0.72, edgecolor="white", linewidth=0.45, label=str(n))
            for n in legend_counts
        ]
        ax.legend(handles=handles, title="Sequences", frameon=False, loc="lower right", labelspacing=0.75)
    panel_label(axes[0], "e")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.19, top=0.82, wspace=0.36)
    save_pub(fig, FIG_MAIN / "Fig5e_polished_new_target_vs_cage_motifs")
    plt.close(fig)


def plot_5f(motif_box: pd.DataFrame) -> None:
    data = motif_box.copy()
    series_order = ["HK target", "HK-CAGE", "DEV target", "DEV-CAGE"]
    summary = (
        data.groupby(["motif_label", "series"], observed=True)["new_hit6_count"]
        .agg(q25=lambda x: float(np.quantile(x, 0.25)), median="median", q75=lambda x: float(np.quantile(x, 0.75)))
        .reset_index()
    )
    motif_order = (
        summary.groupby("motif_label", observed=True)["median"]
        .max()
        .sort_values(ascending=True)
        .index.tolist()
    )
    y_lookup = {motif: i for i, motif in enumerate(motif_order)}
    offsets = {"HK target": -0.24, "HK-CAGE": -0.08, "DEV target": 0.08, "DEV-CAGE": 0.24}

    fig, ax = plt.subplots(1, 1, figsize=(6.9, 3.25), dpi=300)
    for series in series_order:
        sub = summary.loc[summary["series"].eq(series)].copy()
        y = np.array([y_lookup[m] + offsets[series] for m in sub["motif_label"]])
        ax.hlines(y, sub["q25"], sub["q75"], color=SERIES_COLORS[series], lw=1.4, alpha=0.85)
        ax.scatter(
            sub["median"],
            y,
            s=18,
            color=SERIES_COLORS[series],
            edgecolor="white",
            linewidth=0.35,
            label=series,
            zorder=3,
        )
    ax.set_yticks(range(len(motif_order)))
    ax.set_yticklabels(motif_order)
    ax.set_xlabel("New motif hits at hit6, median and IQR")
    ax.set_ylabel("")
    ax.set_title("Per-motif gains from round0 to hit6")
    ax.grid(axis="x", linestyle=":", alpha=0.22)
    ax.legend(frameon=False, title="", ncol=2, loc="lower right", columnspacing=0.9, handlelength=1.4)
    panel_label(ax, "f", x=-0.06)
    fig.subplots_adjust(left=0.23, right=0.99, bottom=0.16, top=0.84)
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
    ax.set_title("CAGE-group representation")
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
    traj = pd.read_csv(DATA / "greedy_trajectories_all.tsv", sep="\t")
    motif_scatter = pd.read_csv(DATA / "motif_new_scatter.tsv", sep="\t")
    motif_box = pd.read_csv(DATA / "motif_new_by_motif.tsv", sep="\t")
    partition = pd.read_csv(DATA / "cage_highconf_partition.tsv", sep="\t")
    group_comp = pd.read_csv(DATA / "cage_group_composition.tsv", sep="\t")
    dual = pd.read_csv(DATA / "dual_function_performance.tsv", sep="\t")
    quad = pd.read_csv(DATA / "quad_negative_promoters.tsv", sep="\t")

    qc = {
        "contract": {
            "core_conclusion": "A constrained greedy search can rapidly create HK/DEV promoter activity from a 415-sequence CAGE-stratified cohort while tracking CAGE-context response and motif gains.",
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
            "Fig5b": "Full 415-sequence greedy cohort; bar counts are CAGE strata within the plotted cohort.",
            "Fig5c": "Mean trajectories with 95% SEM intervals; coincident DEV hit6 rounds are labelled as all strata.",
            "Fig5d": "Redrawn from processed attribution arrays and annotated Fi-NeMo hit windows; contribution letters are rendered directly in matplotlib without external logo dependencies.",
            "Fig5e": "Bubble size encodes the number of sequences at the same new CAGE/target motif-count coordinate; no CAGE-group colour split.",
            "Fig5f": "All motif rows are summarized as median and interquartile range, not downsampled.",
            "FigS5a": "Supplementary cohort-selection and CAGE-stratum composition panel.",
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
    plot_5d_logo()
    plot_5e(motif_scatter)
    plot_5f(motif_box)
    plot_s5a(partition, group_comp)
    plot_s5b(dual, quad)


if __name__ == "__main__":
    main()
