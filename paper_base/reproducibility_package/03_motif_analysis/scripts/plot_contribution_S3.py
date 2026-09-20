#!/usr/bin/env python
# coding: utf-8
"""DeepSTARR proximal/distal per-motif contribution supplement."""

from __future__ import annotations

from math import erf, sqrt

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from paths_S3 import CLEAN_DIR, panel_dirs
from style_S3 import DOUBLE_COL_MM, LOCATION_COLORS, TASK_COLORS, mm_to_in, save_all, set_pub_style


DATA = CLEAN_DIR / "motif_contribution_hits.csv"
ORDER = ["Proximal", "Distal"]
width_mm = 183
EXPORT_EXTS = (".svg", ".pdf", ".tiff", ".png")
EXPORT_DPI = 600


def star(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def bh_adjust(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    valid_indices = np.where(valid)[0]
    q[valid_indices[order]] = adjusted
    return q


def mannwhitneyu_pvalue(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan
    values = np.concatenate([x, y])
    group_x = np.concatenate([np.ones(n1, dtype=bool), np.zeros(n2, dtype=bool)])
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    tie_sum = 0.0
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        tie_count = end - start
        tie_sum += tie_count**3 - tie_count
        start = end
    r1 = ranks[group_x].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mean = n1 * n2 / 2.0
    n = n1 + n2
    tie_correction = 1.0 - tie_sum / (n**3 - n) if n > 1 else 1.0
    sd = sqrt(max(n1 * n2 * (n + 1) * tie_correction / 12.0, 0))
    if sd == 0:
        return 1.0
    z = (abs(u1 - mean) - 0.5) / sd
    return 2.0 * (0.5 * (1.0 - erf(abs(z) / sqrt(2.0))))


def add_sig_bracket(ax, x1, x2, y, h, text, color):
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + h, y + h, y],
        color=color,
        lw=0.9,
        clip_on=False,
        zorder=7,
    )
    ax.text(
        (x1 + x2) / 2,
        y + h * 1.15,
        text,
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=color,
        fontweight="bold",
        clip_on=False,
        zorder=8,
    )


def local_violin_top(g1: pd.Series, g2: pd.Series) -> float:
    values = pd.concat([g1, g2], ignore_index=True).astype(float)
    values = values[np.isfinite(values)]
    if values.empty:
        return 0.0
    return float(values.max())


def group_violin_top(values: pd.Series, upper_quantile: float = 0.94) -> float:
    values = values.astype(float)
    values = values[np.isfinite(values)]
    if values.empty:
        return 0.0
    return float(values.quantile(upper_quantile))


def draw_manual_violin_box(ax, sub, motifs):
    offsets = {"Proximal": -0.18, "Distal": 0.18}
    values_by_loc = {loc: [] for loc in ORDER}
    positions_by_loc = {loc: [] for loc in ORDER}
    label_col = "motif_display" if "motif_display" in sub.columns else "motif_label"
    for i, motif in enumerate(motifs):
        for loc in ORDER:
            values = sub[(sub[label_col].eq(motif)) & (sub["location"].eq(loc))]["hit_importance"].to_numpy(
                dtype=float
            )
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            values_by_loc[loc].append(values)
            positions_by_loc[loc].append(i + offsets[loc])
    for loc in ORDER:
        if not values_by_loc[loc]:
            continue
        parts = ax.violinplot(
            values_by_loc[loc],
            positions=positions_by_loc[loc],
            widths=0.32,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(LOCATION_COLORS[loc])
            body.set_edgecolor("#555555")
            body.set_alpha(0.72)
            body.set_linewidth(0.9)
        box = ax.boxplot(
            values_by_loc[loc],
            positions=positions_by_loc[loc],
            widths=0.11,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            manage_ticks=False,
        )
        for patch in box["boxes"]:
            patch.set_facecolor("#303030")
            patch.set_edgecolor("#111111")
            patch.set_alpha(0.75)
            patch.set_linewidth(0.75)
        for median in box["medians"]:
            median.set_color("white")
            median.set_linewidth(1.45)
        for element in ["whiskers", "caps"]:
            for artist in box[element]:
                artist.set_color("#111111")
                artist.set_linewidth(0.75)
    return offsets


def axis_label(label: str) -> str:
    text = str(label).replace("\n(", "\n").rstrip(")")
    text = text.replace("CREB/ATF", "CREB/\nATF")
    text = text.replace("GAGA-repeat", "GAGA-\nrepeat")
    return text


def plot_task(task: str, df: pd.DataFrame, dirs: dict[str, object]) -> pd.DataFrame:
    sub = df[df["task"].eq(task)].copy()
    label_col = "motif_display" if "motif_display" in sub.columns else "motif_label"
    motifs = (
        sub[[label_col, "motif_index"]]
        .drop_duplicates()
        .sort_values("motif_index")[label_col]
        .tolist()
    )
    set_pub_style(font_size=10.5)
    fig, ax = plt.subplots(figsize=(mm_to_in(DOUBLE_COL_MM * 0.95), mm_to_in(82)))
    offsets = draw_manual_violin_box(ax, sub, motifs)
    ymax = float(sub["hit_importance"].quantile(0.992))
    ymin = float(sub["hit_importance"].quantile(0.005))
    yrange = max(ymax - ymin, 1.0)
    sig_top = ymax
    rows = []
    for i, motif in enumerate(motifs):
        g1 = sub[(sub[label_col].eq(motif)) & (sub["location"].eq("Proximal"))]["hit_importance"]
        g2 = sub[(sub[label_col].eq(motif)) & (sub["location"].eq("Distal"))]["hit_importance"]
        p = mannwhitneyu_pvalue(g1, g2) if len(g1) >= 3 and len(g2) >= 3 else np.nan
        med1, med2 = g1.median(), g2.median()
        rows.append(
            {
                "task": task,
                "motif": motif,
                "n_proximal_hits": int(g1.notna().sum()),
                "n_distal_hits": int(g2.notna().sum()),
                "median_proximal": med1,
                "median_distal": med2,
                "p_mannwhitney": p,
            }
        )

    stats = pd.DataFrame(rows)
    stats["q_bh_within_task"] = bh_adjust(stats["p_mannwhitney"])
    for i, motif in enumerate(motifs):
        g1 = sub[(sub[label_col].eq(motif)) & (sub["location"].eq("Proximal"))]["hit_importance"]
        g2 = sub[(sub[label_col].eq(motif)) & (sub["location"].eq("Distal"))]["hit_importance"]
        q = stats.loc[stats["motif"].eq(motif), "q_bh_within_task"].iloc[0]
        med1 = stats.loc[stats["motif"].eq(motif), "median_proximal"].iloc[0]
        med2 = stats.loc[stats["motif"].eq(motif), "median_distal"].iloc[0]
        if np.isfinite(q) and q < 0.05:
            bias_loc = "Proximal" if med1 > med2 else "Distal"
            color = LOCATION_COLORS[bias_loc]
            y = local_violin_top(g1, g2) + yrange * 0.018
            h = yrange * 0.018
            add_sig_bracket(
                ax,
                i + offsets["Proximal"],
                i + offsets["Distal"],
                y,
                h,
                star(q),
                color,
            )
            sig_top = max(sig_top, y + h * 2.35)

    ax.set_xlim(-0.65, len(motifs) - 0.35)
    ax.set_ylim(max(0, ymin - yrange * 0.05), sig_top + yrange * 0.08)
    ax.set_xticks(np.arange(len(motifs)))
    xtick_fs = 7.1 if len(motifs) > 10 else 8.4
    ax.set_xticklabels(
        [axis_label(motif) for motif in motifs],
        fontsize=xtick_fs,
        rotation=0,
        ha="center",
        rotation_mode="anchor",
    )
    ax.set_ylabel("Fi-NeMo hit importance", fontsize=10.8)
    ax.tick_params(axis="y", labelsize=9.8)
    ax.set_xlabel("")
    ax.set_title(task, color=TASK_COLORS[task], fontweight="bold", fontsize=11.4, pad=9)
    ax.legend(
        handles=[Patch(facecolor=LOCATION_COLORS[loc], label=loc) for loc in ORDER],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.13),
        frameon=False,
        ncol=2,
        fontsize=9.2,
        handlelength=1.0,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.072, right=0.988, bottom=0.27, top=0.83)
    paths = save_all(fig, dirs, f"deepstarr_{task.lower()}_proximal_distal_motif_contribution")
    plt.close(fig)
    print(f"saved: {paths['png']}")
    return stats


def main() -> None:
    dirs = panel_dirs("contribution")
    df = pd.read_csv(DATA)
    rows = []
    for task in ["HK", "DEV"]:
        rows.append(plot_task(task, df, dirs))
    pd.concat(rows, ignore_index=True).to_csv(dirs["qa"] / "contribution_stats.csv", index=False)


if __name__ == "__main__":
    main()
