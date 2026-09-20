#!/usr/bin/env python
# coding: utf-8
"""Fig 3A-1 — Subcluster 全局相似度热图（带 MC 边界 + Task 标注 + colorbar + 双图例）。

来源: MC000_Motif_Trimming.ipynb cell 37 (fig3A 第一部分, 24bp)

数据（clean CSV，对齐 assemble_figure_shared_motifs.py canonical 路径）:
  - shared_motif_subcluster_similarity_affinity.csv  (163×163 affinity = raw_sim_mat)
  - shared_motif_subcluster_metadata.csv  (entity_name/task/meta_label)

改动（相对 cell 37）: Task 图例里 CAGE_NEW → CAGE（显示名）。

输出: <REPO>/plot/output/png/figure3a_1/similarity_heatmap.png
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from figure3_paths import CLEAN_DIR, panel_dirs
from figure3_style import DOUBLE_COL_MM, TASK_COLORS as BASE_TASK_COLORS, mm_to_in, save_all, set_pub_style

mpl.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.4, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
})

AFFINITY_CSV = CLEAN_DIR / 'shared_motif_subcluster_similarity_affinity.csv'
METADATA_CSV = CLEAN_DIR / 'shared_motif_subcluster_metadata.csv'

# Task 显示名: CAGE_NEW -> CAGE（用户要求）
TASK_DISPLAY = {"CAGE_NEW": "CAGE", "HK": "HK", "DEV": "DEV"}
TASK_COLORS = {"CAGE": BASE_TASK_COLORS["CAGE"], "HK": BASE_TASK_COLORS["HK"], "DEV": BASE_TASK_COLORS["DEV"]}
width_mm = 183
EXPORT_TARGETS = ("figure.svg", "figure.pdf", "figure.tiff", "figure.png")
EXPORT_DPI = 600


def compress_clustermap_width(g, target_right=0.76):
    """把 clustermap 图体压到左边，给右侧 colorbar/图例留位（复用 cell 37）。"""
    axes = [g.ax_heatmap, g.ax_row_dendrogram, g.ax_col_dendrogram]
    if hasattr(g, "ax_row_colors") and g.ax_row_colors is not None:
        axes.append(g.ax_row_colors)
    if hasattr(g, "ax_col_colors") and g.ax_col_colors is not None:
        axes.append(g.ax_col_colors)
    x0 = min(ax.get_position().x0 for ax in axes)
    x1 = max(ax.get_position().x1 for ax in axes)
    scale = (target_right - x0) / (x1 - x0)
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([x0 + (pos.x0 - x0) * scale, pos.y0, pos.width * scale, pos.height])


def main():
    set_pub_style(font_size=15.5)
    dirs = panel_dirs("figure3a_1")
    # ---- 1. 数据 ----
    sim_df = pd.read_csv(AFFINITY_CSV, index_col=0)
    meta = pd.read_csv(METADATA_CSV).set_index('entity_name').loc[sim_df.index]

    entity_names = list(sim_df.index)
    task_raw = meta['task'].values
    task_list = [TASK_DISPLAY.get(t, t) for t in task_raw]  # CAGE_NEW -> CAGE
    mc_list = [int(x) for x in meta['meta_label'].values]

    # ---- 2. 按 (MC, Task) 排序 ----
    df_order = pd.DataFrame({
        "Index": np.arange(len(mc_list)),
        "Name": entity_names,
        "MC": mc_list,
        "Task": task_list,
    }).sort_values(by=["MC", "Task"], kind="mergesort")
    sorted_idx = df_order["Index"].values
    sorted_names = df_order["Name"].values
    sorted_sim = sim_df.iloc[sorted_idx, sorted_idx].values
    sorted_task = df_order["Task"].values
    sorted_mc = df_order["MC"].values
    sim_plot = pd.DataFrame(sorted_sim, index=sorted_names, columns=sorted_names)

    # ---- 3. 颜色标注 ----
    unique_mcs = sorted(set(mc_list))
    mc_palette = sns.color_palette("tab10", n_colors=len(unique_mcs))
    mc_color_map = {mc: mc_palette[i] for i, mc in enumerate(unique_mcs)}

    anno_df = pd.DataFrame({
        "Task": pd.Series(sorted_task, index=sorted_names).map(TASK_COLORS),
        "MC": pd.Series(sorted_mc, index=sorted_names).map(mc_color_map),
    }, index=sorted_names)

    # ---- 4. clustermap（对齐 cell 37）----
    g = sns.clustermap(
        sim_plot, cmap="viridis", vmin=0, vmax=1,
        figsize=(mm_to_in(DOUBLE_COL_MM), mm_to_in(136)),
        row_colors=anno_df, col_colors=anno_df,
        row_cluster=True, col_cluster=True,
        xticklabels=False, yticklabels=False,
        linewidths=0,
        dendrogram_ratio=(0.13, 0.13), colors_ratio=(0.025, 0.025),
        cbar_pos=None, tree_kws={"linewidths": 0.35},
    )
    compress_clustermap_width(g, target_right=0.70)

    # ---- 5. 样式 ----
    g.ax_heatmap.set_xlabel(""); g.ax_heatmap.set_ylabel("")
    for sp in g.ax_heatmap.spines.values():
        sp.set_visible(False)
    for ax in [g.ax_row_dendrogram, g.ax_col_dendrogram]:
        for coll in ax.collections:
            coll.set_linewidth(0.35); coll.set_color("black")
    if hasattr(g, "ax_col_colors") and g.ax_col_colors is not None:
        g.ax_col_colors.tick_params(labelsize=11, length=0)
    if hasattr(g, "ax_row_colors") and g.ax_row_colors is not None:
        g.ax_row_colors.tick_params(labelsize=11, length=0)

    # ---- 6. 手动 colorbar（对齐 cell 37 第 6 步）----
    mappable = g.ax_heatmap.collections[0]
    cax = g.fig.add_axes([0.77, 0.70, 0.024, 0.20])
    cb = g.fig.colorbar(mappable, cax=cax, ticks=[0, 0.25, 0.50, 0.75, 1.00])
    cb.ax.tick_params(labelsize=13.8, length=2.8, width=0.6)
    cb.set_label("Similarity", fontsize=14.5, labelpad=7)

    # ---- 7. 双图例（Task + Meta-cluster，对齐 cell 37 第 7 步）----
    task_handles = [Patch(facecolor=TASK_COLORS[t], edgecolor="none", label=t)
                    for t in ["CAGE", "HK", "DEV"]]  # ★ CAGE 不是 CAGE_NEW
    mc_handles = [Patch(facecolor=mc_color_map[m], edgecolor="none", label=f"MC_{m:03d}")
                  for m in unique_mcs]

    g.fig.legend(handles=task_handles, title="Task", loc="upper left",
                 bbox_to_anchor=(0.77, 0.58), frameon=False, fontsize=14.2,
                 title_fontsize=14.5, handlelength=1.4, handleheight=0.8,
                 handletextpad=0.5, labelspacing=0.45, borderaxespad=0)
    g.fig.legend(handles=mc_handles, title="Meta-cluster", loc="upper left",
                 bbox_to_anchor=(0.77, 0.40), frameon=False, fontsize=13.6,
                 title_fontsize=14.2, handlelength=1.4, handleheight=0.8,
                 handletextpad=0.5, labelspacing=0.35, borderaxespad=0, ncol=1)

    paths = save_all(g.fig, dirs, 'similarity_heatmap')
    pd.DataFrame({"entity_name": sorted_names, "task": sorted_task, "mc": sorted_mc}).to_csv(
        dirs["qa"] / "similarity_order.csv", index=False
    )
    plt.close('all')
    print(f"saved: {paths['png']}  ({len(entity_names)} subclusters, {len(unique_mcs)} MCs)")


if __name__ == '__main__':
    main()
