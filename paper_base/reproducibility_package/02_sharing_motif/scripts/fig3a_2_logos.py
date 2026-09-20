#!/usr/bin/env python
# coding: utf-8
"""Fig 3A-2 — Meta-cluster motif logo grid（IC-trimmed, 只留正值, 合并标签）。

来源: MC000_Motif_Trimming.ipynb cell 49 (fig3A 第二部分)

微调（用户标注要求）:
  1. CWM 只展示正值（np.clip(cwm, 0, None)），去掉朝下负贡献
  2. 标签合并：标题用 "MC_ID (Family_Type) - TF"（替换原下方冗余的紫色 TF 行）
  3. logomaker 美化（center_values/fade_below/shade_below，沿用 cell 49）

数据:
  - Final_Annotated_MetaClusters_IC_Trimmed.tsv  (CWM 列字符串存储, parse_cwm 解析)
    列: MC_ID, CWM, Family_Type, Match_1 (TF)

输出: <REPO>/plot/output/png/figure3a_2/motif_logos_grid.png
依赖: logomaker（用 Deepstarr env）。
"""

import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logomaker

from figure3_paths import RAW_DIR, panel_dirs
from figure3_style import DOUBLE_COL_MM, mm_to_in, save_all, set_pub_style

TSV = RAW_DIR / 'ic_trimmed_results' / 'Final_Annotated_MetaClusters_IC_Trimmed.tsv'
width_mm = 183
EXPORT_TARGETS = ("figure.svg", "figure.pdf", "figure.tiff", "figure.png")
EXPORT_DPI = 600

TF_DISPLAY = {
    "CREB/ATF/3": "CREB/ATF",
    "Ebox/CAGCTG/CACCTG": "Ebox/CAGCTG",
}


def parse_cwm(value):
    """解析 TSV 里字符串存储的 CWM（对齐 assemble 模块 parse_cwm）。"""
    text = str(value)
    arr = np.fromstring(text.replace("[", " ").replace("]", " "), sep=" ")
    if arr.size % 4:
        raise ValueError(f"CWM length not divisible by 4: {arr.size}")
    return arr.reshape((-1, 4))


def trim_cwm(cwm, threshold_ratio=0.3, min_len=6, max_len=18):
    """IC 自适应裁剪（复用 MC000 trim_cwm_for_meme）。"""
    score = np.sum(np.abs(cwm), axis=1)
    threshold = np.max(score) * threshold_ratio
    keep = score >= threshold
    if not np.any(keep):
        return cwm
    start = np.argmax(keep)
    end = len(keep) - np.argmax(keep[::-1])
    trimmed = cwm[start:end]
    if len(trimmed) < min_len:
        c = np.argmax(score); h = min_len // 2
        trimmed = cwm[max(0, c - h):min(len(cwm), c + h)]
    if len(trimmed) > max_len:
        c = np.argmax(score); h = max_len // 2
        trimmed = cwm[max(0, c - h):min(len(cwm), c + h)]
    return trimmed


def draw_logo(ax, cwm):
    """Draw one motif logo directly on its final vector axis."""
    df_logo = pd.DataFrame(cwm, columns=['A', 'C', 'G', 'T'])
    logomaker.Logo(df_logo, ax=ax, width=.86, color_scheme='classic')
    ax.set_ylim(bottom=0)
    ax.set_yticks([])
    ax.set_xticks([])
    for sp in ['right', 'top']:
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_linewidth(0.65)
    ax.spines['bottom'].set_linewidth(0.65)


def wrap_tf_label(label):
    text = TF_DISPLAY.get(str(label), str(label))
    if len(text) <= 14:
        return text
    parts = text.split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2]) + "\n" + "/".join(parts[2:])
    return text[:14] + "\n" + text[14:]


def main():
    set_pub_style(font_size=17.7)
    df = pd.read_csv(TSV, sep='\t')
    df['MC_num'] = df['MC_ID'].str.replace('MC_', '', regex=False).astype(int)
    df = df.sort_values('MC_num').reset_index(drop=True)

    dirs = panel_dirs("figure3a_2")

    n = len(df)
    motifs = []
    for _, row in df.iterrows():
        cwm = parse_cwm(row['CWM'])
        trimmed = trim_cwm(cwm)
        trimmed = np.clip(trimmed, 0, None)  # ★ 只展示正值
        motifs.append(trimmed)

    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(mm_to_in(DOUBLE_COL_MM * 0.94), mm_to_in(36 * nrows)), dpi=300)
    axes = axes.flatten()
    for ax in axes:
        ax.axis('off')

    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i]
        ax.axis('on')
        mc_id = row['MC_ID']
        tf = row.get('TF_LABEL', row.get('Match_1', ''))
        draw_logo(ax, motifs[i])
        label = f"{mc_id}"
        ax.text(0.0, 1.07, label, transform=ax.transAxes,
                ha='left', va='bottom', fontsize=15.6, color='#222222')
        if pd.notna(tf) and str(tf).strip():
            ax.text(1.0, 1.07, wrap_tf_label(tf), transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=14.6, color='#4D4D4D', linespacing=0.9)

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    fig.subplots_adjust(left=0.05, right=0.99, top=0.96, bottom=0.04,
                        wspace=0.18, hspace=0.86)
    paths = save_all(fig, dirs, 'motif_logos_grid')
    plt.close()
    df[['MC_ID', 'Family_Type', 'Match_1']].to_csv(dirs["qa"] / "motif_logo_labels.csv", index=False)
    print(f"saved: {paths['png']}  ({n} meta-clusters)")


if __name__ == '__main__':
    main()
