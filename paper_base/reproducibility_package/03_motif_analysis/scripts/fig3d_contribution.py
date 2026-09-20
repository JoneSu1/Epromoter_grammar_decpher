#!/usr/bin/env python
# coding: utf-8
"""Fig 3D — per-motif 贡献分 violin（去 dot 改 box, median 加粗, Mann-Whitney highlight 倾向）。

来源: Motif_Density_Visualization.ipynb cell 10/12 (fig3 D)

微调（用户标注要求）:
  1. 去掉 stripplot（dot 渲染）
  2. 内置深黑 box、明显，median 加粗
  3. 只留 HK / DEV vs CAGE 的重要性得分对比
  4. ★ Mann-Whitney U 检验判断倾向 + 局部 bracket highlight：
     对每个 motif 比较 HK vs CAGE（或 DEV vs CAGE），p<0.05 且中位数差方向决定倾向；
     颜色与 DeepSTARR 模块统一：CAGE=red, DEV=blue, HK=green。

数据:
  - shared_motif_hit_contribution_tss.csv (hit_importance_norm + mc_id + tf + task)

输出: <REPO>/plot/output/png/figure3d/per_motif_score_{hk,dev}_vs_cage.png
"""

import re
from math import erfc, sqrt
import matplotlib
matplotlib.use('Agg')
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

from figure3_paths import CLEAN_DIR, panel_dirs
from figure3_style import DOUBLE_COL_MM, TASK_COLORS, mm_to_in, save_all, set_pub_style

HIT_CSV = CLEAN_DIR / 'shared_motif_hit_contribution_tss.csv'
PAL = {'CAGE': TASK_COLORS['CAGE'], 'HK': TASK_COLORS['HK'], 'DEV': TASK_COLORS['DEV']}
DISPLAY = {'CAGE_NEW': 'CAGE', 'HK': 'HK', 'DEV': 'DEV'}
width_mm = 183
EXPORT_TARGETS = ("figure.svg", "figure.pdf", "figure.tiff", "figure.png")
EXPORT_DPI = 600


def _sort_mc(mc_label):
    """按 MC 编号排序：'MC_000\\nDRE' -> 0。"""
    m = re.search(r'MC_(\d+)', mc_label)
    return int(m.group(1)) if m else 999


def mannwhitneyu_pvalue(x, y):
    """Two-sided Mann-Whitney U p-value with normal approximation and tie correction."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan

    values = np.concatenate([x, y])
    order = np.argsort(values, kind='mergesort')
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    tie_sum = 0.0
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        tie_count = end - start
        tie_sum += tie_count ** 3 - tie_count
        start = end

    rank_sum_x = ranks[:n1].sum()
    u1 = rank_sum_x - n1 * (n1 + 1) / 2.0
    mean_u = n1 * n2 / 2.0
    n = n1 + n2
    tie_correction = 1.0 - tie_sum / (n ** 3 - n) if n > 1 else 1.0
    sd_u = sqrt(n1 * n2 * (n + 1) * tie_correction / 12.0)
    if sd_u == 0:
        return 1.0
    z = (abs(u1 - mean_u) - 0.5) / sd_u
    return erfc(abs(z) / sqrt(2.0))


def format_label(mc_id, tf):
    """mc_id + tf 组合标签（对齐 cell 12 format_motif_label 风格）。"""
    name = str(tf) if pd.notna(tf) else ''
    if name.count('/') > 1:
        name = name.split('/')[0]
    return f"{mc_id}\n{name}"


def add_sig_bracket(ax, x1, x2, y, h, text, color):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            color=color, lw=1.15, clip_on=False, zorder=7)
    ax.text((x1 + x2) / 2, y + h * 1.35, text,
            ha='center', va='bottom', fontsize=9.2, color=color,
            fontweight='bold', clip_on=False, zorder=8)


def draw_manual_violin_box(ax, sub, motifs, pair, pal):
    offsets = {pair[0]: -0.18, pair[1]: 0.18}
    positions_by_task = {task: [] for task in pair}
    values_by_task = {task: [] for task in pair}

    for i, lab in enumerate(motifs):
        for task in pair:
            values = sub[(sub['label'] == lab) & (sub['task'] == task)]['hit_importance_norm'].dropna().to_numpy()
            if len(values) == 0:
                continue
            positions_by_task[task].append(i + offsets[task])
            values_by_task[task].append(values)

    for task in pair:
        if not values_by_task[task]:
            continue
        parts = ax.violinplot(values_by_task[task], positions=positions_by_task[task],
                              widths=0.32, showmeans=False, showmedians=False,
                              showextrema=False)
        for body in parts['bodies']:
            body.set_facecolor(pal[task])
            body.set_edgecolor('#555555')
            body.set_linewidth(1.0)
            body.set_alpha(0.72)
            body.set_zorder(2)

        box = ax.boxplot(values_by_task[task], positions=positions_by_task[task],
                         widths=0.105, patch_artist=True, showfliers=False,
                         whis=(5, 95), manage_ticks=False)
        for patch in box['boxes']:
            patch.set_facecolor('#2F2F2F')
            patch.set_edgecolor('#111111')
            patch.set_alpha(0.82)
            patch.set_linewidth(0.85)
            patch.set_zorder(4)
        for median in box['medians']:
            median.set_color('white')
            median.set_linewidth(1.75)
            median.set_zorder(5)
        for element in ['whiskers', 'caps']:
            for artist in box[element]:
                artist.set_color('#111111')
                artist.set_linewidth(0.85)
                artist.set_zorder(4)

    return offsets


def plot_per_motif(df, pair, out_path):
    """per-motif 贡献分 violin，HK/DEV vs CAGE，去 dot + 深黑 box + Mann-Whitney highlight。

    pair = ['HK','CAGE'] 或 ['DEV','CAGE']

    ★ 聚合对齐 notebook cell 9：groupby(['peak_id','mc_id'])['hit_importance_norm'].mean()
       即每个 peak 每个 motif 一条平均分（而非每条 hit 一条）。
    """
    t1, t2 = pair
    sub = df[df['task'].isin(pair)].copy()
    if sub.empty:
        print(f"  {t1}_vs_{t2}: 无数据"); return

    # ★ 聚合：(peak_id, mc_id, task) → mean hit_importance_norm（对齐 notebook cell 9）
    agg = (sub.groupby(['peak_id', 'mc_id', 'task'], observed=True)['hit_importance_norm']
              .mean()
              .reset_index())
    # tf 标签：每个 mc_id 取一个 tf（同 mc_id 内 tf 应一致）
    tf_map = sub.drop_duplicates('mc_id').set_index('mc_id')['tf']
    agg['tf'] = agg['mc_id'].map(tf_map)
    # label 格式 + 排序
    agg['label'] = agg.apply(lambda r: format_label(r['mc_id'], r['tf']), axis=1)
    motifs = sorted(agg['label'].dropna().unique().tolist(), key=_sort_mc)
    agg['label'] = pd.Categorical(agg['label'], categories=motifs, ordered=True)
    pal = {t: PAL[t] for t in pair}
    sub = agg  # 以下是聚合后数据
    print(f"  {t1}_vs_{t2}: 聚合后 {len(sub)} 条 (peak×motif)，绘制 {len(motifs)} 个 motif")

    set_pub_style(font_size=12.7)
    fig, ax = plt.subplots(figsize=(mm_to_in(DOUBLE_COL_MM * 0.95), mm_to_in(86)))

    # 手动定位 violin 和 box，保证 box 与对应 violin 严格同轴。
    offsets = draw_manual_violin_box(ax, sub, motifs, pair, pal)
    ax.legend(handles=[Patch(facecolor=PAL[t], label=t) for t in pair],
              frameon=False, fontsize=11.0, loc="upper right", bbox_to_anchor=(1.0, 1.18), ncol=2,
              handlelength=1.0, columnspacing=0.9)

    ymax = sub['hit_importance_norm'].max()
    ymin = sub['hit_importance_norm'].min()
    yrange = max(1.0, ymax - ymin)
    sig_h = yrange * 0.035
    marker_top = ymax
    bias_color = {t1: PAL[t1], t2: PAL[t2]}  # 倾向哪边就用哪边色
    for i, lab in enumerate(motifs):
        g1 = sub[(sub['label'] == lab) & (sub['task'] == t1)]['hit_importance_norm'].dropna()
        g2 = sub[(sub['label'] == lab) & (sub['task'] == t2)]['hit_importance_norm'].dropna()
        if len(g1) < 3 or len(g2) < 3:
            continue
        p = mannwhitneyu_pvalue(g1, g2)
        if p >= 0.05:
            continue  # 不显著，不标
        m1, m2 = g1.median(), g2.median()
        bias = t1 if m1 > m2 else t2  # 倾向方
        local_top = max(g1.max(), g2.max())
        sig_y = local_top + yrange * 0.018
        add_sig_bracket(ax, i + offsets[t1], i + offsets[t2], sig_y, sig_h,
                        _star(p), PAL[bias])
        marker_top = max(marker_top, sig_y + sig_h * 2.5)

    ax.set_xlim(-0.6, len(motifs) - 0.4)
    ax.set_xticks(np.arange(len(motifs)))
    ax.set_xticklabels(motifs)
    ax.set_ylim(None, marker_top + yrange * 0.06)
    ax.set_title('')
    ax.set_xlabel(''); ax.set_ylabel('Median-normalized importance', fontsize=12.0)
    ax.tick_params(axis='x', labelsize=9.5, rotation=0)
    ax.tick_params(axis='y', labelsize=11.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.84)
    paths = save_all(fig, out_path, out_path["stem"])
    plt.close()
    print(f"saved: {paths['png']}")


def _star(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


def main():
    dirs = panel_dirs("figure3d")
    df = pd.read_csv(HIT_CSV)
    df['task'] = df['task'].map(DISPLAY).fillna(df['task'])
    # clean CSV 已有 hit_importance_norm，过滤空值
    df = df[df['hit_importance_norm'].notna()].copy()
    print(f"加载 {len(df)} 条记录, task: {df['task'].value_counts().to_dict()}")

    hk_dirs = {**dirs, "stem": "per_motif_score_hk_vs_cage"}
    dev_dirs = {**dirs, "stem": "per_motif_score_dev_vs_cage"}
    plot_per_motif(df, ['HK', 'CAGE'], hk_dirs)
    plot_per_motif(df, ['DEV', 'CAGE'], dev_dirs)


if __name__ == '__main__':
    main()
