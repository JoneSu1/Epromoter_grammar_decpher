#!/usr/bin/env python
# coding: utf-8
"""Fig 3C — motif 密度 + 复杂度 violin（去 dot，改内置深黑 box，median 加粗）。

来源: Motif_Density_Visualization.ipynb cell 8/9 (fig3 C)

微调（用户标注要求）:
  1. 去掉 stripplot（dot 渲染）
  2. 内置 box 改深黑色、明显（原 facecolor:none 太细）
  3. median 加粗（medianprops linewidth 粗）
保留: violin + mannwhitneyu 显著性星标。

数据:
  - shared_motif_density_complexity.csv
    task=CAGE_NEW/HK/DEV（显示为 CAGE/HK/DEV）

输出: <REPO>/plot/output/png/figure3c/{density,complexity}_{hk,dev}_vs_cage.png
"""

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
import matplotlib.pyplot as plt
from math import erf, sqrt
import numpy as np
import pandas as pd
import seaborn as sns

from figure3_paths import CLEAN_DIR, panel_dirs
from figure3_style import SINGLE_COL_MM, TASK_COLORS, mm_to_in, save_all, set_pub_style

DENSITY_CSV = CLEAN_DIR / 'shared_motif_density_complexity.csv'
PAL = {'CAGE': TASK_COLORS['CAGE'], 'HK': TASK_COLORS['HK'], 'DEV': TASK_COLORS['DEV']}
DISPLAY = {'CAGE_NEW': 'CAGE', 'HK': 'HK', 'DEV': 'DEV'}
width_mm = 183
EXPORT_TARGETS = ("figure.svg", "figure.pdf", "figure.tiff", "figure.png")
EXPORT_DPI = 600


def _star(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


def mannwhitneyu_two_sided(x, y):
    """Return Mann-Whitney U and an asymptotic two-sided P value.

    This local implementation keeps the formal script independent of scipy.
    It uses average ranks for ties and the standard tie-corrected normal
    approximation, which is appropriate for the large peak-level sample sizes
    used in this panel.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan

    values = np.concatenate([x, y])
    groups = np.concatenate([np.zeros(n1, dtype=bool), np.ones(n2, dtype=bool)])
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks_sorted = np.empty_like(sorted_values, dtype=float)
    tie_counts = []
    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks_sorted[start:end] = (start + 1 + end) / 2.0
        tie_counts.append(end - start)
        start = end
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    r1 = ranks[~groups].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    n = n1 + n2
    tie_term = sum(t**3 - t for t in tie_counts)
    variance = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
    if variance <= 0:
        return u, 1.0
    mean = n1 * n2 / 2.0
    z = (u - mean) / sqrt(variance)
    p = 2.0 * (0.5 * (1.0 - erf(abs(z) / sqrt(2.0))))
    return u, p


def draw_violin_compare(df, val, order, pal, title, ylabel, out_path):
    """微调版: 半透明任务色 violin + 深色实心窄 box（白色 median），去掉 stripplot。

    嵌入式 box 风格（参考用户参考图）:
    - violin 用任务色但 alpha=0.55 半透明，让深色 box 透出来
    - box 用深色实心填充（不是透明描边）+ 白色粗 median（反白醒目）
    - box 宽度约 violin 的 1/4（width=0.18），真正"嵌"在中央
    """
    set_pub_style(font_size=13.2)
    fig, ax = plt.subplots(figsize=(mm_to_in(SINGLE_COL_MM * 0.78), mm_to_in(72)))
    # violin：任务色，按 x='task' 直接分组（不用 hue，避免 seaborn 自动 dodge 让 box 错位）
    sns.violinplot(data=df, x='task', y=val, order=order, palette=pal,
                   cut=0, density_norm='width', inner=None, linewidth=1.05, ax=ax)
    for polycoll in ax.collections:
        polycoll.set_alpha(0.78)
        polycoll.set_edgecolor('#555555')
    # ★ 深色实心 box + 白色粗 median（无 hue 也无 dodge，居中嵌在 violin 中央）
    sns.boxplot(data=df, x='task', y=val, order=order, width=0.18,
                boxprops={'facecolor': '#333333', 'edgecolor': '#333333', 'linewidth': 0.75, 'alpha': 0.32},
                whiskerprops={'color': '#333333', 'linewidth': 0.9, 'alpha': 0.75},
                capprops={'color': '#333333', 'linewidth': 0.9, 'alpha': 0.75},
                medianprops={'color': '#2F2F2F', 'linewidth': 0},
                fliersize=0, ax=ax)
    for xpos, task in enumerate(order):
        mean_value = df.loc[df['task'] == task, val].mean()
        if pd.notna(mean_value):
            ax.hlines(mean_value, xpos - 0.22, xpos + 0.22, color='black', linewidth=2.8, zorder=7)
    # 显著性星标
    if len(order) == 2:
        g1 = df[df['task'] == order[0]][val].dropna()
        g2 = df[df['task'] == order[1]][val].dropna()
        if len(g1) > 1 and len(g2) > 1:
            _, p = mannwhitneyu_two_sided(g1, g2)
            ymax = df[val].max()
            ax.plot([0, 0, 1, 1], [ymax * 1.05, ymax * 1.1, ymax * 1.1, ymax * 1.05],
                    'k-', lw=0.8)
            ax.text(0.5, ymax * 1.12, _star(p), ha='center', fontsize=12.0)
    ax.set_title('')
    ax.set_xlabel(''); ax.set_ylabel(ylabel, fontsize=12.5)
    ax.tick_params(axis='both', labelsize=12.0)
    sns.despine(offset=5, trim=True)
    save_all(fig, out_path, title)
    plt.close()
    print(f"saved: {out_path['png'] / (title + '.png')}")


def main():
    dirs = panel_dirs("figure3c")
    df = pd.read_csv(DENSITY_CSV)
    df['task'] = df['task_label'].fillna(df['task'].map(DISPLAY)).fillna(df['task'])

    s2_dens = df.rename(columns={'motif_density': 'mc'})[['peak_id', 'task', 'mc']].copy()
    s2_comp = df.rename(columns={'motif_complexity': 'uc'})[['peak_id', 'task', 'uc']].copy()

    for pair_name, pair in [('hk_vs_cage', ['HK', 'CAGE']), ('dev_vs_cage', ['DEV', 'CAGE'])]:
        pal = {t: PAL[t] for t in pair}
        t1 = pair[0]
        draw_violin_compare(s2_dens[s2_dens['task'].isin(pair)], 'mc', pair, pal,
                            f'density_{pair_name}', 'Hits/peak',
                            dirs)
        draw_violin_compare(s2_comp[s2_comp['task'].isin(pair)], 'uc', pair, pal,
                            f'complexity_{pair_name}', 'Unique motifs/peak',
                            dirs)


if __name__ == '__main__':
    main()
