#!/usr/bin/env python
# coding: utf-8
"""Fig 3B — motif-TSS 距离分析（4 个子图，scatter 跳过）。

来源: Ep_ISA_Analysis.ipynb cell 23 标注 # fig3 B
  - cell 24: KDE (top 10 motif)
  - cell 25: EP vs Specific boxplot (3 panel)
  - cell 26: TF median 距离 heatmap
  - cell 28: per-motif boxplot (MC_000..MC_009)
  - cell 27: TSS vs cooperativity scatter —— 【跳过】依赖 coop CSV（ISA 产出，未落盘）

数据:
  - shared_motif_hit_contribution_tss.csv
    列: task, peak_id, tf, tss_distance_bp, hit_importance
    task 取值: CAGE_NEW / HK / DEV

输出: <REPO>/plot/output/png/figure3b/{kde, ep_boxplot, tf_heatmap, mc_boxplot}.png

注意:
  - clean CSV 的 tss_distance_bp 是 proxy 假数据（motif_center - 固定窗口中心）；
    真实 motif→TSS 距离 = motif_center - real_tss_bp（等价 notebook cell 20 tss_dist_seq）
  - real_tss_bp 来自 bioframe dm3 refGene overlap（22% 为 NaN = 窗口内无 TSS，notebook 同样丢弃）
  - clean CSV task=CAGE_NEW（notebook 显示为 CAGE），标题统一用 CAGE
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
import numpy as np
import pandas as pd
import seaborn as sns

from figure3_paths import CLEAN_DIR, RAW_DIR, panel_dirs
from figure3_style import DOUBLE_COL_MM, TASK_COLORS, GRID, mm_to_in, save_all, set_pub_style

HIT_CSV = CLEAN_DIR / 'shared_motif_hit_contribution_tss.csv'

# top10 motif 固定顺序（对应 MC_000..MC_009）
FIXED_ORDER = ['DRE/3', 'Ohler1', 'ATA', 'MAF/2', 'Ohler7',
               'SREBP/2', 'kni/1', 'CREB/ATF/3', 'Ebox/CAGCTG/CACCTG', 'HD/16']
TF_DISPLAY = {'CREB/ATF/3': 'CREB/ATF', 'Ebox/CAGCTG/CACCTG': 'Ebox'}

# 显示名: CAGE_NEW -> CAGE
DISPLAY = {'CAGE_NEW': 'CAGE', 'HK': 'HK', 'DEV': 'DEV'}
COLORS = {'CAGE': TASK_COLORS['CAGE'], 'HK': TASK_COLORS['HK'], 'DEV': TASK_COLORS['DEV']}
width_mm = 183
EXPORT_TARGETS = ("figure.svg", "figure.pdf", "figure.tiff", "figure.png")
EXPORT_DPI = 600


def load():
    df = pd.read_csv(HIT_CSV)
    # ★ 真实 motif→TSS 距离 = motif_center - real_tss_bp（不用 proxy tss_distance_bp）
    df['tss_dist'] = df['motif_center'] - df['real_tss_bp']
    df = df[df['tss_dist'].notna()].copy()  # 丢 NaN（窗口无 TSS，对齐 notebook）
    df['task_disp'] = df['task'].map(DISPLAY).fillna(df['task'])
    return df


def plot_kde(df, dirs):
    """cell 24: top10 motif TSS 距离 KDE (2×5), with one shared legend."""
    set_pub_style(font_size=10.8)
    fig, axes = plt.subplots(
        2,
        5,
        figsize=(mm_to_in(DOUBLE_COL_MM), mm_to_in(82)),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()
    for i, tf in enumerate(FIXED_ORDER):
        ax = axes[i]
        for name in ['CAGE', 'HK', 'DEV']:
            v = df[(df['task_disp'] == name) & (df['tf'] == tf)]
            if len(v) > 1:
                sns.kdeplot(
                    data=v,
                    x='tss_dist',
                    color=COLORS[name],
                    ax=ax,
                    alpha=0.62,
                    lw=1.45,
                    label=name,
                )
        ax.axvline(0, color='#333333', ls='--', lw=0.75)
        ax.set_xlim(-150, 150)
        ax.set_title(TF_DISPLAY.get(tf, tf), fontsize=11.0, pad=5)
        ax.set_xlabel('')
        ax.set_ylabel('Density' if i % 5 == 0 else '', fontsize=10.2)
        ax.tick_params(labelsize=9.2)
        ax.xaxis.grid(True, color=GRID, linewidth=0.35, alpha=0.55)
        ax.set_axisbelow(True)

    handles = [plt.Line2D([0], [0], color=COLORS[t], lw=2.0, label=t) for t in ['CAGE', 'DEV', 'HK']]
    fig.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.53, 0.995),
        frameon=False,
        fontsize=10.8,
        ncol=3,
        handlelength=1.6,
        columnspacing=1.1,
    )
    fig.text(0.53, 0.045, 'Distance to TSS (bp)', ha='center', va='center', fontsize=10.2)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.17, top=0.84, wspace=0.30, hspace=0.55)
    paths = save_all(fig, dirs, 'kde')
    plt.close()
    print(f"saved: {paths['png']}")


def plot_tf_heatmap(df, dirs):
    """cell 26: per-TF 中位 TSS 距离 heatmap（跨 task）。"""
    tf_tss = {}
    for name in ['CAGE', 'HK', 'DEV']:
        v = df[df['task_disp'] == name]
        if not v.empty:
            tf_tss[name] = v.groupby('tf')['tss_dist'].median()
    tf_df = pd.DataFrame(tf_tss)
    shared = tf_df.dropna(thresh=2).sort_values('CAGE')
    if shared.empty:
        print("  tf_heatmap: 无共享 TF，跳过"); return
    set_pub_style(font_size=11.5)
    fig, ax = plt.subplots(figsize=(mm_to_in(72), mm_to_in(max(86, len(shared) * 7.2))))
    sns.heatmap(shared, cmap='coolwarm', center=0, ax=ax,
                cbar_kws={'label': 'Median distance to TSS (bp)'}, linewidths=0.3)
    ax.set_title('')
    ax.tick_params(axis='both', labelsize=10.5)
    fig.subplots_adjust(left=0.40, right=0.94, bottom=0.08, top=0.98)
    paths = save_all(fig, dirs, 'tf_heatmap')
    plt.close()
    print(f"saved: {paths['png']}")


def plot_mc_boxplot(df, dirs):
    """cell 28: per-motif TSS 距离 boxplot (3 panel CAGE/DEV/HK)。"""
    set_pub_style(font_size=11.8)
    fig, axes = plt.subplots(1, 3, figsize=(mm_to_in(DOUBLE_COL_MM), mm_to_in(82)), sharex=True)
    for idx, (ax, name) in enumerate(zip(axes, ['CAGE', 'DEV', 'HK'])):
        v = df[df['task_disp'] == name]
        v10 = v[v['tf'].isin(FIXED_ORDER)]
        if v10.empty:
            ax.set_title(f'{name}: no data'); continue
        sns.boxplot(data=v10, y='tf', x='tss_dist', order=FIXED_ORDER, ax=ax,
                    color=COLORS[name], fliersize=0, linewidth=0.8)
        ax.axvline(0, color='#333333', ls='--', lw=0.8)
        ax.set_xlim(-200, 200)
        ax.set_xlabel('Distance to TSS (bp)', fontsize=11.0)
        ax.set_ylabel(''); ax.set_title(name, fontsize=12.5)
        if idx == 0:
            ax.set_yticklabels([TF_DISPLAY.get(t, t) for t in FIXED_ORDER])
            ax.tick_params(axis='y', labelsize=9.2)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)
        ax.xaxis.grid(True, color=GRID, linewidth=0.45, alpha=0.7)
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.16, top=0.86, wspace=0.18)
    paths = save_all(fig, dirs, 'mc_boxplot')
    plt.close()
    print(f"saved: {paths['png']}")


def plot_tss_vs_coop(df, dirs):
    """cell 27: TSS 距离 vs cooperativity 散点（coop CSV 已从 Drive 复制到本地）。"""
    COOP_DIR = RAW_DIR / 'ep_isa'
    set_pub_style(font_size=11.8)
    fig, ax = plt.subplots(figsize=(mm_to_in(84), mm_to_in(72)))
    plotted = False
    def mean_pair_tss(pair_text, lookup):
        values = [lookup.get(t, np.nan) for t in str(pair_text).split('|')]
        values = [v for v in values if pd.notna(v)]
        return float(np.mean(values)) if values else np.nan

    for name, coop_file, color in [('CAGE', 'coop_cage.csv', COLORS['CAGE']),
                                    ('DEV', 'coop_dev.csv', COLORS['DEV']),
                                    ('HK', 'coop_hk.csv', COLORS['HK'])]:
        coop_p = COOP_DIR / coop_file
        if not coop_p.exists():
            continue
        cp = pd.read_csv(coop_p)
        # 显著性筛选：mw_p < 0.05（等价 notebook 的 cooperativity != 'Independent'）
        if 'mw_p' in cp.columns:
            sig = cp[cp['mw_p'] < 0.05].copy()
        else:
            sig = cp.copy()
        if sig.empty:
            continue
        # 每 TF 的中位 TSS 距离（用真实 tss_dist）
        tss_lookup = (df[df['task_disp'] == name]
                      .groupby('tf')['tss_dist'].median().to_dict())
        sig['tss_dist'] = sig['tf_pair'].apply(lambda p: mean_pair_tss(p, tss_lookup))
        valid = sig[sig['tss_dist'].notna()]
        if len(valid) > 0:
            ax.scatter(valid['tss_dist'], valid['coop_score'], color=color,
                       label=name, alpha=0.5, s=15)
            plotted = True
    ax.axvline(0, color='grey', ls='--', lw=0.5)
    ax.set_xlabel('Avg Motif–TSS Distance (bp)')
    ax.set_ylabel('Coop Score')
    ax.set_title('TSS Distance vs Cooperativity')
    if plotted:
        ax.legend(frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.16, top=0.95)
    paths = save_all(fig, dirs, 'tss_vs_coop')
    plt.close()
    print(f"saved: {paths['png']}")


def main():
    dirs = panel_dirs("figure3b")
    df = load()
    print(f"加载 {len(df)} 条 hit 记录（真实 TSS 距离）, task 分布: {df['task_disp'].value_counts().to_dict()}")
    plot_kde(df, dirs)
    plot_tf_heatmap(df, dirs)
    plot_mc_boxplot(df, dirs)
    plot_tss_vs_coop(df, dirs)
    print("\n⚠️ ep_boxplot 需 ep_seqs（EP motif peak 集合），clean CSV 无此信息，跳过")


if __name__ == '__main__':
    main()
