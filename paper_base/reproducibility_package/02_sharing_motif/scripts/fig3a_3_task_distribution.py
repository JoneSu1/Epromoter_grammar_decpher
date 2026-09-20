#!/usr/bin/env python
# coding: utf-8
"""Fig 3A-3 — 兵力分布（Task 占比堆叠柱状图，带 TF 标注）。

来源: MC000_Motif_Trimming.ipynb cell 46 (fig3A 第三部分)

数据:
  - shared_motif_meta_enrichment.csv
    列: MC_ID, HK/CAGE_NEW/DEV_Frac_In_Family, Match_1..4, Family_Type

输出: <REPO>/plot/output/png/figure3a_3/task_distribution.png
"""

from collections import defaultdict
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

from figure3_paths import CLEAN_DIR, panel_dirs
from figure3_style import DOUBLE_COL_MM, TASK_COLORS, mm_to_in, save_all, set_pub_style

ENRICH_CSV = CLEAN_DIR / 'shared_motif_meta_enrichment.csv'
width_mm = 183
EXPORT_TARGETS = ("figure.svg", "figure.pdf", "figure.tiff", "figure.png")
EXPORT_DPI = 600

TF_DISPLAY = {
    "CREB/ATF/3": "CREB/ATF",
    "Ebox/CAGCTG/CACCTG": "Ebox/CAGCTG",
}


def assign_unique_tf_labels(df):
    """对齐 cell 46: TF_LABEL 去重逻辑（Match_1..4 优先取未用过的）。"""
    used = set()
    counter = defaultdict(int)
    labels = []
    for _, row in df.iterrows():
        candidates = [str(row[k]) for k in ["Match_1", "Match_2", "Match_3", "Match_4"]
                      if pd.notna(row.get(k))]
        chosen = None
        for c in candidates:
            if c not in used:
                chosen = c; used.add(c); break
        if chosen is None:
            base = candidates[0] if candidates else "Unknown"
            counter[base] += 1
            chosen = f"{base}_{counter[base]:02d}"
        labels.append(chosen)
    return labels


def task_columns(df):
    return sorted({c.replace('_Frac_In_Family', '')
                   for c in df.columns if '_Frac_In_Family' in c})


def display_task(task):
    return "CAGE" if task == "CAGE_NEW" else task


def display_tf_label(label):
    return TF_DISPLAY.get(str(label), str(label))


def draw_within_mc_distribution(df, dirs):
    fig, ax = plt.subplots(figsize=(mm_to_in(DOUBLE_COL_MM * 0.64), mm_to_in(92)), dpi=300)
    y_pos = np.arange(len(df))
    left = np.zeros(len(df))

    tasks = task_columns(df)
    for task in tasks:
        col = f"{task}_Frac_In_Family"
        if col in df.columns:
            y = df[col].values
            label = display_task(task)
            ax.barh(y_pos, y, left=left, color=TASK_COLORS.get(task, '#333'),
                    label=label, height=0.72, edgecolor='white', linewidth=0.55)
            left += y

    y_labels = [f"{row.MC_ID}  {display_tf_label(row.TF_LABEL)}" for row in df.itertuples()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=10.4)
    ax.invert_yaxis()

    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Fraction of seqlet weight', fontsize=12.5)
    ax.set_ylabel('')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='both', length=3, labelsize=11.2)
    ax.xaxis.grid(True, color='#D9D9D9', linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=11.0, loc='upper center',
              bbox_to_anchor=(0.5, 1.08), ncol=3, handlelength=1.0,
              columnspacing=1.0)
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.12, top=0.90)
    paths = save_all(fig, dirs, 'task_distribution')
    plt.close()
    print(f"saved: {paths['png']}  ({len(df)} MCs)")


def compute_task_normalized(df):
    rows = []
    for _, row in df.iterrows():
        for task in task_columns(df):
            weight = float(row['Total_Weight']) * float(row[f'{task}_Frac_In_Family'])
            rows.append({
                'MC_ID': row['MC_ID'],
                'TF_LABEL': row['TF_LABEL'],
                'task': display_task(task),
                'task_raw': task,
                'weight': weight,
            })
    long_df = pd.DataFrame(rows)
    totals = long_df.groupby('task', observed=True)['weight'].transform('sum')
    long_df['fraction_in_task'] = long_df['weight'] / totals
    return long_df


def draw_task_normalized_distribution(df, dirs):
    long_df = compute_task_normalized(df)
    task_order = ['CAGE', 'DEV', 'HK']
    y_pos = np.arange(len(df))
    offsets = {'CAGE': -0.24, 'DEV': 0.0, 'HK': 0.24}
    height = 0.20

    fig, ax = plt.subplots(figsize=(mm_to_in(DOUBLE_COL_MM * 0.70), mm_to_in(98)), dpi=300)
    for task in task_order:
        values = (long_df[long_df['task'] == task]
                  .set_index('MC_ID')
                  .reindex(df['MC_ID'])['fraction_in_task']
                  .fillna(0)
                  .to_numpy())
        color_key = 'CAGE_NEW' if task == 'CAGE' else task
        ax.barh(y_pos + offsets[task], values, height=height,
                color=TASK_COLORS.get(color_key, '#333333'), label=task,
                edgecolor='white', linewidth=0.45)

    y_labels = [f"{row.MC_ID}  {display_tf_label(row.TF_LABEL)}" for row in df.itertuples()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=10.4)
    ax.invert_yaxis()
    ax.set_xlabel('Fraction within task', fontsize=12.5)
    ax.set_ylabel('')
    ax.set_xlim(0, max(0.01, float(long_df['fraction_in_task'].max()) * 1.12))
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='both', length=3, labelsize=11.2)
    ax.xaxis.grid(True, color='#D9D9D9', linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=11.0, loc='upper center',
              bbox_to_anchor=(0.5, 1.08), ncol=3, handlelength=1.0,
              columnspacing=1.0)
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.12, top=0.90)
    paths = save_all(fig, dirs, 'task_normalized_distribution')
    plt.close()
    long_df.to_csv(dirs["qa"] / "task_normalized_distribution_source.csv", index=False)
    print(f"saved: {paths['png']}  ({len(df)} MCs)")


def main():
    set_pub_style(font_size=13.2)
    df = pd.read_csv(ENRICH_CSV)
    df['MC_num'] = df['MC_ID'].str.replace('MC_', '', regex=False).astype(int)
    df = df.sort_values('MC_num').reset_index(drop=True)
    df['TF_LABEL'] = assign_unique_tf_labels(df)

    dirs = panel_dirs("figure3a_3")
    draw_within_mc_distribution(df, dirs)
    draw_task_normalized_distribution(df, dirs)
    df[['MC_ID', 'TF_LABEL', 'Family_Type']].to_csv(dirs["qa"] / "task_distribution_labels.csv", index=False)


if __name__ == '__main__':
    main()
