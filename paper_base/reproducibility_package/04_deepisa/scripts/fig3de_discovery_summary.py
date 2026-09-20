from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
import matplotlib.pyplot as plt

from deepisa_paths import DATA_ROOT, panel_dirs
from deepisa_style import DOUBLE_COL_MM, SIGN_COLORS, TASK_COLORS, mm_to_in, save_all, set_pub_style


CLASS_TABLE = DATA_ROOT / "fig3" / "fig3_tf_pair_class_table.csv"
COUNT_TABLE = DATA_ROOT / "fig3" / "fig3c_direction_supported_class_counts.csv"
TASKS = ["CAGE", "DEV", "HK"]
width_mm = 183
EXPORT_TARGETS = ("figure.svg", "figure.pdf", "figure.tiff", "figure.png")
EXPORT_DPI = 600


def plot_ecdf(df: pd.DataFrame) -> None:
    dirs = panel_dirs("fig3d_direction_supported_ecdf")
    set_pub_style(font_size=12.8)
    fig, ax = plt.subplots(figsize=(mm_to_in(92), mm_to_in(78)), dpi=300)
    for task in TASKS:
        values = (
            df[(df["task"] == task) & (df["direction_supported"])]
            ["C_effective"]
            .dropna()
            .sort_values()
            .to_numpy()
        )
        if len(values) == 0:
            continue
        y = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, y, where="post", color=TASK_COLORS[task], lw=2.3, label=f"{task} (n={len(values)})")

    ax.axvline(0, color="#888888", lw=1.0, linestyle=(0, (2, 2)))
    ax.set_xlabel("TF-pair net non-additivity")
    ax.set_ylabel("ECDF")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=10.5, handlelength=1.4)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.98)
    paths = save_all(fig, dirs, "fig3d_direction_supported_ecdf")
    plt.close(fig)
    print(paths["png"])


def plot_sign_counts(counts: pd.DataFrame) -> None:
    dirs = panel_dirs("fig3e_sign_counts")
    set_pub_style(font_size=12.8)
    pivot = (
        counts.pivot(index="task", columns="direction_class", values="n")
        .reindex(TASKS)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(mm_to_in(88), mm_to_in(78)), dpi=300)
    bottom = np.zeros(len(pivot))
    for sign in ["Negative", "Positive"]:
        values = pivot.get(sign, pd.Series(0, index=pivot.index)).to_numpy()
        bars = ax.bar(
            pivot.index,
            values,
            bottom=bottom,
            color=SIGN_COLORS[sign],
            edgecolor="white",
            linewidth=0.7,
            width=0.64,
            label=sign,
        )
        for bar, value, base in zip(bars, values, bottom):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, base + value / 2, f"{int(value)}",
                        ha="center", va="center", fontsize=10.5,
                        color="white" if value >= 8 else "#222222")
        bottom += values

    for tick, task in zip(ax.get_xticklabels(), TASKS):
        tick.set_color(TASK_COLORS[task])
    ax.set_ylabel("Direction-supported TF-pair classes", fontsize=11.4)
    ax.set_xlabel("")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.05), ncol=2, fontsize=10.5, handlelength=1.0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.15, top=0.91)
    paths = save_all(fig, dirs, "fig3e_sign_counts")
    plt.close(fig)
    print(paths["png"])


def main() -> None:
    class_df = pd.read_csv(CLASS_TABLE)
    counts = pd.read_csv(COUNT_TABLE)
    plot_ecdf(class_df)
    plot_sign_counts(counts)
    class_df.to_csv(panel_dirs("fig3d_direction_supported_ecdf")["qa"] / "source_class_table.csv", index=False)
    counts.to_csv(panel_dirs("fig3e_sign_counts")["qa"] / "source_counts.csv", index=False)


if __name__ == "__main__":
    main()
