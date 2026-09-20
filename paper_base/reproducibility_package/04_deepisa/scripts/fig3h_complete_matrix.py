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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from deepisa_paths import DATA_ROOT, panel_dirs
from deepisa_style import DOUBLE_COL_MM, TASK_COLORS, mm_to_in, save_all, set_pub_style


SOURCE = DATA_ROOT / "fig3h" / "fig3h_complete_tf_pair_nonadditivity_matrix_source.csv"
TASKS = ["CAGE", "DEV", "HK"]
width_mm = 183
EXPORT_TARGETS = ("figure.svg", "figure.pdf", "figure.tiff", "figure.png")
EXPORT_DPI = 600


def display_tf(name: str) -> str:
    text = str(name)
    mapping = {
        "CREB/ATF/3": "CREB",
        "EBOX/CAGCTG/CACCTG": "E-box",
        "KNI/1": "KNI",
        "MAF/2": "MAF",
        "HD/16": "HD",
        "DRE/3": "DRE",
        "OHLER1": "Ohler1",
        "OHLER7": "Ohler7",
        "SREBP/2": "SREBP",
    }
    return mapping.get(text, text)


def task_matrix(df: pd.DataFrame, task: str, tf_order: list[str]) -> pd.DataFrame:
    sub = df[(df["task"] == task) & (df["direction_supported"])].copy()
    mat = sub.pivot_table(index="tf1", columns="tf2", values="C_effective", aggfunc="first")
    mat = mat.reindex(index=tf_order, columns=tf_order)
    return mat


def main() -> None:
    set_pub_style(font_size=12.0)
    dirs = panel_dirs("fig3h_complete_tf_pair_matrix")
    df = pd.read_csv(SOURCE)
    tf_order = ["ATA", "CREB/ATF/3", "DRE/3", "EBOX/CAGCTG/CACCTG", "HD/16",
                "KNI/1", "MAF/2", "OHLER1", "OHLER7", "SREBP/2"]
    labels = [display_tf(x) for x in tf_order]

    cmap = LinearSegmentedColormap.from_list("deepisa_effect", ["#276899", "#F6F6F6", "#B33D33"])
    cmap.set_bad("#F4F4F4")
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    fig, axes = plt.subplots(1, 4, figsize=(mm_to_in(DOUBLE_COL_MM), mm_to_in(58)),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.038], "wspace": 0.095},
                             dpi=300)
    images = []
    for ax, task in zip(axes[:3], TASKS):
        mat = task_matrix(df, task, tf_order)
        image = ax.imshow(mat.to_numpy(dtype=float), cmap=cmap, norm=norm, aspect="equal")
        images.append(image)
        for i in range(len(tf_order) + 1):
            ax.axhline(i - 0.5, color="#F8F8F8", lw=0.55)
            ax.axvline(i - 0.5, color="#F8F8F8", lw=0.55)
        ax.add_patch(plt.Rectangle((-0.5, -0.5), len(tf_order), len(tf_order),
                                   fill=False, edgecolor="#D8D8D8", linewidth=0.7,
                                   clip_on=False))
        ax.set_title(task, color=TASK_COLORS[task], fontsize=12.8, pad=7)
        ax.set_xticks(np.arange(len(tf_order)))
        ax.set_xticklabels(labels, rotation=52, ha="right", fontsize=8.4)
        ax.set_yticks(np.arange(len(tf_order)))
        if ax is axes[0]:
            ax.set_yticklabels(labels, fontsize=8.9)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    cb = fig.colorbar(images[0], cax=axes[3])
    cb.set_label("C_effective", fontsize=10.6)
    cb.ax.tick_params(labelsize=9.4, length=2.5)
    cb.outline.set_linewidth(0.7)
    fig.subplots_adjust(left=0.075, right=0.958, bottom=0.24, top=0.90)
    paths = save_all(fig, dirs, "fig3h_complete_tf_pair_matrix")
    plt.close(fig)
    df.to_csv(dirs["qa"] / "source_matrix.csv", index=False)
    print(paths["png"])


if __name__ == "__main__":
    main()
