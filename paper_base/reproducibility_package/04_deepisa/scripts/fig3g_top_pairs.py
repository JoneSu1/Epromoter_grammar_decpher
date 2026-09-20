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
from matplotlib.patches import Rectangle

from deepisa_paths import DATA_ROOT, panel_dirs
from deepisa_style import DOUBLE_COL_MM, TASK_COLORS, mm_to_in, save_all, set_pub_style


LONG_CSV = DATA_ROOT / "fig3g" / "fig3g_top_tf_pair_resolution_long_source.csv"
PAIR_CSV = DATA_ROOT / "fig3g" / "fig3g_top_tf_pair_resolution_selected_pairs.csv"
TASKS = ["CAGE", "DEV", "HK"]
width_mm = 183
EXPORT_TARGETS = ("figure.svg", "figure.pdf", "figure.tiff", "figure.png")
EXPORT_DPI = 600


def task_ordered_pairs() -> list[str]:
    selected = pd.read_csv(PAIR_CSV)
    return selected["short_label"].drop_duplicates().tolist()


def main() -> None:
    set_pub_style(font_size=12.4)
    dirs = panel_dirs("fig3g_top_tf_pair_resolution")
    df = pd.read_csv(LONG_CSV)
    pair_order = task_ordered_pairs()

    heat = (
        df.pivot_table(index="task", columns="short_label", values="C_effective", aggfunc="first")
        .reindex(TASKS)
        .reindex(columns=pair_order)
    )
    support = (
        df.pivot_table(index="task", columns="short_label", values="direction_supported", aggfunc="first")
        .reindex(TASKS)
        .reindex(columns=pair_order)
        .fillna(False)
        .astype(bool)
    )
    n_eff = (
        df.pivot_table(index="task", columns="short_label", values="n_effective_instances", aggfunc="first")
        .reindex(TASKS)
        .reindex(columns=pair_order)
    )

    cmap = LinearSegmentedColormap.from_list("deepisa_effect", ["#2F6FA3", "#F5F5F5", "#B9473A"])
    cmap.set_bad("#F0F0F0")
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    n_cols = len(pair_order)
    fig = plt.figure(figsize=(mm_to_in(DOUBLE_COL_MM), mm_to_in(80)), dpi=300)
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 0.032], height_ratios=[0.61, 0.20, 0.19],
                          wspace=0.035, hspace=0.08)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_sup = fig.add_subplot(gs[1, 0], sharex=ax_heat)
    ax_n = fig.add_subplot(gs[2, 0], sharex=ax_heat)
    cax = fig.add_subplot(gs[0, 1])

    image = ax_heat.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap=cmap, norm=norm)
    for i in range(len(TASKS) + 1):
        ax_heat.axhline(i - 0.5, color="#F8F8F8", lw=0.65)
    for j in range(n_cols + 1):
        ax_heat.axvline(j - 0.5, color="#F8F8F8", lw=0.65)
    ax_heat.set_yticks(np.arange(len(TASKS)))
    ax_heat.set_yticklabels(TASKS, fontsize=11.8)
    for label, task in zip(ax_heat.get_yticklabels(), TASKS):
        label.set_color(TASK_COLORS[task])
    ax_heat.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_heat.set_ylabel("")

    cb = fig.colorbar(image, cax=cax)
    cb.set_label("C_effective", fontsize=10.8)
    cb.ax.tick_params(labelsize=9.5, length=2.5)
    cb.outline.set_linewidth(0.7)

    ax_sup.set_xlim(-0.5, n_cols - 0.5)
    ax_sup.set_ylim(-0.5, len(TASKS) - 0.5)
    for i, task in enumerate(TASKS):
        for j, pair in enumerate(pair_order):
            face = TASK_COLORS[task] if support.loc[task, pair] else "#F1F1F1"
            ax_sup.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face,
                                       edgecolor="#F8F8F8", linewidth=0.65))
    ax_sup.set_yticks(np.arange(len(TASKS)))
    ax_sup.set_yticklabels(TASKS, fontsize=10.8)
    ax_sup.invert_yaxis()
    for label, task in zip(ax_sup.get_yticklabels(), TASKS):
        label.set_color(TASK_COLORS[task])
    ax_sup.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_sup.set_ylabel("Supported", fontsize=10.8)
    for spine in ax_sup.spines.values():
        spine.set_visible(False)

    n_values = n_eff.to_numpy(dtype=float)
    n_total = np.nansum(n_values, axis=0)
    n_scaled = np.log10(np.clip(n_total, 1, None))[None, :]
    ax_n.imshow(n_scaled, aspect="auto", cmap="Greys", vmin=0, vmax=np.nanmax(n_scaled))
    for j in range(n_cols + 1):
        ax_n.axvline(j - 0.5, color="#F8F8F8", lw=0.65)
    for j, total in enumerate(n_total):
        if np.isfinite(total) and total >= 1000:
            text = f"{int(total):,}"
        elif np.isfinite(total) and total > 0:
            text = f"{int(total)}"
        else:
            text = ""
        ax_n.text(j, 0, text, ha="center", va="center", fontsize=8.2,
                  color="white" if total >= np.nanpercentile(n_total, 70) else "#222222")
    ax_n.set_yticks([0])
    ax_n.set_yticklabels(["n_eff"], fontsize=10.6)
    ax_n.set_xticks(np.arange(n_cols))
    ax_n.set_xticklabels(pair_order, rotation=50, ha="right", fontsize=9.4)
    for spine in ax_n.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.075, right=0.95, bottom=0.22, top=0.98)
    paths = save_all(fig, dirs, "fig3g_top_tf_pair_resolution")
    plt.close(fig)
    df.to_csv(dirs["qa"] / "source_long.csv", index=False)
    print(paths["png"])


if __name__ == "__main__":
    main()
