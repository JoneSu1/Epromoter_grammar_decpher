from __future__ import annotations

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
from deepisa_style import DOUBLE_COL_MM, SIGN_COLORS, GRID, mm_to_in, save_all, set_pub_style


SUMMARY = DATA_ROOT / "fig4_positive_negative_distance" / "fig4_positive_negative_distance_summary.csv"
TASKS = ["CAGE", "DEV", "HK"]
CONTEXTS = {
    "Flank-dense": {"linestyle": "-", "marker": "o", "label": "dense"},
    "Flank-sparse": {"linestyle": "--", "marker": "s", "label": "sparse"},
}
width_mm = 183
EXPORT_TARGETS = ("figure.svg", "figure.pdf", "figure.tiff", "figure.png")
EXPORT_DPI = 600


def main() -> None:
    set_pub_style(font_size=10.8)
    dirs = panel_dirs("fig4b_positive_negative_distance")
    df = pd.read_csv(SUMMARY)
    plot_df = df[df["passes_n_filter"]].copy()

    ymax = plot_df["median_interaction"].abs().quantile(0.98) * 1.15
    ymax = max(float(ymax), 0.04)

    fig, axes = plt.subplots(1, 3, figsize=(mm_to_in(DOUBLE_COL_MM * 0.98), mm_to_in(66)),
                             sharey=True, dpi=300)
    for ax, task in zip(axes, TASKS):
        sub_task = plot_df[plot_df["task"] == task]
        for sign in ["positive", "negative"]:
            for context, style in CONTEXTS.items():
                sub = (
                    sub_task[(sub_task["sign_class"] == sign) & (sub_task["context"] == context)]
                    .sort_values("distance_bin")
                )
                if sub.empty:
                    continue
                ax.plot(
                    sub["distance_bin"],
                    sub["median_interaction"],
                    color=SIGN_COLORS[sign],
                    lw=1.8,
                    ms=4.0,
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    label=f"{sign}, {style['label']}",
                )
        ax.axhline(0, color="#9A9A9A", lw=0.85, linestyle=(0, (2, 2)))
        ax.set_title(task, fontsize=12.8, pad=5)
        ax.set_xlabel("Motif-pair distance bin (bp)", fontsize=9.8)
        ax.set_xlim(-5, 140)
        ax.set_ylim(-ymax, ymax)
        ax.set_xticks(range(0, 141, 20))
        ax.tick_params(axis="x", rotation=45, labelsize=9.0)
        ax.tick_params(axis="y", labelsize=9.4)
        ax.yaxis.grid(True, color=GRID, linewidth=0.45, alpha=0.65)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Median interaction within sign class", fontsize=10.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1.02),
               ncol=4, fontsize=9.0, handlelength=1.35, columnspacing=0.85)
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.25, top=0.80, wspace=0.18)
    paths = save_all(fig, dirs, "fig4b_positive_negative_distance")
    plt.close(fig)
    df.to_csv(dirs["qa"] / "fig4_positive_negative_distance_summary.csv", index=False)
    print(paths["png"])


if __name__ == "__main__":
    main()
