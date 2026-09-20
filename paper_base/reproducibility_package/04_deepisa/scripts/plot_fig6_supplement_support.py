from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
DATA_IN = BASE / "data" / "fig6"
PLOT_OUT = BASE / "figures" / "supplement"
DATA_OUT = DATA_IN / "supporting"

MM_PER_INCH = 25.4
EXPORT_DPI = 600
TASK_ORDER = ["CAGE", "DEV", "HK"]
SIGN_ORDER = ["positive", "negative"]
CONTEXT_ORDER = ["Flank-dense", "Flank-sparse"]

GROUP_SPECS = [
    ("triple_shared", "fig6a_triple_shared_tf_pair_distance", "Triple shared"),
    ("cage_dev_shared", "fig6b_cage_dev_shared_tf_pair_distance", "CAGE-DEV"),
    ("cage_hk_shared", "fig6c_cage_hk_shared_tf_pair_distance", "CAGE-HK"),
    ("dev_hk_shared", "fig6d_dev_hk_shared_tf_pair_distance", "DEV-HK"),
]

TASK_COLORS = {"CAGE": "#D64F4F", "DEV": "#4C78A8", "HK": "#54A24B"}
SUPPORT_CMAP = "Blues"
EFFECT_CMAP = "RdBu_r"


def mm_to_in(value: float) -> float:
    return value / MM_PER_INCH


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.dpi": EXPORT_DPI,
        }
    )


def save_all(fig: plt.Figure, stem: str) -> None:
    paths = {
        "svg": PLOT_OUT / "svg" / f"{stem}.svg",
        "pdf": PLOT_OUT / "pdf" / f"{stem}.pdf",
        "png": PLOT_OUT / "png" / f"{stem}.png",
        "tiff": PLOT_OUT / "tiff" / f"{stem}.tiff",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["png"], dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(paths["tiff"], dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def display_tf(name: str) -> str:
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
    return mapping.get(str(name), str(name))


def display_pair(pair: str) -> str:
    left, right = str(pair).split("|", 1)
    return f"{display_tf(left)} | {display_tf(right)}"


def add_cell_grid(ax: plt.Axes, n_rows: int, n_cols: int) -> None:
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)


def annotate_matrix(ax: plt.Axes, matrix: np.ndarray, fmt: str, vmax: float, fontsize: float = 6.5) -> None:
    threshold = vmax * 0.58
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            color = "white" if value >= threshold else "#1F1F1F"
            ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=fontsize, color=color)


def read_group_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    pairs = []
    for group_key, stem, group_label in GROUP_SPECS:
        summary = pd.read_csv(DATA_IN / f"{stem}_context_binned_summary.csv")
        summary["shared_group"] = group_key
        summary["shared_group_label"] = group_label
        summaries.append(summary)

        membership = pd.read_csv(DATA_IN / f"{stem}_shared_pairs.csv")
        membership["shared_group"] = group_key
        membership["shared_group_label"] = group_label
        pairs.append(membership)
    return pd.concat(summaries, ignore_index=True), pd.concat(pairs, ignore_index=True)


def write_support_tables(summary: pd.DataFrame, membership: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    plotted = summary[summary["passes_n_filter"]].copy()
    support_rows = []
    for group_key, _, group_label in GROUP_SPECS:
        pairs = membership.loc[membership["shared_group"] == group_key, "tf_pair"].nunique()
        sub_all = summary[summary["shared_group"] == group_key]
        sub_plot = plotted[plotted["shared_group"] == group_key]
        for sign in SIGN_ORDER:
            for context in CONTEXT_ORDER:
                part_all = sub_all[(sub_all["sign_class"] == sign) & (sub_all["context"] == context)]
                part_plot = sub_plot[(sub_plot["sign_class"] == sign) & (sub_plot["context"] == context)]
                support_rows.append(
                    {
                        "shared_group": group_key,
                        "shared_group_label": group_label,
                        "n_tf_pairs": pairs,
                        "sign_class": sign,
                        "context": context,
                        "n_binned_rows": len(part_all),
                        "n_plotted_rows": len(part_plot),
                        "n_instances_in_plotted_bins": int(part_plot["n"].sum()) if not part_plot.empty else 0,
                        "n_tf_pair_panels_with_plotted_bins": part_plot["tf_pair"].nunique(),
                    }
                )
    support = pd.DataFrame(support_rows)

    panel_rows = []
    for row in membership.itertuples(index=False):
        sub_pair = plotted[
            (plotted["shared_group"] == row.shared_group)
            & (plotted["tf_pair"] == row.tf_pair)
        ]
        for sign in SIGN_ORDER:
            for context in CONTEXT_ORDER:
                part = sub_pair[(sub_pair["sign_class"] == sign) & (sub_pair["context"] == context)]
                panel_rows.append(
                    {
                        "shared_group": row.shared_group,
                        "shared_group_label": row.shared_group_label,
                        "tf_pair": row.tf_pair,
                        "display_pair": display_pair(row.tf_pair),
                        "sign_class": sign,
                        "context": context,
                        "n_plotted_bins": len(part),
                        "n_instances_in_plotted_bins": int(part["n"].sum()) if not part.empty else 0,
                        "has_plotted_bins": bool(len(part)),
                    }
                )
    panel_coverage = pd.DataFrame(panel_rows)

    support.to_csv(DATA_OUT / "Supp_Fig_S6c_support_summary_by_group_sign_context.csv", index=False)
    panel_coverage.to_csv(DATA_OUT / "Supp_Fig_S6c_tf_pair_panel_coverage.csv", index=False)
    return support, panel_coverage


def plot_support_summary(support: pd.DataFrame, panel_coverage: pd.DataFrame) -> None:
    labels = [spec[2] for spec in GROUP_SPECS]
    xlabels = ["Pos.\ndense", "Pos.\nsparse", "Neg.\ndense", "Neg.\nsparse"]
    count_matrix = np.zeros((len(labels), len(xlabels)))
    instance_matrix = np.zeros_like(count_matrix)
    for i, (_, _, group_label) in enumerate(GROUP_SPECS):
        for j, (sign, context) in enumerate(
            [("positive", "Flank-dense"), ("positive", "Flank-sparse"), ("negative", "Flank-dense"), ("negative", "Flank-sparse")]
        ):
            row = support[
                (support["shared_group_label"] == group_label)
                & (support["sign_class"] == sign)
                & (support["context"] == context)
            ].iloc[0]
            count_matrix[i, j] = row["n_plotted_rows"]
            instance_matrix[i, j] = row["n_instances_in_plotted_bins"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(mm_to_in(183), mm_to_in(64)),
        dpi=300,
        gridspec_kw={"width_ratios": [1.0, 1.08]},
    )
    for ax, matrix, title, fmt in [
        (axes[0], count_matrix, "Supported distance bins", "{:.0f}"),
        (axes[1], instance_matrix, "Instances in supported bins", "{:.0f}"),
    ]:
        vmax = max(float(np.nanmax(matrix)), 1.0)
        im = ax.imshow(matrix, cmap=SUPPORT_CMAP, vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=8.5, pad=5)
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels([])
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        if ax is axes[0]:
            for i, label in enumerate(labels):
                ax.text(-0.62, i, label, ha="right", va="center", fontsize=7.5, clip_on=False)
        annotate_matrix(ax, matrix, fmt, vmax)
        add_cell_grid(ax, matrix.shape[0], matrix.shape[1])
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
        cbar.ax.tick_params(labelsize=6.5, length=2)

    fig.suptitle("Support for shared TF-pair distance/context panels", fontsize=9.5, y=0.985)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.19, top=0.82, wspace=0.30)
    save_all(fig, "Supp_Fig_S6c_support_coverage")


def dense_sparse_effect(summary: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    plotted = summary[summary["passes_n_filter"]].copy()
    rows = []
    for group_key, _, group_label in GROUP_SPECS:
        group_pairs = membership[membership["shared_group"] == group_key].copy()
        for pair in group_pairs["tf_pair"]:
            sub_pair = plotted[(plotted["shared_group"] == group_key) & (plotted["tf_pair"] == pair)]
            for task in TASK_ORDER:
                for sign in SIGN_ORDER:
                    dense = sub_pair[
                        (sub_pair["task"] == task)
                        & (sub_pair["sign_class"] == sign)
                        & (sub_pair["context"] == "Flank-dense")
                    ]["median_interaction"]
                    sparse = sub_pair[
                        (sub_pair["task"] == task)
                        & (sub_pair["sign_class"] == sign)
                        & (sub_pair["context"] == "Flank-sparse")
                    ]["median_interaction"]
                    rows.append(
                        {
                            "shared_group": group_key,
                            "shared_group_label": group_label,
                            "tf_pair": pair,
                            "display_pair": display_pair(pair),
                            "task": task,
                            "sign_class": sign,
                            "dense_median_of_bins": float(dense.median()) if len(dense) else np.nan,
                            "sparse_median_of_bins": float(sparse.median()) if len(sparse) else np.nan,
                            "dense_minus_sparse": (
                                float(dense.median() - sparse.median()) if len(dense) and len(sparse) else np.nan
                            ),
                            "n_dense_bins": len(dense),
                            "n_sparse_bins": len(sparse),
                        }
                    )
    effect = pd.DataFrame(rows)
    effect.to_csv(DATA_OUT / "Supp_Fig_S6d_dense_sparse_effect_summary.csv", index=False)
    return effect


def plot_dense_sparse_effect(effect: pd.DataFrame) -> None:
    row_keys = [(group_key, group_label) for group_key, _, group_label in GROUP_SPECS]
    summary_rows = []
    for group_key, _, group_label in GROUP_SPECS:
        for sign in SIGN_ORDER:
            for task in TASK_ORDER:
                values = effect.loc[
                    (effect["shared_group"] == group_key)
                    & (effect["sign_class"] == sign)
                    & (effect["task"] == task),
                    "dense_minus_sparse",
                ].dropna()
                summary_rows.append(
                    {
                        "shared_group": group_key,
                        "group_label": group_label,
                        "sign_class": sign,
                        "task": task,
                        "median_dense_minus_sparse": float(values.median()) if len(values) else np.nan,
                        "n_tf_pairs": int(len(values)),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(DATA_OUT / "Supp_Fig_S6d_dense_sparse_effect_group_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(mm_to_in(183), mm_to_in(78)), dpi=300, sharey=True)
    vlim = np.nanpercentile(np.abs(summary["median_dense_minus_sparse"]), 95)
    if not np.isfinite(vlim) or vlim == 0:
        vlim = 0.05
    vlim = max(float(vlim), 0.04)

    for ax, sign in zip(axes, SIGN_ORDER):
        matrix = np.full((len(row_keys), len(TASK_ORDER)), np.nan)
        n_matrix = np.zeros_like(matrix)
        for i, (group_key, _) in enumerate(row_keys):
            for j, task in enumerate(TASK_ORDER):
                row = summary[
                    (summary["shared_group"] == group_key)
                    & (summary["task"] == task)
                    & (summary["sign_class"] == sign)
                ]
                if not row.empty:
                    matrix[i, j] = row.iloc[0]["median_dense_minus_sparse"]
                    n_matrix[i, j] = row.iloc[0]["n_tf_pairs"]

        im = ax.imshow(matrix, cmap=EFFECT_CMAP, vmin=-vlim, vmax=vlim, aspect="auto")
        ax.set_title(f"{sign.capitalize()} interactions", fontsize=8.5, pad=5)
        ax.set_xticks(np.arange(len(TASK_ORDER)))
        ax.set_xticklabels(TASK_ORDER, fontsize=7.5)
        ax.set_yticks(np.arange(len(row_keys)))
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        if ax is axes[0]:
            for i, (_, label) in enumerate(row_keys):
                ax.text(-0.60, i, label, ha="right", va="center", fontsize=7.2, clip_on=False)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    text_color = "white" if abs(matrix[i, j]) > vlim * 0.62 else "#1F1F1F"
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.2f}\nn={int(n_matrix[i, j])}",
                        ha="center",
                        va="center",
                        fontsize=6.2,
                        color=text_color,
                        linespacing=0.95,
                    )
        add_cell_grid(ax, matrix.shape[0], matrix.shape[1])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Task", fontsize=7.5)

    cax = fig.add_axes([0.90, 0.22, 0.016, 0.55])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Dense - sparse\nmedian interaction", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5, length=2)
    fig.suptitle("Dense-sparse context effect across shared TF-pairs", fontsize=9.5, y=0.985)
    fig.subplots_adjust(left=0.20, right=0.86, bottom=0.17, top=0.80, wspace=0.16)
    save_all(fig, "Supp_Fig_S6d_dense_sparse_effect_summary")


def main() -> None:
    set_style()
    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    summary, membership = read_group_tables()
    support, panel_coverage = write_support_tables(summary, membership)
    plot_support_summary(support, panel_coverage)
    effect = dense_sparse_effect(summary, membership)
    plot_dense_sparse_effect(effect)
    print(PLOT_OUT / "svg" / "Supp_Fig_S6c_support_coverage.svg")
    print(PLOT_OUT / "svg" / "Supp_Fig_S6d_dense_sparse_effect_summary.svg")


if __name__ == "__main__":
    main()
