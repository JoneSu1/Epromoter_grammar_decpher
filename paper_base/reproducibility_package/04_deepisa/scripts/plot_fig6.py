from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
DATA_ROOT = BASE / "data"
DATA_OUT = DATA_ROOT / "fig6"
PLOT_OUT = BASE / "figures" / "supplement"
RESULT_ROOT = Path(
    r"G:\我的云端硬盘\DeepEpromote\Drosophila\Motif_cluster\ic_trimmed_results\ep_isa_new_rerun_results"
)
CLASS_TABLE = DATA_ROOT / "fig3h" / "fig3h_complete_tf_pair_nonadditivity_matrix_source.csv"

MM_PER_INCH = 25.4
DOUBLE_COL_MM = 183
DISTANCE_BIN_SIZE = 10
FLANK_WIDTH_BP = 50
SEQUENCE_WINDOW_END = 249
MIN_BIN_N = 3
EXPORT_DPI = 600

TASK_COLORS = {
    "CAGE": "#D64F4F",
    "DEV": "#4C78A8",
    "HK": "#54A24B",
}
TASK_SIGN_STYLES = {
    ("CAGE", "positive"): {"color": "#B9473A", "linestyle": "-", "marker": "o"},
    ("CAGE", "negative"): {"color": "#2F6FA3", "linestyle": "-", "marker": "o"},
    ("DEV", "positive"): {"color": "#D9822B", "linestyle": "--", "marker": "s"},
    ("DEV", "negative"): {"color": "#4C78A8", "linestyle": "--", "marker": "s"},
    ("HK", "positive"): {"color": "#8C6D1F", "linestyle": ":", "marker": "^"},
    ("HK", "negative"): {"color": "#2B8C7E", "linestyle": ":", "marker": "^"},
}
GRID = "#D9D9D9"


@dataclass(frozen=True)
class TaskSpec:
    label: str
    folder: str
    track: int


TASKS = [
    TaskSpec("CAGE", "results_cage_newisa", 0),
    TaskSpec("DEV", "results_dev_newisa", 0),
    TaskSpec("HK", "results_hk_newisa", 1),
]
TASK_ORDER = [task.label for task in TASKS]


def mm_to_in(mm: float) -> float:
    return mm / MM_PER_INCH


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.85,
            "legend.frameon": False,
            "savefig.dpi": EXPORT_DPI,
        }
    )


def save_all(fig: plt.Figure, stem: str) -> dict[str, Path]:
    paths = {
        "svg": PLOT_OUT / "svg" / f"{stem}.svg",
        "pdf": PLOT_OUT / "pdf" / f"{stem}.pdf",
        "tiff": PLOT_OUT / "tiff" / f"{stem}.tiff",
        "png": PLOT_OUT / "png" / f"{stem}.png",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["tiff"], dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    return paths


def interaction_col(task: TaskSpec) -> str:
    return f"interaction_t{task.track}"


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


def canonical_tf_pair(tf1: pd.Series, tf2: pd.Series) -> pd.Series:
    left = np.minimum(tf1.astype(str), tf2.astype(str))
    right = np.maximum(tf1.astype(str), tf2.astype(str))
    return left + "|" + right


def read_task_table(task: TaskSpec, filename: str) -> pd.DataFrame:
    return pd.read_csv(RESULT_ROOT / task.folder / "Data" / filename)


def build_distance_instances() -> pd.DataFrame:
    frames = []
    for task in TASKS:
        pairs = read_task_table(task, "motif_combi_isa.csv").copy()
        locs = read_task_table(task, "motif_locs.csv")
        centers = {
            region: ((group["start_rel"] + group["end_rel"]) / 2).to_numpy(float)
            for region, group in locs.groupby("region", observed=True)
        }
        inter = interaction_col(task)
        pairs["tf_pair"] = canonical_tf_pair(pairs["tf1"], pairs["tf2"])
        rows = []
        for row in pairs.itertuples(index=False):
            left = min(float(row.start1_rel), float(row.start2_rel))
            right = max(float(row.end1_rel), float(row.end2_rel))
            if left < FLANK_WIDTH_BP or right + FLANK_WIDTH_BP > SEQUENCE_WINDOW_END:
                continue
            motif_centers = centers.get(row.region)
            if motif_centers is None:
                continue
            flank = (
                ((motif_centers >= left - FLANK_WIDTH_BP) & (motif_centers < left))
                | ((motif_centers > right) & (motif_centers <= right + FLANK_WIDTH_BP))
            )
            interaction = getattr(row, inter)
            if pd.isna(interaction):
                continue
            rows.append(
                {
                    "task": task.label,
                    "tf_pair": row.tf_pair,
                    "context": "Flank-dense" if int(flank.sum()) >= 1 else "Flank-sparse",
                    "distance": float(row.distance),
                    "distance_bin": int(np.floor(float(row.distance) / DISTANCE_BIN_SIZE) * DISTANCE_BIN_SIZE),
                    "interaction": float(interaction),
                }
            )
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def shared_pair_sets() -> tuple[dict[str, set[str]], pd.DataFrame]:
    classes = pd.read_csv(CLASS_TABLE)
    supported = classes[classes["direction_supported"]].copy()
    sets = {task: set(supported.loc[supported["task"] == task, "tf_pair"]) for task in TASK_ORDER}
    triple = set.intersection(*(sets[task] for task in TASK_ORDER))
    groups = {
        "fig6a_triple_shared": triple,
        "fig6b_cage_dev_shared": (sets["CAGE"] & sets["DEV"]) - triple,
        "fig6c_cage_hk_shared": (sets["CAGE"] & sets["HK"]) - triple,
        "fig6d_dev_hk_shared": (sets["DEV"] & sets["HK"]) - triple,
    }
    rows = []
    for group, pairs in groups.items():
        for pair in sorted(pairs):
            rows.append({"shared_group": group, "tf_pair": pair, "display_pair": display_pair(pair)})
    return groups, pd.DataFrame(rows)


def summarize(instances: pd.DataFrame, pairs: set[str], tasks: list[str]) -> pd.DataFrame:
    sub = instances[instances["tf_pair"].isin(pairs) & instances["task"].isin(tasks)].copy()
    sub = sub.dropna(subset=["interaction"])
    sub = sub[sub["interaction"] != 0].copy()
    sub["sign_class"] = np.where(sub["interaction"] > 0, "positive", "negative")
    summary = (
        sub.groupby(["task", "tf_pair", "distance_bin", "sign_class"], observed=True)
        .agg(
            n=("interaction", "size"),
            mean_interaction=("interaction", "mean"),
            median_interaction=("interaction", "median"),
            median_abs_interaction=("interaction", lambda x: float(np.median(np.abs(x)))),
            q25_interaction=("interaction", lambda x: float(np.percentile(x, 25))),
            q75_interaction=("interaction", lambda x: float(np.percentile(x, 75))),
        )
        .reset_index()
    )
    summary["passes_n_filter"] = summary["n"] >= MIN_BIN_N
    return summary


def summarize_by_context(instances: pd.DataFrame, pairs: set[str], tasks: list[str]) -> pd.DataFrame:
    sub = instances[instances["tf_pair"].isin(pairs) & instances["task"].isin(tasks)].copy()
    sub = sub.dropna(subset=["interaction"])
    sub = sub[sub["interaction"] != 0].copy()
    sub["sign_class"] = np.where(sub["interaction"] > 0, "positive", "negative")
    summary = (
        sub.groupby(["task", "context", "tf_pair", "distance_bin", "sign_class"], observed=True)
        .agg(
            n=("interaction", "size"),
            mean_interaction=("interaction", "mean"),
            median_interaction=("interaction", "median"),
            median_abs_interaction=("interaction", lambda x: float(np.median(np.abs(x)))),
            q25_interaction=("interaction", lambda x: float(np.percentile(x, 25))),
            q75_interaction=("interaction", lambda x: float(np.percentile(x, 75))),
        )
        .reset_index()
    )
    summary["passes_n_filter"] = summary["n"] >= MIN_BIN_N
    return summary


def y_limit(summary: pd.DataFrame) -> float:
    plotted = summary.loc[summary["passes_n_filter"], "median_interaction"].abs()
    if plotted.empty:
        return 0.05
    return max(float(plotted.quantile(0.98)) * 1.2, 0.05)


def plot_shared_group(summary: pd.DataFrame, pair_order: list[str], tasks: list[str], stem: str) -> Path:
    n_pairs = len(pair_order)
    if n_pairs <= 4:
        ncols = 2
    else:
        ncols = 3
    nrows = ceil(n_pairs / ncols)
    fig_h = 36 + 38 * nrows
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(mm_to_in(DOUBLE_COL_MM), mm_to_in(fig_h)),
        sharex=True,
        sharey=True,
        dpi=300,
    )
    axes_arr = np.atleast_1d(axes).ravel()
    ylim = y_limit(summary)

    for ax, pair in zip(axes_arr, pair_order):
        sub_pair = summary[(summary["tf_pair"] == pair) & (summary["passes_n_filter"])].copy()
        for task in tasks:
            for sign in ["positive", "negative"]:
                sub = (
                    sub_pair[(sub_pair["task"] == task) & (sub_pair["sign_class"] == sign)]
                    .sort_values("distance_bin")
                )
                if sub.empty:
                    continue
                ax.plot(
                    sub["distance_bin"],
                    sub["median_interaction"],
                    lw=1.45,
                    ms=3.0,
                    label=f"{task} {sign}",
                    **TASK_SIGN_STYLES[(task, sign)],
                )
        ax.axhline(0, color="#999999", lw=0.75, linestyle=(0, (2, 2)))
        ax.set_title(display_pair(pair), fontsize=8.5, pad=4)
        ax.set_xlim(-5, 140)
        ax.set_ylim(-ylim, ylim)
        ax.set_xticks(np.arange(0, 141, 40))
        ax.tick_params(axis="x", labelsize=7.6, length=2.5)
        ax.tick_params(axis="y", labelsize=7.6, length=2.5)
        ax.yaxis.grid(True, color=GRID, linewidth=0.38, alpha=0.65)
        ax.set_axisbelow(True)

    for ax in axes_arr[n_pairs:]:
        ax.axis("off")
    fig.text(
        0.012,
        0.50,
        "Median interaction within sign class",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=8.8,
    )
    for ax in axes_arr[max(0, (nrows - 1) * ncols): nrows * ncols]:
        if ax.has_data():
            ax.set_xlabel("Distance bin (bp)", fontsize=8.6)

    handles = []
    for task in tasks:
        for sign in ["positive", "negative"]:
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    lw=1.7,
                    ms=3.2,
                    label=f"{task} {sign}",
                    **TASK_SIGN_STYLES[(task, sign)],
                )
            )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(len(handles), 6),
        frameon=False,
        bbox_to_anchor=(0.52, 0.995),
        fontsize=8.3,
        handlelength=1.5,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.095, top=0.90, wspace=0.26, hspace=0.54)
    paths = save_all(fig, stem)
    plt.close(fig)
    return paths["png"]


def sign_y_limits(summary: pd.DataFrame, sign: str) -> tuple[float, float]:
    plotted = summary.loc[
        (summary["passes_n_filter"]) & (summary["sign_class"] == sign),
        "median_interaction",
    ]
    if plotted.empty:
        return (-0.05, 0.05)
    if sign == "positive":
        y_max = max(float(plotted.quantile(0.98)) * 1.18, 0.04)
        return (0.0, y_max)
    y_min = min(float(plotted.quantile(0.02)) * 1.18, -0.04)
    return (y_min, 0.0)


def plot_shared_group_sign(
    summary: pd.DataFrame,
    pair_order: list[str],
    tasks: list[str],
    sign: str,
    stem: str,
) -> Path:
    n_pairs = len(pair_order)
    if n_pairs <= 4:
        ncols = 2
    else:
        ncols = 3
    nrows = ceil(n_pairs / ncols)
    fig_h = 34 + 38 * nrows
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(mm_to_in(DOUBLE_COL_MM), mm_to_in(fig_h)),
        sharex=True,
        sharey=True,
        dpi=300,
    )
    axes_arr = np.atleast_1d(axes).ravel()
    y_min, y_max = sign_y_limits(summary, sign)

    for ax, pair in zip(axes_arr, pair_order):
        sub_pair = summary[
            (summary["tf_pair"] == pair)
            & (summary["passes_n_filter"])
            & (summary["sign_class"] == sign)
        ].copy()
        for task in tasks:
            sub = sub_pair[sub_pair["task"] == task].sort_values("distance_bin")
            if sub.empty:
                continue
            ax.plot(
                sub["distance_bin"],
                sub["median_interaction"],
                color=TASK_COLORS[task],
                lw=1.55,
                marker="o",
                ms=2.7,
                label=task,
            )
        ax.axhline(0, color="#9A9A9A", lw=0.75, linestyle=(0, (2, 2)))
        ax.set_title(display_pair(pair), fontsize=8.8, pad=4)
        ax.set_xlim(-5, 140)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(np.arange(0, 141, 40))
        ax.tick_params(axis="x", labelsize=7.6, length=2.5)
        ax.tick_params(axis="y", labelsize=7.6, length=2.5)
        ax.yaxis.grid(True, color=GRID, linewidth=0.35, alpha=0.55)
        ax.set_axisbelow(True)

    for ax in axes_arr[n_pairs:]:
        ax.axis("off")

    fig.text(
        0.012,
        0.50,
        f"Median {sign} interaction",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=8.8,
    )
    for ax in axes_arr[max(0, (nrows - 1) * ncols): nrows * ncols]:
        if ax.has_data():
            ax.set_xlabel("Distance bin (bp)", fontsize=8.6)

    handles = [
        plt.Line2D([0], [0], color=TASK_COLORS[task], lw=1.8, marker="o", ms=3.0, label=task)
        for task in tasks
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(tasks),
        frameon=False,
        bbox_to_anchor=(0.52, 0.995),
        fontsize=8.8,
        handlelength=1.45,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.095, top=0.90, wspace=0.28, hspace=0.58)
    paths = save_all(fig, stem)
    plt.close(fig)
    return paths["png"]


def plot_shared_group_sign_context(
    summary: pd.DataFrame,
    pair_order: list[str],
    tasks: list[str],
    sign: str,
    stem: str,
) -> Path:
    contexts = ["Flank-dense", "Flank-sparse"]
    n_pairs = len(pair_order)
    block_cols = 2 if n_pairs <= 4 else 3
    nrows = ceil(n_pairs / block_cols)
    ncols = block_cols * len(contexts)
    fig_h = 38 + 42 * nrows
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(mm_to_in(DOUBLE_COL_MM), mm_to_in(fig_h)),
        sharex=True,
        sharey=True,
        dpi=300,
    )
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    y_min, y_max = sign_y_limits(summary, sign)
    pair_title_axes: list[tuple[plt.Axes, plt.Axes, str]] = []

    for pair_i, pair in enumerate(pair_order):
        row = pair_i // block_cols
        block_col = pair_i % block_cols
        first_ax = axes_arr[row, block_col * len(contexts)]
        second_ax = axes_arr[row, block_col * len(contexts) + 1]
        pair_title_axes.append((first_ax, second_ax, display_pair(pair)))
        for context_i, context in enumerate(contexts):
            col = block_col * len(contexts) + context_i
            ax = axes_arr[row, col]
            sub_panel = summary[
                (summary["tf_pair"] == pair)
                & (summary["context"] == context)
                & (summary["sign_class"] == sign)
                & (summary["passes_n_filter"])
            ].copy()
            for task in tasks:
                sub = sub_panel[sub_panel["task"] == task].sort_values("distance_bin")
                if sub.empty:
                    continue
                ax.plot(
                    sub["distance_bin"],
                    sub["median_interaction"],
                    color=TASK_COLORS[task],
                    lw=1.35,
                    marker="o",
                    ms=2.4,
                    label=task,
                )
            ax.axhline(0, color="#9A9A9A", lw=0.65, linestyle=(0, (2, 2)))
            ax.set_title(context.replace("Flank-", ""), fontsize=7.0, pad=2)
            ax.set_xlim(-5, 140)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(np.arange(0, 141, 70))
            ax.tick_params(axis="x", labelsize=6.6, length=2.2)
            ax.tick_params(axis="y", labelsize=6.6, length=2.2)
            ax.yaxis.grid(True, color=GRID, linewidth=0.3, alpha=0.5)
            ax.set_axisbelow(True)

    for pair_i in range(n_pairs, nrows * block_cols):
        row = pair_i // block_cols
        block_col = pair_i % block_cols
        for context_i in range(len(contexts)):
            axes_arr[row, block_col * len(contexts) + context_i].axis("off")

    fig.text(
        0.012,
        0.50,
        f"Median {sign} interaction",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=8.3,
    )
    for ax in axes_arr[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Distance bin (bp)", fontsize=7.6)

    handles = [
        plt.Line2D([0], [0], color=TASK_COLORS[task], lw=1.7, marker="o", ms=2.9, label=task)
        for task in tasks
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(tasks),
        frameon=False,
        bbox_to_anchor=(0.52, 0.995),
        fontsize=8.4,
        handlelength=1.35,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.080, right=0.995, bottom=0.090, top=0.875, wspace=0.24, hspace=0.86)
    for left_ax, right_ax, pair_label in pair_title_axes:
        left_box = left_ax.get_position()
        right_box = right_ax.get_position()
        x_center = (left_box.x0 + right_box.x1) / 2
        y_top = max(left_box.y1, right_box.y1) + 0.018
        fig.text(x_center, y_top, pair_label, ha="center", va="bottom", fontsize=7.7)
    paths = save_all(fig, stem)
    plt.close(fig)
    return paths["png"]


def write_audit_report(instances: pd.DataFrame, groups: dict[str, set[str]], outputs: list[Path]) -> None:
    lines = [
        "# Fig6 shared TF-pair distance audit",
        "",
        "Shared TF-pairs are defined from direction-supported TF-pair classes in Fig3h source data.",
        "",
        "## Shared-set sizes",
    ]
    for group, pairs in groups.items():
        lines.append(f"- {group}: {len(pairs)} TF-pairs")
    lines.append("")
    lines.append("## Flank-valid distance instances")
    for task, sub in instances.groupby("task", observed=True):
        lines.append(f"- {task}: {len(sub):,} instances")
    lines.append("")
    lines.append(
        f"Only distance bins with n >= {MIN_BIN_N} motif-pair instances within each positive/negative sign class are plotted."
    )
    lines.append("")
    lines.append("## Output files")
    for path in outputs:
        lines.append(f"- `{path}`")
    (DATA_OUT / "fig6_shared_tf_pair_distance_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    set_style()
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    PLOT_OUT.mkdir(parents=True, exist_ok=True)

    groups, membership = shared_pair_sets()
    instances = build_distance_instances()
    instances.to_csv(DATA_OUT / "fig6_distance_instances_with_tf_pair.csv", index=False)
    membership.to_csv(DATA_OUT / "fig6_shared_tf_pair_membership.csv", index=False)

    specs = [
        ("fig6a_triple_shared", "fig6a_triple_shared_tf_pair_distance", TASK_ORDER),
        ("fig6b_cage_dev_shared", "fig6b_cage_dev_shared_tf_pair_distance", ["CAGE", "DEV"]),
        ("fig6c_cage_hk_shared", "fig6c_cage_hk_shared_tf_pair_distance", ["CAGE", "HK"]),
        ("fig6d_dev_hk_shared", "fig6d_dev_hk_shared_tf_pair_distance", ["DEV", "HK"]),
    ]

    outputs = []
    for group, stem, tasks in specs:
        pairs = groups[group]
        summary = summarize(instances, pairs, tasks)
        context_summary = summarize_by_context(instances, pairs, tasks)
        order = sorted(pairs, key=display_pair)
        summary.to_csv(DATA_OUT / f"{stem}_binned_summary.csv", index=False)
        context_summary.to_csv(DATA_OUT / f"{stem}_context_binned_summary.csv", index=False)
        membership[membership["shared_group"] == group].to_csv(DATA_OUT / f"{stem}_shared_pairs.csv", index=False)
        outputs.append(plot_shared_group(summary, order, tasks, stem))
        outputs.append(plot_shared_group_sign(summary, order, tasks, "positive", f"{stem}_positive_only"))
        outputs.append(plot_shared_group_sign(summary, order, tasks, "negative", f"{stem}_negative_only"))
        outputs.append(
            plot_shared_group_sign_context(context_summary, order, tasks, "positive", f"{stem}_positive_context")
        )
        outputs.append(
            plot_shared_group_sign_context(context_summary, order, tasks, "negative", f"{stem}_negative_context")
        )

    write_audit_report(instances, groups, outputs)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
