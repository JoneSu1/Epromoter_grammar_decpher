from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOP_K = 6
REGION_LEN = 250
POSITION_BIN_SIZE = 50
MIN_EXAMPLE_SPAN_BP = 80
TASKS = [
    ("CAGE", "results_cage_newisa", 0, "#BA611B"),
    ("DEV", "results_dev_newisa", 0, "#148A6A"),
    ("HK", "results_hk_newisa", 1, "#166C9C"),
]


@dataclass(frozen=True)
class TaskData:
    task: str
    track: int
    color: str
    selected: pd.DataFrame
    selected_pairs: pd.DataFrame


def find_newisa_root() -> Path:
    local = Path(
        r"F:\phd\Drophila\3Model_motif_discovering\ISA\result"
    ) / "nature_rebuilt_figures_newisa_from_gdrive" / "ep_isa_new_rerun_results"
    if local.exists():
        return local

    for drive in [Path("G:/"), Path("F:/Google")]:
        if not drive.exists():
            continue
        candidates = [drive]
        try:
            candidates.extend([p for p in drive.iterdir() if p.is_dir()])
        except OSError:
            pass
        for root in candidates:
            p = (
                root
                / "DeepEpromote"
                / "Drosophila"
                / "Motif_cluster"
                / "ic_trimmed_results"
                / "ep_isa_new_rerun_results"
            )
            if p.exists():
                return p
    raise FileNotFoundError("Could not find ep_isa_new_rerun_results.")


def output_root(input_root: Path) -> Path:
    out = input_root.parent / "position_aware_topk_interaction_architecture"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "source_data").mkdir(parents=True, exist_ok=True)
    return out


def motif_key(region: pd.Series, tf: pd.Series, start: pd.Series, end: pd.Series) -> pd.Series:
    return (
        region.astype(str)
        + "|"
        + tf.astype(str)
        + "|"
        + start.astype(int).astype(str)
        + "-"
        + end.astype(int).astype(str)
    )


def short_tf(x: str) -> str:
    mapping = {
        "EBOX/CAGCTG/CACCTG": "E-box",
        "CREB/ATF/3": "CREB",
        "DRE/3": "DRE",
        "HD/16": "HD",
        "KNI/1": "KNI",
        "MAF/2": "MAF",
        "SREBP/2": "SREBP",
        "OHLER1": "Ohler1",
        "OHLER7": "Ohler7",
    }
    return mapping.get(str(x), str(x))


def load_task(root: Path, task: str, folder: str, track: int, color: str) -> TaskData:
    data_dir = root / folder / "Data"
    isa_col = f"isa_t{track}"
    inter_col = f"interaction_t{track}"

    single = pd.read_csv(data_dir / "motif_single_isa.csv")
    combi = pd.read_csv(data_dir / "motif_combi_isa.csv")

    single = single.dropna(subset=[isa_col]).copy()
    single["motif_key"] = motif_key(single["region"], single["tf"], single["start_rel"], single["end_rel"])
    single["center"] = (single["start_rel"].astype(float) + single["end_rel"].astype(float)) / 2
    single = single.sort_values(["region", isa_col], ascending=[True, False])
    selected = select_topk_nonoverlap(single, isa_col, TOP_K)
    selected["rank_in_region"] = selected.groupby("region")[isa_col].rank(method="first", ascending=False).astype(int)
    selected["task"] = task
    selected["track"] = track
    selected["selected_by"] = f"top{TOP_K}_single_ISA"

    selected_keys = set(selected["motif_key"])
    combi = combi.copy()
    combi["motif1_key"] = motif_key(combi["region"], combi["tf1"], combi["start1_rel"], combi["end1_rel"])
    combi["motif2_key"] = motif_key(combi["region"], combi["tf2"], combi["start2_rel"], combi["end2_rel"])
    pairs = combi[
        combi["motif1_key"].isin(selected_keys)
        & combi["motif2_key"].isin(selected_keys)
        & combi[inter_col].notna()
    ].copy()
    pairs["task"] = task
    pairs["track"] = track
    pairs["interaction"] = pairs[inter_col].astype(float)
    pairs["center1"] = (pairs["start1_rel"].astype(float) + pairs["end1_rel"].astype(float)) / 2
    pairs["center2"] = (pairs["start2_rel"].astype(float) + pairs["end2_rel"].astype(float)) / 2
    pairs["bin1"] = np.floor(pairs["center1"] / POSITION_BIN_SIZE).astype(int).clip(0, 4)
    pairs["bin2"] = np.floor(pairs["center2"] / POSITION_BIN_SIZE).astype(int).clip(0, 4)
    pairs["abs_interaction"] = pairs["interaction"].abs()
    return TaskData(task, track, color, selected, pairs)


def select_topk_nonoverlap(single: pd.DataFrame, isa_col: str, top_k: int) -> pd.DataFrame:
    rows = []
    for _, group in single.groupby("region", sort=False):
        kept = []
        for row in group.sort_values(isa_col, ascending=False).itertuples(index=False):
            start, end = int(row.start_rel), int(row.end_rel)
            # Match pair-ISA intuition: selected instances should be non-overlapping and non-abutting.
            ok = all((end < s) or (start > e) for s, e in kept)
            if ok:
                rows.append(row._asdict())
                kept.append((start, end))
            if len(kept) >= top_k:
                break
    return pd.DataFrame(rows)


def build_position_bin_summary(all_pairs: pd.DataFrame) -> pd.DataFrame:
    # Pair interactions are instance-level and have no biological direction from
    # "motif1" to "motif2" here. Aggregate unordered position-bin pairs, then
    # mirror off-diagonal cells for display without duplicating diagonal counts.
    unordered = all_pairs.copy()
    unordered["bin_low"] = np.minimum(unordered["bin1"], unordered["bin2"])
    unordered["bin_high"] = np.maximum(unordered["bin1"], unordered["bin2"])
    summary = (
        unordered.groupby(["task", "bin_low", "bin_high"], observed=True)
        .agg(
            median_interaction=("interaction", "median"),
            mean_interaction=("interaction", "mean"),
            n_pairs=("interaction", "size"),
            median_abs_interaction=("abs_interaction", "median"),
        )
        .reset_index()
    )
    rows = []
    for row in summary.itertuples(index=False):
        base = {
            "task": row.task,
            "median_interaction": row.median_interaction,
            "mean_interaction": row.mean_interaction,
            "n_pairs": row.n_pairs,
            "median_abs_interaction": row.median_abs_interaction,
        }
        rows.append({**base, "bin1": int(row.bin_low), "bin2": int(row.bin_high)})
        if int(row.bin_low) != int(row.bin_high):
            rows.append({**base, "bin1": int(row.bin_high), "bin2": int(row.bin_low)})
    return pd.DataFrame(rows)[
        ["task", "bin1", "bin2", "median_interaction", "mean_interaction", "n_pairs", "median_abs_interaction"]
    ]


def choose_example(selected: pd.DataFrame, pairs: pd.DataFrame) -> str | None:
    counts = (
        selected.groupby("region")
        .agg(
            n_selected=("motif_key", "size"),
            min_center=("center", "min"),
            max_center=("center", "max"),
        )
        .reset_index()
    )
    counts["span"] = counts["max_center"] - counts["min_center"]
    pair_stats = (
        pairs.groupby("region")
        .agg(n_pairs=("interaction", "size"), mean_abs_interaction=("abs_interaction", "mean"))
        .reset_index()
    )
    stats = counts.merge(pair_stats, on="region", how="inner")
    stats = stats[
        (stats["n_selected"] >= TOP_K)
        & (stats["n_pairs"] >= min(8, TOP_K * (TOP_K - 1) // 2))
        & (stats["span"] >= MIN_EXAMPLE_SPAN_BP)
    ].copy()
    if stats.empty:
        stats = counts.merge(pair_stats, on="region", how="inner")
    if stats.empty:
        return None
    stats["example_score"] = stats["mean_abs_interaction"] * np.log1p(stats["n_pairs"]) * np.sqrt(stats["span"].clip(lower=1))
    return stats.sort_values(["example_score", "mean_abs_interaction"], ascending=False).iloc[0]["region"]


def panel_label(ax, label: str) -> None:
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def draw_method_schematic(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(-0.02, 1.08, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
    ax.set_title("Position-aware selected-motif map", fontsize=7, pad=8)

    y = 0.72
    ax.hlines(y, 0.08, 0.92, color="#CFCFCF", lw=1.5)
    xs = [0.18, 0.28, 0.43, 0.58, 0.72, 0.83]
    scores = [0.72, 0.54, 0.90, 0.62, 0.82, 0.48]
    colors = ["#BA611B", "#BA611B", "#148A6A", "#148A6A", "#166C9C", "#166C9C"]
    for i, (x, score, c) in enumerate(zip(xs, scores, colors), start=1):
        ax.add_patch(plt.Rectangle((x - 0.025, y - 0.055), 0.05, 0.11, facecolor=c, edgecolor="#4D4D4D", lw=0.45))
        ax.text(x, y + 0.085, f"M{i}", ha="center", fontsize=5.5)
        ax.plot([x, x], [y - 0.08, y - 0.14 - score * 0.08], color=c, lw=1.0)
    ax.text(0.5, 0.94, f"per region: select top {TOP_K} non-overlapping motif instances by single ISA", ha="center", fontsize=5.5)

    for x1, x2, val in [(xs[0], xs[2], -0.5), (xs[2], xs[4], 0.45), (xs[1], xs[5], -0.25)]:
        rad = 0.25 if val > 0 else -0.2
        col = "#C9473D" if val > 0 else "#4E79A7"
        ax.annotate(
            "",
            xy=(x2, y + 0.02),
            xytext=(x1, y + 0.02),
            arrowprops=dict(arrowstyle="-", connectionstyle=f"arc3,rad={rad}", lw=1.4, color=col, alpha=0.9),
        )
    ax.text(0.5, 0.34, "retain position, identity and normalized interaction", ha="center", fontsize=5.6)

    # Mini matrix
    x0, y0, w = 0.32, 0.07, 0.055
    for r in range(6):
        for c in range(6):
            face = "#F5F5F5" if r == c else ("#D4E1F2" if (r + c) % 2 else "#F4D2CD")
            ax.add_patch(plt.Rectangle((x0 + c * w, y0 + (5 - r) * w), w, w, facecolor=face, edgecolor="white", lw=0.4))
    ax.text(0.5, 0.02, "local top-k interaction matrix per region", ha="center", fontsize=5.6)


def draw_position_heatmaps(fig, outer_gs, summary: pd.DataFrame) -> None:
    labels = [f"{i*50}-{(i+1)*50}" for i in range(5)]
    vmax = np.nanpercentile(np.abs(summary["median_interaction"]), 95)
    vmax = max(vmax, 0.05)
    axes = []
    for i, (task, _, _, _) in enumerate(TASKS):
        ax = fig.add_subplot(outer_gs[i])
        axes.append(ax)
        sub = summary[summary["task"] == task]
        mat = np.full((5, 5), np.nan)
        nmat = np.zeros((5, 5), dtype=int)
        for row in sub.itertuples(index=False):
            mat[int(row.bin1), int(row.bin2)] = row.median_interaction
            nmat[int(row.bin1), int(row.bin2)] = int(row.n_pairs)
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_title(task, fontsize=7)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5.2)
        ax.set_yticklabels(labels, fontsize=5.2)
        ax.set_xlabel("Motif 2 position bin (bp)", fontsize=5.8)
        if i == 0:
            ax.set_ylabel("Motif 1 position bin (bp)", fontsize=5.8)
            panel_label(ax, "b")
        else:
            ax.set_ylabel("")
        for r in range(5):
            for c in range(5):
                if nmat[r, c] > 0:
                    ax.text(c, r, str(nmat[r, c]), ha="center", va="center", fontsize=4.8, color="#202020")
    cax = fig.add_axes([0.92, 0.51, 0.012, 0.27])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Median interaction", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.2, length=2)


def draw_example(
    ax_track,
    ax_mat,
    task_data: TaskData,
    region: str,
    label: str | None = None,
    matrix_vmax: float | None = None,
):
    selected = task_data.selected[task_data.selected["region"] == region].copy()
    pairs = task_data.selected_pairs[task_data.selected_pairs["region"] == region].copy()
    selected = selected.sort_values("center").head(TOP_K).copy()
    keys = selected["motif_key"].tolist()
    key_to_idx = {k: i for i, k in enumerate(keys)}

    if label:
        panel_label(ax_track, label)
    ax_track.set_xlim(0, REGION_LEN)
    ax_track.set_ylim(0, 1)
    ax_track.hlines(0.5, 0, REGION_LEN, color="#CFCFCF", lw=1.4)
    max_isa = max(float(selected[f"isa_t{task_data.track}"].max()), 1e-6)
    for row in selected.itertuples(index=False):
        c = task_data.color
        h = 0.14 + 0.16 * (getattr(row, f"isa_t{task_data.track}") / max_isa)
        ax_track.add_patch(
            plt.Rectangle((row.start_rel, 0.5 - h / 2), row.end_rel - row.start_rel, h, facecolor=c, edgecolor="#4D4D4D", lw=0.45)
        )

    label_rows = selected.sort_values("center").reset_index(drop=True)
    if not label_rows.empty:
        label_x = label_rows["center"].astype(float).to_numpy().copy()
        min_sep = 16.0
        for i in range(1, len(label_x)):
            if label_x[i] - label_x[i - 1] < min_sep:
                label_x[i] = label_x[i - 1] + min_sep
        overflow = label_x[-1] - (REGION_LEN - 8)
        if overflow > 0:
            label_x -= overflow
        for i in range(len(label_x) - 2, -1, -1):
            if label_x[i + 1] - label_x[i] < min_sep:
                label_x[i] = label_x[i + 1] - min_sep
        label_x = np.clip(label_x, 8, REGION_LEN - 8)
        y_levels = [0.74, 0.82, 0.90]
        for i, row in enumerate(label_rows.itertuples(index=False)):
            y = y_levels[i % len(y_levels)]
            ax_track.plot([row.center, label_x[i]], [0.66, y - 0.02], color="#9A9A9A", lw=0.35, zorder=1)
            ax_track.text(
                label_x[i],
                y,
                short_tf(row.tf),
                ha="center",
                va="bottom",
                fontsize=4.4,
                rotation=55,
                zorder=2,
            )

    # Draw the strongest selected-pair interactions as position-aware arcs.
    arc_pairs = pairs[
        pairs["motif1_key"].isin(key_to_idx) & pairs["motif2_key"].isin(key_to_idx)
    ].nlargest(4, "abs_interaction")
    for p in arc_pairs.itertuples(index=False):
        x1, x2 = float(p.center1), float(p.center2)
        if x1 == x2:
            continue
        sign_color = "#C9473D" if p.interaction > 0 else "#4E79A7"
        lw = 0.65 + 1.4 * min(abs(float(p.interaction)) / 0.5, 1.0)
        rad = 0.18 if p.interaction > 0 else -0.18
        ax_track.annotate(
            "",
            xy=(x2, 0.58),
            xytext=(x1, 0.58),
            arrowprops=dict(
                arrowstyle="-",
                connectionstyle=f"arc3,rad={rad}",
                color=sign_color,
                lw=lw,
                alpha=0.85,
            ),
        )
    ax_track.set_yticks([])
    ax_track.set_xticks([0, 50, 100, 150, 200, 250])
    ax_track.tick_params(axis="x", labelsize=5.5, length=2)
    ax_track.set_title(f"{task_data.task} example region", fontsize=6.5)
    for spine in ["left", "right", "top"]:
        ax_track.spines[spine].set_visible(False)

    mat = np.full((len(keys), len(keys)), np.nan)
    for p in pairs.itertuples(index=False):
        if p.motif1_key in key_to_idx and p.motif2_key in key_to_idx:
            i, j = key_to_idx[p.motif1_key], key_to_idx[p.motif2_key]
            mat[i, j] = mat[j, i] = p.interaction
    vmax = max(
        float(matrix_vmax) if matrix_vmax is not None else np.nanpercentile(np.abs(mat), 95) if np.isfinite(mat).any() else 0.1,
        0.05,
    )
    masked = np.ma.masked_invalid(mat)
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad("#EFEFEF")
    im = ax_mat.imshow(masked, cmap=cmap, vmin=-vmax, vmax=vmax)
    labs = [f"{short_tf(r.tf)}\n{int(r.center)}" for r in selected.itertuples(index=False)]
    ax_mat.set_xticks(range(len(labs)))
    ax_mat.set_yticks(range(len(labs)))
    ax_mat.set_xticklabels(labs, fontsize=4.8, rotation=45, ha="right")
    ax_mat.set_yticklabels(labs, fontsize=4.8)
    ax_mat.tick_params(length=0)
    ax_mat.set_title("selected-pair matrix", fontsize=6.2)
    for i in range(len(labs)):
        ax_mat.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, facecolor="#F8F8F8", edgecolor="white", lw=0.3))
    return im


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
        }
    )
    root = find_newisa_root()
    out = output_root(root)

    task_data = [load_task(root, *spec) for spec in TASKS]
    selected_all = pd.concat([td.selected for td in task_data], ignore_index=True)
    pairs_all = pd.concat([td.selected_pairs for td in task_data], ignore_index=True)
    summary = build_position_bin_summary(pairs_all)

    selected_all.to_csv(out / "source_data" / "topk_selected_motif_instances.csv", index=False)
    pairs_all.to_csv(out / "source_data" / "topk_selected_pair_interactions.csv", index=False)
    summary.to_csv(out / "source_data" / "position_bin_interaction_summary.csv", index=False)

    examples = []
    for td in task_data:
        region = choose_example(td.selected, td.selected_pairs)
        if region is not None:
            examples.append((td, region))
    pd.DataFrame([{"task": td.task, "region": region} for td, region in examples]).to_csv(
        out / "source_data" / "representative_regions.csv", index=False
    )

    fig = plt.figure(figsize=(7.4, 6.0))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 1.35, 1.35], width_ratios=[1.25, 1, 1, 1], hspace=0.62, wspace=0.48)
    ax_a = fig.add_subplot(gs[0, 0])
    draw_method_schematic(ax_a)
    draw_position_heatmaps(fig, [gs[0, 1], gs[0, 2], gs[0, 3]], summary)

    for idx, (td, region) in enumerate(examples[:3]):
        ax_track = fig.add_subplot(gs[1, idx + 1])
        ax_mat = fig.add_subplot(gs[2, idx + 1])
        draw_example(ax_track, ax_mat, td, region, label="c" if idx == 0 else None)
    ax_blank1 = fig.add_subplot(gs[1, 0])
    ax_blank2 = fig.add_subplot(gs[2, 0])
    for ax in [ax_blank1, ax_blank2]:
        ax.axis("off")
    ax_blank1.text(
        0.02,
        0.85,
        "Representative regions\nshow local position\nand pair interaction",
        ha="left",
        va="top",
        fontsize=6.2,
    )
    ax_blank1.text(
        0.02,
        0.35,
        "Exploratory panel:\nselected top-k motifs only;\nnot a global grammar claim.",
        ha="left",
        va="top",
        fontsize=5.8,
        color="#555555",
    )

    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out / "figures" / f"position_aware_topk_interaction_architecture.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    fig_hm = plt.figure(figsize=(6.8, 2.15))
    gs_hm = fig_hm.add_gridspec(1, 3, wspace=0.46)
    draw_position_heatmaps(fig_hm, [gs_hm[0, 0], gs_hm[0, 1], gs_hm[0, 2]], summary)
    fig_hm.suptitle("Position-bin aggregation of selected top-k motif-pair interactions", fontsize=7.5, y=1.04)
    for ext in ["png", "svg", "pdf"]:
        fig_hm.savefig(out / "figures" / f"position_bin_interaction_heatmaps.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig_hm)

    fig_ex = plt.figure(figsize=(6.9, 3.8))
    gs_ex = fig_ex.add_gridspec(2, 3, height_ratios=[0.85, 1.25], hspace=0.72, wspace=0.55)
    for idx, (td, region) in enumerate(examples[:3]):
        ax_track = fig_ex.add_subplot(gs_ex[0, idx])
        ax_mat = fig_ex.add_subplot(gs_ex[1, idx])
        draw_example(ax_track, ax_mat, td, region, label="a" if idx == 0 else None)
    fig_ex.suptitle("Representative position-aware selected-motif interaction maps", fontsize=7.5, y=1.02)
    for ext in ["png", "svg", "pdf"]:
        fig_ex.savefig(out / "figures" / f"representative_topk_region_examples.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig_ex)

    print("Input:", root)
    print("Output:", out)
    print("Selected motif instances:", len(selected_all))
    print("Selected pair interactions:", len(pairs_all))
    print("Examples:", [(td.task, region) for td, region in examples])


if __name__ == "__main__":
    main()
