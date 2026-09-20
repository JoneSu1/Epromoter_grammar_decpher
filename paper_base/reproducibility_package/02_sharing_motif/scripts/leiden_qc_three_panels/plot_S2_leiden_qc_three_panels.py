#!/usr/bin/env python
"""
Polish Supplementary Leiden QC figure for shared-motif analysis.

Input:
  formal_script/sharing_motif/data/supplement_leiden_qc/

Output:
  formal_script/sharing_motif/outputs/supplement_leiden_qc/polished/

Scientific contract:
  - A: Leiden modularity is stable across random seeds.
  - B: density adaptation sharpens the CWM similarity landscape into coherent
       meta-cluster blocks.
  - C: sequence-level validation supports the final ten meta-clusters.

The task-composition bar from the raw Colab output is intentionally not used
in the polished primary supplement to avoid repeating the main Fig. 3C
fraction-within-task panel.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "supplement_leiden_qc"
FINAL_DATA = DATA / "final_three_panels"
OUT = ROOT / "outputs" / "supplement_leiden_qc" / "polished"


COLORS = {
    "neutral": "#3A3A3A",
    "muted": "#6E6E6E",
    "blue": "#4E79A7",
    "blue_dark": "#1F5A92",
    "red": "#D64F4F",
    "green": "#59A14F",
    "teal": "#00A88A",
    "grid": "#D9D9D9",
    "boundary": "#FFFFFF",
}


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "savefig.facecolor": "white",
    }
)


def mm_to_in(x: float) -> float:
    return x / 25.4


def read_inputs() -> dict[str, pd.DataFrame]:
    def csv_any(*names: str, index_col=None) -> pd.DataFrame | None:
        for base in (FINAL_DATA, DATA):
            for name in names:
                path = base / name
                if path.exists():
                    return pd.read_csv(path, index_col=index_col)
        return None

    quality = csv_any("S2a_leiden_quality_history.csv", "leiden_quality_history.csv")
    entity = csv_any("common_leiden_entity_manifest.csv", "leiden_entity_manifest.csv")
    report = csv_any("common_leiden_meta_cluster_report.csv", "leiden_meta_cluster_report.csv")
    validation = csv_any("S2c_sequence_level_cluster_validation.csv", "sequence_level_cluster_validation.csv")
    order = csv_any("density_adaptation_heatmap_order.csv")
    raw = csv_any("raw_cwm_affinity_matrix.csv", index_col=0)
    adapted = csv_any("density_adapted_affinity_matrix.csv", index_col=0)
    norm_qc = csv_any("S2b_median_l2_norm_per_entity_qc.csv", "median_l2_norm_per_entity_qc.csv")
    assignment_pairwise = csv_any(
        "S2a_leiden_assignment_pairwise_ari_nmi.csv",
        "leiden_assignment_pairwise_ari_nmi.csv",
    )
    assignment_history = csv_any(
        "S2a_leiden_assignment_seed_history.csv",
        "leiden_assignment_seed_history.csv",
    )
    median_scale = csv_any("S2b_median_norm_task_scale.csv", "median_norm_task_scale.csv")

    required = {
        "quality": quality,
        "entity": entity,
        "report": report,
        "validation": validation,
        "norm_qc": norm_qc,
        "assignment_pairwise": assignment_pairwise,
        "assignment_history": assignment_history,
        "median_scale": median_scale,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise FileNotFoundError(f"Missing required S2 Leiden QC inputs: {', '.join(missing)}")
    return {
        "quality": quality,
        "entity": entity,
        "report": report,
        "validation": validation,
        "order": order,
        "raw": raw,
        "adapted": adapted,
        "norm_qc": norm_qc,
        "assignment_pairwise": assignment_pairwise,
        "assignment_history": assignment_history,
        "median_scale": median_scale,
    }


def order_matrices(
    raw: pd.DataFrame, adapted: pd.DataFrame, order: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # The Colab export stores sorted integer positions into the original matrix.
    pos = order["sorted_index"].astype(int).to_numpy()
    labels = order["meta_label"].astype(int).to_numpy()
    raw_values = raw.to_numpy()[np.ix_(pos, pos)]
    adapted_values = adapted.to_numpy()[np.ix_(pos, pos)]
    return raw_values, adapted_values, labels


def cluster_boundaries(labels: np.ndarray) -> list[int]:
    changes = np.where(labels[1:] != labels[:-1])[0] + 1
    return changes.tolist()


def add_panel_label(ax, label: str, x: float = -0.08, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="black",
    )


def panel_seed_stability(ax, quality: pd.DataFrame) -> dict[str, float]:
    x = quality["seed_iteration"].to_numpy()
    modularity = quality["modularity"].to_numpy()
    mean = float(np.mean(modularity))
    sd = float(np.std(modularity, ddof=1)) if len(modularity) > 1 else 0.0
    y = (modularity - mean) * 1e6
    ymax_abs = max(float(np.max(np.abs(y))), 1.0)

    ax.plot(x, y, color=COLORS["blue"], lw=1.1, zorder=2)
    ax.scatter(x, y, s=18, color=COLORS["blue_dark"], edgecolor="white", lw=0.35, zorder=3)

    ax.set_title("Seed stability of Leiden clustering", pad=5, fontsize=8.5, loc="center")
    ax.set_xlabel("Random-seed run")
    ax.set_ylabel("Δ modularity from mean (1e-6 units)")
    ax.set_xlim(0.5, len(x) + 0.5)
    ax.set_ylim(-ymax_abs * 1.5, ymax_abs * 1.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(axis="y", color=COLORS["grid"], lw=0.45, alpha=0.8)
    ax.text(
        0.03,
        0.92,
        f"n={len(y)} runs\nmean modularity={mean:.6f}\ns.d.={sd:.2e}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=COLORS["neutral"],
    )
    return {"n_seed_runs": int(len(y)), "mean_modularity": mean, "sd_modularity": sd}


def panel_heatmaps(fig, spec, raw_values, adapted_values, labels) -> dict[str, float]:
    sub = GridSpecFromSubplotSpec(
        1, 4, subplot_spec=spec, width_ratios=[1, 0.045, 1, 0.045], wspace=0.09
    )
    ax_raw = fig.add_subplot(sub[0, 0])
    cax_raw = fig.add_subplot(sub[0, 1])
    ax_ad = fig.add_subplot(sub[0, 2])
    cax_ad = fig.add_subplot(sub[0, 3])

    raw_upper = float(np.quantile(raw_values, 0.995))
    adapted_upper = float(np.quantile(adapted_values, 0.995))
    im0 = ax_raw.imshow(raw_values, cmap="viridis", vmin=0, vmax=raw_upper, interpolation="nearest", rasterized=True)
    im1 = ax_ad.imshow(adapted_values, cmap="viridis", vmin=0, vmax=adapted_upper, interpolation="nearest", rasterized=True)

    for ax, title in [(ax_raw, "Raw CWM affinity"), (ax_ad, "Density-adapted affinity")]:
        ax.set_title(title, fontsize=8.5, pad=5, loc="center")
        ax.set_xticks([])
        ax.set_yticks([])
        for b in cluster_boundaries(labels):
            ax.axhline(b - 0.5, color=COLORS["boundary"], lw=0.45, alpha=0.8)
            ax.axvline(b - 0.5, color=COLORS["boundary"], lw=0.45, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.45)
            spine.set_color("#BBBBBB")

    cb0 = fig.colorbar(im0, cax=cax_raw)
    cb1 = fig.colorbar(im1, cax=cax_ad)
    for cb, label in [(cb0, "Similarity"), (cb1, "Adapted weight")]:
        cb.ax.tick_params(labelsize=6, length=2)
        cb.set_label(label, fontsize=6.5, labelpad=3)

    ax_raw.text(
        -0.13,
        1.09,
        "B",
        transform=ax_raw.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    return {
        "raw_affinity_q995": raw_upper,
        "density_adapted_affinity_q995": adapted_upper,
    }


def panel_sequence_validation(ax, validation: pd.DataFrame, report: pd.DataFrame) -> dict[str, float]:
    df = validation.merge(report[["MC_ID", "Family_Type", "Total_Weight"]], on="MC_ID", how="left")
    df = df.sort_values("Mean_Seq_Similarity", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))

    motif_min = float(df["Motif_Count"].min())
    motif_max = float(df["Motif_Count"].max())
    if motif_max > motif_min:
        sizes = 32 + (df["Motif_Count"] - motif_min) / (motif_max - motif_min) * (115 - 32)
    else:
        sizes = np.repeat(70, len(df))

    ax.hlines(y, df["Min_Seq_Similarity"], df["Max_Seq_Similarity"], color="#BFBFBF", lw=1.2, zorder=1)
    ax.scatter(df["Mean_Seq_Similarity"], y, s=sizes, color=COLORS["blue"], edgecolor="white", lw=0.55, zorder=3)
    ax.axvline(df["Mean_Seq_Similarity"].median(), color=COLORS["red"], lw=0.8, ls="--", zorder=0)

    ax.set_title("Sequence-level validation of meta-clusters", pad=5, fontsize=8.5, loc="center")
    ax.set_xlabel("Mean pairwise sequence similarity")
    ax.set_ylabel("Meta-cluster")
    ax.set_yticks(y)
    ax.set_yticklabels(df["MC_ID"])
    ax.set_xlim(0.25, 0.72)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.45, alpha=0.8)
    return {
        "min_mean_sequence_similarity": float(df["Mean_Seq_Similarity"].min()),
        "median_mean_sequence_similarity": float(df["Mean_Seq_Similarity"].median()),
        "max_mean_sequence_similarity": float(df["Mean_Seq_Similarity"].max()),
    }


def save_all(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def save_single_seed_stability(quality: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(mm_to_in(89), mm_to_in(58)))
    fig.subplots_adjust(left=0.22, right=0.97, bottom=0.22, top=0.86)
    panel_seed_stability(ax, quality)
    save_all(fig, OUT / "supplement_leiden_qc_seed_stability")
    plt.close(fig)


def save_single_assignment_stability(pairwise: pd.DataFrame, history: pd.DataFrame | None = None) -> dict[str, float]:
    """Polished assignment-stability panel using pairwise ARI and NMI across seeds."""
    metrics = [("ARI", "Adjusted Rand index"), ("NMI", "Normalized mutual information")]
    fig, axes = plt.subplots(1, 2, figsize=(mm_to_in(89), mm_to_in(54)), sharey=True)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.24, top=0.78, wspace=0.2)
    color = COLORS["blue"]
    stats: dict[str, float] = {}

    for ax, (col, title) in zip(axes, metrics):
        vals = pairwise[col].to_numpy(dtype=float)
        jitter = np.linspace(-0.13, 0.13, len(vals))
        ax.boxplot(
            [vals],
            positions=[0],
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=0.85),
            boxprops=dict(facecolor=color, alpha=0.42, edgecolor="#555555", linewidth=0.7),
            whiskerprops=dict(color="#777777", linewidth=0.75),
            capprops=dict(color="#777777", linewidth=0.75),
        )
        ax.scatter(jitter, vals, s=8.5, color=color, alpha=0.42, edgecolor="white", linewidth=0.18, rasterized=True)
        ax.set_title(title, fontsize=8, pad=5, loc="center")
        ax.set_xticks([0])
        ax.set_xticklabels([f"seed pairs\nn={len(vals)}"])
        ax.set_xlim(-0.32, 0.32)
        ax.set_ylim(0.95, 1.005)
        ax.grid(axis="y", color=COLORS["grid"], lw=0.45, alpha=0.85)
        ax.text(
            0.5,
            0.1,
            f"median={np.median(vals):.3f}\nmin={np.min(vals):.3f}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=COLORS["neutral"],
        )
        stats[f"{col}_median"] = float(np.median(vals))
        stats[f"{col}_min"] = float(np.min(vals))
        stats[f"{col}_max"] = float(np.max(vals))

    axes[0].set_ylabel("Pairwise assignment similarity")
    title = "Leiden assignments are identical across random seeds"
    if history is not None and "n_clusters" in history.columns:
        n_seed = int(history.shape[0])
        n_cluster_unique = sorted(history["n_clusters"].unique().tolist())
        title += f" ({n_seed} seeds; {n_cluster_unique[0]} clusters)"
        stats["n_seed_runs"] = n_seed
        stats["n_clusters_min"] = int(min(n_cluster_unique))
        stats["n_clusters_max"] = int(max(n_cluster_unique))
    fig.suptitle(title, x=0.5, y=0.955, ha="center", va="top", fontsize=8.8)
    save_all(fig, OUT / "supplement_leiden_qc_assignment_stability_qc_polished")
    plt.close(fig)
    stats["n_pairwise_seed_comparisons"] = int(pairwise.shape[0])
    return stats


def save_single_density_heatmaps(raw_values: np.ndarray, adapted_values: np.ndarray, labels: np.ndarray) -> None:
    fig = plt.figure(figsize=(mm_to_in(120), mm_to_in(58)))
    gs = GridSpec(
        1,
        4,
        figure=fig,
        width_ratios=[1, 0.045, 1, 0.045],
        left=0.04,
        right=0.98,
        bottom=0.08,
        top=0.82,
        wspace=0.09,
    )
    panel_heatmaps(fig, gs[0, :], raw_values, adapted_values, labels)
    # The helper adds panel label B for composite context; remove only the panel
    # label-like text while retaining axes titles/colorbars.
    for ax in fig.axes:
        for text in list(ax.texts):
            if text.get_text() == "B":
                text.remove()
    save_all(fig, OUT / "supplement_leiden_qc_density_adaptation_heatmaps")
    plt.close(fig)


def save_single_sequence_validation(validation: pd.DataFrame, report: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(mm_to_in(89), mm_to_in(78)))
    fig.subplots_adjust(left=0.24, right=0.97, bottom=0.16, top=0.88)
    panel_sequence_validation(ax, validation, report)
    save_all(fig, OUT / "supplement_leiden_qc_sequence_validation")
    plt.close(fig)


def save_single_median_norm(scale: pd.DataFrame) -> None:
    values = scale.iloc[0].rename(index={"CAGE_NEW": "CAGE"}).sort_values(ascending=False)
    colors = [COLORS["red"] if i == "CAGE" else COLORS["green"] if i == "HK" else COLORS["teal"] for i in values.index]

    fig, ax = plt.subplots(figsize=(mm_to_in(89), mm_to_in(50)))
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.25, top=0.82)
    x = np.arange(len(values))
    ax.bar(x, values.to_numpy(), color=colors, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(values.index)
    ax.set_ylabel("Median L2 scale factor")
    ax.set_title("Task-wise median L2 normalization", fontsize=8.5, pad=5, loc="center")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.45, alpha=0.8)
    for xi, yi in zip(x, values.to_numpy()):
        ax.text(xi, yi + max(values) * 0.035, f"{yi:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, max(values) * 1.22)
    save_all(fig, OUT / "supplement_leiden_qc_median_l2_scale")
    plt.close(fig)


def save_single_median_l2_norm_qc(df_norm: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Distribution QC for raw, task-median-scaled, and final L2 CWM norms."""
    df = df_norm.copy()
    df["task_display"] = df["task_display"].replace({"CAGE_NEW": "CAGE"})
    order = ["CAGE", "HK", "DEV"]
    task_colors = {"CAGE": COLORS["red"], "HK": COLORS["green"], "DEV": COLORS["teal"]}
    stages = [
        ("raw_l2_norm", "Raw CWM norm"),
        ("median_scaled_l2_norm", "After task-median scaling"),
        ("final_l2_norm", "After final L2 normalization"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(mm_to_in(150), mm_to_in(55)), sharex=False)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.22, top=0.78, wspace=0.32)
    summary: dict[str, dict[str, float]] = {}

    for ax, (col, title) in zip(axes, stages):
        stage_summary: dict[str, float] = {}
        data = [df.loc[df["task_display"] == task, col].to_numpy(dtype=float) for task in order]
        bp = ax.boxplot(
            data,
            positions=np.arange(len(order)),
            widths=0.44,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=0.85),
            whiskerprops=dict(color="#777777", linewidth=0.75),
            capprops=dict(color="#777777", linewidth=0.75),
        )
        for patch, task in zip(bp["boxes"], order):
            patch.set_facecolor(task_colors[task])
            patch.set_alpha(0.48)
            patch.set_edgecolor("#555555")
            patch.set_linewidth(0.65)

        ymin, ymax = np.inf, -np.inf
        for xi, task in enumerate(order):
            vals = data[xi]
            ymin = min(ymin, float(np.min(vals)))
            ymax = max(ymax, float(np.max(vals)))
            jitter = np.linspace(-0.055, 0.055, len(vals))
            ax.scatter(
                np.full(len(vals), xi) + jitter,
                vals,
                s=7.5,
                color=task_colors[task],
                alpha=0.48,
                edgecolor="white",
                linewidth=0.18,
                rasterized=True,
                zorder=3,
            )
            med = float(np.median(vals))
            stage_summary[f"{task}_median"] = med
            stage_summary[f"{task}_n"] = int(len(vals))

        ypad = max((ymax - ymin) * 0.11, 0.05)
        ax.set_ylim(ymin - ypad, ymax + ypad)
        for xi, task in enumerate(order):
            vals = data[xi]
            ax.text(
                xi,
                ymax + ypad * 0.18,
                f"n={len(vals)}",
                ha="center",
                va="bottom",
                fontsize=6.2,
                color=COLORS["muted"],
            )
        ax.set_title(title, fontsize=8, pad=5, loc="center")
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order)
        ax.grid(axis="y", color=COLORS["grid"], lw=0.45, alpha=0.85)
        summary[col] = stage_summary

    axes[0].set_ylabel("CWM L2 norm")
    fig.suptitle(
        "Median/L2 normalization controls task-scale differences",
        x=0.5,
        y=0.955,
        ha="center",
        va="top",
        fontsize=9.2,
    )
    save_all(fig, OUT / "supplement_leiden_qc_median_l2_norm_qc_polished")
    plt.close(fig)
    return summary


def main() -> None:
    d = read_inputs()
    save_single_seed_stability(d["quality"])
    save_single_sequence_validation(d["validation"], d["report"])
    save_single_median_norm(d["median_scale"])
    norm_summary = None
    if d["norm_qc"] is not None:
        norm_summary = save_single_median_l2_norm_qc(d["norm_qc"])
    assignment_summary = None
    if d["assignment_pairwise"] is not None:
        assignment_summary = save_single_assignment_stability(d["assignment_pairwise"], d["assignment_history"])

    stats = {}
    stem = OUT / "supplement_leiden_qc_polished_ABD"
    if d["raw"] is not None and d["adapted"] is not None and d["order"] is not None:
        raw_values, adapted_values, labels = order_matrices(d["raw"], d["adapted"], d["order"])
        save_single_density_heatmaps(raw_values, adapted_values, labels)
        fig = plt.figure(figsize=(mm_to_in(183), mm_to_in(128)))
        gs = GridSpec(
            2,
            2,
            figure=fig,
            width_ratios=[0.72, 1.28],
            height_ratios=[0.82, 1.18],
            left=0.055,
            right=0.985,
            bottom=0.07,
            top=0.94,
            hspace=0.43,
            wspace=0.28,
        )
        ax_a = fig.add_subplot(gs[0, 0])
        stats.update(panel_seed_stability(ax_a, d["quality"]))
        add_panel_label(ax_a, "A")
        stats.update(panel_heatmaps(fig, gs[:, 1], raw_values, adapted_values, labels))
        ax_c = fig.add_subplot(gs[1, 0])
        stats.update(panel_sequence_validation(ax_c, d["validation"], d["report"]))
        add_panel_label(ax_c, "C")
        save_all(fig, stem)
        plt.close(fig)

    manifest = {
        "figure": "supplement_leiden_qc_polished_ABD",
        "purpose": "Supplementary QC for shared-motif Leiden/meta-cluster construction.",
        "included_panels": {
            "A": "Leiden modularity across random-seed runs.",
            "B": "Raw and density-adapted CWM affinity matrices sorted by final meta-cluster.",
            "C": "Sequence-level pairwise similarity validation for the ten meta-clusters.",
        },
        "excluded_from_primary_polished_composite": {
            "composition_bar": "Omitted because the main figure already shows task fraction within meta-clusters."
        },
        "source_data": [
            str(DATA / "leiden_quality_history.csv"),
            str(DATA / "raw_cwm_affinity_matrix.csv"),
            str(DATA / "density_adapted_affinity_matrix.csv"),
            str(DATA / "density_adaptation_heatmap_order.csv"),
            str(DATA / "sequence_level_cluster_validation.csv"),
            str(DATA / "leiden_meta_cluster_report.csv"),
        ],
        "outputs": [str(stem.with_suffix(ext)) for ext in [".svg", ".pdf", ".png", ".tiff"]],
        "split_outputs": {
            "seed_stability": [str((OUT / "supplement_leiden_qc_seed_stability").with_suffix(ext)) for ext in [".svg", ".pdf", ".png", ".tiff"]],
            "density_adaptation_heatmaps": [str((OUT / "supplement_leiden_qc_density_adaptation_heatmaps").with_suffix(ext)) for ext in [".svg", ".pdf", ".png", ".tiff"]],
            "sequence_validation": [str((OUT / "supplement_leiden_qc_sequence_validation").with_suffix(ext)) for ext in [".svg", ".pdf", ".png", ".tiff"]],
            "median_l2_scale": [str((OUT / "supplement_leiden_qc_median_l2_scale").with_suffix(ext)) for ext in [".svg", ".pdf", ".png", ".tiff"]],
            "S2a_median_l2_normalization_qc": [str((OUT / "supplement_leiden_qc_median_l2_norm_qc_polished").with_suffix(ext)) for ext in [".svg", ".pdf", ".png", ".tiff"]],
            "S2b_leiden_assignment_stability": [str((OUT / "supplement_leiden_qc_assignment_stability_qc_polished").with_suffix(ext)) for ext in [".svg", ".pdf", ".png", ".tiff"]],
        },
        "summary_statistics": stats,
        "median_l2_norm_qc_summary": norm_summary,
        "assignment_stability_qc_summary": assignment_summary,
    }
    with open(OUT / "supplement_leiden_qc_polished_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
