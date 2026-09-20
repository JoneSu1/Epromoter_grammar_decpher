"""Fig. 3e: shared-motif markers on core-promoter contribution logos.

This panel belongs to the Shared_motif module. Motif markers are read from the
shared-motif Fi-NeMo stage2 core-promoter scans stored under:

    Shared_motif/plot/data/raw/stage2_core_promoter/{HK,DEV,CAGE_NEW}/hits.tsv

The logo matrix uses the matching Fi-NeMo input arrays from the mounted G: drive
because the local Shared_motif copy contains the scan tables but not the large
finemo_input.npz arrays. Matrix plotted by logomaker:

    one_hot sequence * contribution
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import logomaker
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PLOT = ROOT / "plot"
OUT = PLOT / "output"
QA = PLOT / "qa" / "polished_figure_code" / "figure3e"
TRACE = PLOT / "data" / "intermediate" / "fig3e_core_promoter_logo_trace.csv"

LOCAL_STAGE2 = PLOT / "data" / "raw" / "stage2_core_promoter"
GDRIVE_STAGE2 = Path(
    r"G:\我的云端硬盘\DeepEpromote\Drosophila\Motif_cluster"
    r"\ic_trimmed_results\finemo_validation_scans\stage2_core_promoter"
)
PROM_RUN = Path(
    r"G:\我的云端硬盘\DeepEpromote\Drosophila\Motif_cluster"
    r"\prom_scan_results\prom_run_20260415_041448"
)
PROM_INPUT_INFO = PROM_RUN / "prom_input_info.tsv"
PROMOTER_PREDICTIONS = Path(
    r"G:\我的云端硬盘\DeepEpromote\Drosophila\DeepCAGE"
    r"\DATA\PROMOTER_Dominant_Predictions_Splits.tsv"
)

INPUTS = {
    "HK": GDRIVE_STAGE2 / "finemo_input" / "HK" / "finemo_input.npz",
    "DEV": GDRIVE_STAGE2 / "finemo_input" / "DEV" / "finemo_input.npz",
    "CAGE": GDRIVE_STAGE2 / "finemo_input" / "CAGE_NEW" / "finemo_input.npz",
}
HITS = {
    "HK": LOCAL_STAGE2 / "HK" / "hits.tsv",
    "DEV": LOCAL_STAGE2 / "DEV" / "hits.tsv",
    "CAGE": LOCAL_STAGE2 / "CAGE_NEW" / "hits.tsv",
}
REGIONS = GDRIVE_STAGE2 / "finemo_input" / "HK" / "regions.bed"
PRED_H5 = {
    "HK": PROM_RUN / "HK" / "attributions.h5",
    "DEV": PROM_RUN / "DEV" / "attributions.h5",
    "CAGE": PROM_RUN / "CAGE" / "attributions.h5",
}

SUBSET_REGIONS = {
    "HK-specific": PROM_RUN / "finemo_subset_scans_separated" / "HK_Model" / "HK_only" / "input" / "regions.bed",
    "DEV-specific": PROM_RUN / "finemo_subset_scans_separated" / "DEV_Model" / "DEV_only" / "input" / "regions.bed",
    "Sharing": PROM_RUN / "finemo_subset_scans_separated" / "HK_Model" / "Shared" / "input" / "regions.bed",
}

TASK_ORDER = ["HK", "DEV", "CAGE"]
CLASS_ORDER = ["HK-specific", "DEV-specific", "Sharing"]
TASK_COLORS = {"HK": "#0072B2", "DEV": "#009E73", "CAGE": "#D55E00"}
BASE_COLORS = {"A": "#109618", "C": "#3366CC", "G": "#FF9900", "T": "#DC3912"}

MOTIF_LABELS = {
    "pos_patterns.pattern_000": "DRE/3",
    "pos_patterns.pattern_001": "Ohler1",
    "pos_patterns.pattern_002": "ATA",
    "pos_patterns.pattern_003": "MAF/2",
    "pos_patterns.pattern_004": "Ohler7",
    "pos_patterns.pattern_005": "SREBP/2",
    "pos_patterns.pattern_006": "kni/1",
    "pos_patterns.pattern_007": "CREB/ATF3",
    "pos_patterns.pattern_008": "Ebox",
    "pos_patterns.pattern_009": "HD/16",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.75,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


@dataclass(frozen=True)
class Example:
    class_label: str
    peak_id: int
    region_name: str
    genomic_id: str
    pred_hk: float
    pred_dev: float
    pred_cage: float
    selection_score: float
    n_hits_total: int


def ensure_dirs() -> None:
    for suffix in ["png", "svg", "pdf", "tiff"]:
        (OUT / suffix).mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    TRACE.parent.mkdir(parents=True, exist_ok=True)


def check_inputs() -> None:
    paths = [
        *INPUTS.values(),
        *HITS.values(),
        *PRED_H5.values(),
        REGIONS,
        PROM_INPUT_INFO,
        PROMOTER_PREDICTIONS,
        *SUBSET_REGIONS.values(),
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required Fig3e inputs:\n" + "\n".join(missing))


def read_prom_indices(path: Path) -> list[int]:
    regions = pd.read_csv(path, sep="\t", header=None)
    out: list[int] = []
    for name in regions.iloc[:, 3].astype(str):
        match = re.search(r"_(\d+)$", name)
        if match:
            out.append(int(match.group(1)))
    return out


def load_region_names() -> dict[int, str]:
    regions = pd.read_csv(REGIONS, sep="\t", header=None)
    return {idx: str(name) for idx, name in enumerate(regions.iloc[:, 3].astype(str))}


def load_prediction_values() -> dict[int, dict[str, float]]:
    trace_ids = sorted({idx for path in SUBSET_REGIONS.values() for idx in read_prom_indices(path)})
    preds: dict[int, dict[str, float]] = {idx: {} for idx in trace_ids}
    for task, path in PRED_H5.items():
        with h5py.File(path, "r") as h5:
            pred = h5[task]["pred"]
            for idx in trace_ids:
                preds[idx][task] = float(pred[idx])
    return preds


def load_genomic_ids() -> dict[int, str]:
    prom = pd.read_csv(PROM_INPUT_INFO, sep="\t", usecols=["seq"])
    pred = pd.read_csv(
        PROMOTER_PREDICTIONS,
        sep="\t",
        usecols=["chrom", "start", "end", "strand", "seq"],
    )
    pred = pred.drop_duplicates("seq", keep="first").set_index("seq")
    genomic_ids: dict[int, str] = {}
    for idx, seq in prom["seq"].items():
        if seq in pred.index:
            row = pred.loc[seq]
            genomic_ids[int(idx)] = f"{row['chrom']}:{int(row['start'])}-{int(row['end'])}({row['strand']})"
        else:
            genomic_ids[int(idx)] = f"prom_{idx}"
    return genomic_ids


def genomic_id_for_display(genomic_id: str) -> str:
    match = re.match(r"([^:]+):(\d+)-(\d+)\(([+-])\)$", genomic_id)
    if not match:
        return genomic_id
    chrom, start, end, strand = match.groups()
    strand_label = "plus" if strand == "+" else "minus"
    return f"{chrom}_{start}_{end}_{strand_label}"


def genomic_id_for_panel(genomic_id: str) -> str:
    match = re.match(r"([^:]+):(\d+)-(\d+)\(([+-])\)$", genomic_id)
    if not match:
        return genomic_id
    chrom, start, end, strand = match.groups()
    return f"{chrom}:{start}-{end}{strand}"


def load_hits() -> dict[str, pd.DataFrame]:
    usecols = [
        "start_untrimmed",
        "end_untrimmed",
        "motif_name",
        "hit_importance",
        "hit_coefficient",
        "peak_id",
    ]
    out: dict[str, pd.DataFrame] = {}
    for task, path in HITS.items():
        df = pd.read_csv(path, sep="\t", usecols=usecols)
        df["peak_id"] = df["peak_id"].astype(int)
        df["hit_importance"] = pd.to_numeric(df["hit_importance"], errors="coerce").fillna(0.0)
        df["hit_coefficient"] = pd.to_numeric(df["hit_coefficient"], errors="coerce").fillna(0.0)
        out[task] = df
    return out


def select_examples(
    hits_by_task: dict[str, pd.DataFrame],
    region_names: dict[int, str],
    genomic_ids: dict[int, str],
    preds: dict[int, dict[str, float]],
) -> list[Example]:
    """Choose one representative sequence per biological group.

    The candidate sets come from the existing separated promoter run. The score
    is computed only from the shared-motif stage2 scan hits loaded from the
    Shared_motif module.
    """
    examples: list[Example] = []
    for class_label in CLASS_ORDER:
        candidates = read_prom_indices(SUBSET_REGIONS[class_label])
        ranked: list[tuple[int, float, int, int]] = []
        for peak_id in candidates:
            total_hits = 0
            total_importance = 0.0
            task_presence = 0
            for task in TASK_ORDER:
                sub = hits_by_task[task].loc[hits_by_task[task]["peak_id"].eq(peak_id)]
                n_hits = len(sub)
                total_hits += n_hits
                if n_hits:
                    task_presence += 1
                    total_importance += float(sub["hit_importance"].nlargest(min(4, n_hits)).sum())
            if total_hits:
                ranked.append((task_presence, total_importance, total_hits, peak_id))
        if not ranked:
            raise RuntimeError(f"No shared-motif scan hits found for {class_label}.")
        ranked.sort(reverse=True)
        task_presence, score, n_hits, peak_id = ranked[0]
        examples.append(
            Example(
                class_label=class_label,
                peak_id=int(peak_id),
                region_name=region_names.get(int(peak_id), f"prom_{peak_id}"),
                genomic_id=genomic_ids.get(int(peak_id), f"prom_{peak_id}"),
                pred_hk=preds[int(peak_id)]["HK"],
                pred_dev=preds[int(peak_id)]["DEV"],
                pred_cage=preds[int(peak_id)]["CAGE"],
                selection_score=float(score),
                n_hits_total=int(n_hits),
            )
        )
    return examples


def load_logo_arrays() -> dict[str, dict[str, np.ndarray]]:
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for task, path in INPUTS.items():
        z = np.load(path)
        seq = z["sequences"]
        contrib = z["contributions"]
        if seq.shape != contrib.shape:
            raise ValueError(f"{task} sequence/contribution shape mismatch: {seq.shape} vs {contrib.shape}")
        arrays[task] = {"seq": seq, "contrib": contrib}
    return arrays


def logo_df(arrays: dict[str, np.ndarray], peak_id: int) -> pd.DataFrame:
    actual = arrays["seq"][peak_id].astype(float) * arrays["contrib"][peak_id].astype(float)
    return pd.DataFrame(actual.T, columns=list("ACGT"))


def overlap_bp(first: tuple[int, int], second: tuple[int, int]) -> int:
    return max(0, min(first[1], second[1]) - max(first[0], second[0]))


def top_motifs(hits: pd.DataFrame, peak_id: int, max_n: int = 4) -> list[tuple[int, int, str, float]]:
    sub = hits.loc[hits["peak_id"].eq(peak_id)].copy()
    if sub.empty:
        return []
    sub = sub.sort_values(["hit_importance", "hit_coefficient"], ascending=False)
    kept: list[tuple[int, int, str, float]] = []
    for row in sub.itertuples(index=False):
        start = max(0, min(248, int(row.start_untrimmed)))
        end = max(0, min(248, int(row.end_untrimmed)))
        if end <= start:
            continue
        if any(overlap_bp((start, end), (ks, ke)) > 2 for ks, ke, _, _ in kept):
            continue
        label = MOTIF_LABELS.get(str(row.motif_name), str(row.motif_name).replace("pos_patterns.", ""))
        kept.append((start, end, label, float(row.hit_importance)))
        if len(kept) >= max_n:
            break
    return sorted(kept, key=lambda item: (item[0], item[1]))


def draw_motif_markers(ax: plt.Axes, motifs: list[tuple[int, int, str, float]], y_top: float, y_span: float) -> None:
    occupied: list[float] = []
    # Keep motif labels inside the logo area but away from the top frame.
    # Overlapping labels are stacked downward so enlarged text does not hit
    # the panel border.
    base_y = y_top - y_span * 0.27
    step = y_span * 0.105
    for start, end, label, _score in motifs:
        ax.axvspan(start, end, facecolor="#9ECAE1", alpha=0.28, edgecolor="none", zorder=-10)
        center = (start + end) / 2.0
        half_width = max(14.0, len(label) * 4.0)
        left = center - half_width
        right = center + half_width
        level = 0
        while level < len(occupied) and left <= occupied[level] + 2:
            level += 1
        if level == len(occupied):
            occupied.append(right)
        else:
            occupied[level] = right
        y = base_y - step * level
        ax.plot([start, end], [y - step * 0.12, y - step * 0.12], color="#2166AC", lw=1.7, solid_capstyle="butt")
        ax.text(center, y, label, ha="center", va="bottom", fontsize=8.2, color="black")


def draw_track(
    ax: plt.Axes,
    df: pd.DataFrame,
    motifs: list[tuple[int, int, str, float]],
    task: str,
    pred_value: float,
    panel_label: str,
    show_ylabel: bool,
    show_xlabel: bool,
    ylim: tuple[float, float],
) -> None:
    logo = logomaker.Logo(
        df,
        ax=ax,
        color_scheme=BASE_COLORS,
        vpad=0.01,
        width=0.92,
        baseline_width=0.45,
    )
    logo.style_spines(visible=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.55)
    ax.axhline(0, color="0.25", lw=0.55, zorder=0)
    draw_motif_markers(ax, motifs, ylim[1], ylim[1] - ylim[0])
    ax.text(
        0.01,
        1.26,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.4,
        color="black",
        linespacing=1.08,
    )
    ax.text(
        0.99,
        1.055,
        f"{task} pred={pred_value:.2f}",
        transform=ax.transAxes,
        color=TASK_COLORS[task],
        ha="right",
        va="bottom",
        fontsize=11.1,
        fontweight="bold",
    )
    ax.set_xlim(0, 248)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_yticks([0, round(ylim[1] * 0.55, 2)])
    ax.tick_params(axis="both", labelsize=9, length=2)
    ax.set_ylabel("Contribution" if show_ylabel else "", fontsize=9.2)
    ax.set_xlabel("Position in region" if show_xlabel else "", fontsize=9.2)


def compute_ylim(examples: list[Example], arrays_by_task: dict[str, dict[str, np.ndarray]]) -> tuple[float, float]:
    vals = []
    for example in examples:
        for task in TASK_ORDER:
            vals.append(logo_df(arrays_by_task[task], example.peak_id).to_numpy().reshape(-1))
    arr = np.concatenate(vals)
    lo = min(float(np.nanpercentile(arr, 0.5)) * 1.25, -0.04)
    hi = max(float(np.nanpercentile(arr, 99.5)) * 2.4, 0.12)
    return lo, hi


def save_all(fig: plt.Figure, stem: str) -> list[Path]:
    paths = [
        OUT / "png" / f"{stem}.png",
        OUT / "svg" / f"{stem}.svg",
        OUT / "pdf" / f"{stem}.pdf",
        OUT / "tiff" / f"{stem}.tiff",
        QA / f"{stem}_preview.png",
    ]
    fig.savefig(paths[0], bbox_inches="tight", dpi=600)
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], bbox_inches="tight")
    fig.savefig(paths[3], bbox_inches="tight", dpi=600)
    fig.savefig(paths[4], bbox_inches="tight", dpi=600)
    return paths


def main() -> None:
    ensure_dirs()
    check_inputs()
    region_names = load_region_names()
    hits_by_task = load_hits()
    arrays_by_task = load_logo_arrays()
    genomic_ids = load_genomic_ids()
    preds = load_prediction_values()
    examples = select_examples(hits_by_task, region_names, genomic_ids, preds)
    ylim = compute_ylim(examples, arrays_by_task)

    fig, axes = plt.subplots(len(TASK_ORDER), len(CLASS_ORDER), figsize=(7.25, 4.05), sharex=True, sharey=False)
    trace_rows = []
    for col_idx, example in enumerate(examples):
        column_label = (
            f"{example.class_label}\n"
            f"{genomic_id_for_display(example.genomic_id)}"
        )
        for row_idx, task in enumerate(TASK_ORDER):
            motifs = top_motifs(hits_by_task[task], example.peak_id)
            pred_value = {"HK": example.pred_hk, "DEV": example.pred_dev, "CAGE": example.pred_cage}[task]
            draw_track(
                axes[row_idx, col_idx],
                logo_df(arrays_by_task[task], example.peak_id),
                motifs,
                task,
                pred_value,
                panel_label=column_label if row_idx == 0 else "",
                show_ylabel=col_idx == 0,
                show_xlabel=row_idx == len(TASK_ORDER) - 1,
                ylim=ylim,
            )
            if row_idx < len(TASK_ORDER) - 1:
                axes[row_idx, col_idx].tick_params(labelbottom=False)
            trace_rows.append(
                {
                    "class": example.class_label,
                    "task": task,
                    "peak_id": example.peak_id,
                    "region_name": example.region_name,
                    "genomic_id": example.genomic_id,
                    "pred_HK": example.pred_hk,
                    "pred_DEV": example.pred_dev,
                    "pred_CAGE": example.pred_cage,
                    "selection_score_top_shared_motif_importance": example.selection_score,
                    "n_shared_motif_hits_total_all_tasks": example.n_hits_total,
                    "logo_input_npz": str(INPUTS[task]),
                    "shared_motif_hits_tsv": str(HITS[task]),
                    "n_motifs_displayed": len(motifs),
                    "motifs_displayed": ";".join(m[2] for m in motifs),
                }
            )

    fig.subplots_adjust(left=0.075, right=0.99, top=0.86, bottom=0.13, wspace=0.18, hspace=1.08)

    stem = "figure3e_core_promoter_shared_motif_logos"
    outputs = save_all(fig, stem)
    plt.close(fig)

    pd.DataFrame(trace_rows).to_csv(TRACE, index=False)
    manifest = {
        "panel": "Fig3e",
        "module": "Shared_motif",
        "motif_marker_source": {task: str(path) for task, path in HITS.items()},
        "logo_matrix_source": {task: str(path) for task, path in INPUTS.items()},
        "prediction_source": {task: str(path) for task, path in PRED_H5.items()},
        "genomic_id_source": str(PROMOTER_PREDICTIONS),
        "selection_rule": (
            "HK-specific from HK_Model/HK_only, DEV-specific from DEV_Model/DEV_only, "
            "Sharing from HK_Model/Shared; representative selected by largest summed top "
            "shared-motif hit_importance across HK/DEV/CAGE."
        ),
        "outputs": [str(path) for path in outputs],
    }
    (TRACE.parent / "fig3e_core_promoter_logo_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Wrote trace:", TRACE)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
