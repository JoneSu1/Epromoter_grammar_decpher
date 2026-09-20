#!/usr/bin/env python
# coding: utf-8
"""CAGE promoter sequence-level contribution-logo examples for Supplementary Fig. 3."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd

from paths_S3 import CLEAN_DIR, panel_dirs
from style_S3 import DOUBLE_COL_MM, TASK_COLORS, mm_to_in, save_all, set_pub_style


MODULE_ROOT = CLEAN_DIR.parents[1]
PACKAGE_ROOT = MODULE_ROOT.parent
HITS = PACKAGE_ROOT / "02_sharing_motif" / "data" / "raw_reference" / "stage2_core_promoter" / "CAGE_NEW" / "hits.tsv"
PROM_INFO = PACKAGE_ROOT / "frozen_data" / "sharing_motif_logos" / "prom_input_info.tsv"
FINEMO_INPUT = PACKAGE_ROOT / "frozen_data" / "sharing_motif_logos" / "finemo_input" / "CAGE_NEW" / "finemo_input.npz"
PREDICTIONS = (
    PACKAGE_ROOT
    / "01_model_build"
    / "deepcage"
    / "data"
    / "DATA"
    / "PROMOTER_Dominant_Predictions_Splits.tsv"
)
SELECTED_CSV = CLEAN_DIR / "cage_sequence_logo_examples.csv"
BASES = ["A", "C", "G", "T"]
BASE_COLORS = {"A": "#109618", "C": "#3366CC", "G": "#FF9900", "T": "#DC3912"}
MOTIF_LABELS = {
    "pos_patterns.pattern_000": "DRE/3",
    "pos_patterns.pattern_001": "Ohler1",
    "pos_patterns.pattern_004": "Ebox/CAGCTG",
}
PLOT_MOTIF_LABELS = {"Ebox/CAGCTG": "Ebox"}
TARGET_MOTIFS = ["Ohler1", "Ebox/CAGCTG", "DRE/3"]
EXPORT_DPI = 600


@dataclass(frozen=True)
class Example:
    peak_id: int
    chrom: str
    start: int
    end: int
    strand: str
    true_cage: float
    pred_cage: float
    hits: pd.DataFrame


def orient_base_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape[0] == 4:
        return arr.T
    if arr.shape[1] == 4:
        return arr
    raise ValueError(f"Expected base array with one axis of length 4, got {arr.shape}")


def load_logo_array(z: np.lib.npyio.NpzFile, peak_id: int) -> np.ndarray:
    seq = orient_base_array(z["sequences"][peak_id]).astype(float)
    contrib = orient_base_array(z["contributions"][peak_id]).astype(float)
    if seq.shape != contrib.shape:
        raise ValueError(f"sequence/contribution shape mismatch: {seq.shape} vs {contrib.shape}")
    return seq * contrib


def draw_base_letter(ax, base: str, x: float, y0: float, height: float) -> None:
    if abs(height) < 1e-4:
        return
    font_prop = FontProperties(family="Arial", weight="bold")
    path = TextPath((0, 0), base, size=1, prop=font_prop)
    bbox = path.get_extents()
    sx = 0.80 / max(bbox.width, 1e-6)
    sy = abs(height) / max(bbox.height, 1e-6)
    y = y0 if height >= 0 else y0 + height
    transform = Affine2D().scale(sx, sy).translate(x - 0.40, y)
    patch = PathPatch(path, transform=transform + ax.transData, color=BASE_COLORS[base], lw=0)
    ax.add_patch(patch)


def draw_logo(ax, arr: np.ndarray) -> None:
    for pos in range(arr.shape[0]):
        values = arr[pos, :]
        x = pos + 1
        pos_bottom = 0.0
        neg_bottom = 0.0
        for base, value in sorted(zip(BASES, values), key=lambda item: item[1]):
            if value < 0:
                draw_base_letter(ax, base, x, neg_bottom, float(value))
                neg_bottom += float(value)
        for base, value in sorted(zip(BASES, values), key=lambda item: item[1]):
            if value > 0:
                draw_base_letter(ax, base, x, pos_bottom, float(value))
                pos_bottom += float(value)


def pick_target_hits(group: pd.DataFrame) -> pd.DataFrame:
    picked = []
    for motif in TARGET_MOTIFS:
        sub = group[group["motif_tf"].eq(motif)].sort_values(
            ["hit_importance", "hit_coefficient"], ascending=False
        )
        if sub.empty:
            raise ValueError(f"Missing target motif {motif} in peak {group['peak_id'].iloc[0]}")
        picked.append(sub.iloc[0])
    return pd.DataFrame(picked).reset_index(drop=True)


def load_examples() -> tuple[dict[str, Example], pd.DataFrame]:
    hits = pd.read_csv(
        HITS,
        sep="\t",
        usecols=[
            "start",
            "end",
            "start_untrimmed",
            "end_untrimmed",
            "motif_name",
            "hit_importance",
            "hit_coefficient",
            "strand",
            "peak_id",
        ],
    )
    hits = hits[hits["motif_name"].isin(MOTIF_LABELS)].copy()
    hits["motif_tf"] = hits["motif_name"].map(MOTIF_LABELS)
    prom = pd.read_csv(PROM_INFO, sep="\t", usecols=["seq", "y"])
    pred = pd.read_csv(PREDICTIONS, sep="\t", usecols=["chrom", "start", "end", "strand", "seq", "Predicted_CAGE"])
    pred = pred.drop_duplicates("seq", keep="first").set_index("seq")

    rows = []
    for peak_id, group in hits.groupby("peak_id"):
        if not set(TARGET_MOTIFS).issubset(set(group["motif_tf"])):
            continue
        seq = prom.iloc[int(peak_id)]["seq"]
        if seq not in pred.index:
            continue
        picked = pick_target_hits(group)
        row = pred.loc[seq]
        rows.append(
            {
                "peak_id": int(peak_id),
                "sum_target_hit_importance": float(picked["hit_importance"].sum()),
                "min_target_hit_importance": float(picked["hit_importance"].min()),
                "true_CAGE_log2TPM": float(prom.iloc[int(peak_id)]["y"]),
                "pred_CAGE_log2TPM": float(row["Predicted_CAGE"]),
                "chrom": str(row["chrom"]),
                "start": int(row["start"]),
                "end": int(row["end"]),
                "strand": str(row["strand"]),
                "hits": ";".join(
                    f"{r.motif_tf}:{int(r.start_untrimmed)}-{int(r.end_untrimmed)}:{float(r.hit_importance):.3f}"
                    for r in picked.itertuples(index=False)
                ),
            }
        )
    candidates = pd.DataFrame(rows).sort_values(
        ["pred_CAGE_log2TPM", "true_CAGE_log2TPM", "sum_target_hit_importance"],
        ascending=False,
    )
    chosen_ids = list(candidates.head(2)["peak_id"].astype(int))
    examples: dict[str, Example] = {}
    for i, peak_id in enumerate(chosen_ids, start=1):
        seq = prom.iloc[peak_id]["seq"]
        row = pred.loc[seq]
        picked = pick_target_hits(hits[hits["peak_id"].eq(peak_id)])
        examples[f"CAGE promoter {i}"] = Example(
            peak_id=peak_id,
            chrom=str(row["chrom"]),
            start=int(row["start"]),
            end=int(row["end"]),
            strand=str(row["strand"]),
            true_cage=float(prom.iloc[peak_id]["y"]),
            pred_cage=float(row["Predicted_CAGE"]),
            hits=picked,
        )
    return examples, candidates


def draw_hit_spans(ax, example: Example, y_ref: float, pad: float) -> None:
    label_ends: list[float] = []
    color = TASK_COLORS["CAGE"]
    for row in example.hits.sort_values("start_untrimmed").itertuples(index=False):
        x0 = max(0, min(248, int(row.start_untrimmed))) + 1
        x1 = max(0, min(248, int(row.end_untrimmed))) + 1
        label = PLOT_MOTIF_LABELS.get(str(row.motif_tf), str(row.motif_tf))
        label_x = (x0 + x1) / 2
        half_width = max(18.0, len(label) * 1.15)
        left = label_x - half_width
        level = 0
        while level < len(label_ends) and left <= label_ends[level] + 1.8:
            level += 1
        right = label_x + half_width
        if level == len(label_ends):
            label_ends.append(right)
        else:
            label_ends[level] = right
        y_text = max(y_ref, 0.0) + pad * (0.04 + 1.35 * level)
        y_bar = y_text - pad * 0.10
        ax.plot([x0, x1], [y_bar, y_bar], color=color, lw=0.9, solid_capstyle="butt", zorder=4)
        ax.text(label_x, y_text, label, ha="center", va="bottom", fontsize=6.4, color=color)


def format_region(example: Example) -> str:
    return f"({example.chrom}:{example.start}-{example.end}{example.strand})"


def draw_example_panel(
    ax,
    label: str,
    example: Example,
    arr: np.ndarray,
    y_lim: tuple[float, float],
    y_max: float,
    pad: float,
) -> None:
    draw_logo(ax, arr)
    draw_hit_spans(ax, example, y_ref=y_max, pad=pad)
    ax.axhline(0, color="#4D4D4D", lw=0.45, zorder=1)
    ax.set_xlim(0, arr.shape[0])
    ax.set_ylim(*y_lim)
    ax.text(
        0.0,
        1.02,
        label.replace("promoter", "prom."),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="normal",
        color=TASK_COLORS["CAGE"],
    )
    ax.text(
        0.52,
        1.02,
        format_region(example),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#8A8A8A",
    )
    ax.set_ylabel("CAGE score", fontsize=8.2)
    ax.set_xlabel("")
    ax.set_xticks([25, 75, 125, 175, 225])
    ax.tick_params(axis="both", labelsize=7.1, length=2.2, width=0.6, pad=1.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#777777")
        spine.set_linewidth(0.55)


def plot_examples(examples: dict[str, Example], dirs: dict[str, object]) -> None:
    z = np.load(FINEMO_INPUT)
    arrays = {label: load_logo_array(z, example.peak_id) for label, example in examples.items()}
    all_values = np.concatenate([arr.reshape(-1) for arr in arrays.values()])
    y_min = float(np.nanpercentile(all_values, 0.5))
    y_max = float(np.nanpercentile(all_values, 99.5))
    span = max(y_max - y_min, 0.18)
    pad = max(span * 0.12, 0.025)
    y_lim = (min(y_min - pad * 0.45, -0.05), y_max + pad * 2.45)

    set_pub_style(font_size=8.0)
    fig, axes = plt.subplots(1, 2, figsize=(mm_to_in(DOUBLE_COL_MM), mm_to_in(28)), dpi=300)
    for ax, label in zip(axes, examples):
        draw_example_panel(ax, label, examples[label], arrays[label], y_lim, y_max, pad)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.24, top=0.84, wspace=0.14)
    paths = save_all(fig, dirs, "deepcage_cage_finemo_sequence_logo_examples_row")
    plt.close(fig)
    print(f"saved: {paths['png']}")


def write_tables(examples: dict[str, Example], candidates: pd.DataFrame, dirs: dict[str, object]) -> None:
    selected_rows = []
    hit_rows = []
    for label, example in examples.items():
        selected_rows.append(
            {
                "example_label": label,
                "peak_id": example.peak_id,
                "Chrom": example.chrom,
                "Start": example.start,
                "End": example.end,
                "Strand": example.strand,
                "true_CAGE_log2TPM": example.true_cage,
                "pred_CAGE_log2TPM": example.pred_cage,
                "target_motifs": ";".join(TARGET_MOTIFS),
            }
        )
        hits = example.hits.copy()
        hits["example_label"] = label
        hit_rows.append(hits)
    pd.DataFrame(selected_rows).to_csv(SELECTED_CSV, index=False)
    pd.concat(hit_rows, ignore_index=True).to_csv(dirs["qa"] / "cage_sequence_logo_example_hits.csv", index=False)
    candidates.to_csv(dirs["qa"] / "cage_sequence_logo_candidate_rankings.csv", index=False)


def main() -> None:
    dirs = panel_dirs("cage_sequence_examples")
    examples, candidates = load_examples()
    write_tables(examples, candidates, dirs)
    plot_examples(examples, dirs)
    print("selected:", ", ".join(f"{label}=peak_id {ex.peak_id}" for label, ex in examples.items()))


if __name__ == "__main__":
    main()
