#!/usr/bin/env python
# coding: utf-8
"""Representative DeepSTARR Fi-NeMo sequence contribution logos.

The figure shows real 249 bp DeepSTARR sequences selected from the annotated
Fi-NeMo hit table: one HK sequence containing Dref and Ohler1 hits, and one DEV
sequence containing AP-1 and GATA hits. Full-sequence contribution arrays are
masked by the observed base; Fi-NeMo hit spans are highlighted as annotations.
"""

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

from paths_S3 import CLEAN_DIR, RAW_DIR, panel_dirs
from style_S3 import DOUBLE_COL_MM, TASK_COLORS, mm_to_in, save_all, set_pub_style


DATA = CLEAN_DIR / "motif_hits_clean.csv"
SELECTED_CSV = CLEAN_DIR / "finemo_sequence_logo_examples.csv"
ATTR_SOURCE_DIR = RAW_DIR / "finemo_full_attribution_source"
BASES = ["A", "C", "G", "T"]
BASE_COLORS = {"A": "#109618", "C": "#3366CC", "G": "#FF9900", "T": "#DC3912"}
TARGETS = {
    "HK": ["Dref", "Ohler1"],
    "DEV": ["AP-1", "GATA"],
}
width_mm = 183
EXPORT_DPI = 600
EXPORT_TARGETS = ("figure.svg", "figure.pdf", "figure.tiff", "figure.png")


@dataclass(frozen=True)
class Example:
    task: str
    sequence_id: str
    chrom: str
    start: int
    end: int
    strand: str
    location: str
    sequence: str
    hits: pd.DataFrame


def interval_overlap_bp(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def orient_base_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape == (4, 249):
        return arr.T
    if arr.shape == (249, 4):
        return arr
    raise ValueError(f"Expected a 249 bp x 4 base array, got {arr.shape}")


def full_contribution_array(task: str, peak_index: int) -> np.ndarray:
    prefix = "hk" if task == "HK" else "dev"
    attr = np.load(ATTR_SOURCE_DIR / f"attr_{prefix}_pos.npy", mmap_mode="r")
    ohe = np.load(ATTR_SOURCE_DIR / f"ohe_{prefix}_pos.npy", mmap_mode="r")
    contrib = orient_base_array(attr[peak_index]).astype(float)
    sequence_mask = orient_base_array(ohe[peak_index]).astype(float)
    return contrib * sequence_mask


def pick_target_hits(group: pd.DataFrame, motifs: list[str]) -> pd.DataFrame:
    picked = []
    for motif in motifs:
        sub = group[group["motif_tf"].eq(motif)].sort_values(
            ["hit_importance", "motif_contrib_sum"], ascending=False
        )
        if sub.empty:
            raise ValueError(f"Missing target motif {motif} in {group['sequence_id'].iloc[0]}")
        picked.append(sub.iloc[0])
    return pd.DataFrame(picked).reset_index(drop=True)


def target_mask_for_hits(target_hits: pd.DataFrame, pad: int = 3) -> np.ndarray:
    mask = np.zeros(249, dtype=bool)
    for row in target_hits.itertuples(index=False):
        rel_start = int(row.start) - int(row.Start)
        rel_end = int(row.end) - int(row.Start)
        start = max(0, rel_start - pad)
        end = min(249, rel_end + pad)
        mask[start:end] = True
    return mask


def outside_peak_ratio(task: str, target_hits: pd.DataFrame) -> float:
    peak_index = int(target_hits.iloc[0]["peak_id"])
    arr = full_contribution_array(task, peak_index)
    positive_profile = np.maximum(arr, 0).sum(axis=1)
    target_mask = target_mask_for_hits(target_hits)
    target_max = float(positive_profile[target_mask].max()) if target_mask.any() else 0.0
    outside_max = float(positive_profile[~target_mask].max()) if (~target_mask).any() else 0.0
    return outside_max / max(target_max, 1e-9)


def choose_examples(df: pd.DataFrame) -> dict[str, Example]:
    examples: dict[str, Example] = {}
    for task, motifs in TARGETS.items():
        target_hits = df[df["task"].eq(task) & df["motif_tf"].isin(motifs) & df["Strand"].eq("+")].copy()
        rows = []
        for sequence_id, group in target_hits.groupby("sequence_id", observed=True):
            if not set(motifs).issubset(set(group["motif_tf"])):
                continue
            picked = pick_target_hits(group, motifs)
            intervals = [(int(row.start), int(row.end)) for row in picked.itertuples(index=False)]
            overlap = interval_overlap_bp(intervals[0], intervals[1])
            min_width = min(end - start for start, end in intervals)
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "combined_hit_importance": float(picked["hit_importance"].sum()),
                    "overlap_fraction": float(overlap / max(min_width, 1)),
                    "outside_peak_ratio": outside_peak_ratio(task, picked),
                }
            )
        candidates = pd.DataFrame(rows).sort_values(
            ["outside_peak_ratio", "overlap_fraction", "combined_hit_importance"],
            ascending=[True, True, False],
        )
        if candidates.empty:
            raise RuntimeError(f"No {task} sequence contains all target motifs: {motifs}")
        sequence_id = str(candidates.iloc[0]["sequence_id"])
        group = df[df["task"].eq(task) & df["sequence_id"].eq(sequence_id)].copy()
        picked = pick_target_hits(group, motifs)
        first = picked.iloc[0]
        examples[task] = Example(
            task=task,
            sequence_id=sequence_id,
            chrom=str(first["Chrom"]),
            start=int(first["Start"]),
            end=int(first["End"]),
            strand=str(first["Strand"]),
            location=str(first["location"]),
            sequence=str(first["Sequence"]),
            hits=picked,
        )
    return examples


def sequence_from_ohe(ohe: np.ndarray) -> str:
    oriented = orient_base_array(ohe)
    idx = oriented.argmax(axis=1)
    valid = oriented.max(axis=1) > 0
    return "".join(BASES[idx[i]] if valid[i] else "N" for i in range(oriented.shape[0]))


def load_full_logo_array(example: Example) -> np.ndarray:
    prefix = "hk" if example.task == "HK" else "dev"
    attr = np.load(ATTR_SOURCE_DIR / f"attr_{prefix}_pos.npy", mmap_mode="r")
    ohe = np.load(ATTR_SOURCE_DIR / f"ohe_{prefix}_pos.npy", mmap_mode="r")
    idx = int(example.sequence_id.split(":")[1])
    arr = full_contribution_array(example.task, idx)
    observed = sequence_from_ohe(ohe[idx])
    if observed != example.sequence:
        raise ValueError(
            f"{example.sequence_id}: full attribution source sequence does not match motif hit table"
        )
    return arr


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


def display_label(row) -> str:
    return str(row.motif_tf)


def draw_hit_spans(ax, example: Example, y_ref: float, pad: float) -> None:
    label_ends: list[float] = []
    for row in example.hits.sort_values("start").itertuples(index=False):
        x0 = int(row.start) - example.start + 1
        x1 = int(row.end) - example.start + 1
        label = display_label(row)
        label_x = (x0 + x1) / 2
        half_width = max(5.2, len(label) * 0.36)
        left = label_x - half_width
        level = 0
        while level < len(label_ends) and left <= label_ends[level] + 1.8:
            level += 1
        right = label_x + half_width
        if level == len(label_ends):
            label_ends.append(right)
        else:
            label_ends[level] = right
        y_text = max(y_ref, 0.0) + pad * (0.04 + 0.78 * level)
        y_bar = y_text - pad * 0.10
        ax.plot([x0, x1], [y_bar, y_bar], color=TASK_COLORS[example.task], lw=0.9, solid_capstyle="butt", zorder=4)
        ax.text(
            label_x,
            y_text,
            label,
            ha="center",
            va="bottom",
            fontsize=6.8,
            color=TASK_COLORS[example.task],
        )


def format_region(example: Example) -> str:
    return f"({example.chrom}:{example.start}-{example.end})"


def draw_example_panel(
    ax,
    task: str,
    example: Example,
    arr: np.ndarray,
    y_lim: tuple[float, float],
    y_max: float,
    pad: float,
    *,
    compact: bool = False,
) -> None:
    draw_logo(ax, arr)
    draw_hit_spans(ax, example, y_ref=y_max, pad=pad)
    ax.axhline(0, color="#4D4D4D", lw=0.45, zorder=1)
    ax.set_xlim(0, len(example.sequence))
    ax.set_ylim(*y_lim)
    ax.text(
        0.0,
        1.02,
        f"{task} enhancer",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7 if compact else 9.2,
        fontweight="normal",
        color=TASK_COLORS[task],
    )
    ax.text(
        0.50,
        1.02,
        format_region(example),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.2 if compact else 8.2,
        color="#8A8A8A",
    )
    ax.set_ylabel(f"{task} score", fontsize=8.2 if compact else 9.0)
    ax.set_xlabel("")
    ax.set_xticks([25, 75, 125, 175, 225] if compact else [25, 50, 75, 100, 125, 150, 175, 200, 225])
    ax.tick_params(axis="both", labelsize=7.1 if compact else 8.0, length=2.2, width=0.6, pad=1.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#777777")
        spine.set_linewidth(0.55)


def plot_single_example(
    task: str,
    example: Example,
    arr: np.ndarray,
    y_lim: tuple[float, float],
    y_max: float,
    pad: float,
    dirs: dict[str, object],
) -> None:
    set_pub_style(font_size=9.2)
    fig, ax = plt.subplots(1, 1, figsize=(mm_to_in(DOUBLE_COL_MM), mm_to_in(27)), dpi=300)
    draw_example_panel(ax, task, example, arr, y_lim, y_max, pad)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.24, top=0.86)
    paths = save_all(fig, dirs, f"deepstarr_{task.lower()}_finemo_sequence_logo_example")
    plt.close(fig)
    print(f"saved: {paths['png']}")


def plot_combined_examples(
    examples: dict[str, Example],
    arrays: dict[str, np.ndarray],
    y_lim: tuple[float, float],
    y_max: float,
    pad: float,
    dirs: dict[str, object],
) -> None:
    set_pub_style(font_size=8.0)
    fig, axes = plt.subplots(1, 2, figsize=(mm_to_in(DOUBLE_COL_MM), mm_to_in(28)), dpi=300)
    for ax, task in zip(axes, ["DEV", "HK"]):
        draw_example_panel(ax, task, examples[task], arrays[task], y_lim, y_max, pad, compact=True)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.24, top=0.84, wspace=0.14)
    paths = save_all(fig, dirs, "deepstarr_finemo_sequence_logo_examples_row")
    plt.close(fig)
    print(f"saved: {paths['png']}")


def plot_examples(examples: dict[str, Example], dirs: dict[str, object]) -> None:
    arrays = {task: load_full_logo_array(example) for task, example in examples.items()}
    all_values = np.concatenate([arr.reshape(-1) for arr in arrays.values()])
    y_min = float(np.nanpercentile(all_values, 0.5))
    y_max = float(np.nanpercentile(all_values, 99.5))
    span = max(y_max - y_min, 0.18)
    pad = max(span * 0.12, 0.025)
    y_lim = (min(y_min - pad * 0.45, -0.05), y_max + pad * 1.95)

    plot_combined_examples(examples, arrays, y_lim, y_max, pad, dirs)


def write_selection_tables(examples: dict[str, Example], dirs: dict[str, object]) -> None:
    selected_rows = []
    hit_rows = []
    for example in examples.values():
        selected_rows.append(
            {
                "task": example.task,
                "sequence_id": example.sequence_id,
                "Chrom": example.chrom,
                "Start": example.start,
                "End": example.end,
                "Strand": example.strand,
                "location": example.location,
                "Sequence": example.sequence,
                "target_motifs": ";".join(TARGETS[example.task]),
            }
        )
        hits = example.hits.copy()
        hits["relative_start_0based"] = hits["start"].astype(int) - example.start
        hits["relative_end_0based_exclusive"] = hits["end"].astype(int) - example.start
        hits["profile_start_0based"] = hits["start_untrimmed"].astype(int) - example.start
        hit_rows.append(hits)
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(SELECTED_CSV, index=False)
    pd.concat(hit_rows, ignore_index=True).to_csv(dirs["qa"] / "finemo_sequence_logo_example_hits.csv", index=False)


def main() -> None:
    dirs = panel_dirs("finemo_sequence_examples")
    cols = [
        "task",
        "sequence_id",
        "peak_id",
        "motif_display",
        "motif_tf",
        "hit_importance",
        "motif_contrib_sum",
        "start",
        "end",
        "start_untrimmed",
        "end_untrimmed",
        "Chrom",
        "Start",
        "End",
        "Strand",
        "location",
        "Sequence",
    ]
    df = pd.read_csv(DATA, usecols=cols, low_memory=False)
    examples = choose_examples(df)
    write_selection_tables(examples, dirs)
    plot_examples(examples, dirs)


if __name__ == "__main__":
    main()
