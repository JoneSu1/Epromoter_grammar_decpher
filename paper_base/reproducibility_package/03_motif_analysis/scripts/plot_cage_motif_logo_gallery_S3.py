#!/usr/bin/env python
# coding: utf-8
"""CAGE motif logo gallery for Supplementary Fig. 3.

This panel uses the package-local DeepCAGE/CTSS TF-MoDISco output and the
matching tomtom annotation table. It mirrors the DEV/HK motif-logo rows in
plot_motif_logo_rows_S3.py, but is kept as a separate CAGE panel because the
CAGE input is a core-promoter readout rather than a DeepSTARR enhancer subset.
"""

from __future__ import annotations

import re

import h5py
import logomaker as lm
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

from paths_S3 import RAW_DIR, panel_dirs
from style_S3 import DOUBLE_COL_MM, TASK_COLORS, mm_to_in, set_pub_style


SOURCE_DIR = RAW_DIR / "motif_logo_source"
H5 = SOURCE_DIR / "CTSS_modisco_results.h5"
ANNOTATION = SOURCE_DIR / "CAGE_motif_tomtom_annotation.tsv"
LOGO_COLS = 3
LOGO_ROWS = 5
LOGO_CELL_HEIGHT_MM = 16.3
LOGO_TITLE_SIZE = 11.0
EXPORT_DPI = 600
EPS = 1e-12
MIN_IC = 0.30
MIN_LEN = 6
PAD = 1

TF_DISPLAY = {
    "Ebox/CAGCTG/3": "Ebox/CAGCTG",
}


def save_logo_all(fig: plt.Figure, dirs: dict[str, object], stem: str) -> dict[str, object]:
    paths = {
        "svg": dirs["svg"] / f"{stem}.svg",
        "pdf": dirs["pdf"] / f"{stem}.pdf",
        "tiff": dirs["tiff"] / f"{stem}.tiff",
        "png": dirs["png"] / f"{stem}.png",
    }
    fig.savefig(paths["svg"])
    fig.savefig(paths["pdf"])
    fig.savefig(paths["tiff"], dpi=EXPORT_DPI)
    fig.savefig(paths["png"], dpi=300)
    return paths


def pattern_index(name: str) -> int:
    match = re.search(r"pattern_(\d+)$", str(name))
    if not match:
        raise ValueError(f"Unexpected pattern name: {name}")
    return int(match.group(1))


def sequence_ic(sequence_pwm: np.ndarray) -> np.ndarray:
    freq = np.asarray(sequence_pwm, dtype=float)
    freq = freq / np.clip(freq.sum(axis=1, keepdims=True), EPS, None)
    return 2.0 + np.sum(freq * np.log2(np.clip(freq, EPS, None)), axis=1)


def trim_indices(ic: np.ndarray, *, min_ic: float = MIN_IC, min_len: int = MIN_LEN, pad: int = PAD) -> tuple[int, int]:
    hits = np.flatnonzero(ic >= min_ic)
    if len(hits) == 0:
        hits = np.argsort(ic)[-min_len:]
    runs = []
    run_start = int(hits[0])
    previous = int(hits[0])
    for current in map(int, hits[1:]):
        if current == previous + 1:
            previous = current
            continue
        runs.append((run_start, previous + 1))
        run_start = current
        previous = current
    runs.append((run_start, previous + 1))
    core_start, core_end = max(runs, key=lambda run: float(ic[run[0] : run[1]].sum()))
    start = max(0, core_start - pad)
    end = min(len(ic), core_end + pad)
    if end - start < min_len:
        center = (start + end) // 2
        start = max(0, center - min_len // 2)
        end = min(len(ic), start + min_len)
        start = max(0, end - min_len)
    return start, end


def load_annotations() -> dict[int, str]:
    df = pd.read_csv(ANNOTATION, sep="\t")
    labels = {}
    for _, row in df.iterrows():
        idx = pattern_index(row["pattern"])
        match = str(row.get("Match_1", "")).strip()
        labels[idx] = TF_DISPLAY.get(match, match)
    return labels


def load_patterns():
    with h5py.File(H5, "r") as h5:
        root = h5["pos_patterns"]
        names = sorted(root.keys(), key=pattern_index)
        for name in names:
            pattern = root[name]
            sequence = np.asarray(pattern["sequence"])
            hypothetical = np.asarray(pattern["hypothetical_contribs"])
            n_seqlets = int(pattern["seqlets"]["n_seqlets"][0])
            yield name, sequence, hypothetical, n_seqlets


def contribution_logo_matrix(sequence: np.ndarray, hypothetical: np.ndarray) -> pd.DataFrame:
    masked = np.asarray(sequence, dtype=float) * np.asarray(hypothetical, dtype=float)
    positive = np.maximum(masked, 0)
    return pd.DataFrame(positive, columns=["A", "C", "G", "T"])


def centered_logo_axes(fig: plt.Figure, n_items: int, ncols: int = LOGO_COLS, nrows: int = LOGO_ROWS) -> list[plt.Axes]:
    grid = GridSpec(nrows, ncols * 2, figure=fig, wspace=0.32, hspace=0.72)
    axes = []
    for row in range(nrows):
        remaining = n_items - row * ncols
        if remaining <= 0:
            break
        count = min(ncols, remaining)
        starts = [0, 2, 4] if count == 3 else ([1, 3] if count == 2 else [2])
        for col in starts:
            axes.append(fig.add_subplot(grid[row, col : col + 2]))
    return axes


def main() -> None:
    dirs = panel_dirs("cage_motif_logo_gallery")
    annotations = load_annotations()
    rows = []
    matrices = []
    labels = []
    for name, sequence, hypothetical, n_seqlets in load_patterns():
        idx = pattern_index(name)
        ic = sequence_ic(sequence)
        start, end = trim_indices(ic)
        logo = contribution_logo_matrix(sequence, hypothetical).iloc[start:end].reset_index(drop=True)
        if float(logo.to_numpy().max()) <= 0:
            logo = pd.DataFrame(sequence[start:end], columns=["A", "C", "G", "T"]).reset_index(drop=True)
        tf = annotations.get(idx, "")
        matrices.append(logo)
        labels.append(f"C_P{idx} | {tf}" if tf else f"C_P{idx}")
        rows.append(
            {
                "task": "CAGE",
                "motif_index": idx,
                "motif_label": f"C_P{idx}",
                "motif_tf": tf,
                "n_seqlets": int(n_seqlets),
                "trim_start_0based": int(start),
                "trim_end_0based_exclusive": int(end),
                "trimmed_length": int(end - start),
                "max_ic_in_window": float(ic[start:end].max()),
                "max_positive_contribution": float(logo.to_numpy().max()),
            }
        )

    set_pub_style(font_size=9.2)
    fig = plt.figure(figsize=(mm_to_in(DOUBLE_COL_MM * 0.78), mm_to_in(LOGO_ROWS * LOGO_CELL_HEIGHT_MM + 10)))
    axes = centered_logo_axes(fig, len(matrices))
    y_max = max(float(mat.to_numpy().max()) for mat in matrices)
    y_max = max(y_max, 0.01)

    for ax, mat, label in zip(axes, matrices, labels):
        lm.Logo(mat, ax=ax, color_scheme="classic", baseline_width=0.25)
        ax.set_title(
            label,
            color=TASK_COLORS["CAGE"],
            fontsize=LOGO_TITLE_SIZE,
            fontfamily="Arial",
            fontweight="bold",
            pad=2,
        )
        ax.set_ylim(0, y_max * 1.05)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ["left", "right", "top"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.45)
        ax.tick_params(length=0)

    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.06, top=0.92)
    paths = save_logo_all(fig, dirs, "deepcage_cage_motif_logo_gallery")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(dirs["qa"] / "cage_motif_logo_gallery_trim_audit.csv", index=False)
    print(f"saved: {paths['png']} ({len(matrices)} CAGE motifs)")


if __name__ == "__main__":
    main()
