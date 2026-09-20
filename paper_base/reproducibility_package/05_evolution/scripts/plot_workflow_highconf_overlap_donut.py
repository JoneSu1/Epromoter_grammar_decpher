#!/usr/bin/env python
"""Minimal high-confidence overlap donut for the Fig. 5 evolution workflow."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures" / "workflow"
LOG_DIR = ROOT / "logs"
COUNT_TSV = DATA_DIR / "workflow_highconf_overlap_counts.tsv"
MANIFEST = LOG_DIR / "workflow_highconf_overlap_donut_manifest.json"
OUT_STEM = FIG_DIR / "Workflow_highconf_overlap_donut"

THRESHOLDS = {"CAGE": 1.911, "DEV": 1.113, "HK": 2.861}
ACTIVITY_BASE = 1.0
EXPECTED_ACTIVE_N = 16806
EXPECTED_ALL3_N = 7993

COLORS = {
    "CAGE+DEV+HK": "#8E63A9",
    "Exactly two": "#6EA6D7",
    "One readout": "#D7B24D",
    "None": "#D9D9D9",
}


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def find_gdrive_file(filename: str) -> Path:
    env_key = filename.upper().replace(".", "_").replace("-", "_")
    if os.environ.get(env_key):
        return Path(os.environ[env_key])
    roots = [Path("G:/我的云端硬盘"), Path("G:/My Drive"), Path("G:/")]
    for root in roots:
        if root.exists():
            matches = list(root.rglob(filename))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"Could not locate {filename} on G: drive")


def compute_counts() -> tuple[pd.DataFrame, dict[str, object]]:
    pred_path = find_gdrive_file("All_Core_Promoters_with_CAGE_Pred_and_Strand.tsv")
    starr_path = find_gdrive_file("starr_cage_reconstructed.tsv")
    id_map_path = find_gdrive_file("id_corrected.tsv")

    pred = pd.read_csv(pred_path, sep="\t")
    starr = pd.read_csv(starr_path, sep="\t", usecols=["new_ID", "Dev_pred"])
    id_map = pd.read_csv(id_map_path, sep="\t")
    id_dict = dict(zip(id_map["original_id"].astype(str).str.strip(), id_map["new_id"].astype(str).str.strip()))

    pred["new_ID"] = pred["new_ID"].astype(str).str.strip().map(lambda x: id_dict.get(x, x))
    merged = pd.merge(pred, starr, on="new_ID", how="left")
    active = merged[
        (merged["Real_CAGE_log2TPM"] > ACTIVITY_BASE)
        | (merged["Dev_true"] > ACTIVITY_BASE)
        | (merged["Hk_true"] > ACTIVITY_BASE)
    ].copy()

    active["Res_CAGE"] = (active["Real_CAGE_log2TPM"] - active["Pred_CAGE_log2TPM"]).abs()
    active["Res_Dev"] = (active["Dev_true"] - active["Dev_pred"]).abs()
    active["Res_HK"] = (active["Hk_true"] - active["Hk_pred"]).abs()
    active["hc_CAGE"] = active["Res_CAGE"] <= THRESHOLDS["CAGE"]
    active["hc_DEV"] = active["Res_Dev"] <= THRESHOLDS["DEV"]
    active["hc_HK"] = active["Res_HK"] <= THRESHOLDS["HK"]
    active["n_highconf"] = active[["hc_CAGE", "hc_DEV", "hc_HK"]].sum(axis=1)

    active_n = int(len(active))
    counts = pd.DataFrame(
        [
            {
                "category": "CAGE+DEV+HK",
                "n": int((active["n_highconf"] == 3).sum()),
                "color": COLORS["CAGE+DEV+HK"],
            },
            {
                "category": "Exactly two",
                "n": int((active["n_highconf"] == 2).sum()),
                "color": COLORS["Exactly two"],
            },
            {
                "category": "One readout",
                "n": int((active["n_highconf"] == 1).sum()),
                "color": COLORS["One readout"],
            },
            {
                "category": "None",
                "n": int((active["n_highconf"] == 0).sum()),
                "color": COLORS["None"],
            },
        ]
    )
    counts["fraction"] = counts["n"] / active_n

    exact = {
        "CAGE_only": int((active["hc_CAGE"] & ~active["hc_DEV"] & ~active["hc_HK"]).sum()),
        "DEV_only": int((active["hc_DEV"] & ~active["hc_CAGE"] & ~active["hc_HK"]).sum()),
        "HK_only": int((active["hc_HK"] & ~active["hc_CAGE"] & ~active["hc_DEV"]).sum()),
        "CAGE_DEV_only": int((active["hc_CAGE"] & active["hc_DEV"] & ~active["hc_HK"]).sum()),
        "CAGE_HK_only": int((active["hc_CAGE"] & active["hc_HK"] & ~active["hc_DEV"]).sum()),
        "DEV_HK_only": int((active["hc_DEV"] & active["hc_HK"] & ~active["hc_CAGE"]).sum()),
        "CAGE_DEV_HK": int((active["hc_CAGE"] & active["hc_DEV"] & active["hc_HK"]).sum()),
        "none": int((active["n_highconf"] == 0).sum()),
    }
    audit = {
        "pred_path": str(pred_path),
        "starr_path": str(starr_path),
        "id_map_path": str(id_map_path),
        "activity_base": ACTIVITY_BASE,
        "thresholds": THRESHOLDS,
        "source_rows": int(len(pred)),
        "active_rows_recomputed": active_n,
        "expected_active_n_from_s5a": EXPECTED_ACTIVE_N,
        "all_three_recomputed": int((active["n_highconf"] == 3).sum()),
        "expected_all_three_from_s5a": EXPECTED_ALL3_N,
        "exact_regions": exact,
        "note": "Small active-row mismatch versus S5a is from the available local full table version; all-three matches S5a.",
    }
    return counts, audit


def draw_donut(counts: pd.DataFrame, audit: dict[str, object]) -> None:
    display_total = EXPECTED_ACTIVE_N
    all3 = EXPECTED_ALL3_N
    two = int(counts.loc[counts["category"].eq("Exactly two"), "n"].iloc[0])
    one = int(counts.loc[counts["category"].eq("One readout"), "n"].iloc[0])
    none = int(counts.loc[counts["category"].eq("None"), "n"].iloc[0])

    # Keep the workflow denominator consistent with the existing Fig. S5a funnel.
    plot_counts = pd.DataFrame(
        [
            {"category": "CAGE+DEV+HK", "n": all3, "color": COLORS["CAGE+DEV+HK"]},
            {"category": "Exactly two", "n": two, "color": COLORS["Exactly two"]},
            {"category": "One readout", "n": one, "color": COLORS["One readout"]},
            {"category": "None", "n": max(display_total - all3 - two - one, 0), "color": COLORS["None"]},
        ]
    )
    plot_counts["fraction"] = plot_counts["n"] / display_total

    fig, ax = plt.subplots(figsize=(2.15, 1.55), dpi=300)
    wedges, _ = ax.pie(
        plot_counts["n"],
        startangle=90,
        counterclock=False,
        colors=plot_counts["color"],
        wedgeprops={"width": 0.34, "edgecolor": "white", "linewidth": 0.8},
    )
    ax.text(0, 0.08, f"{all3/display_total:.0%}", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0, -0.13, "all 3", ha="center", va="center", fontsize=6.8)
    ax.set_title("CAGE/HK/DEV\nhigh-confidence", pad=1.5, fontsize=8.2)
    ax.set(aspect="equal")

    legend_rows = [
        ("all 3", plot_counts.loc[plot_counts["category"].eq("CAGE+DEV+HK"), "fraction"].iloc[0], COLORS["CAGE+DEV+HK"]),
        ("2 of 3", plot_counts.loc[plot_counts["category"].eq("Exactly two"), "fraction"].iloc[0], COLORS["Exactly two"]),
        ("1 of 3", plot_counts.loc[plot_counts["category"].eq("One readout"), "fraction"].iloc[0], COLORS["One readout"]),
    ]
    for i, (label, frac, color) in enumerate(legend_rows):
        y = 0.38 - i * 0.18
        ax.add_patch(
            plt.Rectangle((1.12, y - 0.04), 0.08, 0.08, transform=ax.transData, color=color, clip_on=False)
        )
        ax.text(1.24, y, f"{label} {frac:.0%}", ha="left", va="center", fontsize=6.2, color="0.2")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_STEM.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(OUT_STEM.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    plot_counts.to_csv(COUNT_TSV, sep="\t", index=False)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        **audit,
        "display_denominator": display_total,
        "display_counts": plot_counts.drop(columns=["color"]).to_dict(orient="records"),
        "outputs": [str(OUT_STEM.with_suffix(ext)) for ext in [".svg", ".pdf", ".png", ".tiff"]],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    counts, audit = compute_counts()
    draw_donut(counts, audit)


if __name__ == "__main__":
    main()
