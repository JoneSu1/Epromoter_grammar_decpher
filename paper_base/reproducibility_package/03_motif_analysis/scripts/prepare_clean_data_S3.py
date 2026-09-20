#!/usr/bin/env python
# coding: utf-8
"""Prepare DeepSTARR motif-distribution V2 lambda=0.7 supplement tables.

Source logic follows motif_distribution_v2.ipynb:
read DEV/HK annotated Fi-NeMo lambda=0.7 hit tables, derive Region from
Dev_promoter_type or Hk_promoter_type, and aggregate only hit-positive peaks.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from paths_S3 import CLEAN_DIR, RAW_DIR


RAW_HITS_DIR = RAW_DIR / "motif_distribution_v2_annotated_hits"
LAMBDA = 0.7
DATASETS = {
    "DEV": {
        "dataset_key": "dev",
        "path": RAW_HITS_DIR / "DEV_finemo_lambda_0.7_hits_annotated.tsv",
        "region_col": "Dev_promoter_type",
        "prefix": "D",
        "motifs": {
            0: "AP-1",
            1: "GATA",
            2: "Dref",
            3: "Ohler1",
            4: "CREB/ATF",
            5: "AP-1",
            6: "twist",
            7: "SREBP",
            8: "CREB/ATF",
            9: "ETS",
            10: "STAT",
            11: "Ohler6",
            12: "Dip3",
            13: "PAX4",
            14: "GAGA-repeat",
        },
    },
    "HK": {
        "dataset_key": "hk",
        "path": RAW_HITS_DIR / "HK_finemo_lambda_0.7_hits_annotated.tsv",
        "region_col": "Hk_promoter_type",
        "prefix": "H",
        "motifs": {
            0: "Dref",
            1: "Ohler1",
            2: "Ohler7",
            3: "Ohler6",
            4: "Ohler5",
            5: "Unknown",
            6: "CEBPB",
        },
    },
}
ORDER = ["proximal", "distal"]


def format_region(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).lower()
    if "proximal" in text:
        return "proximal"
    if "distal" in text:
        return "distal"
    return None


def motif_index(motif_name: str) -> int:
    match = re.search(r"pattern_(\d+)", str(motif_name))
    return int(match.group(1)) if match else 999


def load_hits(task: str) -> pd.DataFrame:
    spec = DATASETS[task]
    df = pd.read_csv(spec["path"], sep="\t")
    df["task"] = task
    df["dataset_key"] = spec["dataset_key"]
    df["lambda"] = LAMBDA
    df["Region"] = df[spec["region_col"]].apply(format_region)
    df = df[df["Region"].isin(ORDER)].copy()
    df["location"] = df["Region"].map({"proximal": "Proximal", "distal": "Distal"})
    df["motif_index"] = df["motif_name"].map(motif_index)
    df = df[df["motif_index"].ne(999)].copy()
    df["motif_label"] = df["motif_index"].map(lambda idx: f"{spec['prefix']}_P{int(idx)}")
    df["motif_tf"] = df["motif_index"].map(spec["motifs"]).fillna("Unknown")
    df["motif_display"] = df["motif_label"] + "\n(" + df["motif_tf"] + ")"
    df["sequence_id"] = task + ":" + df["peak_id"].astype(str)
    return df


def build_tables() -> dict[str, pd.DataFrame]:
    all_frames = []
    audit_rows = []

    for task, spec in DATASETS.items():
        raw = pd.read_csv(spec["path"], sep="\t", usecols=["peak_id", "motif_name", spec["region_col"]])
        raw_region = raw[spec["region_col"]].apply(format_region)
        hits = load_hits(task)
        all_frames.append(hits)
        audit_rows.append(
            {
                "task": task,
                "source_file": spec["path"].name,
                "raw_hit_rows": int(len(raw)),
                "region_hit_rows": int(raw_region.isin(ORDER).sum()),
                "excluded_hit_rows_without_region": int((~raw_region.isin(ORDER)).sum()),
                "raw_peak_ids": int(raw["peak_id"].nunique()),
                "region_peak_ids": int(raw.loc[raw_region.isin(ORDER), "peak_id"].nunique()),
                "region_counts": ";".join(f"{k}:{v}" for k, v in raw_region.value_counts(dropna=False).items()),
            }
        )

    hits_all = pd.concat(all_frames, ignore_index=True)

    density = (
        hits_all.groupby(["task", "peak_id"], observed=True)
        .agg(
            Region=("Region", "first"),
            location=("location", "first"),
            sequence_id=("sequence_id", "first"),
            motif_density=("motif_name", "size"),
            motif_complexity=("motif_name", pd.Series.nunique),
            total_importance=("hit_importance", "sum"),
            mean_importance=("hit_importance", "mean"),
            total_motif_contrib_sum=("motif_contrib_sum", "sum"),
            mean_motif_contrib_sum=("motif_contrib_sum", "mean"),
        )
        .reset_index()
    )

    per_motif_density = (
        hits_all.groupby(
            ["task", "location", "Region", "peak_id", "sequence_id", "motif_display", "motif_label", "motif_index", "motif_tf"],
            observed=True,
        )
        .size()
        .reset_index(name="motif_count")
    )

    contribution_cols = [
        "task",
        "location",
        "Region",
        "peak_id",
        "sequence_id",
        "motif_display",
        "motif_label",
        "motif_index",
        "motif_tf",
        "motif_name",
        "hit_importance",
        "motif_contrib_sum",
    ]
    contribution = hits_all[[col for col in contribution_cols if col in hits_all.columns]].copy()

    groups = (
        density.groupby(["task", "location"], observed=True)
        .agg(
            n_hit_positive_peaks=("peak_id", "nunique"),
            mean_density=("motif_density", "mean"),
            median_density=("motif_density", "median"),
            mean_complexity=("motif_complexity", "mean"),
            median_complexity=("motif_complexity", "median"),
            mean_hit_importance=("mean_importance", "mean"),
            mean_motif_contrib_sum=("mean_motif_contrib_sum", "mean"),
        )
        .reset_index()
    )
    motifs = (
        per_motif_density.groupby(["task", "location", "motif_label", "motif_display", "motif_index", "motif_tf"], observed=True)
        .agg(
            n_peak_motif=("peak_id", "nunique"),
            median_motif_count=("motif_count", "median"),
        )
        .reset_index()
        .sort_values(["task", "motif_index", "location"])
    )

    return {
        "motif_hits_clean": hits_all,
        "motif_density_complexity_by_sequence": density,
        "motif_per_motif_density_by_sequence": per_motif_density,
        "motif_contribution_hits": contribution,
        "motif_group_summary": groups,
        "motif_level_summary": motifs,
        "input_audit": pd.DataFrame(audit_rows),
    }


def main() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    tables = build_tables()
    for name, table in tables.items():
        table.to_csv(CLEAN_DIR / f"{name}.csv", index=False)
        print(f"wrote {name}: {len(table):,} rows")

    manifest = {
        "source": "DeepSTARR_finemo_lambda07/motif_distribution/motif_distribution_v2.ipynb",
        "lambda": LAMBDA,
        "density_complexity_definition": "hit-positive peaks only; groupby peak_id after filtering proximal/distal rows",
        "region_definition": "Dev_promoter_type or Hk_promoter_type contains proximal/distal",
        "contribution_definition": "raw hit-level hit_importance and motif_contrib_sum from annotated hits; no sequence-level averaging for contribution plots",
        "datasets": {task: {"source_file": spec["path"].name, "region_col": spec["region_col"]} for task, spec in DATASETS.items()},
    }
    (CLEAN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
