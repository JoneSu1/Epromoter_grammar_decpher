#!/usr/bin/env python
# coding: utf-8
"""Shared data loading for Figure 1B and 1C."""

from __future__ import annotations

import pandas as pd

from common import DATA_ROOT, require_columns


PRED_DIR = DATA_ROOT / "prediction_output"
ANNOT_DIR = DATA_ROOT / "overlap_annotation"

OVERLAP_COLS = [
    "chr",
    "start",
    "end",
    "ID",
    "summit_chr",
    "summit_start",
    "summit_end",
    "region_annotation",
    "score",
    "strand",
]

PRED_REQUIRED = ["ID", "Dev_true", "Dev_pred", "Hk_true", "Hk_pred"]
PROMOTER_REGIONS = {"core_promoter", "proximal_promoter"}


def _read_prediction(split: str) -> pd.DataFrame:
    path = PRED_DIR / f"prediction_{split}.tsv"
    frame = pd.read_csv(path, sep="\t")
    require_columns(frame, PRED_REQUIRED, label=path.name)
    frame["set"] = split
    return frame


def _read_annotation(filename: str, region_col: str, promoter_col: str) -> pd.DataFrame:
    path = ANNOT_DIR / filename
    frame = pd.read_csv(path, sep="\t", header=None, names=OVERLAP_COLS)
    require_columns(frame, ["ID", "region_annotation"], label=path.name)
    annot = (
        frame[["ID", "region_annotation"]]
        .drop_duplicates()
        .rename(columns={"region_annotation": region_col})
    )
    annot[promoter_col] = annot[region_col].where(
        annot[region_col].isin(PROMOTER_REGIONS), "distal_promoter"
    )
    annot.loc[annot[region_col].isin(PROMOTER_REGIONS), promoter_col] = "proximal_promoter"
    return annot


def load_prediction_and_annotations() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction = pd.concat(
        [_read_prediction(split) for split in ("train", "valid", "test")],
        ignore_index=True,
    )
    dev_annot = _read_annotation(
        "deepstarr_overlap_development.tsv", "Dev_region", "Dev_promoter_type"
    )
    hk_annot = _read_annotation(
        "deepstarr_overlap_housekeeping.tsv", "Hk_region", "Hk_promoter_type"
    )
    return prediction, dev_annot, hk_annot


def load_fig1b_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction, dev_annot, hk_annot = load_prediction_and_annotations()
    dev_df = prediction.merge(dev_annot, on="ID", how="inner")
    hk_df = prediction.merge(hk_annot, on="ID", how="inner")
    return dev_df, hk_df


def load_fig1c_data() -> pd.DataFrame:
    prediction, dev_annot, hk_annot = load_prediction_and_annotations()
    dev_ids = set(dev_annot["ID"].unique())
    hk_ids = set(hk_annot["ID"].unique())
    dh_ids = dev_ids | hk_ids

    frame = prediction[prediction["ID"].isin(dh_ids)].copy()
    frame = frame.merge(dev_annot, on="ID", how="left").merge(hk_annot, on="ID", how="left")
    frame["Source"] = "Both"
    frame.loc[frame["ID"].isin(dev_ids - hk_ids), "Source"] = "Dev_only"
    frame.loc[frame["ID"].isin(hk_ids - dev_ids), "Source"] = "HK_only"
    frame["Promoter_group"] = "distal_promoter"
    proximal = (frame["Dev_promoter_type"] == "proximal_promoter") | (
        frame["Hk_promoter_type"] == "proximal_promoter"
    )
    frame.loc[proximal, "Promoter_group"] = "proximal_promoter"
    return frame
