#!/usr/bin/env python
"""Prepare v2 Figure 5 and supplementary plotting tables from the current full run."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


PLOT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PLOT_ROOT / "data" / "raw"
DATA_PROCESSED = PLOT_ROOT / "data" / "processed"
LOG_DIR = PLOT_ROOT / "logs"

RUN_LABEL = "reviewer_greedy_20260724_115905"
ANALYSIS_LABEL = "attribution_finemo_24bp_annotated"
ACTIVITY_BASE = 1.0
TH_CAGE = 1.911
TH_DEV = 1.113
TH_HK = 2.861
CAGE_BINS = [-np.inf, 2.0, 4.0, np.inf]
CAGE_LABELS = ["Low (<2)", "Medium (2-4)", "High (>4)"]


def find_project_root() -> Path:
    candidates = sorted(Path("G:/").glob("*/DeepEpromote/Drosophila"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one DeepEpromote/Drosophila root on G:, found {candidates}")
    return candidates[0]


def atomic_write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, sep="\t", index=False)
    tmp.replace(path)


def copy_raw(src: Path) -> Path:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    dst = DATA_RAW / src.name
    shutil.copy2(src, dst)
    return dst


def add_cage_group(df: pd.DataFrame, value_col: str = "Pred_CAGE_log2TPM") -> pd.DataFrame:
    out = df.copy()
    out["CAGE_Group"] = pd.cut(out[value_col], bins=CAGE_BINS, labels=CAGE_LABELS)
    return out


def load_active_table(root: Path) -> pd.DataFrame:
    cage_pred = pd.read_csv(root / "DeepCAGE" / "DATA" / "All_Core_Promoters_with_CAGE_Pred_and_Strand.tsv", sep="\t")
    starr = pd.read_csv(root / "DeepCAGE" / "DATA" / "starr_cage_reconstructed.tsv", sep="\t")
    starr_add = starr[[c for c in ["ID", "Dev_pred", "new_ID"] if c in starr.columns]].drop_duplicates("ID")
    merged = cage_pred.merge(starr_add, on="ID", how="left", suffixes=("", "_starr"), validate="one_to_one")
    if "new_ID_starr" in merged.columns:
        merged["new_ID"] = merged["new_ID"].fillna(merged["new_ID_starr"])
        merged = merged.drop(columns=["new_ID_starr"])
    clean = merged.dropna(subset=["Real_CAGE_log2TPM", "Pred_CAGE_log2TPM", "Dev_true", "Hk_true", "Dev_pred", "Hk_pred"]).copy()
    clean["Res_CAGE"] = (clean["Real_CAGE_log2TPM"] - clean["Pred_CAGE_log2TPM"]).abs()
    clean["Res_Dev"] = (clean["Dev_true"] - clean["Dev_pred"]).abs()
    clean["Res_Hk"] = (clean["Hk_true"] - clean["Hk_pred"]).abs()
    clean = add_cage_group(clean)
    return clean


def prepare_cohort_tables(root: Path, greedy: pd.DataFrame) -> dict[str, pd.DataFrame]:
    active_table = load_active_table(root)
    active = active_table.loc[
        (active_table["Real_CAGE_log2TPM"] > ACTIVITY_BASE)
        | (active_table["Dev_true"] > ACTIVITY_BASE)
        | (active_table["Hk_true"] > ACTIVITY_BASE)
    ].copy()
    cage_good = active.loc[active["Res_CAGE"] <= TH_CAGE].copy()
    dev_good = active.loc[active["Res_Dev"] <= TH_DEV].copy()
    hk_good = active.loc[active["Res_Hk"] <= TH_HK].copy()

    ids_cage = set(cage_good["new_ID"].astype(str))
    ids_dev = set(dev_good["new_ID"].astype(str))
    ids_hk = set(hk_good["new_ID"].astype(str))
    partition = pd.DataFrame(
        [
            {"category": "Dev-specific", "n_sequences": len((ids_cage & ids_dev) - ids_hk)},
            {"category": "Dual function", "n_sequences": len(ids_cage & ids_dev & ids_hk)},
            {"category": "Hk-specific", "n_sequences": len((ids_cage & ids_hk) - ids_dev)},
            {"category": "Unclassified", "n_sequences": len(ids_cage - ids_dev - ids_hk)},
        ]
    )
    partition["fraction"] = partition["n_sequences"] / partition["n_sequences"].sum()

    super_consensus = pd.read_csv(
        root / "DeepSTARR" / "promoter_mut" / "data" / "CAGE_high_coff" / "Mutation_Super_Consensus.tsv",
        sep="\t",
    )
    super_consensus = add_cage_group(super_consensus)
    greedy_unique = greedy.drop_duplicates("ID").copy()
    greedy_round0 = greedy_unique.merge(
        super_consensus[
            [
                "ID",
                "Real_CAGE_log2TPM",
                "Pred_CAGE_log2TPM",
                "Dev_true",
                "Hk_true",
                "Dev_pred",
                "Hk_pred",
            ]
        ],
        left_on="source_ID",
        right_on="ID",
        how="left",
        suffixes=("_greedy", "_super"),
        validate="one_to_one",
    )
    greedy_round0["CAGE_Group_Short"] = greedy_round0["CAGE_Group"].map(
        {"Low (<2)": "Low", "Medium (2-4)": "Medium", "High (>4)": "High"}
    )
    greedy_round0 = greedy_round0.rename(
        columns={
            "ID_greedy": "ID",
            "ID_super": "super_consensus_ID",
            "init_cage": "Greedy_round0_CAGE_pred",
            "init_score": "Greedy_round0_target_pred",
        }
    )
    quad_neg = super_consensus.loc[
        (super_consensus["Dev_true"] <= 0)
        & (super_consensus["Hk_true"] <= 0)
        & (super_consensus["Dev_pred"] <= 0)
        & (super_consensus["Hk_pred"] <= 0)
    ].copy()
    quad_neg["CAGE_Residual"] = (quad_neg["Real_CAGE_log2TPM"] - quad_neg["Pred_CAGE_log2TPM"]).abs()
    quad_neg["Accuracy_Level"] = pd.cut(
        quad_neg["CAGE_Residual"],
        bins=[0, 0.5, 1.0, np.inf],
        labels=["Accurate (<0.5)", "Acceptable (0.5-1.0)", "Divergent (>1.0)"],
        include_lowest=True,
    )

    stages = pd.DataFrame(
        [
            {"stage_order": 1, "stage": "Core promoters with CAGE prediction", "n_sequences": active_table["ID"].nunique()},
            {"stage_order": 2, "stage": "Active in CAGE or STARR", "n_sequences": active["ID"].nunique()},
            {"stage_order": 3, "stage": "CAGE high-confidence", "n_sequences": cage_good["ID"].nunique()},
            {"stage_order": 4, "stage": "CAGE+DEV+HK super-consensus", "n_sequences": super_consensus["ID"].nunique()},
            {"stage_order": 5, "stage": "Reviewer-grade greedy cohort", "n_sequences": greedy["ID"].nunique()},
        ]
    )

    group_comp = []
    for name, df in [("Super-consensus", super_consensus), ("Greedy cohort", greedy.drop_duplicates("ID"))]:
        counts = df["CAGE_Group"].value_counts(sort=False)
        total = counts.sum()
        for group in CAGE_LABELS:
            n = int(counts.get(group, 0))
            group_comp.append({"source": name, "CAGE_Group": group, "n_sequences": n, "fraction": n / total})

    dual_ids = ids_cage & ids_dev & ids_hk
    dual_perf = active.loc[active["new_ID"].astype(str).isin(dual_ids)].copy()
    return {
        "cohort_filter_counts": stages,
        "cage_highconf_partition": partition,
        "cage_group_composition": pd.DataFrame(group_comp),
        "fig5b_greedy_round0_cohort": greedy_round0,
        "quad_negative_promoters": quad_neg,
        "dual_function_performance": dual_perf,
    }


def aggregate_trajectories(run_root: Path) -> pd.DataFrame:
    cache = DATA_PROCESSED / "greedy_trajectories_all.tsv"
    if cache.exists():
        return pd.read_csv(cache, sep="\t")
    files = sorted((run_root / "checkpoints" / "trajectories").glob("*.trajectory.tsv"))
    if len(files) != 830:
        print(f"WARNING: expected 830 trajectory files, found {len(files)}")
    use_cols = [
        "task",
        "seed",
        "ID",
        "source_ID",
        "CAGE_Group",
        "Round",
        "Score_Opt",
        "Dev_pred",
        "Hk_pred",
        "CAGE_pred",
        "total_muts",
        "max_10bp_muts",
        "gc_delta",
    ]
    def read_one(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, sep="\t", usecols=lambda c: c in use_cols)

    with ThreadPoolExecutor(max_workers=24) as pool:
        frames = list(pool.map(read_one, files))
    trajectories = pd.concat(frames, ignore_index=True)
    trajectories["CAGE_Group_Short"] = trajectories["CAGE_Group"].map(
        {"Low (<2)": "Low", "Medium (2-4)": "Medium", "High (>4)": "High"}
    )
    atomic_write_tsv(trajectories, cache)
    return trajectories


def prepare_motif_tables(run_root: Path, greedy: pd.DataFrame) -> dict[str, pd.DataFrame]:
    motif = pd.read_csv(run_root / ANALYSIS_LABEL / "summaries" / "motif_gain_loss_vs_round0.tsv", sep="\t")
    anno = pd.read_csv(run_root / ANALYSIS_LABEL / "summaries" / "median_motif_delta_by_track_branch.tsv", sep="\t")[
        ["MC_ID", "Family_Type", "Match_1", "Query_Consensus"]
    ].drop_duplicates()
    motif = motif.merge(anno, on="MC_ID", how="left", validate="many_to_one")
    motif["motif_label"] = motif["MC_ID"] + " " + motif["Match_1"]
    motif["new_hit6_count"] = motif["delta_hit6_vs_round0"].clip(lower=0)
    motif = motif.merge(greedy[["task", "ID", "CAGE_Group"]].drop_duplicates(), left_on=["branch_task", "ID"], right_on=["task", "ID"], how="left")

    branch_pairs = []
    for branch, track in [("HK", "HK"), ("DEV", "DEV")]:
        target = motif.loc[(motif["branch_task"] == branch) & (motif["track"] == track)]
        cage = motif.loc[(motif["branch_task"] == branch) & (motif["track"] == "CAGE")]
        target_sum = target.groupby(["branch_task", "ID"], as_index=False).agg(target_new_motifs=("new_hit6_count", "sum"))
        cage_sum = cage.groupby(["branch_task", "ID"], as_index=False).agg(cage_new_motifs=("new_hit6_count", "sum"))
        merged = target_sum.merge(cage_sum, on=["branch_task", "ID"], how="inner")
        merged = merged.merge(greedy[["task", "ID", "CAGE_Group"]].drop_duplicates(), left_on=["branch_task", "ID"], right_on=["task", "ID"], how="left")
        branch_pairs.append(merged.drop(columns=["task"]))
    motif_scatter = pd.concat(branch_pairs, ignore_index=True)

    box_rows = []
    labels = [
        ("HK target", "HK", "HK"),
        ("DEV target", "DEV", "DEV"),
        ("HK-CAGE", "CAGE", "HK"),
        ("DEV-CAGE", "CAGE", "DEV"),
    ]
    for label, track, branch in labels:
        sub = motif.loc[(motif["track"] == track) & (motif["branch_task"] == branch)].copy()
        sub["series"] = label
        box_rows.append(sub)
    motif_box = pd.concat(box_rows, ignore_index=True)
    return {"motif_new_scatter": motif_scatter, "motif_new_by_motif": motif_box, "motif_gain_loss_raw": motif}


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = find_project_root()
    run_root = root / "DeepSTARR" / "promoter_mut" / "results_context_dependent" / "reviewer_grade_runs" / RUN_LABEL
    greedy_src = run_root / "summaries" / "primary_greedy_summary.tsv"
    greedy = pd.read_csv(greedy_src, sep="\t")

    copied = {
        "primary_greedy_summary": str(copy_raw(greedy_src)),
        "mutation_super_consensus": str(copy_raw(root / "DeepSTARR" / "promoter_mut" / "data" / "CAGE_high_coff" / "Mutation_Super_Consensus.tsv")),
        "motif_gain_loss_vs_round0": str(copy_raw(run_root / ANALYSIS_LABEL / "summaries" / "motif_gain_loss_vs_round0.tsv")),
        "median_motif_delta_by_track_branch": str(copy_raw(run_root / ANALYSIS_LABEL / "summaries" / "median_motif_delta_by_track_branch.tsv")),
    }

    cohort_tables = prepare_cohort_tables(root, greedy)
    trajectories = aggregate_trajectories(run_root)
    motif_tables = prepare_motif_tables(run_root, greedy)

    atomic_write_tsv(greedy, DATA_PROCESSED / "primary_greedy_summary.tsv")
    for name, df in cohort_tables.items():
        atomic_write_tsv(df, DATA_PROCESSED / f"{name}.tsv")
    atomic_write_tsv(trajectories, DATA_PROCESSED / "greedy_trajectories_all.tsv")
    for name, df in motif_tables.items():
        atomic_write_tsv(df, DATA_PROCESSED / f"{name}.tsv")

    manifest = {
        "source_root": str(root),
        "run_label": RUN_LABEL,
        "analysis_label": ANALYSIS_LABEL,
        "thresholds": {
            "activity_base": ACTIVITY_BASE,
            "cage_residual_max": TH_CAGE,
            "dev_residual_max": TH_DEV,
            "hk_residual_max": TH_HK,
            "cage_group_bins": CAGE_BINS,
            "cage_group_labels": CAGE_LABELS,
        },
        "copied_raw": copied,
        "processed_dir": str(DATA_PROCESSED),
    }
    (LOG_DIR / "fig5_v2_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Prepared v2 Fig5 data.")
    print("Greedy:", greedy.shape)
    print("Trajectories:", trajectories.shape, "unique IDs:", trajectories["ID"].nunique())
    print("Quad-negative:", cohort_tables["quad_negative_promoters"].shape)
    print("Motif scatter:", motif_tables["motif_new_scatter"].shape)


if __name__ == "__main__":
    main()
