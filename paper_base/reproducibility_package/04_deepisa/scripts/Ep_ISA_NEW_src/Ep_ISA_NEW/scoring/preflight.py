import numpy as np
import pandas as pd
from loguru import logger
from typing import Optional


DEDUP_KEYS = ["chrom", "start", "end", "region"]


def deduplicate_motif_locs(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in DEDUP_KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"motif_locs is missing required columns for audit: {missing}")

    if "score" in df.columns:
        return (
            df.sort_values("score", ascending=False)
            .drop_duplicates(subset=DEDUP_KEYS, keep="first")
            .reset_index(drop=True)
        )
    return df.drop_duplicates(subset=DEDUP_KEYS, keep="first").reset_index(drop=True)


def motif_family(tf_name) -> str:
    """Coarse label for overlap diagnostics; exact TF names remain in the detail table."""
    text = str(tf_name)
    return text.split("/")[0] if "/" in text else text


def _count_region_pairs(region_df: pd.DataFrame, receptive_field: int) -> dict:
    region_df = region_df.sort_values(["start_rel", "end_rel"]).reset_index(drop=True)
    n = len(region_df)
    all_pairs = n * (n - 1) // 2
    rf_pairs = 0
    overlapping_or_adjacent = 0
    too_far = 0
    sampled_distances = []
    starts = region_df["start_rel"].to_numpy(dtype=float)
    ends = region_df["end_rel"].to_numpy(dtype=float)

    for i in range(n - 1):
        first_nonoverlap = max(i + 1, int(np.searchsorted(starts, ends[i], side="right")))
        first_too_far = max(i + 1, int(np.searchsorted(starts, ends[i] + receptive_field, side="right")))
        overlap_n = max(0, first_nonoverlap - (i + 1))
        valid_n = max(0, first_too_far - first_nonoverlap)
        far_n = max(0, n - first_too_far)

        overlapping_or_adjacent += overlap_n
        rf_pairs += valid_n
        too_far += far_n

        if valid_n and len(sampled_distances) < 100000:
            sampled_distances.extend((starts[first_nonoverlap:first_too_far] - ends[i]).tolist())

    return {
        "n_motifs": n,
        "all_pairs_same_region": all_pairs,
        "receptive_field_pairs": rf_pairs,
        "overlapping_or_adjacent_pairs": overlapping_or_adjacent,
        "too_far_pairs": too_far,
        "median_valid_pair_distance": float(np.median(sampled_distances)) if sampled_distances else np.nan,
        "median_valid_pair_distance_note": "exact if <=100000 sampled valid pairs per region; otherwise first 100000 after start sorting",
    }


def _collect_overlap_pairs(region: str, region_df: pd.DataFrame) -> list[dict]:
    region_df = region_df.sort_values(["start_rel", "end_rel"]).reset_index(drop=True)
    starts = region_df["start_rel"].to_numpy(dtype=float)
    ends = region_df["end_rel"].to_numpy(dtype=float)
    rows = []

    for i in range(len(region_df) - 1):
        first_nonoverlap = max(i + 1, int(np.searchsorted(starts, ends[i], side="right")))
        for j in range(i + 1, first_nonoverlap):
            m1 = region_df.iloc[i]
            m2 = region_df.iloc[j]
            dist = float(m2.start_rel - m1.end_rel)
            fam1 = motif_family(m1.tf)
            fam2 = motif_family(m2.tf)
            rows.append({
                "region": region,
                "tf1": m1.tf,
                "tf2": m2.tf,
                "family1": fam1,
                "family2": fam2,
                "same_tf": bool(m1.tf == m2.tf),
                "same_family": bool(fam1 == fam2),
                "start1_rel": int(m1.start_rel),
                "end1_rel": int(m1.end_rel),
                "start2_rel": int(m2.start_rel),
                "end2_rel": int(m2.end_rel),
                "distance": dist,
                "overlap_bp": max(0, int(min(m1.end_rel, m2.end_rel) - max(m1.start_rel, m2.start_rel))),
                "overlap_type": "abutting" if dist == 0 else "overlapping",
            })
    return rows


def run_preflight_pair_audit(
    motif_locs_path: str,
    out_summary_path: str,
    out_region_path: str,
    out_overlap_path: Optional[str] = None,
    receptive_field: int = 255,
) -> pd.DataFrame:
    df_raw = pd.read_csv(motif_locs_path)
    df = deduplicate_motif_locs(df_raw)

    required = {"region", "start_rel", "end_rel"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"motif_locs is missing required columns for pair audit: {sorted(missing)}")

    region_rows = []
    overlap_rows = []
    for region, group in df.groupby("region"):
        row = {"region": region}
        row.update(_count_region_pairs(group, receptive_field))
        region_rows.append(row)
        if row["overlapping_or_adjacent_pairs"] > 0:
            overlap_rows.extend(_collect_overlap_pairs(region, group))

    by_region = pd.DataFrame(region_rows)
    by_region.to_csv(out_region_path, index=False)
    overlap_df = pd.DataFrame(overlap_rows)
    if out_overlap_path is not None:
        overlap_df.to_csv(out_overlap_path, index=False)

    total_single = int(len(df))
    total_all_pairs = int(by_region["all_pairs_same_region"].sum()) if not by_region.empty else 0
    total_rf_pairs = int(by_region["receptive_field_pairs"].sum()) if not by_region.empty else 0
    singleton_regions = int((by_region["n_motifs"] == 1).sum()) if not by_region.empty else 0
    no_valid_pair_regions = int((by_region["receptive_field_pairs"] == 0).sum()) if not by_region.empty else 0
    overlap_same_tf = int(overlap_df["same_tf"].sum()) if not overlap_df.empty else 0
    overlap_same_family = int(overlap_df["same_family"].sum()) if not overlap_df.empty else 0
    overlap_same_family_different_tf = int(((overlap_df["same_family"]) & (~overlap_df["same_tf"])).sum()) if not overlap_df.empty else 0
    overlap_different_family = int((~overlap_df["same_family"]).sum()) if not overlap_df.empty else 0
    true_overlapping_pairs = int((overlap_df["overlap_type"] == "overlapping").sum()) if not overlap_df.empty else 0
    abutting_pairs = int((overlap_df["overlap_type"] == "abutting").sum()) if not overlap_df.empty else 0

    summary = pd.DataFrame([{
        "motif_locs_rows_raw": int(len(df_raw)),
        "motif_locs_rows_after_new_dedup": total_single,
        "dedup_removed_rows": int(len(df_raw) - total_single),
        "regions_with_motifs": int(df["region"].nunique()),
        "singleton_regions": singleton_regions,
        "regions_without_receptive_field_pair": no_valid_pair_regions,
        "all_pairs_same_region": total_all_pairs,
        "receptive_field_pairs": total_rf_pairs,
        "overlapping_or_adjacent_pairs": int(by_region["overlapping_or_adjacent_pairs"].sum()) if not by_region.empty else 0,
        "overlap_exact_same_tf_pairs": overlap_same_tf,
        "overlap_same_family_pairs": overlap_same_family,
        "overlap_same_family_different_tf_pairs": overlap_same_family_different_tf,
        "overlap_different_family_pairs": overlap_different_family,
        "true_overlapping_pairs": true_overlapping_pairs,
        "abutting_pairs": abutting_pairs,
        "too_far_pairs": int(by_region["too_far_pairs"].sum()) if not by_region.empty else 0,
        "pair_to_single_ratio_all_same_region": total_all_pairs / total_single if total_single else np.nan,
        "pair_to_single_ratio_receptive_field": total_rf_pairs / total_single if total_single else np.nan,
        "receptive_field": int(receptive_field),
        "pair_rule": "same region, sorted by start_rel, distance=start2_rel-end1_rel, 1<=distance<=receptive_field",
    }])
    summary.to_csv(out_summary_path, index=False)

    logger.info(
        "Preflight motif/pair audit: "
        f"single={total_single}, all_same_region_pairs={total_all_pairs}, "
        f"rf_pairs={total_rf_pairs}, rf_pair/single={summary.loc[0, 'pair_to_single_ratio_receptive_field']:.3f}"
    )
    if total_rf_pairs < total_single:
        logger.warning(
            "Preflight audit found fewer receptive-field motif pairs than single motifs. "
            "This can be expected when many regions have one motif, overlapping motifs, or motif intervals farther apart than the receptive field."
        )

    return summary
