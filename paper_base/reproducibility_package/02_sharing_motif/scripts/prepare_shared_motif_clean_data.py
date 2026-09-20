"""Prepare clean plotting tables for the Shared motifs figure.

Inputs are the Stage 2 Fi-NeMo scan tables copied from the mounted Google
Drive. Coordinates in hits.tsv are scan-window coordinates; the notebook figure
uses a -150..150 bp TSS-centered axis, so motif center distance is computed as
center - 150.

Strict-revert note (2026-07-14):
  - Panel a affinity: previously we stored a naive in-place `continuous_jaccard`
    similarity; the original notebook stores `results["raw_sim_mat"]`, which is
    `compute_global_affinity_official(scaled_cwms, min_overlap_fraction=0.5)`
    = task median-norm scaling + shift/RC-invariant continuous Jaccard. We now
    reproduce that affinity exactly and store it in the same CSV, but renamed
    `shared_motif_subcluster_similarity_affinity.csv` (the legacy
    continuous_jaccard CSV is preserved for comparison).
  - Panel d fraction: original cell #44 uses `<TASK>_Frac_In_Family` columns
    (already-family-normalized), not `_Weight` normalized in-place. We surface
    those columns from the meta-cluster workbook into the clean metadata table.
  - Panel c TSS: original cell #21 uses `real_tss_bp` = bioframe dm3 refGene
    distance (cell #3 promoter load + sc=='match' filter, cell #19 dm3 refGene
    download, cell #20 bioframe.overlap). The local promoter genomic coords are
    now sourced from DeepCAGE All_Core_Promoters_with_CAGE_Pred_and_Strand.tsv
    (the starr_cage_reconstructed.tsv equivalent); its sc=='match' filter yields
    exactly 19777 rows, matching the Fi-NeMo peak_id range 0..19776, so
    peak_id == filtered promoter row index and the bioframe overlap is exact.
    `_load_promoter_tss_map()` reproduces cell #3/#19/#20. If bioframe is not
    installed or the dm3 download fails, `real_tss_bp` falls back to the
    window-relative proxy (tss_distance_bp) so the panel still renders.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
PLOT_DIR = PROJECT / "plot"
RAW_DIR = PLOT_DIR / "data" / "raw" / "stage2_core_promoter"
RAW_IC_TRIMMED_DIR = PLOT_DIR / "data" / "raw" / "ic_trimmed_results"
META_CLUSTER_TABLE = PLOT_DIR / "data" / "raw" / "drive_tables" / "Final_Annotated_MetaClusters_IC_Trimmed.xlsx"
CLEAN_DIR = PLOT_DIR / "data" / "clean"
INTERMEDIATE_DIR = PLOT_DIR / "data" / "intermediate"

# Panel c real-TSS bridge (reproduces notebook cell #3/#19/#20):
#   PROM_FILE locally = DeepCAGE All_Core_Promoters_with_CAGE_Pred_and_Strand.tsv
#   (the Drive-side starr_cage_reconstructed.tsv equivalent). Its rows carry
#   chrom/start/end/strand/refseq_tss_strand; after cell #3's sc=='match' filter
#   (refseq_tss_strand == strand) it yields exactly 19777 rows, matching the
#   Fi-NeMo peak_id range 0..19776 — so peak_id == filtered promoter row index.
DEEPCAGE_MODULE = PROJECT.parent / "DeepCAGE"
PROM_FILE = DEEPCAGE_MODULE / "plot" / "data" / "raw" / "DATA" / "All_Core_Promoters_with_CAGE_Pred_and_Strand.tsv"
DM3_REFGENE_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/dm3/database/refGene.txt.gz"
DM3_REFGENE_CACHE = PLOT_DIR / "data" / "raw" / "dm3_refGene.txt.gz"

TASKS = ["CAGE_NEW", "HK", "DEV"]
TSS_CENTER_BP = 150.0
MIN_OVERLAP_FRACTION = 0.5  # mirrors notebook constant for compute_global_affinity_official


def motif_to_mc_id(motif_name: object) -> str:
    match = re.search(r"pattern_(\d+)", str(motif_name))
    if match is None:
        return "unmapped"
    return f"MC_{int(match.group(1)):03d}"


def task_label(task: str) -> str:
    return "CAGE" if task == "CAGE_NEW" else task


def _load_promoter_tss_map() -> tuple[pd.DataFrame, bool]:
    """Reproduce notebook cell #3 + #19 + #20 to map peak_id -> real_tss_bp.

    cell #3:  load PROM_FILE, keep rows where refseq_tss_strand == strand
              (sc == 'match'), reset_index -> peak_id == row index. Verified to
              yield exactly 19777 rows, matching the Fi-NeMo peak_id range.
    cell #19: download dm3 refGene from UCSC, build tss_bed (chrom/start/end).
    cell #20: bioframe.overlap(promoter, tss_bed) -> nearest TSS offset per peak.

    Returns (DataFrame[peak_id,chrom,start,end,real_tss_bp], bridge_active).
    bridge_active=True once the bioframe overlap actually ran (even if some
    promoters have no TSS inside their interval and stay NaN — this is the
    original notebook behavior). bridge_active=False only when bioframe is not
    installed, the dm3 download fails, or PROM_FILE is missing — in that case
    the caller falls back to the window-relative proxy so the panel still renders.
    """
    if not PROM_FILE.exists():
        return pd.DataFrame(columns=["peak_id", "chrom", "start", "end", "real_tss_bp"]), False

    # --- cell #3: load + sc=='match' filter (yields 19777 rows == peak_id range) ---
    df_prom = pd.read_csv(PROM_FILE, sep="\t")
    df_prom["sc"] = "mismatch"
    df_prom.loc[df_prom["refseq_tss_strand"] == df_prom["strand"], "sc"] = "match"
    df_prom.loc[df_prom[["refseq_tss_strand", "strand"]].isna().any(axis=1), "sc"] = "NA"
    df_prom = df_prom[df_prom["sc"] == "match"].reset_index(drop=True)
    df_prom["peak_id"] = df_prom.index  # _pidx in cell #3

    # --- cell #19: dm3 refGene -> tss_bed ---
    try:
        import bioframe as bf  # noqa: F401
    except ImportError:
        print("[panel c TSS] bioframe not installed; real_tss_bp will be NaN (proxy used).")
        df_prom["real_tss_bp"] = np.nan
        return df_prom[["peak_id", "chrom", "start", "end", "real_tss_bp"]], False

    if not DM3_REFGENE_CACHE.exists():
        DM3_REFGENE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        try:
            print(f"[panel c TSS] downloading dm3 refGene from UCSC -> {DM3_REFGENE_CACHE}")
            df_ref = pd.read_csv(
                DM3_REFGENE_URL, sep="\t", header=None, compression="gzip",
                usecols=[1, 2, 3, 4, 5],
                names=["transcript", "chrom", "strand", "txStart", "txEnd"],
            )
            df_ref.to_csv(DM3_REFGENE_CACHE, sep="\t", header=False, index=False, compression="gzip")
        except Exception as exc:  # network blocked etc.
            print(f"[panel c TSS] dm3 refGene download failed: {exc}; real_tss_bp will be NaN.")
            df_prom["real_tss_bp"] = np.nan
            return df_prom[["peak_id", "chrom", "start", "end", "real_tss_bp"]], False
    else:
        df_ref = pd.read_csv(
            DM3_REFGENE_CACHE, sep="\t", header=None, compression="gzip",
            names=["transcript", "chrom", "strand", "txStart", "txEnd"],
        )

    df_ref["tss"] = df_ref.apply(
        lambda r: r["txStart"] if r["strand"] == "+" else r["txEnd"], axis=1
    )
    tss_bed = df_ref[["chrom", "tss", "tss", "transcript", "strand"]].copy()
    tss_bed.columns = ["chrom", "start", "end", "transcript", "strand"]
    tss_bed["end"] = tss_bed["start"] + 1
    tss_bed["start"] = tss_bed["start"].astype(int)
    tss_bed["end"] = tss_bed["end"].astype(int)
    tss_bed = tss_bed.drop_duplicates(subset=["chrom", "start", "strand"]).reset_index(drop=True)

    # --- cell #20: bioframe.overlap -> nearest TSS offset per promoter ---
    # bioframe 0.8: suffixes apply to ALL columns of both dfs (df1 -> _p, df2 -> _t).
    df_regions = df_prom[["chrom", "start", "end", "peak_id"]].copy()
    ov = bf.overlap(
        df_regions, tss_bed, suffixes=("_p", "_t"), how="inner"
    )
    valid = ov[ov["start_t"].notna()].copy() if "start_t" in ov.columns else ov.iloc[0:0]
    if valid.empty:
        print("[panel c TSS] no promoter-TSS overlaps; real_tss_bp will be NaN.")
        df_prom["real_tss_bp"] = np.nan
        # bridge ran (overlap executed, just no hits) — keep NaN, do NOT proxy.
        return df_prom[["peak_id", "chrom", "start", "end", "real_tss_bp"]], True

    # real_tss_bp = TSS genomic pos - promoter start (cell #20 tss_offset, sign-aware)
    valid["real_tss_bp"] = valid["start_t"].astype(int) - valid["start_p"].astype(int)
    # When multiple TSS overlap a promoter, take the nearest to the promoter center.
    valid["prom_center"] = (valid["start_p"] + valid["end_p"]) / 2.0
    valid["dist_abs"] = (valid["real_tss_bp"] - (valid["prom_center"] - valid["start_p"])).abs()
    nearest = valid.sort_values("dist_abs").drop_duplicates("peak_id_p")
    tss_map = nearest.set_index("peak_id_p")["real_tss_bp"]
    df_prom["real_tss_bp"] = df_prom["peak_id"].map(tss_map)
    print(
        f"[panel c TSS] real_tss_bp computed for "
        f"{df_prom['real_tss_bp'].notna().sum()}/{len(df_prom)} promoters "
        f"(median {df_prom['real_tss_bp'].median():.1f} bp; the rest stay NaN as "
        f"in the original notebook — their 249bp interval contains no refGene TSS)."
    )
    return df_prom[["peak_id", "chrom", "start", "end", "real_tss_bp"]], True


def continuous_jaccard(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    """Continuous Jaccard on non-negative CWM weights.

    Legacy in-place metric (no scaling, no shift/RC invariance). Kept only to
    reproduce the previous `shared_motif_subcluster_similarity.csv`; panel a now
    uses `compute_global_affinity` (see below) to match the notebook's
    `results["raw_sim_mat"]`.
    """
    a = np.maximum(np.asarray(matrix_a, dtype=float), 0.0).ravel()
    b = np.maximum(np.asarray(matrix_b, dtype=float), 0.0).ravel()
    denominator = np.maximum(a, b).sum()
    if denominator <= 0:
        return 0.0
    return float(np.minimum(a, b).sum() / denominator)


def apply_median_norm_scaling(cwm_array: np.ndarray, tasks: list[str]) -> tuple[np.ndarray, dict[str, float]]:
    """Reproduce notebook `apply_median_norm_scaling`.

    1) per-task median L2 norm -> task_scale
    2) divide each CWM by its task scale
    3) L2-normalize each scaled CWM row-wise
    Returns (scaled_cwms, task_scale).
    """
    cwm_array = np.asarray(cwm_array, dtype=float)
    flat = cwm_array.reshape(len(cwm_array), -1)
    motif_norms = np.linalg.norm(flat, axis=1)
    task_norms: dict[str, list[float]] = {}
    for norm, task in zip(motif_norms, tasks):
        task_norms.setdefault(task, []).append(float(norm))
    task_scale = {task: float(np.median(values)) for task, values in task_norms.items()}
    scaled = np.zeros_like(cwm_array)
    for i, task in enumerate(tasks):
        scale = task_scale.get(task, 1.0)
        scaled[i] = cwm_array[i] / scale if scale > 1e-8 else cwm_array[i]
    norms = np.linalg.norm(scaled.reshape(len(scaled), -1), axis=1, keepdims=True)
    scaled = scaled / (norms[:, None] + 1e-6)
    return scaled, task_scale


def _continuous_jaccard_aligned(c1: np.ndarray, c2: np.ndarray) -> float:
    """Continuous Jaccard for one (shift, strand) alignment.

    Mirrors the inner loop of notebook `compute_global_affinity_official`:
    sign-aware (signed min / abs max), supports negative CWM weights.
    """
    sign = np.sign(c1) * np.sign(c2)
    abs_x = np.abs(c1)
    abs_y = np.abs(c2)
    min_sum = np.minimum(abs_x, abs_y) * sign
    max_sum = np.maximum(abs_x, abs_y)
    denom = max_sum.sum()
    if denom <= 1e-6:
        return 0.0
    return float(min_sum.sum() / denom)


def compute_global_affinity(cwms: np.ndarray, min_overlap_fraction: float = MIN_OVERLAP_FRACTION) -> np.ndarray:
    """Pure-numpy port of notebook `compute_global_affinity_official` (numba).

    For each (i, j) pair, tries both strands (fwd / reverse-complement) and all
    shifts that keep >= min_overlap_fraction overlap; takes the max similarity.
    Diagonal = 1.0. Symmetric output.
    """
    cwms = np.asarray(cwms, dtype=float)
    n = len(cwms)
    L, C = cwms[0].shape
    min_overlap = int(L * min_overlap_fraction)
    max_shift = L - min_overlap
    rc_cwms = cwms[:, ::-1, ::-1]
    sim_mat = np.zeros((n, n), dtype=float)
    np.fill_diagonal(sim_mat, 1.0)
    for i in range(n):
        c1 = cwms[i]
        for j in range(i + 1, n):
            c2_fwd = cwms[j]
            c2_rev = rc_cwms[j]
            best = -1.0
            for c2_target in (c2_fwd, c2_rev):
                for shift in range(-max_shift, max_shift + 1):
                    start_idx = min(0, -shift)
                    end_idx = max(L, L - shift)
                    rows1 = []
                    rows2 = []
                    for r in range(start_idx, end_idx):
                        r1 = r
                        r2 = r + shift
                        if 0 <= r1 < L and 0 <= r2 < L:
                            rows1.append(c1[r1])
                            rows2.append(c2_target[r2])
                    if not rows1:
                        continue
                    a = np.array(rows1)
                    b = np.array(rows2)
                    sim = _continuous_jaccard_aligned(a, b)
                    if sim > best:
                        best = sim
            sim_mat[i, j] = best
            sim_mat[j, i] = best
    return sim_mat


def build_subcluster_similarity_tables() -> None:
    entities_path = RAW_IC_TRIMMED_DIR / "entities_trimmed.pkl"
    results_path = RAW_IC_TRIMMED_DIR / "results_complete.pkl"
    if not entities_path.exists() or not results_path.exists():
        return

    with entities_path.open("rb") as handle:
        entities = pickle.load(handle)
    with results_path.open("rb") as handle:
        results = pickle.load(handle)

    entity_names = list(results["entity_names"])
    meta_labels = np.asarray(results["meta_labels"], dtype=int)
    missing = [name for name in entity_names if name not in entities]
    if missing:
        raise KeyError(f"Missing {len(missing)} entities required by results_complete.pkl")

    metadata = pd.DataFrame(
        {
            "entity_name": entity_names,
            "task": [entities[name].get("task", str(name).split("|", 1)[0]) for name in entity_names],
            "task_label": [task_label(entities[name].get("task", str(name).split("|", 1)[0])) for name in entity_names],
            "meta_label": meta_labels,
            "mc_id": [f"MC_{label:03d}" for label in meta_labels],
            "weight": [entities[name].get("weight", np.nan) for name in entity_names],
            "core_start": [entities[name].get("core_start", np.nan) for name in entity_names],
            "core_end": [entities[name].get("core_end", np.nan) for name in entity_names],
            "core_len": [entities[name].get("core_len", np.nan) for name in entity_names],
            "ic_ratio": [entities[name].get("ic_ratio", np.nan) for name in entity_names],
        }
    )
    task_rank = {task: idx for idx, task in enumerate(TASKS)}
    metadata["_task_rank"] = metadata["task"].map(task_rank).fillna(len(task_rank)).astype(int)
    metadata = metadata.sort_values(["meta_label", "_task_rank", "entity_name"]).reset_index(drop=True)
    order = [entity_names.index(name) for name in metadata["entity_name"]]

    cwms = [entities[name]["cwm"] for name in entity_names]
    cwm_array = np.array(cwms, dtype=float)
    tasks_in_order = [entities[name].get("task", str(name).split("|", 1)[0]) for name in entity_names]

    # Legacy: naive in-place continuous Jaccard (no scaling, no shift/RC).
    similarity_legacy = np.eye(len(cwms), dtype=float)
    for i in range(len(cwms)):
        for j in range(i + 1, len(cwms)):
            value = continuous_jaccard(cwms[i], cwms[j])
            similarity_legacy[i, j] = value
            similarity_legacy[j, i] = value

    # Strict revert: notebook `results["raw_sim_mat"]` =
    # compute_global_affinity_official(scaled_cwms, min_overlap_fraction=0.5).
    scaled_cwms, _task_scale = apply_median_norm_scaling(cwm_array, tasks_in_order)
    affinity = compute_global_affinity(scaled_cwms, min_overlap_fraction=MIN_OVERLAP_FRACTION)

    similarity_legacy = similarity_legacy[np.ix_(order, order)]
    affinity = affinity[np.ix_(order, order)]

    metadata = metadata.drop(columns=["_task_rank"])
    metadata.insert(0, "subcluster_order", np.arange(len(metadata), dtype=int))
    metadata.to_csv(CLEAN_DIR / "shared_motif_subcluster_metadata.csv", index=False)
    # Legacy similarity CSV (kept for back-compat; assemble script no longer reads it).
    pd.DataFrame(
        similarity_legacy,
        index=metadata["entity_name"],
        columns=metadata["entity_name"],
    ).to_csv(CLEAN_DIR / "shared_motif_subcluster_similarity.csv")
    # New affinity CSV (raw_sim_mat equivalent) — panel a source of truth.
    pd.DataFrame(
        affinity,
        index=metadata["entity_name"],
        columns=metadata["entity_name"],
    ).to_csv(CLEAN_DIR / "shared_motif_subcluster_similarity_affinity.csv")


def read_task_tables(task: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_dir = RAW_DIR / task
    hits_path = task_dir / "hits.tsv"
    peaks_path = task_dir / "peaks_qc.tsv"
    motif_path = task_dir / "motif_data.tsv"
    for path in [hits_path, peaks_path, motif_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    hits = pd.read_csv(hits_path, sep="\t")
    peaks = pd.read_csv(peaks_path, sep="\t")
    motifs = pd.read_csv(motif_path, sep="\t")
    return hits, peaks, motifs


def build_clean_tables() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    # Load meta-cluster workbook once for TF labels (Match_1) — used by panels c & d.
    mc_to_tf: dict[str, str] = {}
    if META_CLUSTER_TABLE.exists():
        _meta = pd.read_excel(META_CLUSTER_TABLE)
        for _, row in _meta.iterrows():
            mc_to_tf[str(row["MC_ID"])] = str(row.get("Match_1", "")).split("|")[0]

    # Real-TSS bridge (notebook cell #3/#19/#20): peak_id -> genomic real_tss_bp.
    # `_load_promoter_tss_map` reproduces the notebook exactly: cell #20 uses
    # `bf.overlap(..., how='inner')`, so promoters with NO refGene TSS inside
    # their 249bp interval get real_tss_bp = NaN — this is the ORIGINAL behavior
    # (~26% of promoters; the original notebook also leaves them NaN and the
    # per-TF boxplot simply drops them). We therefore keep NaN where the bridge
    # ran but found no overlap. Only when bioframe/download itself is unavailable
    # (bridge never ran) do we fall back to the window-relative proxy so the
    # panel still renders.
    prom_tss, tss_bridge_active = _load_promoter_tss_map()
    peak_to_real_tss: dict[int, float] = {}
    if not prom_tss.empty and "real_tss_bp" in prom_tss.columns:
        peak_to_real_tss = (
            prom_tss.dropna(subset=["real_tss_bp"])
            .set_index("peak_id")["real_tss_bp"]
            .to_dict()
        )

    density_rows: list[pd.DataFrame] = []
    contribution_rows: list[pd.DataFrame] = []
    motif_rows: list[pd.DataFrame] = []
    validation_rows = []

    for task in TASKS:
        hits, peaks, motifs = read_task_tables(task)
        hits = hits.copy()
        hits["task"] = task
        hits["task_label"] = task_label(task)
        hits["mc_id"] = hits["motif_name"].map(motif_to_mc_id)
        hits["motif_center"] = (hits["start"].astype(float) + hits["end"].astype(float)) / 2.0
        hits["tss_distance_bp"] = hits["motif_center"] - TSS_CENTER_BP
        # real_tss_bp: genomic TSS distance from cell #20 (bioframe dm3 overlap).
        # When the bridge ran, NaN means "no TSS inside this promoter" (original
        # behavior — kept as NaN, boxplot drops it). When the bridge did NOT run
        # (bioframe/download unavailable), fall back to window-relative proxy.
        hits["real_tss_bp"] = hits["peak_id"].map(peak_to_real_tss)
        if not tss_bridge_active:
            missing = hits["real_tss_bp"].isna()
            if missing.any():
                hits.loc[missing, "real_tss_bp"] = hits.loc[missing, "tss_distance_bp"]
        # TF label per hit (TomTom Match_1) — drives panel c per-TF boxplot.
        hits["tf"] = hits["mc_id"].map(mc_to_tf).fillna("")

        median_importance = hits.loc[hits["hit_importance"] > 0, "hit_importance"].median()
        if not np.isfinite(median_importance) or median_importance <= 0:
            median_importance = 1.0
        hits["hit_importance_norm"] = hits["hit_importance"] / median_importance

        all_peak_ids = pd.Index(peaks["peak_id"].astype(int).unique(), name="peak_id")
        hit_count = hits.groupby("peak_id").size().reindex(all_peak_ids, fill_value=0)
        motif_complexity = hits.groupby("peak_id")["mc_id"].nunique().reindex(all_peak_ids, fill_value=0)
        density = pd.DataFrame(
            {
                "task": task,
                "task_label": task_label(task),
                "peak_id": all_peak_ids.to_numpy(),
                "motif_density": hit_count.to_numpy(dtype=int),
                "motif_complexity": motif_complexity.to_numpy(dtype=int),
            }
        )
        density_rows.append(density)

        keep_cols = [
            "task",
            "task_label",
            "peak_id",
            "motif_name",
            "mc_id",
            "tf",
            "start",
            "end",
            "strand",
            "hit_importance",
            "hit_importance_norm",
            "motif_center",
            "tss_distance_bp",
            "real_tss_bp",
        ]
        contribution_rows.append(hits[keep_cols])

        motifs = motifs.copy()
        motifs["task"] = task
        motifs["task_label"] = task_label(task)
        motifs["mc_id"] = motifs["motif_name"].map(motif_to_mc_id)
        motif_rows.append(motifs)

        validation_rows.append(
            {
                "task": task,
                "hits_rows": len(hits),
                "peaks_rows": len(peaks),
                "motifs_rows": len(motifs),
                "unique_peak_ids_in_hits": hits["peak_id"].nunique(),
                "unique_peak_ids_in_peaks": peaks["peak_id"].nunique(),
                "unique_mc_ids": hits["mc_id"].nunique(),
                "median_hit_importance": median_importance,
            }
        )

    density_all = pd.concat(density_rows, ignore_index=True)
    contribution_all = pd.concat(contribution_rows, ignore_index=True)
    motif_all = pd.concat(motif_rows, ignore_index=True)
    validation = pd.DataFrame(validation_rows)

    density_all.to_csv(CLEAN_DIR / "shared_motif_density_complexity.csv", index=False)
    contribution_all.to_csv(CLEAN_DIR / "shared_motif_hit_contribution_tss.csv", index=False)
    motif_all.to_csv(CLEAN_DIR / "shared_motif_finemo_motif_data.csv", index=False)
    validation.to_csv(INTERMEDIATE_DIR / "shared_motif_clean_data_validation.csv", index=False)
    # Surface Frac_In_Family + TF labels for panel d (vertical bar + _Frac_In_Family).
    _write_meta_enrichment_table()
    build_subcluster_similarity_tables()


def _write_meta_enrichment_table() -> None:
    """Write a clean per-meta-cluster enrichment table for panel d.

    Mirrors the columns consumed by notebook cell #44: MC_ID, Match_1..Match_4
    (for TF_LABEL dedup) and `<TASK>_Frac_In_Family` (already family-normalized
    fractions). Local panel d reads this instead of re-deriving fractions from
    `_Weight` (which is what the previous horizontal barh did).
    """
    if not META_CLUSTER_TABLE.exists():
        return
    meta = pd.read_excel(META_CLUSTER_TABLE)
    keep = ["MC_ID", "Total_Weight", "Num_Members", "Family_Type",
            "HK_Frac_In_Family", "CAGE_NEW_Frac_In_Family", "DEV_Frac_In_Family",
            "Match_1", "Match_2", "Match_3", "Match_4"]
    keep = [c for c in keep if c in meta.columns]
    meta[keep].to_csv(CLEAN_DIR / "shared_motif_meta_enrichment.csv", index=False)


if __name__ == "__main__":
    build_clean_tables()
