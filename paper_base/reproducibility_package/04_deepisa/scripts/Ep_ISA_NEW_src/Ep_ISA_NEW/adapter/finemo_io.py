import numpy as np
import pandas as pd
import bioframe as bf
from loguru import logger


FINEMO_HITS_COLUMNS = [
    'chr', 'start', 'end', 'start_untrimmed', 'end_untrimmed',
    'motif_name', 'hit_coefficient', 'hit_coefficient_global',
    'hit_similarity', 'hit_correlation', 'hit_importance',
    'hit_importance_sq', 'strand', 'peak_name', 'peak_id',
]

MOTIF_LOCS_COLUMNS = [
    'chrom', 'start', 'end', 'start_rel', 'end_rel',
    'tf', 'score', 'strand', 'region',
]

RELATIVE_COORD_MARKERS = ('NA', 'chrFake', 'nan', '', 'None')


def load_finemo_hits(hits_tsv_path):
    df = pd.read_csv(hits_tsv_path, sep='\t')
    missing = set(FINEMO_HITS_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Fi-NeMo hits.tsv missing columns: {missing}")
    return df


def load_motif_annotation(finemo_h5_path):
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required only when finemo_h5_path is provided for motif annotation."
        ) from exc

    mapping = {}
    with h5py.File(finemo_h5_path, 'r') as f:
        if 'pos_patterns' not in f:
            logger.warning("No 'pos_patterns' group in H5")
            return mapping
        for pat_name, grp in f['pos_patterns'].items():
            mc_id = grp.attrs.get('mc_id', '')
            tf_label = grp.attrs.get('tf_label', '')
            if isinstance(mc_id, bytes):
                mc_id = mc_id.decode()
            if isinstance(tf_label, bytes):
                tf_label = tf_label.decode()
            if not tf_label:
                tf_label = 'Unknown'
            mapping[f"pos_patterns.{pat_name}"] = {
                'MC_ID': mc_id,
                'TF_Name': tf_label,
            }
    return mapping


def annotate_hits(df_hits, finemo_h5_path):
    mapping = load_motif_annotation(finemo_h5_path)
    df = df_hits.copy()
    df['MC_ID'] = df['motif_name'].map(
        lambda x: mapping.get(x, {}).get('MC_ID', ''))
    df['TF_Name'] = df['motif_name'].map(
        lambda x: mapping.get(x, {}).get('TF_Name', 'Unknown'))
    n_unknown = (df['TF_Name'] == 'Unknown').sum()
    if n_unknown > 0:
        logger.warning(f"{n_unknown} hits have unknown TF annotation")
    return df


def prepare_region_map(region_df):
    df = region_df.copy()
    if 'peak_id' not in df.columns:
        df['peak_id'] = df.index
    if 'region' not in df.columns:
        df['region'] = df.apply(
            lambda r: f"{r['chrom']}:{int(r['start'])}-{int(r['end'])}",
            axis=1)
    region_map = {}
    for _, row in df.iterrows():
        region_map[int(row['peak_id'])] = {
            'chrom': str(row['chrom']),
            'start': int(row['start']),
            'end': int(row['end']),
            'region': str(row['region']),
        }
    return region_map


def _is_relative_coords(df):
    if len(df) == 0 or 'chr' not in df.columns:
        return True
    sample = str(df['chr'].iloc[0])
    return sample in RELATIVE_COORD_MARKERS


def hits_to_motif_locs(
    df_hits,
    region_map=None,
    score_col='hit_coefficient',
    score_threshold=None,
    finemo_h5_path=None,
    similarity_threshold=None,
):
    """
    Convert Fi-NeMo hits to deepISA motif_locs.csv format.

    region_map: {peak_id: {chrom, start, end, region}}.
        Auto-detects whether hits use relative coords (chr=NA/chrFake)
        or absolute genomic coords, and converts accordingly.

    score_threshold: minimum |score| to keep. Uses absolute value
        because Fi-NeMo hit_coefficient can be negative (repressive).

    similarity_threshold: minimum hit_similarity to keep (e.g. 0.8).
    """
    df = df_hits.copy()

    if score_col not in df.columns:
        score_cols = [c for c in df.columns if 'coefficient' in c or 'score' in c]
        raise KeyError(
            f"score_col='{score_col}' not in hits columns. "
            f"Available score-like columns: {score_cols}")

    if finemo_h5_path is not None and 'TF_Name' not in df.columns:
        df = annotate_hits(df, finemo_h5_path)

    tf_col = 'TF_Name' if 'TF_Name' in df.columns else 'motif_name'

    if similarity_threshold is not None and 'hit_similarity' in df.columns:
        before = len(df)
        df = df[df['hit_similarity'] >= similarity_threshold].copy()
        logger.info(f"Similarity filter (>={similarity_threshold}): "
                     f"{before} -> {len(df)}")

    is_relative = _is_relative_coords(df)

    if region_map is not None:
        _validate_peak_ids(df, region_map)
        region_starts = df['peak_id'].map(
            lambda x: region_map.get(int(x), {}).get('start', 0))

        if is_relative:
            df['start_rel'] = df['start'].astype(int)
            df['end_rel'] = df['end'].astype(int)
            df['start'] = (region_starts + df['start_rel']).astype(int)
            df['end'] = (region_starts + df['end_rel']).astype(int)
        else:
            df['start'] = df['start'].astype(int)
            df['end'] = df['end'].astype(int)
            df['start_rel'] = df['start'] - region_starts.astype(int)
            df['end_rel'] = df['end'] - region_starts.astype(int)

        df['chrom'] = df['peak_id'].map(
            lambda x: region_map.get(int(x), {}).get('chrom', 'NA'))
        df['region'] = df['peak_id'].map(
            lambda x: region_map.get(int(x), {}).get('region', f'peak_{x}'))
    else:
        df['chrom'] = df['chr']
        df['start'] = df['start'].astype(int)
        df['end'] = df['end'].astype(int)
        df['start_rel'] = 0
        df['end_rel'] = df['end'] - df['start']
        df['region'] = df.apply(
            lambda r: f"{r['chrom']}:{r['start']}-{r['end']}", axis=1)

    df['score'] = df[score_col]
    df['tf'] = df[tf_col].astype(str).str.upper()
    df['strand'] = df['strand'].astype(str)

    if score_threshold is not None:
        before = len(df)
        df = df[df['score'].abs() >= score_threshold].copy()
        logger.info(f"Score filter (|score| >= {score_threshold}): "
                     f"{before} -> {len(df)}")

    result = df[MOTIF_LOCS_COLUMNS].drop_duplicates().reset_index(drop=True)
    logger.info(f"motif_locs: {len(result)} rows, "
                 f"{result['tf'].nunique()} TFs, "
                 f"{result['region'].nunique()} regions")
    return result


def _validate_peak_ids(df, region_map):
    hit_ids = set(df['peak_id'].apply(lambda x: int(x) if pd.notna(x) else -1))
    map_ids = set(region_map.keys())
    missing = hit_ids - map_ids
    if missing:
        n_before = len(df)
        df.drop(df[df['peak_id'].apply(lambda x: int(x) if pd.notna(x) else -1).isin(missing)].index, inplace=True)
        logger.warning(
            f"{len(missing)} peak_ids in hits not in region_map (filtered). "
            f"Hits: {n_before} -> {len(df)}. Use full df_regions for complete analysis.")


def compute_non_motif_regions(df_motif_locs, region_map, flank=5, min_len=4, max_len=500):
    rows = []
    for peak_id, info in region_map.items():
        chrom = info['chrom']
        reg_start = info['start']
        reg_end = info['end']
        region_str = info['region']

        region_row = pd.DataFrame([{
            'chrom': chrom, 'start': reg_start, 'end': reg_end,
            'region': region_str
        }])

        motif_mask = df_motif_locs[df_motif_locs['region'] == region_str].copy()
        if not motif_mask.empty:
            motif_mask['start'] = motif_mask['start'] - flank
            motif_mask['end'] = motif_mask['end'] + flank
            motif_mask = motif_mask[['chrom', 'start', 'end']]
        else:
            motif_mask = pd.DataFrame(columns=['chrom', 'start', 'end'])

        non_motif = region_row.copy() if motif_mask.empty else bf.subtract(region_row, motif_mask)
        if non_motif.empty:
            continue
        non_motif = non_motif[['chrom', 'start', 'end']].copy()
        non_motif['region'] = region_str
        non_motif['start_rel'] = non_motif['start'] - reg_start
        non_motif['end_rel'] = non_motif['end'] - reg_start
        non_motif['len'] = non_motif['end'] - non_motif['start']
        non_motif = non_motif[
            (non_motif['len'] >= min_len) & (non_motif['len'] < max_len)]
        rows.append(non_motif)

    if not rows:
        logger.warning("No non-motif regions generated")
        return pd.DataFrame(columns=[
            'chrom', 'start', 'end', 'region', 'start_rel', 'end_rel', 'len'])

    result = pd.concat(rows, ignore_index=True)
    logger.info(f"non_motif_locs: {len(result)} intervals")
    return result


def suggest_score_threshold(df_hits, score_col='hit_coefficient', percentile=50):
    """
    Suggest a score threshold from |score| distribution.
    Fi-NeMo hit_coefficient can be negative (repressive motif);
    we use absolute value so both activating and repressive hits
    above the intensity threshold are kept.
    """
    vals = df_hits[score_col].dropna().abs()
    suggested = float(np.percentile(vals, percentile))
    logger.info(f"Suggested |{score_col}| threshold (P{percentile}): {suggested:.4f}")
    return suggested


def load_finemo_scan(
    hits_tsv_path,
    region_df,
    finemo_h5_path=None,
    motif_outpath=None,
    non_motif_outpath=None,
    score_col='hit_coefficient',
    score_threshold=None,
    similarity_threshold=None,
):
    """
    High-level API: load Fi-NeMo hits and produce motif_locs + non_motif_locs.

    region_df: DataFrame [peak_id, chrom, start, end].
    Returns: (df_motif_locs, df_non_motif_locs)
    """
    logger.info(f"Loading Fi-NeMo hits: {hits_tsv_path}")
    df_hits = load_finemo_hits(hits_tsv_path)
    logger.info(f"  {len(df_hits)} raw hits, "
                 f"{df_hits['motif_name'].nunique()} motifs, "
                 f"{df_hits['peak_id'].nunique()} peaks")

    region_map = prepare_region_map(region_df)

    df_motif_locs = hits_to_motif_locs(
        df_hits,
        region_map=region_map,
        score_col=score_col,
        score_threshold=score_threshold,
        finemo_h5_path=finemo_h5_path,
        similarity_threshold=similarity_threshold,
    )

    df_non_motif = compute_non_motif_regions(df_motif_locs, region_map)

    if motif_outpath:
        df_motif_locs.to_csv(motif_outpath, index=False)
        logger.info(f"Saved motif_locs: {motif_outpath}")
    if non_motif_outpath:
        df_non_motif.to_csv(non_motif_outpath, index=False)
        logger.info(f"Saved non_motif_locs: {non_motif_outpath}")

    return df_motif_locs, df_non_motif
