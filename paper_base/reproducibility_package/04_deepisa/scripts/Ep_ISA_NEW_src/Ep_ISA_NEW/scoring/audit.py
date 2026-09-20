from pathlib import Path

import numpy as np
import pandas as pd


def positive_tail_threshold(values, percentile):
    values = pd.Series(values).dropna().astype(float)
    pos = values[values > 0]
    return float(np.percentile(pos, percentile)) if len(pos) else np.nan


def tail_thresholds(values, percentile):
    values = pd.Series(values).dropna().astype(float)
    pos = values[values > 0]
    neg = values[values < 0]
    return {
        "pos": float(np.percentile(pos, percentile)) if len(pos) else np.nan,
        "neg": float(np.percentile(neg, 100 - percentile)) if len(neg) else np.nan,
    }


def audit_single_filter(data_dir, track, null_percentile=80):
    data_dir = Path(data_dir)
    single_path = data_dir / "motif_single_isa.csv"
    null_path = data_dir / "null_isa.csv"
    if not single_path.exists() or not null_path.exists():
        return {
            "single_audit_status": "missing",
            "single_rows": np.nan,
            "single_target_positive_threshold": np.nan,
            "single_rows_above_target_positive_threshold": np.nan,
            "single_rows_below_or_equal_target_positive_threshold": np.nan,
            "single_positive_filter_consistent": False,
        }

    single = pd.read_csv(single_path)
    null = pd.read_csv(null_path)
    col = f"isa_t{track}"
    thr = positive_tail_threshold(null[col], null_percentile)
    above = int((single[col] >= thr).sum())
    below = int((single[col] < thr).sum())
    return {
        "single_audit_status": "ok",
        "single_rows": int(len(single)),
        "single_target_positive_threshold": thr,
        "single_rows_above_target_positive_threshold": above,
        "single_rows_below_target_positive_threshold": below,
        "single_positive_filter_consistent": bool(below == 0),
    }


def audit_interaction_gate(data_dir, track, null_percentile=80):
    data_dir = Path(data_dir)
    combi_path = data_dir / "motif_combi_isa.csv"
    null_isa_path = data_dir / "null_isa.csv"
    null_inter_path = data_dir / "null_interaction.csv"
    if not combi_path.exists() or not null_isa_path.exists() or not null_inter_path.exists():
        return {
            "interaction_audit_status": "missing",
            "combi_rows": np.nan,
            "interaction_non_nan_rows": np.nan,
            "interaction_nan_rows": np.nan,
        }, pd.DataFrame()

    combi = pd.read_csv(combi_path)
    null_isa = pd.read_csv(null_isa_path)
    null_inter = pd.read_csv(null_inter_path)
    isa_col = f"isa_t{track}"
    isa1 = f"isa1_t{track}"
    isa2 = f"isa2_t{track}"
    both = f"isa_both_t{track}"
    inter = f"interaction_t{track}"
    isa_thr = positive_tail_threshold(null_isa[isa_col], null_percentile)

    df = combi.copy()
    df["isa1_wo2"] = df[both] - df[isa2]
    df["isa2_wo1"] = df[both] - df[isa1]
    df["pass_isa1"] = df[isa1] >= isa_thr
    df["pass_isa2"] = df[isa2] >= isa_thr
    df["pass_isa1_wo2"] = df["isa1_wo2"] >= 0
    df["pass_isa2_wo1"] = df["isa2_wo1"] >= 0
    df["pass_all_new_interaction_gate"] = (
        df["pass_isa1"]
        & df["pass_isa2"]
        & df["pass_isa1_wo2"]
        & df["pass_isa2_wo1"]
    )

    if inter in df.columns:
        non_nan_mask = df[inter].notna()
        raw_formula = df[isa1] + df[isa2] - df[both]
        median_abs_diff = (
            float((df.loc[non_nan_mask, inter] - raw_formula.loc[non_nan_mask]).abs().median())
            if non_nan_mask.any()
            else np.nan
        )
        non_nan = int(non_nan_mask.sum())
        nan = int(df[inter].isna().sum())
    else:
        median_abs_diff = np.nan
        non_nan = 0
        nan = int(len(df))

    summary = {
        "interaction_audit_status": "ok",
        "combi_rows": int(len(df)),
        "null_interaction_rows": int(len(null_inter)),
        "interaction_non_nan_rows": non_nan,
        "interaction_nan_rows": nan,
        "interaction_nan_fraction": float(nan / len(df)) if len(df) else np.nan,
        "interaction_gate_pass_all_rows": int(df["pass_all_new_interaction_gate"].sum()),
        "interaction_gate_matches_non_nan": bool(non_nan == int(df["pass_all_new_interaction_gate"].sum())),
        "pass_isa1_rows": int(df["pass_isa1"].sum()),
        "pass_isa2_rows": int(df["pass_isa2"].sum()),
        "pass_both_single_rows": int((df["pass_isa1"] & df["pass_isa2"]).sum()),
        "pass_isa1_wo2_rows": int(df["pass_isa1_wo2"].sum()),
        "pass_isa2_wo1_rows": int(df["pass_isa2_wo1"].sum()),
        "pass_both_conditional_rows": int((df["pass_isa1_wo2"] & df["pass_isa2_wo1"]).sum()),
        "median_abs_diff_interaction_vs_raw_formula": median_abs_diff,
    }

    fail_reasons = pd.DataFrame({
        "reason": [
            "fail_isa1_below_single_null",
            "fail_isa2_below_single_null",
            "fail_isa1_wo2_negative",
            "fail_isa2_wo1_negative",
        ],
        "n_failed": [
            int((~df["pass_isa1"]).sum()),
            int((~df["pass_isa2"]).sum()),
            int((~df["pass_isa1_wo2"]).sum()),
            int((~df["pass_isa2_wo1"]).sum()),
        ],
    })
    return summary, fail_reasons


def audit_task_results(data_dir, track, null_percentile=80, expected_null_n=8192):
    data_dir = Path(data_dir)
    single = audit_single_filter(data_dir, track, null_percentile)
    inter, fail_reasons = audit_interaction_gate(data_dir, track, null_percentile)

    null_isa_path = data_dir / "null_isa.csv"
    null_inter_path = data_dir / "null_interaction.csv"
    pred_orig_path = data_dir / "pred_orig.csv"
    null_isa_n = len(pd.read_csv(null_isa_path)) if null_isa_path.exists() else np.nan
    null_inter_n = len(pd.read_csv(null_inter_path)) if null_inter_path.exists() else np.nan
    if pred_orig_path.exists():
        pred_orig_cols = list(pd.read_csv(pred_orig_path, nrows=1).columns)
    else:
        pred_orig_cols = []
    pred_col = f"pred_t{track}"

    out = {
        "data_dir": str(data_dir),
        "track": int(track),
        "null_isa_rows": int(null_isa_n) if pd.notna(null_isa_n) else np.nan,
        "null_interaction_rows": int(null_inter_n) if pd.notna(null_inter_n) else np.nan,
        "null_isa_near_expected": bool(pd.notna(null_isa_n) and null_isa_n >= expected_null_n * 0.95),
        "null_interaction_near_expected": bool(pd.notna(null_inter_n) and null_inter_n >= expected_null_n * 0.95),
        "pred_orig_has_target_track": bool(pred_col in pred_orig_cols),
        "pred_orig_columns": ",".join(pred_orig_cols),
    }
    out.update(single)
    out.update(inter)
    return out, fail_reasons
