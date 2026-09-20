#!/usr/bin/env python
"""Audit Fig. 4 pairwise tests with Benjamini-Hochberg FDR correction.

This script does not redraw Fig. 4. It recomputes the displayed pairwise
Mann-Whitney U tests from the frozen source tables and writes explicit
raw-P and BH-adjusted q-value tables for manuscript/statistical reporting.
"""

from __future__ import annotations

from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = MODULE_ROOT / "data" / "clean"
QA_DIR = MODULE_ROOT / "qa" / "fig4_bh_fdr"
DENSITY_CSV = CLEAN_DIR / "shared_motif_density_complexity.csv"
HIT_CSV = CLEAN_DIR / "shared_motif_hit_contribution_tss.csv"
DISPLAY = {"CAGE_NEW": "CAGE", "HK": "HK", "DEV": "DEV"}


def bh_adjust(pvalues: list[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    valid_idx = np.where(valid)[0]
    q[valid_idx[order]] = adjusted
    return q


def mannwhitneyu_two_sided(x, y) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan

    values = np.concatenate([x, y])
    groups = np.concatenate([np.zeros(n1, dtype=bool), np.ones(n2, dtype=bool)])
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks_sorted = np.empty_like(sorted_values, dtype=float)
    tie_counts = []
    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks_sorted[start:end] = (start + 1 + end) / 2.0
        tie_counts.append(end - start)
        start = end
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    r1 = ranks[~groups].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    n = n1 + n2
    tie_term = sum(t**3 - t for t in tie_counts)
    variance = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
    if variance <= 0:
        return u, 1.0
    mean = n1 * n2 / 2.0
    z = (u - mean) / sqrt(variance)
    p = 2.0 * (0.5 * (1.0 - erf(abs(z) / sqrt(2.0))))
    return u, p


def star(q: float) -> str:
    if not np.isfinite(q):
        return "na"
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return "ns"


def summarize_pair(df: pd.DataFrame, value_col: str, comparison: str, task_a: str, task_b: str) -> dict:
    a = df.loc[df["task"] == task_a, value_col].dropna()
    b = df.loc[df["task"] == task_b, value_col].dropna()
    u, p = mannwhitneyu_two_sided(a, b)
    return {
        "comparison": comparison,
        "task_a": task_a,
        "task_b": task_b,
        "metric": value_col,
        "n_a": len(a),
        "n_b": len(b),
        "median_a": a.median(),
        "median_b": b.median(),
        "mean_a": a.mean(),
        "mean_b": b.mean(),
        "mannwhitney_u": u,
        "p_raw": p,
        "effect_median_a_minus_b": a.median() - b.median(),
    }


def audit_density_complexity() -> pd.DataFrame:
    df = pd.read_csv(DENSITY_CSV)
    df["task"] = df["task_label"].fillna(df["task"].map(DISPLAY)).fillna(df["task"])
    rows = []
    for metric in ["motif_density", "motif_complexity"]:
        rows.append(summarize_pair(df, metric, "HK_vs_CAGE", "HK", "CAGE"))
        rows.append(summarize_pair(df, metric, "DEV_vs_CAGE", "DEV", "CAGE"))
    out = pd.DataFrame(rows)
    out["q_bh_fig4b_family"] = bh_adjust(out["p_raw"].tolist())
    out["significance_bh_0.05"] = out["q_bh_fig4b_family"].map(star)
    return out


def audit_contribution() -> pd.DataFrame:
    df = pd.read_csv(HIT_CSV)
    df["task"] = df["task"].map(DISPLAY).fillna(df["task"])
    df = df[df["hit_importance_norm"].notna()].copy()
    agg = (
        df.groupby(["peak_id", "mc_id", "task"], observed=True)["hit_importance_norm"]
        .mean()
        .reset_index()
    )
    tf_map = df.drop_duplicates("mc_id").set_index("mc_id")["tf"]
    rows = []
    for comparison, task_a, task_b in [("HK_vs_CAGE", "HK", "CAGE"), ("DEV_vs_CAGE", "DEV", "CAGE")]:
        for mc_id in sorted(agg["mc_id"].dropna().unique()):
            sub = agg[agg["mc_id"] == mc_id]
            row = summarize_pair(sub, "hit_importance_norm", comparison, task_a, task_b)
            row["mc_id"] = mc_id
            row["tf"] = tf_map.get(mc_id, "")
            rows.append(row)
    out = pd.DataFrame(rows)
    out["q_bh_fig4cd_family"] = bh_adjust(out["p_raw"].tolist())
    out["significance_bh_0.05"] = out["q_bh_fig4cd_family"].map(star)
    return out


def main() -> None:
    QA_DIR.mkdir(parents=True, exist_ok=True)
    density = audit_density_complexity()
    contribution = audit_contribution()
    density.to_csv(QA_DIR / "fig4b_density_complexity_bh_fdr.csv", index=False)
    contribution.to_csv(QA_DIR / "fig4cd_contribution_bh_fdr.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "table": "fig4b_density_complexity_bh_fdr.csv",
                "n_tests": len(density),
                "n_q_lt_0.05": int((density["q_bh_fig4b_family"] < 0.05).sum()),
                "correction_family": "Fig. 4b density and complexity contrasts; four tests",
            },
            {
                "table": "fig4cd_contribution_bh_fdr.csv",
                "n_tests": len(contribution),
                "n_q_lt_0.05": int((contribution["q_bh_fig4cd_family"] < 0.05).sum()),
                "correction_family": "Fig. 4c-d motif contribution contrasts; 20 tests",
            },
        ]
    )
    summary.to_csv(QA_DIR / "fig4_bh_fdr_summary.csv", index=False)
    print(f"Wrote Fig. 4 BH-FDR audit tables to {QA_DIR}")


if __name__ == "__main__":
    main()
