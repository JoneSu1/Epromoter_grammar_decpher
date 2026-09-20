#!/usr/bin/env python
"""Bootstrap CAGE-HK versus CAGE-DEV correlation differences for Fig. 2c."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from figure_paths import DATA_DIR, POLISH_QA
from figure_stats import pearson


TSV = DATA_DIR / "PROMOTER_Dominant_True_Pred_Verify.tsv"
COLS = ["True_CAGE", "Pred_CAGE", "True_DEV", "Pred_DEV", "True_HK", "Pred_HK"]
DEFAULT_N_BOOTSTRAP = 10_000
DEFAULT_SEED = 20260813


def load_data() -> pd.DataFrame:
    df = pd.read_csv(TSV, sep="\t", usecols=COLS)
    for col in COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=COLS).reset_index(drop=True)


def bootstrap_delta(
    df: pd.DataFrame,
    cage_col: str,
    hk_col: str,
    dev_col: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    values = df[[cage_col, hk_col, dev_col]].to_numpy(dtype=float)
    n = values.shape[0]

    r_cage_hk = pearson(values[:, 0], values[:, 1])
    r_cage_dev = pearson(values[:, 0], values[:, 2])
    observed_delta = r_cage_hk - r_cage_dev

    deltas = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = values[idx]
        deltas[i] = pearson(sample[:, 0], sample[:, 1]) - pearson(sample[:, 0], sample[:, 2])

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    p_two_sided = min(1.0, 2.0 * min(np.mean(deltas <= 0.0), np.mean(deltas >= 0.0)))
    p_one_sided = float(np.mean(deltas <= 0.0))

    return {
        "n": int(n),
        "r_cage_hk": float(r_cage_hk),
        "r_cage_dev": float(r_cage_dev),
        "delta_r_cage_hk_minus_cage_dev": float(observed_delta),
        "bootstrap_ci95_low": float(ci_low),
        "bootstrap_ci95_high": float(ci_high),
        "bootstrap_p_two_sided_delta_eq_0": float(p_two_sided),
        "bootstrap_p_one_sided_delta_le_0": p_one_sided,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out",
        type=Path,
        default=POLISH_QA / "figure2f" / "fig2_correlation_difference_bootstrap_summary.tsv",
    )
    args = parser.parse_args()

    df = load_data()
    rows = []
    measured = bootstrap_delta(
        df,
        cage_col="True_CAGE",
        hk_col="True_HK",
        dev_col="True_DEV",
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    measured["comparison"] = "measured"
    rows.append(measured)

    predicted = bootstrap_delta(
        df,
        cage_col="Pred_CAGE",
        hk_col="Pred_HK",
        dev_col="Pred_DEV",
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 1,
    )
    predicted["comparison"] = "predicted"
    rows.append(predicted)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "comparison",
        "n",
        "r_cage_hk",
        "r_cage_dev",
        "delta_r_cage_hk_minus_cage_dev",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
        "bootstrap_p_two_sided_delta_eq_0",
        "bootstrap_p_one_sided_delta_le_0",
        "n_bootstrap",
        "seed",
    ]
    pd.DataFrame(rows)[columns].to_csv(out, sep="\t", index=False)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
