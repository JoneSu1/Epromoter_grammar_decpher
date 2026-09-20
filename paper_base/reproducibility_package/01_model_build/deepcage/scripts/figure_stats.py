"""Small statistical helpers without scipy/sklearn runtime dependency."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mse(truth, pred) -> float:
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean((truth - pred) ** 2))


def pearson(truth, pred) -> float:
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(truth) < 2:
        return float("nan")
    return float(np.corrcoef(truth, pred)[0, 1])


def spearman(truth, pred) -> float:
    truth_rank = pd.Series(truth).rank(method="average").to_numpy()
    pred_rank = pd.Series(pred).rank(method="average").to_numpy()
    return pearson(truth_rank, pred_rank)


def regression_metrics(truth, pred) -> dict[str, float]:
    return {
        "n": int(len(truth)),
        "pcc": pearson(truth, pred),
        "scc": spearman(truth, pred),
        "mse": mse(truth, pred),
    }
