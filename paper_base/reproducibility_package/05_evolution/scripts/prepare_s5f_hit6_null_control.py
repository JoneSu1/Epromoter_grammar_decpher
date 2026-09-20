#!/usr/bin/env python
"""Prepare hit6 mutation-count-matched random controls for Supplementary Fig. S5f.

This script uses the first trajectory row with target score >= 6.0 as the
guided hit6 state. Random controls are generated from the same round0 sequence
with the same number of mutations as that hit6 state while enforcing the same
local mutation-density and GC constraints used in the reviewer-grade run.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd


SEQ_LEN = 249
NULL_REPLICATES_PER_SEQUENCE = 20
NULL_SEED = 20260723
MAX_10BP_MUTS = 3
MAX_ABS_GC_CHANGE = 0.05
MAX_ATTEMPTS = 5000
TASKS = ["HK", "DEV"]
NUCLEOTIDES = np.array(["A", "C", "G", "T"])
NUC_TO_ONEHOT = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

MODULE_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = MODULE_ROOT / "data" / "processed"
LOG_DIR = MODULE_ROOT / "logs"
RUN_LABEL = "reviewer_greedy_20260724_115905"


def find_run_root() -> Path:
    candidates = sorted(
        Path("G:/").glob(
            f"*/DeepEpromote/Drosophila/DeepSTARR/promoter_mut/results_context_dependent/reviewer_grade_runs/{RUN_LABEL}"
        )
    )
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one {RUN_LABEL} run on G:, found {candidates}")
    return candidates[0]


def find_model_paths() -> tuple[Path, Path, Path]:
    roots = sorted(Path("G:/").glob("*/DeepEpromote/Drosophila"))
    if len(roots) != 1:
        raise FileNotFoundError(f"Expected one DeepEpromote/Drosophila root on G:, found {roots}")
    base = roots[0]
    return (
        base / "DeepSTARR/model_artifacts/DeepSTARR.model.json",
        base / "DeepSTARR/model_artifacts/DeepSTARR.model.h5",
        base / "DeepCAGE/model/model_starr_ctss_T1/best_model.h5",
    )


def atomic_write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, sep="\t", index=False)
    tmp.replace(path)


def reverse_complement(seq: str) -> str:
    return str(seq).translate(COMPLEMENT)[::-1].upper()


def encode_sequence(seq: str) -> np.ndarray:
    seq = str(seq).upper().ljust(SEQ_LEN, "N")[:SEQ_LEN]
    return np.asarray([NUC_TO_ONEHOT.get(base, [0, 0, 0, 0]) for base in seq], dtype=np.float32)


def gc_fraction(seq: str) -> float:
    seq = str(seq).upper()[:SEQ_LEN]
    denom = max(1, sum(base in "ACGT" for base in seq))
    return sum(base in "GC" for base in seq) / denom


def hamming_distance(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(str(a)[:SEQ_LEN], str(b)[:SEQ_LEN]))


def max_mutations_in_window(start_seq: str, seq: str, window: int = 10) -> int:
    diff = np.fromiter((x != y for x, y in zip(start_seq[:SEQ_LEN], seq[:SEQ_LEN])), dtype=np.int8)
    if len(diff) < window:
        return int(diff.sum())
    return int(max(diff[i : i + window].sum() for i in range(len(diff) - window + 1)))


def sequence_hash(seq: str) -> str:
    return hashlib.sha1(str(seq).encode()).hexdigest()[:12]


def constraints_ok(start_seq: str, candidate_seq: str) -> tuple[bool, int, int, float]:
    total_muts = hamming_distance(start_seq, candidate_seq)
    max_window = max_mutations_in_window(start_seq, candidate_seq, 10)
    gc_delta = gc_fraction(candidate_seq) - gc_fraction(start_seq)
    ok = max_window <= MAX_10BP_MUTS and abs(gc_delta) <= MAX_ABS_GC_CHANGE
    return ok, total_muts, max_window, gc_delta


def random_mutant_with_n_mutations(start_seq: str, n_mutations: int, rng: random.Random) -> tuple[str, int, int, float, int]:
    start = str(start_seq).upper()[:SEQ_LEN]
    valid_positions = [i for i, base in enumerate(start) if base in "ACGT"]
    for attempts in range(1, MAX_ATTEMPTS + 1):
        positions = rng.sample(valid_positions, n_mutations)
        seq_list = list(start)
        for pos in positions:
            base = seq_list[pos]
            choices = [x for x in "ACGT" if x != base]
            seq_list[pos] = rng.choice(choices)
        candidate = "".join(seq_list)
        ok, total_muts, max_window, gc_delta = constraints_ok(start, candidate)
        if ok and total_muts == n_mutations:
            return candidate, total_muts, max_window, gc_delta, attempts
    raise RuntimeError(f"Failed to sample a valid {n_mutations}-mutation sequence after {MAX_ATTEMPTS} attempts")


def load_models():
    try:
        import tensorflow as tf
        import tf_keras
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow and tf_keras are required to generate hit6-matched random predictions. "
            "Run this script in the reviewer-grade Colab/runtime environment."
        ) from exc

    deepstarr_json, deepstarr_h5, deepcage_h5 = find_model_paths()
    with open(deepstarr_json, "r") as handle:
        model_json = handle.read()
    model_starr = tf_keras.models.model_from_json(model_json)
    model_starr.load_weights(str(deepstarr_h5))
    model_cage = tf.keras.models.load_model(
        str(deepcage_h5),
        custom_objects={"mse": tf.keras.losses.MeanSquaredError()},
        compile=False,
    )
    return model_starr, model_cage


def predict_batches(model_starr, model_cage, model_sequences: list[str], dataset_sequences: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x_starr = np.stack([encode_sequence(seq) for seq in model_sequences])
    x_cage = np.stack([encode_sequence(seq) for seq in dataset_sequences])
    starr_pred = np.asarray(model_starr.predict(x_starr, batch_size=2048, verbose=0))
    cage_pred = np.asarray(model_cage.predict(x_cage, batch_size=1024, verbose=0)).reshape(-1)
    return starr_pred, cage_pred


def main() -> None:
    random.seed(NULL_SEED)
    np.random.seed(NULL_SEED)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    run_root = find_run_root()
    trajectory_files = sorted((run_root / "checkpoints" / "trajectories").glob("*.trajectory.tsv"))
    if len(trajectory_files) != 830:
        raise RuntimeError(f"Expected 830 trajectory files, found {len(trajectory_files)}")

    frames = [pd.read_csv(path, sep="\t") for path in trajectory_files]
    traj = pd.concat(frames, ignore_index=True)
    init = traj[traj["Round"] == 0].copy()
    hit6 = (
        traj[traj["Score_Opt"] >= 6.0]
        .sort_values(["task", "ID", "Round"])
        .groupby(["task", "ID"], as_index=False)
        .first()
    )
    if len(hit6) != 830:
        raise RuntimeError(f"Expected 830 hit6 states, found {len(hit6)}")

    ref = init[
        [
            "task",
            "ID",
            "source_ID",
            "CAGE_Group",
            "strand",
            "Seq_model_orientation",
            "Seq_dataset_orientation",
            "Score_Opt",
            "Dev_pred",
            "Hk_pred",
            "CAGE_pred",
        ]
    ].rename(
        columns={
            "Seq_model_orientation": "round0_model_seq",
            "Seq_dataset_orientation": "round0_dataset_seq",
            "Score_Opt": "round0_target_pred",
            "Dev_pred": "round0_dev_pred",
            "Hk_pred": "round0_hk_pred",
            "CAGE_pred": "round0_cage_pred",
        }
    )
    hit = hit6[
        [
            "task",
            "ID",
            "Round",
            "Score_Opt",
            "Dev_pred",
            "Hk_pred",
            "CAGE_pred",
            "total_muts",
            "max_10bp_muts",
            "gc_delta",
            "Seq_hash",
        ]
    ].rename(
        columns={
            "Round": "hit6_round",
            "Score_Opt": "hit6_target_pred",
            "Dev_pred": "hit6_dev_pred",
            "Hk_pred": "hit6_hk_pred",
            "CAGE_pred": "hit6_cage_pred",
            "total_muts": "hit6_total_muts",
            "max_10bp_muts": "hit6_max_10bp_muts",
            "gc_delta": "hit6_gc_delta",
            "Seq_hash": "hit6_seq_hash",
        }
    )
    guided = ref.merge(hit, on=["task", "ID"], how="inner", validate="one_to_one")
    guided["hit6_target_delta_vs_round0"] = guided["hit6_target_pred"] - guided["round0_target_pred"]
    guided["hit6_cage_delta_vs_round0"] = guided["hit6_cage_pred"] - guided["round0_cage_pred"]
    atomic_write_tsv(guided, DATA_PROCESSED / "hit6_guided_states.tsv")

    model_starr, model_cage = load_models()
    rng = random.Random(NULL_SEED)
    null_rows = []
    model_sequences = []
    dataset_sequences = []
    row_refs = []
    for _, row in guided.iterrows():
        n_mut = int(row["hit6_total_muts"])
        for replicate in range(NULL_REPLICATES_PER_SEQUENCE):
            seq_model, total_muts, max_window, gc_delta, attempts = random_mutant_with_n_mutations(
                row["round0_model_seq"], n_mut, rng
            )
            seq_dataset = reverse_complement(seq_model) if str(row["strand"]) == "-" else seq_model
            row_refs.append((row, replicate, total_muts, max_window, gc_delta, attempts, seq_model))
            model_sequences.append(seq_model)
            dataset_sequences.append(seq_dataset)

    starr_pred, cage_pred = predict_batches(model_starr, model_cage, model_sequences, dataset_sequences)
    for i, (row, replicate, total_muts, max_window, gc_delta, attempts, seq_model) in enumerate(row_refs):
        dev_pred = float(starr_pred[i, 0])
        hk_pred = float(starr_pred[i, 1])
        target_pred = hk_pred if row["task"] == "HK" else dev_pred
        null_rows.append(
            {
                "method": "Hit6MutationCountMatched_Random",
                "task": row["task"],
                "ID": row["ID"],
                "replicate": replicate,
                "matched_mutations": int(row["hit6_total_muts"]),
                "attempts": attempts,
                "matched_success": True,
                "reached_hit6": target_pred >= 6.0,
                "null_target_pred": target_pred,
                "null_target_delta_vs_round0": target_pred - float(row["round0_target_pred"]),
                "CAGE_pred": float(cage_pred[i]),
                "null_cage_delta_vs_round0": float(cage_pred[i]) - float(row["round0_cage_pred"]),
                "total_muts": total_muts,
                "max_10bp_muts": max_window,
                "gc_delta": gc_delta,
                "Seq_hash": sequence_hash(seq_model),
                "round0_target_pred": float(row["round0_target_pred"]),
                "hit6_target_pred": float(row["hit6_target_pred"]),
                "hit6_target_delta_vs_round0": float(row["hit6_target_delta_vs_round0"]),
                "round0_cage_pred": float(row["round0_cage_pred"]),
                "hit6_cage_pred": float(row["hit6_cage_pred"]),
                "hit6_cage_delta_vs_round0": float(row["hit6_cage_delta_vs_round0"]),
                "hit6_total_muts": int(row["hit6_total_muts"]),
                "hit6_round": int(row["hit6_round"]),
            }
        )
    detail = pd.DataFrame(null_rows)
    atomic_write_tsv(detail, DATA_PROCESSED / "hit6_null_mutation_count_matched_detail.tsv")

    rows = []
    for task in TASKS:
        sub = detail[detail["task"] == task]
        guided_task = guided[guided["task"] == task]
        rows.append(
            {
                "task": task,
                "null_rows": len(sub),
                "null_ids": sub["ID"].nunique(),
                "null_replicates_per_id": len(sub) / sub["ID"].nunique(),
                "null_hit6_rate": sub["reached_hit6"].astype(bool).mean(),
                "null_target_delta_median": sub["null_target_delta_vs_round0"].median(),
                "null_target_delta_q95": sub["null_target_delta_vs_round0"].quantile(0.95),
                "null_cage_delta_median": sub["null_cage_delta_vs_round0"].median(),
                "null_cage_delta_q95": sub["null_cage_delta_vs_round0"].quantile(0.95),
                "guided_ids": len(guided_task),
                "guided_target_delta_median": guided_task["hit6_target_delta_vs_round0"].median(),
                "guided_cage_delta_median": guided_task["hit6_cage_delta_vs_round0"].median(),
                "guided_hit6_total_muts_median": guided_task["hit6_total_muts"].median(),
                "null_total_muts_median": sub["total_muts"].median(),
            }
        )
    summary = pd.DataFrame(rows)
    atomic_write_tsv(summary, DATA_PROCESSED / "hit6_null_mutation_count_matched_summary.tsv")
    manifest = {
        "run_label": RUN_LABEL,
        "null_seed": NULL_SEED,
        "replicates_per_trajectory": NULL_REPLICATES_PER_SEQUENCE,
        "hit6_definition": "first trajectory row with target Score_Opt >= 6.0",
        "constraints": {"max_10bp_muts": MAX_10BP_MUTS, "max_abs_gc_change": MAX_ABS_GC_CHANGE},
        "detail_table": str(DATA_PROCESSED / "hit6_null_mutation_count_matched_detail.tsv"),
        "summary_table": str(DATA_PROCESSED / "hit6_null_mutation_count_matched_summary.tsv"),
    }
    (LOG_DIR / "hit6_null_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote hit6-matched null tables to {DATA_PROCESSED}")


if __name__ == "__main__":
    main()
