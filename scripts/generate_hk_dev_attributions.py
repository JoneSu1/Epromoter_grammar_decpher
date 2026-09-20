#!/usr/bin/env python3
"""Resumable, model-specific dinucleotide DeepLIFT/SHAP attribution producer.

Writes a checkpoint HDF5 during calculation, then atomically publishes the
``sequences``/``hyp_scores`` NPZ consumed by hk_dev_shared_pipeline.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


TRACK_MODEL = {
    "s3_hk": ("deepstarr", "Dense_Hk"),
    "s3_dev": ("deepstarr", "Dense_Dev"),
    "deepisa_hk": ("deepstarr", "Dense_Hk"),
    "deepisa_dev": ("deepstarr", "Dense_Dev"),
    "deepisa_cage": ("deepcage", None),
}


def one_hot(sequences: pd.Series) -> np.ndarray:
    table = np.zeros((len(sequences), 249, 4), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        for pos, base in enumerate(sequence):
            index = "ACGT".find(base)
            if index < 0:
                raise ValueError(f"Non-ACGT base in row {row}: {base!r}")
            table[row, pos, index] = 1.0
    return table


def seed_from_onehot(sequence: np.ndarray) -> int:
    return int(hashlib.md5(sequence.tobytes()).hexdigest()[:8], 16)


def references(sequence: np.ndarray, n: int, dinuc_shuffle) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed_from_onehot(sequence))
    refs = np.asarray([dinuc_shuffle(sequence, rng=rng) for _ in range(n)], dtype=np.float32)
    return refs, refs.mean(axis=0)


def load_model(config: dict, kind: str, head: str | None):
    # The stored DeepSTARR JSON/H5 pair is a Keras-2 artifact; set this before
    # importing TensorFlow so its legacy loader is selected.
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    import tensorflow as tf
    from tf_keras.models import model_from_json

    root = Path(config["data_root"])
    if kind == "deepcage":
        model_path = root / config["sources"]["deepcage_model"]
        model = tf.keras.models.load_model(model_path, custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
        return model, model.output
    weights = root / config["sources"]["deepstarr_model"]
    architecture = weights.with_suffix(".json")
    if not architecture.exists():
        architecture = weights.parent / "DeepSTARR.model.json"
    model = model_from_json(architecture.read_text(encoding="utf-8"))
    model.load_weights(weights)
    return model, model.get_layer(head).output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--track", required=True, choices=sorted(TRACK_MODEL))
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["data_root"] = args.data_root
    manifest = args.output_root / "manifests" / f"{args.track}.tsv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing {manifest}; run prepare first")
    frame = pd.read_csv(manifest, sep="\t", dtype=str)
    x = one_hot(frame["sequence"].str.upper())
    destination = args.output_root / "reviewed_attributions" / f"{args.track}.npz"
    checkpoint = args.output_root / "reviewed_attributions" / f"{args.track}.checkpoint.h5"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with np.load(destination, allow_pickle=False) as done:
            if np.array_equal(done["sequences"], x) and done["hyp_scores"].shape == x.shape:
                print(f"{args.track}: completed attribution NPZ is valid; no work needed")
                return
        raise RuntimeError(f"Existing {destination} does not match the canonical manifest; remove it only after review")

    import shap
    from deeplift.dinuc_shuffle import dinuc_shuffle
    kind, head = TRACK_MODEL[args.track]
    model, output = load_model(config, kind, head)
    try:
        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    except AttributeError:
        pass
    n_backgrounds = int(config["attribution_contract"]["dinucleotide_backgrounds"])
    batch_size = int(config["attribution_contract"]["batch_size"])

    def background_callable(inputs):
        refs, _ = references(inputs[0], n_backgrounds, dinuc_shuffle)
        return [refs]

    explainer = shap.DeepExplainer((model.input, output), data=background_callable)
    if not checkpoint.exists():
        with h5py.File(checkpoint, "w") as handle:
            handle.create_dataset("sequences", data=x, compression="gzip")
            handle.create_dataset("hyp_scores", shape=x.shape, dtype="float32", compression="gzip")
            handle.attrs["processed_count"] = 0
            handle.attrs["track"] = args.track
            handle.attrs["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with h5py.File(checkpoint, "r+") as handle:
        if not np.array_equal(handle["sequences"][:], x):
            raise RuntimeError("Attribution checkpoint sequences differ from the canonical manifest")
        start = int(handle.attrs["processed_count"])
        for offset in range(start, len(x), batch_size):
            end = min(offset + batch_size, len(x))
            batch = x[offset:end]
            values = explainer.shap_values(batch, check_additivity=False)
            phi = values[0] if isinstance(values, list) else values
            if phi.ndim == 4 and phi.shape[-1] == 1:
                phi = phi[..., 0]
            reference_means = np.asarray([references(row, n_backgrounds, dinuc_shuffle)[1] for row in batch], dtype=np.float32)
            multiplier = np.divide(phi, batch - reference_means, out=np.zeros_like(phi), where=np.abs(batch - reference_means) > 1e-6)
            hypothetical = (multiplier - np.sum(multiplier * reference_means, axis=2, keepdims=True)).astype(np.float32)
            handle["hyp_scores"][offset:end] = hypothetical
            handle.attrs["processed_count"] = end
            handle.flush()
            print(f"{args.track}: {end}/{len(x)}")
    with h5py.File(checkpoint, "r") as handle:
        np.savez_compressed(destination, sequences=handle["sequences"][:], hyp_scores=handle["hyp_scores"][:])
    print(f"{args.track}: published {destination}")


if __name__ == "__main__":
    main()
