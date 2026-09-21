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
import shutil
import tempfile
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
                # Assembly-gap N keeps the greedy/CAGE-rerun convention of an
                # all-zero row; any other letter stays a loud failure.
                if base != "N":
                    raise ValueError(f"Non-ACGTN base in row {row}: {base!r}")
                continue
            table[row, pos, index] = 1.0
    return table


def seed_from_onehot(sequence: np.ndarray) -> int:
    return int(hashlib.md5(sequence.tobytes()).hexdigest()[:8], 16)


def dinuc_shuffle(sequence: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Deterministic one-hot shuffle preserving each adjacent-base count.

    This is the Eulerian-trail dinucleotide shuffle used by the historical
    DeepLIFT helper, kept locally so the validated CAGE/Evolution method can
    run on Python 3.12 without installing the legacy ``deeplift`` package.
    """
    tokens = np.asarray(sequence.argmax(axis=1), dtype=np.int8)
    if len(tokens) < 2:
        return sequence.astype(np.float32, copy=True)
    successors = [[] for _ in range(4)]
    for left, right in zip(tokens[:-1], tokens[1:]):
        successors[int(left)].append(int(right))
    for values in successors:
        if len(values) > 1:
            mutable = values[:-1]
            rng.shuffle(mutable)
            values[:-1] = mutable
    used = [0, 0, 0, 0]
    output = np.empty_like(tokens)
    output[0] = tokens[0]
    for pos in range(len(tokens) - 1):
        base = int(output[pos])
        output[pos + 1] = successors[base][used[base]]
        used[base] += 1
    return np.eye(4, dtype=np.float32)[output]


def references(sequence: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed_from_onehot(sequence))
    refs = np.asarray([dinuc_shuffle(sequence, rng) for _ in range(n)], dtype=np.float32)
    return refs, refs.mean(axis=0)


def load_model(config: dict, kind: str, head: str | None):
    import tensorflow as tf
    from tf_keras.models import model_from_json

    root = Path(config["data_root"])
    if kind == "deepcage":
        model_path = root / config["sources"]["deepcage_model"]
        # Exact CAGE path of the reviewed greedy/24-bp Fi-NeMo rerun: run with
        # TF_USE_LEGACY_KERAS=0 (enforced in main), so tf.keras is Keras 3,
        # which reads the legacy InputLayer `batch_shape` config and passes
        # shap's isinstance(model, tf.keras.Model) check.
        model = tf.keras.models.load_model(
            model_path, custom_objects={"mse": tf.keras.losses.MeanSquaredError()}, compile=False
        )
        return model, None
    weights = root / config["sources"]["deepstarr_model"]
    architecture = weights.with_suffix(".json")
    if not architecture.exists():
        architecture = weights.parent / "DeepSTARR.model.json"
    source_model = model_from_json(architecture.read_text(encoding="utf-8"))
    source_model.load_weights(weights)
    # Match the reviewed Shared_motif / 24-bp Fi-NeMo path: deserialize the
    # legacy artifact with tf_keras, then re-load it as a tf.keras model for
    # SHAP DeepExplainer.
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as handle:
        converted = Path(handle.name)
    try:
        source_model.save(converted)
        model = tf.keras.models.load_model(converted, compile=False)
    finally:
        if converted.exists():
            converted.unlink()
    return model, model.get_layer(head).output


def build_explainer(config: dict, kind: str, head: str | None):
    import shap
    model, output = load_model(config, kind, head)
    try:
        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    except AttributeError:
        pass
    n_backgrounds = int(config["attribution_contract"]["dinucleotide_backgrounds"])

    def background_callable(inputs):
        refs, _ = references(inputs[0], n_backgrounds)
        return [refs]

    if head is None:
        return shap.DeepExplainer(model, data=background_callable)  # Keras-3 CAGE, greedy-rerun form
    return shap.DeepExplainer((model.input, output), data=background_callable)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--track", required=True, choices=sorted(TRACK_MODEL))
    parser.add_argument("--check", action="store_true",
                        help="load the model, build the explainer and smoke-test two sequences; writes nothing")
    args = parser.parse_args()

    # Per-track Keras binding, mirroring the two reviewed rerun paths exactly.
    # shap's type check is isinstance(model, tf.keras.Model): the DeepSTARR
    # tuple-form DeepExplainer ran under the legacy tf_keras binding, while
    # the CAGE model-object form ran on Keras 3 (the legacy tf_keras H5
    # loader also rejects this artifact's InputLayer `batch_shape`).  This
    # must be fixed before `import shap`, which imports TensorFlow eagerly.
    kind, head = TRACK_MODEL[args.track]
    os.environ["TF_USE_LEGACY_KERAS"] = "0" if kind == "deepcage" else "1"

    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["data_root"] = args.data_root
    manifest = args.output_root / "manifests" / f"{args.track}.tsv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing {manifest}; run prepare first")
    frame = pd.read_csv(manifest, sep="\t", dtype=str)
    if args.check:
        frame = frame.iloc[:2]
    x = one_hot(frame["sequence"].str.upper())
    if args.check:
        explainer = build_explainer(config, kind, head)
        values = explainer.shap_values(x, check_additivity=False)
        phi = values[0] if isinstance(values, list) else values
        if phi.ndim == 4 and phi.shape[-1] == 1:
            phi = phi[..., 0]
        print(f"{args.track}: CHECK OK; explainer built, shap_values(2 seqs) -> {phi.shape}")
        return
    destination = args.output_root / "reviewed_attributions" / f"{args.track}.npz"
    remote_checkpoint = args.output_root / "reviewed_attributions" / f"{args.track}.checkpoint.h5"
    # Per-batch flushes go to local scratch; Drive only receives a periodic
    # copy, so an unstable /content/drive FUSE mount cannot stall the compute
    # loop (Errno 103 aborts were observed mid-write on Colab).
    checkpoint = Path(tempfile.gettempdir()) / f"hk_dev_attr_{args.track}.checkpoint.h5"
    sync_interval = 1000
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with np.load(destination, allow_pickle=False) as done:
            if np.array_equal(done["sequences"], x) and done["hyp_scores"].shape == x.shape:
                print(f"{args.track}: completed attribution NPZ is valid; no work needed")
                return
        raise RuntimeError(f"Existing {destination} does not match the canonical manifest; remove it only after review")

    explainer = build_explainer(config, kind, head)
    n_backgrounds = int(config["attribution_contract"]["dinucleotide_backgrounds"])
    batch_size = int(config["attribution_contract"]["batch_size"])

    def checkpoint_is_valid(path: Path) -> bool:
        try:
            with h5py.File(path, "r") as handle:
                return np.array_equal(handle["sequences"][:], x)
        except OSError:
            return False

    if checkpoint.exists() and not checkpoint_is_valid(checkpoint):
        checkpoint.unlink()
    if not checkpoint.exists() and remote_checkpoint.exists() and checkpoint_is_valid(remote_checkpoint):
        shutil.copyfile(remote_checkpoint, checkpoint)
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
            reference_means = np.asarray([references(row, n_backgrounds)[1] for row in batch], dtype=np.float32)
            multiplier = np.divide(phi, batch - reference_means, out=np.zeros_like(phi), where=np.abs(batch - reference_means) > 1e-6)
            hypothetical = (multiplier - np.sum(multiplier * reference_means, axis=2, keepdims=True)).astype(np.float32)
            handle["hyp_scores"][offset:end] = hypothetical
            handle.attrs["processed_count"] = end
            handle.flush()
            if end % sync_interval == 0:
                shutil.copyfile(checkpoint, remote_checkpoint)
            print(f"{args.track}: {end}/{len(x)}")
    shutil.copyfile(checkpoint, remote_checkpoint)
    with h5py.File(checkpoint, "r") as handle:
        np.savez_compressed(destination, sequences=handle["sequences"][:], hyp_scores=handle["hyp_scores"][:])
    print(f"{args.track}: published {destination}")


if __name__ == "__main__":
    main()
