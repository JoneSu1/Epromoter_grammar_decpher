#!/usr/bin/env python3
"""Resumable cohort preparation, Fi-NeMo admission/scanning and DeepISA dispatch.

This is deliberately a strict companion workflow.  It never substitutes the
130-region Fig. 3 logo subset for the Fig. 1 DEV/HK sharing cohort, and it
never fabricates CAGE values for windows absent from the observed CAGE matrix.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


OVERLAP_COLUMNS = [
    "chrom", "start", "end", "ID", "summit_chr", "summit_start",
    "summit_end", "region_annotation", "score", "strand",
]
TRACKS = {
    "s3_hk": {"scope": "union", "sequence": "deepstarr", "dictionary": "standalone_hk_motifs", "model": "deepstarr_model", "purpose": "S3 proximal-versus-distal supplement; retain labels then select sharing rows downstream"},
    "s3_dev": {"scope": "union", "sequence": "deepstarr", "dictionary": "standalone_dev_motifs", "model": "deepstarr_model", "purpose": "S3 proximal-versus-distal supplement; retain labels then select sharing rows downstream"},
    "deepisa_hk": {"scope": "non_distal", "sequence": "deepstarr", "dictionary": "shared_24bp_motifs", "model": "deepstarr_model", "deepisa_tracks": [1], "purpose": "DeepISA prerequisite; all labelled non-distal windows"},
    "deepisa_dev": {"scope": "non_distal", "sequence": "deepstarr", "dictionary": "shared_24bp_motifs", "model": "deepstarr_model", "deepisa_tracks": [0], "purpose": "DeepISA prerequisite; all labelled non-distal windows"},
    "deepisa_cage": {"scope": "cage", "sequence": "cage", "dictionary": "shared_24bp_motifs", "model": "deepcage_model", "deepisa_tracks": [0], "purpose": "DeepISA prerequisite; observed CAGE only"},
}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_config(path: Path, data_root: str | None, output_root: str | None) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if data_root:
        config["data_root"] = data_root
    if output_root:
        config["output_root"] = output_root
    config["_config_path"] = str(path.resolve())
    return config


def source_path(config: dict, key: str) -> Path:
    value = Path(config["sources"][key])
    return value if value.is_absolute() else Path(config["data_root"]) / value


def output_path(config: dict) -> Path:
    return Path(config["output_root"])


def require_explicit_local_output(config: dict) -> None:
    """Avoid treating the Colab default `/content/drive` as a Windows path."""
    if os.name == "nt" and str(config["output_root"]).startswith("/content/drive/"):
        raise ValueError(
            "This is a local Windows run with the Colab output default. "
            "Pass --output-root 'G:\\我的云端硬盘\\DeepEpromote\\Drosophila\\HK_DEV_SHARED_rerun_202609'."
        )


def config_fingerprint(config: dict) -> str:
    public = {k: v for k, v in config.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()


def pipeline_fingerprint(config: dict) -> str:
    """Fingerprint code as well as config, so a code fix cannot reuse stale state."""
    return hashlib.sha256((config_fingerprint(config) + sha256(Path(__file__))).encode()).hexdigest()


def coordinates(identifier: str) -> tuple[str, int, int]:
    pieces = identifier.split("_", 4)
    if len(pieces) < 4:
        raise ValueError(f"Cannot parse model-window ID: {identifier}")
    return pieces[0], int(pieces[1]), int(pieces[2])


def require_sources(config: dict, keys: list[str]) -> dict[str, Path]:
    paths = {key: source_path(config, key) for key in keys}
    absent = [str(path) for path in paths.values() if not path.exists()]
    if absent:
        raise FileNotFoundError("Required source files are missing:\n  " + "\n  ".join(absent))
    return paths


def build_cohort(config: dict) -> tuple[pd.DataFrame, dict]:
    paths = require_sources(config, ["labeled_analysis_universe", "development_overlap", "housekeeping_overlap", "deepstarr_sequences", "cage_oriented_observed"])
    dev = pd.read_csv(paths["development_overlap"], sep="\t", header=None, names=OVERLAP_COLUMNS, dtype={"ID": str})
    hk = pd.read_csv(paths["housekeeping_overlap"], sep="\t", header=None, names=OVERLAP_COLUMNS, dtype={"ID": str})
    shared = set(dev["ID"]) & set(hk["ID"])
    labelled = pd.read_csv(paths["labeled_analysis_universe"], sep="\t", dtype=str)
    required_columns = {"canonical_id", "combined_label", "in_HC7990_TSSORIENTED", "in_HK_DEV_SHARED", "fig1_promoter_group", "cage_status"}
    if not required_columns.issubset(labelled.columns):
        raise ValueError(f"Labelled universe is missing: {sorted(required_columns - set(labelled.columns))}")
    if labelled.canonical_id.duplicated().any():
        raise RuntimeError("Labelled universe contains duplicate canonical IDs")
    labelled = labelled.copy()
    labelled["in_HC7990_TSSORIENTED"] = labelled.in_HC7990_TSSORIENTED.eq("True")
    labelled["in_HK_DEV_SHARED"] = labelled.in_HK_DEV_SHARED.eq("True")
    if set(labelled.loc[labelled.in_HK_DEV_SHARED, "canonical_id"]) != shared:
        raise RuntimeError("Labelled HK_DEV_SHARED membership differs from the two Fig1 overlap source tables")
    seq = pd.read_csv(paths["deepstarr_sequences"], sep="\t", dtype={"ID": str})
    if not {"ID", "Sequence"}.issubset(seq.columns):
        raise ValueError("DeepSTARR table must contain ID and Sequence columns")
    seq = seq.drop_duplicates("ID").set_index("ID")["Sequence"]
    cage = pd.read_csv(paths["cage_oriented_observed"], sep="\t", dtype={"ID": str})
    if not {"ID", "seq"}.issubset(cage.columns):
        raise ValueError("Oriented CAGE table must contain ID and seq columns")
    cage = cage.drop_duplicates("ID").set_index("ID")["seq"]
    rows = []
    for record in labelled.sort_values("canonical_id").to_dict("records"):
        identifier = record["canonical_id"]
        chrom, start, end = coordinates(identifier)
        deepstarr_sequence = str(seq.get(identifier, "")).upper()
        cage_sequence = str(cage.get(identifier, "")).upper()
        rows.append({**record,
            "canonical_id": identifier, "chrom": chrom, "start": start, "end": end,
            "coordinate": f"{chrom}:{start}-{end}",
            "deepstarr_sequence": deepstarr_sequence,
            "cage_sequence": cage_sequence if record["cage_status"] == "READY" else "",
            "non_distal": record["cage_status"] != "EXCLUDED_DISTAL_FOR_CAGE",
            "cage_observed": record["cage_status"] == "READY" and bool(cage_sequence),
        })
    cohort = pd.DataFrame(rows)
    expected = config["cohort_contract"]
    counts = {
        "union": len(cohort),
        "hc7990": int(cohort.in_HC7990_TSSORIENTED.sum()),
        "shared": int(cohort.in_HK_DEV_SHARED.sum()),
        "hc7990_only": int((cohort.combined_label == "HC7990_ONLY").sum()),
        "shared_only": int((cohort.combined_label == "HK_DEV_SHARED_ONLY").sum()),
        "hc7990_and_shared": int((cohort.combined_label == "HC7990_AND_HK_DEV_SHARED").sum()),
        "proximal_or_core": int((cohort.fig1_promoter_group == "proximal_promoter").sum()),
        "distal": int((cohort.fig1_promoter_group == "distal_promoter").sum()),
        "union_non_distal": int(cohort.non_distal.sum()),
        "cage_observed_union": int(cohort.cage_observed.sum()),
    }
    required = {key: expected[f"n_{key}"] for key in counts}
    if counts != required:
        raise RuntimeError(f"Cohort count contract failed: got {counts}; expected {required}")
    if not cohort.deepstarr_sequence.map(len).eq(expected["window_length_bp"]).all():
        raise RuntimeError("One or more HK/DEV sequences are not exact 249-bp model windows")
    if not cohort.loc[cohort.cage_observed, "cage_sequence"].map(len).eq(expected["window_length_bp"]).all():
        raise RuntimeError("One or more observed CAGE sequences are not exact 249-bp TSS-oriented windows")
    provenance = {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()}
    return cohort, {"counts": counts, "sources": provenance}


def track_frame(cohort: pd.DataFrame, track: str) -> pd.DataFrame:
    spec = TRACKS[track]
    if spec["scope"] == "union":
        selected = cohort.copy()
    elif spec["scope"] == "non_distal":
        selected = cohort.loc[cohort.non_distal].copy()
    else:
        selected = cohort.loc[cohort.cage_observed].copy()
    sequence_col = f"{spec['sequence']}_sequence"
    selected["sequence"] = selected[sequence_col]
    # DeepISA resolves sequences from the Fi-NeMo-derived `region` value.
    # Canonical IDs are the FASTA headers and preserve strand/window identity;
    # `chrom:start-end` would collapse distinct orientation-specific windows.
    selected["region"] = selected["canonical_id"]
    selected["track"] = track
    selected["dictionary_type"] = "shared_24bp" if spec["dictionary"] == "shared_24bp_motifs" else "task_specific_standalone"
    selected.insert(0, "peak_id", range(len(selected)))
    return selected[["peak_id", "canonical_id", "region", "combined_label", "in_HC7990_TSSORIENTED", "in_HK_DEV_SHARED", "chrom", "start", "end", "coordinate", "fig1_promoter_group", "cage_status", "non_distal", "cage_observed", "track", "dictionary_type", "sequence"]]


def state_path(config: dict, name: str) -> Path:
    return output_path(config) / "state" / f"{name}.json"


def state_is_current(config: dict, name: str, fingerprint: str, outputs: list[Path]) -> bool:
    state = state_path(config, name)
    if not state.exists() or not all(path.exists() and path.stat().st_size > 0 for path in outputs):
        return False
    payload = json.loads(state.read_text(encoding="utf-8"))
    recorded_hashes = payload.get("output_sha256", {})
    current_hashes = {str(path): sha256(path) for path in outputs}
    return (
        payload.get("fingerprint") == fingerprint
        and payload.get("status") == "complete"
        and recorded_hashes == current_hashes
    )


def mark_state(config: dict, name: str, fingerprint: str, outputs: list[Path], extra: dict | None = None) -> None:
    payload = {"status": "complete", "completed_utc": utcnow(), "fingerprint": fingerprint, "outputs": [str(p) for p in outputs], "output_sha256": {str(path): sha256(path) for path in outputs}}
    if extra:
        payload.update(extra)
    atomic_json(state_path(config, name), payload)


def completed_state_fingerprint(config: dict, name: str) -> str:
    path = state_path(config, name)
    if not path.exists():
        raise RuntimeError(f"Missing required completed checkpoint: {path}")
    value = json.loads(path.read_text(encoding="utf-8")).get("fingerprint")
    if not value:
        raise RuntimeError(f"Checkpoint lacks a fingerprint: {path}")
    return value


def command_inspect(config: dict) -> None:
    cohort, audit = build_cohort(config)
    review = {
        "generated_utc": utcnow(), "config_fingerprint": config_fingerprint(config), "pipeline_sha256": sha256(Path(__file__)),
        "cohort_contract": config["cohort_contract"], "observed_counts": audit["counts"],
        "finemo": config["finemo"], "attribution_contract": config["attribution_contract"],
        "deepisa": config["deepisa"],
        "tracks": {name: {**spec, "expected_n": len(track_frame(cohort, name))} for name, spec in TRACKS.items()},
        "input_sources": audit["sources"],
    }
    print(json.dumps(review, indent=2))


def command_prepare(config: dict, force: bool) -> None:
    require_explicit_local_output(config)
    root = output_path(config)
    manifests = root / "manifests"
    cohort_file = manifests / "hk_dev_shared_cohort.tsv"
    track_files = [manifests / f"{name}.tsv" for name in TRACKS]
    cohort, audit = build_cohort(config)
    fingerprint = hashlib.sha256((pipeline_fingerprint(config) + json.dumps(audit, sort_keys=True)).encode()).hexdigest()
    outputs = [cohort_file, *track_files]
    if not force and state_is_current(config, "prepare", fingerprint, outputs):
        print("prepare: checkpoint valid; no files changed")
        return
    manifests.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(cohort_file, sep="\t", index=False)
    fasta_root = root / "fasta"
    for name in TRACKS:
        frame = track_frame(cohort, name)
        expected = {"s3_hk": 23284, "s3_dev": 23284, "deepisa_hk": 18164, "deepisa_dev": 18164, "deepisa_cage": 10082}[name]
        if len(frame) != expected:
            raise RuntimeError(f"{name} has {len(frame)} windows, expected {expected}")
        frame.to_csv(manifests / f"{name}.tsv", sep="\t", index=False)
        fasta = fasta_root / f"{name}.fa"
        fasta.parent.mkdir(parents=True, exist_ok=True)
        with fasta.open("w", encoding="utf-8") as handle:
            for row in frame.itertuples(index=False):
                handle.write(f">{row.canonical_id}\n{row.sequence}\n")
    review_file = root / "parameter_review.json"
    atomic_json(review_file, {"config_fingerprint": config_fingerprint(config), "pipeline_sha256": sha256(Path(__file__)), "cohort_audit": audit, "tracks": {name: len(track_frame(cohort, name)) for name in TRACKS}, "finemo": config["finemo"], "attribution": config["attribution_contract"], "deepisa": config["deepisa"]})
    mark_state(config, "prepare", fingerprint, outputs, {"cohort_counts": audit["counts"], "parameter_review": str(review_file)})
    print(f"prepare: wrote five manifests below {manifests}")


def manifest(config: dict, track: str) -> pd.DataFrame:
    path = output_path(config) / "manifests" / f"{track}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `prepare` after reviewing `inspect`.")
    return pd.read_csv(path, sep="\t", dtype={"canonical_id": str, "sequence": str})


def command_admit_attributions(config: dict, track: str, source: Path, force: bool) -> None:
    require_explicit_local_output(config)
    if track not in TRACKS:
        raise ValueError(f"Unknown track {track}; choose from {', '.join(TRACKS)}")
    if not source.exists():
        raise FileNotFoundError(source)
    frame = manifest(config, track)
    payload = np.load(source, allow_pickle=False)
    if not {"sequences", "hyp_scores"}.issubset(payload.files):
        raise ValueError("Attribution NPZ must contain `sequences` and `hyp_scores` generated by the reviewed upstream model-specific step")
    sequences, scores = payload["sequences"], payload["hyp_scores"]
    if sequences.ndim != 3 or scores.ndim != 3 or sequences.shape != scores.shape:
        raise ValueError(f"Sequence/hyp-score shape mismatch: {sequences.shape} versus {scores.shape}")
    if sequences.shape[0] != len(frame) or sorted(sequences.shape[1:]) != [4, 249]:
        raise ValueError(f"{track}: expected N x 249 x 4 or N x 4 x 249 for N={len(frame)}; got {sequences.shape}")
    if sequences.shape[1] == 4:
        sequences, scores = np.transpose(sequences, (0, 2, 1)), np.transpose(scores, (0, 2, 1))
    # All-zero rows are the assembly-gap N convention shared with the
    # attribution producer; recover them as N for the identity check.
    alphabet = np.array(list("ACGTN"))
    codes = sequences.argmax(axis=2)
    codes[sequences.sum(axis=2) == 0] = 4
    recovered = ["".join(alphabet[x]) for x in codes]
    if recovered != frame.sequence.tolist():
        raise RuntimeError("Attribution sequences do not exactly match the canonical staged manifest; refusing scan input")
    root = output_path(config) / "finemo_input" / track
    npz_file, bed_file = root / "finemo_input.npz", root / "regions.bed"
    fingerprint = hashlib.sha256((pipeline_fingerprint(config) + sha256(source) + sha256(output_path(config) / "manifests" / f"{track}.tsv")).encode()).hexdigest()
    if not force and state_is_current(config, f"admit_{track}", fingerprint, [npz_file, bed_file]):
        print(f"admit-attributions {track}: checkpoint valid; no files changed")
        return
    root.mkdir(parents=True, exist_ok=True)
    # Fi-NeMo requires an even length.  The main analysis convention trims only
    # the final base after the 249-bp identity check above.
    seq248 = np.transpose(sequences[:, :-1, :], (0, 2, 1)).astype(np.int8)
    score248 = np.transpose(scores[:, :-1, :], (0, 2, 1)).astype(np.float32)
    np.savez_compressed(npz_file, sequences=seq248, contributions=score248)
    with bed_file.open("w", encoding="utf-8") as handle:
        for row in frame.itertuples(index=False):
            handle.write(f"chrFake\t0\t248\t{row.canonical_id}\t0\t+\t0\t0\t0\t124\n")
    mark_state(config, f"admit_{track}", fingerprint, [npz_file, bed_file], {"source_attributions": str(source), "source_sha256": sha256(source), "input_shape": list(sequences.shape), "finemo_shape": list(seq248.shape)})
    print(f"admit-attributions {track}: validated {len(frame)} exact sequences and wrote 248-bp Fi-NeMo input")


def command_scan(config: dict, track: str, force: bool) -> None:
    require_explicit_local_output(config)
    if track not in TRACKS:
        raise ValueError(f"Unknown track {track}")
    npz_file = output_path(config) / "finemo_input" / track / "finemo_input.npz"
    if not npz_file.exists():
        raise FileNotFoundError(f"Missing {npz_file}; run admit-attributions first")
    motif = source_path(config, TRACKS[track]["dictionary"])
    if not motif.exists():
        raise FileNotFoundError(motif)
    scan_dir, hits = output_path(config) / "finemo_scans" / track, output_path(config) / "finemo_scans" / track / "hits.tsv"
    fingerprint = hashlib.sha256((pipeline_fingerprint(config) + sha256(npz_file) + sha256(motif)).encode()).hexdigest()
    if not force and state_is_current(config, f"scan_{track}", fingerprint, [hits]):
        print(f"scan {track}: checkpoint valid; hits.tsv retained")
        return
    # An interrupted Fi-NeMo process does not have a valid state. Preserve its
    # incomplete output rather than scanning into a directory with unknown
    # contents; the retry starts from a fresh target directory.
    if scan_dir.exists():
        archive_root = output_path(config) / "state" / "incomplete" / "finemo"
        archive_root.mkdir(parents=True, exist_ok=True)
        archive = archive_root / f"{track}_{utcnow().replace(':', '').replace('+00:00', 'Z')}"
        scan_dir.replace(archive)
        print(f"scan {track}: archived incomplete attempt to {archive}")
    scan_dir.mkdir(parents=True, exist_ok=True)
    command = ["finemo", "call-hits", "-r", str(npz_file), "-m", str(motif), "-o", str(scan_dir), "-l", str(config["finemo"]["lambda"]), "--max-steps", str(config["finemo"]["max_steps"])]
    print("Executing:", " ".join(command))
    subprocess.run(command, check=True)
    if not hits.exists() or hits.stat().st_size == 0:
        raise RuntimeError(f"Fi-NeMo finished without a non-empty {hits}")
    mark_state(config, f"scan_{track}", fingerprint, [hits], {"motif_database": str(motif), "motif_sha256": sha256(motif), "command": command})
    print(f"scan {track}: complete")


def _deepisa_stage_outputs(runner) -> dict[str, list[Path]]:
    files = {key: Path(value) for key, value in runner.files.items()}
    tracks = runner.tracks
    return {
        "preflight_audit": [files["preflight_pair_audit"], files["preflight_pair_audit_by_region"], files["preflight_overlap_pairs"]],
        "single_isa": [files["isa_single"], files["null_isa"], files["pred_orig"]],
        "combi_isa": [files["isa_combi"]],
        "null_interaction": [files["null_interaction"]],
        "aggregate_isa": [files["isa_combi"], files["null_interaction"], files["imp_tf"], *[Path(files["coop_tf_pair"].as_posix().replace(".csv", f"_t{t}.csv")) for t in tracks], *[Path(files["coop_tf"].as_posix().replace(".csv", f"_t{t}.csv")) for t in tracks]],
    }


def _refresh_state_hashes(config: dict, name: str, outputs: list[Path]) -> None:
    path = state_path(config, name)
    if not path.exists() or not all(item.exists() and item.stat().st_size > 0 for item in outputs):
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["outputs"] = [str(item) for item in outputs]
    payload["output_sha256"] = {str(item): sha256(item) for item in outputs}
    atomic_json(path, payload)


def command_deepisa(config: dict, track: str, isa_source: str | None, force: bool, start_from: str) -> None:
    require_explicit_local_output(config)
    if track not in {"deepisa_hk", "deepisa_dev", "deepisa_cage"}:
        raise ValueError("DeepISA is defined only for deepisa_hk, deepisa_dev and deepisa_cage")
    scan_hits = output_path(config) / "finemo_scans" / track / "hits.tsv"
    if not scan_hits.exists():
        raise FileNotFoundError(f"Missing {scan_hits}; run the reviewed shared-atlas Fi-NeMo scan first")
    source = Path(isa_source) if isa_source else source_path(config, "ep_isa_source")
    if not (source / "Ep_ISA_NEW").exists():
        raise FileNotFoundError(f"Ep_ISA_NEW source not found at {source}. Set sources.ep_isa_source or --isa-source.")
    model_path = source_path(config, TRACKS[track]["model"])
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    results = output_path(config) / "deepisa" / track
    base_fingerprint = hashlib.sha256((pipeline_fingerprint(config) + sha256(scan_hits) + sha256(model_path)).encode()).hexdigest()
    sys.path.insert(0, str(source))
    # Same validated CAGE/Evolution loader as generate_hk_dev_attributions.py.
    # The DeepSTARR H5 is a weights-only Keras-2 artifact, so deserialize its
    # JSON with tf_keras and convert through H5 to tf.keras; the CAGE model is
    # a full legacy save and loads directly.
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    import tensorflow as tf
    from tf_keras.models import model_from_json

    from Ep_ISA_NEW.quickstart import EpQuickStart
    frame = manifest(config, track)
    # Ep_ISA_NEW resolves sequences through `chrom:start-end` region strings
    # (region_str_to_seq splits on ':'), while this workflow's lookup key is
    # the canonical window ID.  Present each window as its own chromosome of
    # length 249: region "canonical_id:0-249" fetches the canonical-ID-keyed
    # FASTA entry, motif coordinates stay window-relative (the ablation code
    # slices fetched sequences by start_rel/end_rel), and distinct windows
    # never collapse onto shared genomic coordinates.
    isa_frame = pd.DataFrame({
        "peak_id": frame["peak_id"],
        "chrom": frame["canonical_id"].astype(str),
        "start": 0,
        "end": 249,
        "region": frame["canonical_id"].astype(str) + ":0-249",
    })
    if TRACKS[track]["model"] == "deepcage_model":
        # Same CAGE loader as generate_hk_dev_attributions.py: the legacy H5's
        # InputLayer `batch_shape` config is rejected by tf_keras but read
        # fine by Keras 3, which EpQuickStart consumes unchanged.
        import keras
        model = keras.models.load_model(
            model_path, custom_objects={"mse": keras.losses.MeanSquaredError()}, compile=False
        )
    else:
        architecture = model_path.with_suffix(".json")
        if not architecture.exists():
            architecture = model_path.parent / "DeepSTARR.model.json"
        source_model = model_from_json(architecture.read_text(encoding="utf-8"))
        source_model.load_weights(model_path)
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as handle:
            converted = Path(handle.name)
        try:
            source_model.save(converted)
            model = tf.keras.models.load_model(converted, compile=False)
        finally:
            if converted.exists():
                converted.unlink()
    runner = EpQuickStart(str(results), str(output_path(config) / "fasta" / f"{track}.fa"), isa_frame)
    runner.define_model(model)
    runner.load_finemo(str(scan_hits), finemo_h5_path=str(source_path(config, "shared_24bp_motifs")))
    stage_outputs = _deepisa_stage_outputs(runner)
    stages = list(stage_outputs)
    if start_from == "auto":
        requested = stages
    else:
        requested = stages[stages.index(start_from):]
    def dependency_tokens_for(stage: str) -> list[str]:
        # Evaluated per stage: later stages hash files that only earlier
        # stages produce, so their tokens must not be built on a fresh run
        # while those files do not exist yet.
        if stage == "preflight_audit":
            return [sha256(scan_hits)]
        if stage == "single_isa":
            return [sha256(Path(runner.files["motif_locs"])), sha256(Path(runner.files["non_motif_locs"]))]
        if stage == "combi_isa":
            return [sha256(Path(runner.files["isa_single"])), sha256(Path(runner.files["null_isa"])), sha256(Path(runner.files["pred_orig"]))]
        if stage == "null_interaction":
            # aggregate_isa intentionally rewrites combi/null tables in place.
            # Use upstream checkpoint identities, not their pre-aggregation
            # byte hashes, and validate the final output hashes separately.
            return [completed_state_fingerprint(config, f"deepisa_{track}_combi_isa"), sha256(Path(runner.files["pred_orig"]))]
        return [
            completed_state_fingerprint(config, f"deepisa_{track}_combi_isa"),
            completed_state_fingerprint(config, f"deepisa_{track}_null_interaction"),
            sha256(Path(runner.files["isa_single"])),
            sha256(Path(runner.files["null_isa"])),
        ]

    # run_isa reads `tracks` from the dict it receives; DeepSTARR is a
    # two-head model (track 0 = DEV, track 1 = HK), so the HK track must not
    # reuse the shared default [0] or its ISA silently measures the DEV head.
    isa_config = dict(config["deepisa"])
    isa_config["tracks"] = TRACKS[track].get("deepisa_tracks", isa_config.get("tracks", [0]))
    for stage in requested:
        dependency_tokens = dependency_tokens_for(stage)
        stage_fingerprint = hashlib.sha256((base_fingerprint + stage + "".join(dependency_tokens)).encode()).hexdigest()
        state_name = f"deepisa_{track}_{stage}"
        if not force and state_is_current(config, state_name, stage_fingerprint, stage_outputs[stage]):
            print(f"deepisa {track} {stage}: checkpoint valid; skipped")
            continue
        print(f"deepisa {track}: running {stage}")
        runner.run_isa(isa_config, start_from=stage, stop_after=stage)
        if not all(path.exists() and path.stat().st_size > 0 for path in stage_outputs[stage]):
            raise RuntimeError(f"DeepISA stage {stage} did not produce its required outputs")
        mark_state(config, state_name, stage_fingerprint, stage_outputs[stage], {"model": str(model_path), "model_sha256": sha256(model_path), "finemo_hits": str(scan_hits)})
        # aggregate_isa normalizes these two tables in place, so refresh the
        # upstream state hashes after that intentional mutation.
        if stage == "aggregate_isa":
            _refresh_state_hashes(config, f"deepisa_{track}_combi_isa", stage_outputs["combi_isa"])
            _refresh_state_hashes(config, f"deepisa_{track}_null_interaction", stage_outputs["null_interaction"])
    required = stage_outputs["aggregate_isa"]
    if not all(path.exists() and path.stat().st_size > 0 for path in required):
        raise RuntimeError("DeepISA did not produce its required final interaction tables")
    final_fingerprint = hashlib.sha256((base_fingerprint + "aggregate_isa" + "".join(sha256(item) for item in required)).encode()).hexdigest()
    mark_state(config, f"deepisa_{track}", final_fingerprint, required, {"model": str(model_path), "model_sha256": sha256(model_path), "finemo_hits": str(scan_hits), "start_from": start_from})
    print(f"deepisa {track}: complete")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-root")
    p.add_argument("--output-root")
    p.add_argument("--force", action="store_true", help="replace a valid checkpoint for the requested stage")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect", help="read sources and print cohort/parameter review; no analysis files are written")
    sub.add_parser("prepare", help="write canonical manifests and FASTA files")
    admit = sub.add_parser("admit-attributions", help="validate a reviewed attribution NPZ and create Fi-NeMo input")
    admit.add_argument("--track", required=True, choices=TRACKS)
    admit.add_argument("--source", required=True, type=Path, help="NPZ with sequences and hyp_scores")
    scan = sub.add_parser("scan", help="run the configured Fi-NeMo scan")
    scan.add_argument("--track", required=True, choices=TRACKS)
    isa = sub.add_parser("deepisa", help="run/resume DeepISA after shared-atlas scanning")
    isa.add_argument("--track", required=True, choices=["deepisa_hk", "deepisa_dev", "deepisa_cage"])
    isa.add_argument("--isa-source", help="directory containing Ep_ISA_NEW/")
    isa.add_argument("--start-from", choices=["auto", "preflight_audit", "single_isa", "combi_isa", "null_interaction", "aggregate_isa"], default="auto")
    return p


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config, args.data_root, args.output_root)
    if args.command == "inspect":
        command_inspect(config)
    elif args.command == "prepare":
        command_prepare(config, args.force)
    elif args.command == "admit-attributions":
        command_admit_attributions(config, args.track, args.source, args.force)
    elif args.command == "scan":
        command_scan(config, args.track, args.force)
    else:
        command_deepisa(config, args.track, args.isa_source, args.force, args.start_from)


if __name__ == "__main__":
    main()
