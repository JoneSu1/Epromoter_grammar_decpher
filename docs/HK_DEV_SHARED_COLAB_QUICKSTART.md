# HK_DEV_SHARED Colab runner

This is the Drive-backed rerun workflow for the sharing analysis.  The Colab
notebook is intentionally only a runner: it mounts Drive, clones the paper
base, installs the declared environment and calls the base command. All cohort
construction, validation, Fi-NeMo admission/scanning and DeepISA dispatch live
once in `scripts/hk_dev_shared_pipeline.py`.

GitHub contains the unified `paper_base/` source tree, rerun code, parameter
contract and small data-free tests. The project tables, model H5 files, motif
H5 files, attributions and all run output stay on Google Drive.

## Cohort contract

The immutable Drive input is the labelled union of HC7990 TSS-oriented
high-confidence windows and `HK_DEV_SHARED`. It contains 23,284 unique 249-bp
model-window IDs with three mutually exclusive labels: HC7990-only 5,904,
HK/DEV-shared-only 15,294, and both 2,086. `HK_DEV_SHARED` itself means the
same unique 249-bp ID occurs in both headerless Fig. 1 DEV and HK
summit-overlap tables; summit coordinates need not be identical. It is not the
130-window Fig. 3 logo subset.

| branch | windows | motif dictionary | purpose |
| --- | ---: | --- | --- |
| `s3_hk`, `s3_dev` | 23,284 each | task-specific standalone HK or DEV | one labelled full-union scan; select sharing rows for the S3 proximal-versus-distal supplement |
| `deepisa_hk`, `deepisa_dev` | 18,164 each | shared 24-bp motif atlas | all labelled non-distal windows |
| `deepisa_cage` | 10,082 observed non-distal windows | shared 24-bp motif atlas | observed CAGE only |

The 8,082 sharing proximal/core windows without an observed CAGE row are
retained in HK/DEV scans but never turned into inferred CAGE input. Fig. 3's
shared-atlas logo subset is explicitly outside this workflow.

## Execution gates

The notebook deliberately stops before each material stage.  Read the
rendered `inspect` report, including source hashes, union-label counts,
sequence orientation and parameters, then change `CONFIRM` from `REVIEW` to
`RUN` for that stage. The fixed settings matching the main analysis are:

- Fi-NeMo: lambda `0.7`, `--max-steps 10000`.
- Input is first verified as exact 249-bp sequences, then only the final base
  is removed to obtain Fi-NeMo's 248-bp even-length input.
- Attribution provenance: dinucleotide DeepLIFT/SHAP, 100 backgrounds and
  batch size 20.
- DeepISA: shared atlas only; null percentile 80, 8,192 single and pair null
  samples, receptive field 255.

## Attribution admission

The model-specific attribution process must export an NPZ with:

```text
sequences   N x 249 x 4 (or N x 4 x 249) one-hot sequence array
hyp_scores  same shape, hypothetical dinucleotide DeepLIFT/SHAP scores
```

Do not replace this with a generic SHAP implementation: that would no longer
match the main-analysis attribution convention.  `admit-attributions` checks
the exact order and base identity against the newly generated canonical
manifest before it permits Fi-NeMo to run.  A mismatch stops the run instead
of silently scanning a partial or reordered cohort.  Assembly-gap `N` bases
(two union windows) keep the greedy/CAGE-rerun convention: an all-zero one-hot
row on the producer side, recovered as `N` during the identity check.

The canonical window ID is also carried as Fi-NeMo/DeepISA `region` and as the
FASTA header. This preserves orientation-specific model windows; genomic
`chrom:start-end` is metadata only and is never used as a sequence lookup key.

## Checkpoints and restart

Every successful stage writes `state/<stage>.json`, containing its config and
input fingerprints.  Re-running the same command skips only if all required
outputs exist and its fingerprint still matches.  Changing a source file,
manifest, motif H5, model H5 or configuration invalidates the relevant
checkpoint.  Use `--force` only after consciously reviewing the change.

DeepISA writes one state file for each of `preflight_audit`, `single_isa`,
`combi_isa`, `null_interaction`, and `aggregate_isa`. The default
`--start-from auto` resumes at the first stage without a valid state; manual
stage selection remains available for deliberate reruns. An incomplete
Fi-NeMo directory is retained under `state/incomplete/finemo/` and retried in
a fresh directory.

## Runtime dependencies

The Colab runner installs `shap tf-keras finemo h5py loguru bioframe pandas leidenalg igraph numba pysam` on the preinstalled runtime. `pysam` is required by bioframe's `load_fasta` inside the DeepISA single-ISA stage; without it the stage fails after the motif-location precompute with `ImportError: pysam is required`.

## Local dry-run

From the repository root, this checks the cohort on the project drive without
writing analysis outputs:

```powershell
python scripts/hk_dev_shared_pipeline.py --config configs/hk_dev_shared_rerun_config.json --data-root 'G:\我的云端硬盘\DeepEpromote\Drosophila' --output-root 'G:\我的云端硬盘\DeepEpromote\Drosophila\HK_DEV_SHARED_rerun_202609' inspect
```

The source paths can be overridden on Colab or locally without editing the
versioned config.  Keep the JSON file itself under version control so the
parameter contract remains explicit.
