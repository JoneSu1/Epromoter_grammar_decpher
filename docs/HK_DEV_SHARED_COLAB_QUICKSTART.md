# HK_DEV_SHARED Colab quickstart

This is the rerun workflow for the sharing analysis.  It is an extension to
the frozen paper figures, not a replacement for their released visual assets.
GitHub contains only this complete analysis code, the parameter contract and
small data-free tests.  The project tables, model H5 files, motif H5 files,
attributions and all run output stay on Google Drive.  The notebook records
the cohort definition, every source hash and every stage checkpoint under one
run directory on Google Drive.

## Cohort contract

`HK_DEV_SHARED` means the unique 249-bp DeepSTARR model-window `ID` present in
both headerless Fig. 1 DEV and HK summit-overlap annotation tables.  It does
not mean that the two summits occupy the identical coordinate, and it is not
the 130-window Fig. 3 logo subset.

| branch | windows | motif dictionary | purpose |
| --- | ---: | --- | --- |
| `s3_hk`, `s3_dev` | 17,380 each (12,260 proximal/core; 5,120 distal) | task-specific standalone HK or DEV | S3 proximal-versus-distal supplement |
| `deepisa_hk`, `deepisa_dev` | 12,260 each | shared 24-bp motif atlas | DeepISA |
| `deepisa_cage` | 4,178 observed proximal/core | shared 24-bp motif atlas | DeepISA |

The 8,082 sharing proximal/core windows without an observed CAGE row are
never turned into inferred CAGE input.  Fig. 3's shared-atlas logo subset is
explicitly outside this workflow.

## Execution gates

The notebook deliberately stops before each material stage.  Read the
rendered `inspect` report, including source hashes, five cohort counts,
sequence orientation and parameters, then change `CONFIRM` from `REVIEW` to
`RUN` for that stage.  The fixed settings matching the main analysis are:

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
of silently scanning a partial or reordered cohort.

## Checkpoints and restart

Every successful stage writes `state/<stage>.json`, containing its config and
input fingerprints.  Re-running the same command skips only if all required
outputs exist and its fingerprint still matches.  Changing a source file,
manifest, motif H5, model H5 or configuration invalidates the relevant
checkpoint.  Use `--force` only after consciously reviewing the change.

For a DeepISA interruption, retain the result directory and call `deepisa`
with `--start-from` at the earliest missing stage.  The available stages are
`preflight_audit`, `single_isa`, `combi_isa`, `null_interaction`, and
`aggregate_isa`.

## Local dry-run

From the repository root, this checks the cohort on the project drive without
writing analysis outputs:

```powershell
python scripts/hk_dev_shared_pipeline.py --config configs/hk_dev_shared_rerun_config.json --data-root 'G:\我的云端硬盘\DeepEpromote\Drosophila' inspect
```

The source paths can be overridden on Colab or locally without editing the
versioned config.  Keep the JSON file itself under version control so the
parameter contract remains explicit.
