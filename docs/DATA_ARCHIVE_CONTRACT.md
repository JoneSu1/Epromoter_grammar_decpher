# Data archive contract

The Git repository contains orchestration code, docs, tests, notebooks, and
small configuration. The frozen scientific release assets are a versioned data
archive because they contain multi-gigabyte tables, model files, attribution
arrays, and publication exports.

## Required archive layout

```text
reproducibility_package/
  01_model_build/
  02_sharing_motif/
  03_motif_analysis/
  04_deepisa/
  05_evolution/
  frozen_data/
    manuscript_visual_assets/
  quickstart/
  MANIFEST.md
```

## Tagging gate

Before a public GitHub tag, record the persistent DOI, archive SHA-256, archive
size, creation date, and compatibility version in the GitHub release notes and
in `release_assets/README.md`. Run `drosophila-repro doctor` against a freshly
downloaded archive before claiming reproducibility.

The local source archive is suitable for development testing but is not a
substitute for a deposited immutable archive.

## Build and verification

Create a ZIP64 archive with `scripts/build_release_archive.py`; do not add the
multi-gigabyte payload to Git. On a fresh download, extract the archive, set
`DROSOPHILA_REPRO_PACKAGE_ROOT` to its `reproducibility_package/` directory,
then run `drosophila-repro doctor` and `drosophila-repro assets-verify` before
running any figure. The latter verifies final Figure 1–6/S1–S5 exports and all
editable source files against the visual SHA-256 inventory.
