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
