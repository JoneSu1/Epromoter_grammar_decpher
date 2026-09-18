# Release assets mount point

This directory is intentionally not committed with the multi-gigabyte frozen
asset archive. For a public release, download the DOI archive here or set:

```bash
export DROSOPHILA_REPRO_PACKAGE_ROOT=/path/to/reproducibility_package
```

The mounted directory must contain `01_model_build/` through `05_evolution/`,
`frozen_data/`, and `quickstart/`. Before rendering, run:

```bash
drosophila-repro doctor
```

The final DOI URL and SHA-256 release manifest must be inserted before tagging.
