# Release assets mount point

This directory is intentionally not committed with the multi-gigabyte frozen
asset archive. For a public release, extract the DOI ZIP here or set:

```bash
export DROSOPHILA_REPRO_PACKAGE_ROOT=/path/to/reproducibility_package
```

The mounted directory must contain `01_model_build/` through `05_evolution/`,
`frozen_data/`, `quickstart/`, `MANIFEST.md`, and the frozen final visual
archive at `frozen_data/manuscript_visual_assets/`. Before rendering, run:

```bash
drosophila-repro doctor
drosophila-repro assets-verify
```

Build the DOI payload locally with:

```bash
python scripts/build_release_archive.py \
  --source /path/to/reproducibility_package \
  --output dist/drosophila-promoter-repro-assets-v0.1.0.zip
```

Publish the generated `.zip` and `.zip.sha256` together. The final DOI URL,
checksum, size, and package version must be inserted here before tagging.
