# Drosophila promoter paper — reproducibility release

This repository reproduces the paper's **frozen-input computational figures**
and provides a stable extension interface. It is organised around the final
manuscript story:

1. sequence models;
2. shared motif atlas;
3. task-specific motif deployment;
4. DeepISA motif-pair interactions;
5. model-guided sequence evolution.

## Three ways to use it

### Reader / reviewer

```bash
pip install -e .
drosophila-repro doctor
drosophila-repro list
```

### Reproduction user

```bash
pip install -e ".[figures]"
drosophila-repro reproduce --figure figure5
drosophila-repro reproduce --figure all --continue-on-error
drosophila-repro assets-verify
```

### Notebook quickstart

```bash
pip install -e ".[figures,notebook]"
jupyter lab notebooks/quickstart_figure5.ipynb
```

The notebook runs the same registered Figure 5 command as the CLI. It is a
small, inspectable starting point—not a separate, diverging workflow.

### HK_DEV_SHARED rerun quickstart

`notebooks/quickstart_hk_dev_shared_finemo_deepisa.ipynb` is the Colab
workflow for the post-freeze sharing extension: it reconstructs the Fig. 1
DEV∩HK 249-bp cohort, verifies the required counts and parameters before each
stage, and resumes Fi-NeMo/DeepISA from hash-checked checkpoints. Its scope is
separate from the frozen figure reproduction: S3 scans standalone HK/DEV
motifs across 17,380 sharing windows, while DeepISA uses the shared 24-bp
atlas only on the proximal/core cohort (12,260 HK/DEV; 4,178 observed CAGE).
See [the Colab contract](docs/HK_DEV_SHARED_COLAB_QUICKSTART.md).
All actual data, model and motif H5 files, attributions and outputs remain on
the project Google Drive; this repository contains only the complete scripts,
configuration and minimal data-free tests.

### Developer

```bash
drosophila-repro extension-init --name my_analysis
drosophila-repro registry-check
```

`doctor` discovers the release assets through `DROSOPHILA_REPRO_PACKAGE_ROOT`.
In the source archive it defaults to `../../reproducibility_package`; after
publication it defaults to the repository's `release_assets/` directory.
`assets-verify` additionally checks every final visual export and editable
source against the frozen SHA-256 inventory.

The GitHub Actions workflow tests the installable package on Python 3.10 and
3.12 without downloading multi-gigabyte assets. Full figure and visual-asset
verification runs after mounting the DOI/release archive.

## Reproduction scope

The quickstart regenerates figure panels from frozen local tables/results. It
does not claim to retrain the neural models, rerun TF-MoDISco/Fi-NeMo, or redo
the full greedy optimisation. Those expensive upstream computations are
represented by versioned inputs and documented provenance.

Manual conceptual panels are release assets, not Python outputs. The published
archive must contain their editable PPTX sources, exported panel files and the
final Figure 1–6 composites.

See [user roles](docs/USER_ROLES.md), [extension guide](docs/EXTENDING.md), and
the [acceptance protocol](docs/ACCEPTANCE_PROTOCOL.md).
