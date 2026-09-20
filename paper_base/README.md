# Paper analysis base

This directory is the source-only base for the full paper analysis.  It
preserves the executable layout of the working `reproducibility_package`:

- `01_model_build/` — DeepSTARR and DeepCAGE figure code;
- `02_sharing_motif/` and `03_motif_analysis/` — shared and task-specific
  motif analyses;
- `04_deepisa/` — figures plus the complete `Ep_ISA_NEW` implementation;
- `05_evolution/` — model-guided sequence-evolution analysis;
- `quickstart/` and `figure_generation/` — figure orchestration.

Only executable source, environment contracts and path configuration are in
Git.  No study data, trained models, motif H5 files, attributions, generated
figures or completed result tables are versioned here.  Those assets live on
the project Google Drive and are selected through the Drive-backed configs.

The new labelled rerun is implemented once in
`scripts/hk_dev_shared_pipeline.py`; the Colab runner merely clones this
repository and dispatches that command.  It must not duplicate cohort or
Fi-NeMo/DeepISA logic in notebook cells.
