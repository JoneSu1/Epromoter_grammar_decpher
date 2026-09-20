#!/usr/bin/env python3
"""Run Fig. 3 promoter logos with the localized frozen logo inputs."""

from __future__ import annotations

import sys
import os
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[2]
MODULE = PACKAGE / "02_sharing_motif"
SCRIPTS = MODULE / "scripts"
FROZEN = PACKAGE / "frozen_data" / "sharing_motif_logos"
sys.path.insert(0, str(SCRIPTS))

import fig3e_core_promoter_logos as renderer  # noqa: E402


renderer.ROOT = MODULE
renderer.PLOT = MODULE
_output_root = Path(os.environ["REPRO_OUTPUT_ROOT"]).resolve() / "figure3" if os.environ.get("REPRO_OUTPUT_ROOT") else MODULE
renderer.OUT = _output_root / "figures"
renderer.QA = _output_root / "figures" / "qa" / "figure3e"
renderer.TRACE = _output_root / "figures" / "traces" / "figure3e_core_promoter_logo_trace.csv"
renderer.LOCAL_STAGE2 = MODULE / "data" / "raw_reference" / "stage2_core_promoter"
renderer.PROM_INPUT_INFO = FROZEN / "prom_input_info.tsv"
renderer.PROMOTER_PREDICTIONS = PACKAGE / "01_model_build" / "deepcage" / "data" / "DATA" / "PROMOTER_Dominant_Predictions_Splits.tsv"
renderer.INPUTS = {
    "HK": FROZEN / "finemo_input" / "HK" / "finemo_input.npz",
    "DEV": FROZEN / "finemo_input" / "DEV" / "finemo_input.npz",
    "CAGE": FROZEN / "finemo_input" / "CAGE_NEW" / "finemo_input.npz",
}
renderer.HITS = {
    "HK": renderer.LOCAL_STAGE2 / "HK" / "hits.tsv",
    "DEV": renderer.LOCAL_STAGE2 / "DEV" / "hits.tsv",
    "CAGE": renderer.LOCAL_STAGE2 / "CAGE_NEW" / "hits.tsv",
}
renderer.REGIONS = FROZEN / "finemo_input" / "HK" / "regions.bed"
renderer.PRED_H5 = {task: FROZEN / "attributions" / task / "attributions.h5" for task in ("HK", "DEV", "CAGE")}
renderer.SUBSET_REGIONS = {
    "HK-specific": FROZEN / "subset_regions" / "HK_only" / "regions.bed",
    "DEV-specific": FROZEN / "subset_regions" / "DEV_only" / "regions.bed",
    "Sharing": FROZEN / "subset_regions" / "Shared" / "regions.bed",
}

for path in [*renderer.INPUTS.values(), renderer.REGIONS, renderer.PROM_INPUT_INFO, *renderer.PRED_H5.values(), *renderer.SUBSET_REGIONS.values()]:
    if not path.exists():
        raise FileNotFoundError(path)

renderer.main()
