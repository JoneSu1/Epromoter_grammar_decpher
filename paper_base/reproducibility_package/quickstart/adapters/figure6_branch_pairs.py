#!/usr/bin/env python3
"""Run Figure 6 representative attribution plates from localized release data."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[2]
MODULE = PACKAGE / "05_evolution"
SCRIPTS = MODULE / "scripts"
FROZEN = PACKAGE / "frozen_data" / "evolution_fig5d"
sys.path.insert(0, str(SCRIPTS))

import plot_fig5_polished as base  # noqa: E402


# The original renderer's science is retained; only release storage locations
# replace historical plot_v1/G-drive locations.
base.POLISH_ROOT = MODULE
base.PLOT_V1_DATA = MODULE / "data" / "attribution"
base.ANNOTATED_ATTR_DIR = FROZEN / "attribution"
base.ANNOTATED_HITS_LONG = FROZEN / "summaries" / "finemo_hits_annotated_long.tsv"

import plot_fig5d_branch_pairs_full249 as renderer  # noqa: E402


for path in (base.PLOT_V1_DATA / "legacy_selected_10_metadata.tsv", base.ANNOTATED_HITS_LONG, *[base.ANNOTATED_ATTR_DIR / track / "attribution_arrays.npz" for track in ("HK", "DEV", "CAGE")]):
    if not path.exists():
        raise FileNotFoundError(path)

renderer.plot_branch_pairs_full249()
