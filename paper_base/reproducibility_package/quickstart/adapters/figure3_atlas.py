#!/usr/bin/env python3
"""Run the three shared-motif-atlas panels against package-local inputs."""

from __future__ import annotations

import runpy
import sys
import os
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[2]
MODULE = PACKAGE / "02_sharing_motif"
SCRIPTS = MODULE / "scripts"
sys.path.insert(0, str(SCRIPTS))

import figure3_paths  # noqa: E402


# Historical scripts used <project>/plot/data.  This release stores the same
# frozen inputs under the module itself.
figure3_paths.REPO_ROOT = MODULE
figure3_paths.CLEAN_DIR = MODULE / "data" / "clean"
figure3_paths.RAW_DIR = MODULE / "data" / "raw_reference"
_output_root = Path(os.environ["REPRO_OUTPUT_ROOT"]).resolve() / "figure3" if os.environ.get("REPRO_OUTPUT_ROOT") else MODULE
figure3_paths.POLISH_OUTPUT = _output_root / "figures"
figure3_paths.POLISH_QA = _output_root / "figures" / "qa"

for filename in ("fig3a_1_similarity.py", "fig3a_2_logos.py", "fig3a_3_task_distribution.py"):
    runpy.run_path(str(SCRIPTS / filename), run_name="__main__")
