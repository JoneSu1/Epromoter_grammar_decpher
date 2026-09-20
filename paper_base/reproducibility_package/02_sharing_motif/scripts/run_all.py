#!/usr/bin/env python
"""Render all polished Shared_motif Figure 3 panels."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "fig3a_1_similarity.py",
    "fig3a_2_logos.py",
    "fig3a_3_task_distribution.py",
    "fig3b_tss_distance.py",
    "fig3c_density_complexity.py",
    "fig3d_contribution.py",
]


def main() -> None:
    here = Path(__file__).resolve().parent
    for script in SCRIPTS:
        print(f"\n== {script} ==")
        subprocess.run([sys.executable, str(here / script)], check=True, cwd=here)


if __name__ == "__main__":
    main()
