#!/usr/bin/env python3
"""Package-local adapter for the Figure 5 distance/context renderer.

The archived renderer retains a historical directory spelling.  This adapter
only supplies the registered frozen CSV location; plotting and statistics stay
inside the canonical renderer unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[2] / "04_deepisa" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import fig4b_positive_negative_distance as renderer  # noqa: E402


renderer.SUMMARY = (
    Path(__file__).resolve().parents[2]
    / "04_deepisa"
    / "data"
    / "fig4_pos_neg"
    / "fig4_positive_negative_distance_summary.csv"
)

if not renderer.SUMMARY.exists():
    raise FileNotFoundError(renderer.SUMMARY)


if __name__ == "__main__":
    renderer.main()
