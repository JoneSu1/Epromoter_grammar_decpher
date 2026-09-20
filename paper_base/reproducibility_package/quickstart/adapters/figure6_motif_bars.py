#!/usr/bin/env python3
"""Run the Figure 6 motif-gain bars with release-compatible series aliases."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[2] / "05_evolution" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import plot_fig5_polish2 as base  # noqa: E402


# This companion panel retained the earlier '-CAGE' labels while the current
# candidate renderer standardized them as '-DeepCAGE'.  Color identity is the
# only release-interface bridge required.
base.SERIES_COLORS.update(
    {
        "HK-CAGE": base.SERIES_COLORS["HK-DeepCAGE"],
        "DEV-CAGE": base.SERIES_COLORS["DEV-DeepCAGE"],
    }
)

import plot_fig5e_2_marginal_bars as renderer  # noqa: E402


renderer.main()
