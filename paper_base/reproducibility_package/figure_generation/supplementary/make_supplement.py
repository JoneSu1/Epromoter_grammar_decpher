#!/usr/bin/env python3
"""
Entry-point wrappers for regenerating human-confirmed supplementary figures.

Same philosophy as main/make_figures.py: invoke canonical scripts without
changing scientific computations. Defers to documentation for fragile/G-drive
-dependent scripts.

Usage:
    python make_supplement.py sharing    # Sharing Motif figure3e_series2 (LEVEL B2)
    python make_supplement.py motif      # Motif Analysis S3 (LEVEL A)
    python make_supplement.py deepisa    # DeepISA S6a-e
    python make_supplement.py evolution  # Evolution S5/S6 (LEVEL A)
    python make_supplement.py --all
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TARGETS = {
    "sharing":   ("02_sharing_motif/scripts",     ["fig3e_series2_activity_contrast_supplement.py"]),
    "motif":     ("03_motif_analysis/scripts",    ["plot_density_complexity_S3.py", "plot_contribution_S3.py", "plot_motif_logo_rows_S3.py", "plot_finemo_sequence_examples_S3.py"]),
    "deepisa":   ("04_deepisa/scripts",           ["plot_fig6.py", "plot_fig6_supplement_support.py", "plot_position_aware_summary_split.py"]),
    "evolution": ("05_evolution/scripts",         ["plot_s5a_cohort_selection.py", "plot_s5b_highconf_activity.py", "plot_s5c_strict_residual_sensitivity.py", "plot_s6_sequence_realism_constraints.py"]),
}

def run(group):
    if group not in TARGETS:
        print(f"Unknown supplement group: {group}. Known: {list(TARGETS)}")
        return False
    mod_dir, scripts = TARGETS[group]
    scripts_dir = ROOT / mod_dir
    for s in scripts:
        script = scripts_dir / s
        if not script.exists():
            print(f"  SKIP (missing): {script}")
            continue
        print(f"  RUN: {script}")
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as e:
            print(f"    -> exited ({e.code})")
        except Exception as e:
            print(f"    -> ERROR: {type(e).__name__}: {e}")
            print(f"       May require historical absolute paths or G-drive assets.")
            print(f"       See external_dependencies/EXTERNAL_ASSETS.md.")
            return False
    return True

def main():
    args = sys.argv[1:]
    groups = list(TARGETS) if (not args or args[0] in ("--all","-a")) else args
    for g in groups:
        print(f"\n=== supplement: {g} ===")
        run(g)

if __name__ == "__main__":
    main()
