#!/usr/bin/env python3
"""
Entry-point wrappers for regenerating human-confirmed manuscript figures.

Each make_figN() runs the canonical plotting script(s) for that manuscript
figure WITHOUT changing scientific computations. Wrappers only:
  - resolve the package-relative module path
  - invoke the canonical script via runpy (same as running it directly)

If a figure's canonical script is fragile to wrapping (e.g. hardcoded
historical absolute paths or G-drive deps), the wrapper defers to documentation
rather than forcing an unsafe call. See README.md in this folder.

Usage:
    python make_figures.py fig1      # DeepSTARR Fig1a/b/c
    python make_figures.py fig2      # DeepCAGE Fig2C/2F
    python make_figures.py fig3a     # Sharing Motif Fig3a-1/3a-2/3a-3/3e
    python make_figures.py fig3bcd   # Motif Analysis Fig3b/c/d
    python make_figures.py deepisa   # DeepISA Fig2/3d/3g/3h/4b
    python make_figures.py fig5      # Evolution Fig5b/d/e/f
    python make_figures.py --all
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # reproducibility_package/

# Map manuscript figure group -> (module scripts dir, [script filenames])
TARGETS = {
    "fig1":      ("01_model_build/deepstarr/scripts",  ["run_all.py"]),
    "fig2":      ("01_model_build/deepcage/scripts",   ["run_all.py"]),
    "fig3a":     ("02_sharing_motif/scripts",          ["run_all.py"]),
    "fig3bcd":   ("03_motif_analysis/scripts",         ["fig3b_tss_distance.py", "fig3c_density_complexity.py", "fig3d_contribution.py"]),
    "deepisa":   ("04_deepisa/scripts",                ["run_all.py"]),
    "fig5":      ("05_evolution/scripts",              ["plot_fig5_polish2.py", "plot_fig5e_2_marginal_bars.py"]),
}

def run(group):
    if group not in TARGETS:
        print(f"Unknown figure group: {group}. Known: {list(TARGETS)}")
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
            print(f"       This script may require historical absolute paths or G-drive assets.")
            print(f"       See README.md and external_dependencies/EXTERNAL_ASSETS.md.")
            return False
    return True

def main():
    args = sys.argv[1:]
    if not args or args[0] in ("--all", "-a"):
        groups = list(TARGETS)
    else:
        groups = args
    for g in groups:
        print(f"\n=== {g} ===")
        run(g)

if __name__ == "__main__":
    main()
